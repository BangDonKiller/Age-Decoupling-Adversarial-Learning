import torch
import torch.nn as nn
import torch.nn.functional as F
from ..feature_extractor.speechbrain_model import SpeakerEmbeddingExtractor

class JFENetwork(nn.Module):
    def __init__(self, MODEL_ID, input_dim=192, spk_dim=128, age_dim=64, num_speakers=1000, num_age_groups=7):
        super(JFENetwork, self).__init__()
        
        self.spk_dim = spk_dim
        self.age_dim = age_dim

        # 1. 骨幹網路 (Backbone Network)
        self.backbone = SpeakerEmbeddingExtractor(
            model_id=MODEL_ID,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.backbone.eval()  # 骨幹網路不進行訓練
        self.backbone.requires_grad_(False)
        
        # 2. 全連接神經層
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, spk_dim + age_dim) # 輸出層被切分為 h1 和 h2
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(spk_dim + age_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )
        
        # 3. 分類器 (Main Tasks & Subtasks)
        # 說話者分類器
        self.classifier_spkr = nn.Sequential(
            nn.Linear(spk_dim, spk_dim//2),
            nn.ReLU(),
            nn.Linear(spk_dim//2, num_speakers)
        )
        # 年齡分類器 (將年齡視為分類任務，例如 10-20歲, 20-30歲...)
        self.classifier_age = nn.Sequential(
            nn.Linear(age_dim, age_dim//2),
            nn.ReLU(),
            nn.Linear(age_dim//2, num_age_groups)
        )

    def forward(self, x, mode):
        with torch.no_grad():
            feature = self.backbone(x)
            
        latent = self.encoder(feature)
            
        # --- 提取 Embedding Vectors ---
        # 1. 提取說話者向量 w_spkr
        h_spk = latent[:, :self.spk_dim] 
        
        # 2. 提取年齡向量 w_age (即論文中的 w_nuis)
        h_age = latent[:, self.spk_dim:]
        
        if mode != "train":
            return {
                "spkr_emb": feature,
                "w_spkr": h_spk,
                "w_age": h_age,
            } 
        
        # --- 進行分類 (Cross-Classification) ---
        # 這裡會產生四種 logits，對應論文 Table 1
        
        # A. 主任務 (Main Tasks)
        # w_spkr -> 預測 說話者
        logits_spkr_main = self.classifier_spkr(h_spk)
        # w_age -> 預測 年齡
        logits_age_main = self.classifier_age(h_age)
        
        # B. 次任務 (Subtasks) - 用於計算 Entropy
        # w_spkr -> 預測 年齡 (希望這個很不準)
        logits_age_sub = self.classifier_age(h_spk)
        # w_age -> 預測 說話者 (希望這個很不準)
        logits_spkr_sub = self.classifier_spkr(h_age)
        
        # C. 重建輸出 (可選)
        latent_combined = torch.cat((h_spk, h_age), dim=1)
        x_recon = self.decoder(latent_combined)
        
        return {
            "spkr_emb": feature,
            "w_spkr": h_spk,
            "w_age": h_age,
            "logits_spkr_main": logits_spkr_main,
            "logits_age_main": logits_age_main,
            "logits_age_sub": logits_age_sub,
            "logits_spkr_sub": logits_spkr_sub,
            "x_recon": x_recon
        }

class JFELoss(nn.Module):
    def __init__(self, lambda_entropy=0.1, lambda_mapc=0.1, lambda_recon=1.0, lambda_hsic=1.0, lambda_gr=0.01):
        super(JFELoss, self).__init__()
        self.lambda_entropy = lambda_entropy
        self.lambda_mapc = lambda_mapc
        self.lambda_recon = lambda_recon
        self.lambda_hsic = lambda_hsic
        self.lambda_gr = lambda_gr
        self.ce_loss_spkr = nn.CrossEntropyLoss()
        self.ce_loss_age = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def compute_entropy(self, logits):
        """
        計算 Entropy。
        公式: H(p) = - sum(p * log(p))
        我們希望最大化 Entropy，也就是最小化 -Entropy。
        但在論文公式 (19) 中是減去 Entropy Loss，所以我們這裡返回 H(p)。
        """
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        entropy = -torch.sum(probs * log_probs, dim=1).mean()
        return entropy

    def compute_mapc(self, v1, v2):
        """
        計算 MAPC (Mean Absolute Pearson's Correlation) -> 論文公式 (18)
        計算兩個 Embedding 向量在 Batch 維度上的相關性。
        """
        # v1, v2 shape: (Batch, Dim)
        
        # 1. 去中心化 (Subtract Mean)
        v1_mean = v1 - v1.mean(dim=0, keepdim=True)
        v2_mean = v2 - v2.mean(dim=0, keepdim=True)
        
        # 2. 計算標準差 (Std), 加上 epsilon 防止除以 0
        v1_std = v1.std(dim=0, keepdim=True) + 1e-8
        v2_std = v2.std(dim=0, keepdim=True) + 1e-8
        
        # 3. 計算 Covariance
        # (Batch, Dim) * (Batch, Dim) -> sum over Batch -> (Dim)
        # 除以 (Batch_Size - 1) 得到協方差，但因為分子分母都會除，可以省略
        covariance = (v1_mean * v2_mean).mean(dim=0)
        
        # 4. 計算 Correlation
        correlation = covariance / (v1_std.squeeze() * v2_std.squeeze())
        
        # 5. 取絕對值並平均
        mapc = torch.abs(correlation).mean()
        
        return mapc
    
    def compute_hsic(self, x, y):
        """
        新增：計算 HSIC (Hilbert-Schmidt Independence Criterion)
        作為互資訊 (MI) 的非線性代理損失。
        x: 身分嵌入 [Batch, 256]
        y: 年齡標籤 [Batch]
        """
        # 1. 準備數據
        if y.dim() == 1:
            y = y.view(-1, 1).float()
        
        n = x.size(0)
        
        # 2. 計算 RBF 核矩陣 (Similarity Matrices)
        def rbf_kernel(mat):
            dist = torch.pdist(mat).pow(2)
            sigma = torch.median(dist) # 使用中位數技巧自動調整頻寬
            k_mat = torch.exp(-dist / (2 * sigma + 1e-8))
            # 這裡簡化為直接矩陣運算
            dists = torch.cdist(mat, mat).pow(2)
            return torch.exp(-dists / (2 * sigma + 1e-8))

        K = rbf_kernel(x)
        L = rbf_kernel(y)

        # 3. 中心化矩陣 H = I - (1/n)11^T
        H = torch.eye(n).to(x.device) - (1.0 / n) * torch.ones((n, n)).to(x.device)

        # 4. HSIC = trace(KHLH) / (n-1)^2
        # 我們希望最小化 HSIC
        hsic = torch.trace(K @ H @ L @ H) / ((n - 1) ** 2)
        return hsic

    def forward(self, outputs, target_spkr, target_age):
        """
        outputs: JFENetwork 的輸出字典
        target_spkr: 說話者真實標籤
        target_age: 年齡真實標籤
        """
        
        # 1. Discriminative Losses (公式 14, 15) - 越小越好
        loss_spkr_main = self.ce_loss_spkr(outputs['logits_spkr_main'], target_spkr)
        loss_age_main = self.ce_loss_age(outputs['logits_age_main'], target_age)
        
        # 2. Entropy Losses (公式 16, 17) - 越大越好
        # 注意：論文公式 (19) 是減去這些 Loss。
        # 這裡我們計算出 entropy 值，稍後在 total loss 做減法
        entropy_age_sub = self.compute_entropy(outputs['logits_age_sub'])     # w_spkr 猜年齡的困惑度
        entropy_spkr_sub = self.compute_entropy(outputs['logits_spkr_sub'])   # w_age 猜人的困惑度
        
        # 3. MAPC Loss (公式 18) - 越小越好 (我們希望相關性是 0)
        # 論文寫 "Negative MAPC based disentanglement losses"，並在公式19用減號
        # 實際上目標是 Minimize MAPC。
        # 為了方便優化器，我們直接加上 MAPC term。
        loss_mapc = self.compute_mapc(outputs['w_spkr'], outputs['w_age'])
        
        # 4. Reconstruction Loss (可選)
        loss_recon = self.mse_loss(outputs['x_recon'], outputs['spkr_emb'])
        
        # 6. Total Loss (公式 19 的變體)
        # Minimize: Main_CE + lambda * MAPC - lambda * Entropy + lambda * Causal + lambda * Recon
        total_loss = (loss_spkr_main + loss_age_main) \
                     + (self.lambda_mapc * loss_mapc) \
                     - (self.lambda_entropy * (entropy_age_sub + entropy_spkr_sub)) \
                     + (self.lambda_recon * loss_recon)
                     
        return total_loss, {
            "loss_spkr": loss_spkr_main.item(),
            "loss_age": loss_age_main.item(),
            "entropy_age": entropy_age_sub.item(),
            "entropy_spkr": entropy_spkr_sub.item(),
            "mapc": loss_mapc.item(),
            "loss_recon": loss_recon.item(),
        }

# --- 模擬數據與測試 ---
if __name__ == "__main__":
    # 設定參數
    BATCH_SIZE = 32
    NUM_SPEAKERS = 100
    NUM_AGE_CLASSES = 5  # 假設將年齡分為 5 個區間 (例如: <20, 20-30, 30-40, 40-50, >50)
    INPUT_DIM = 30       # MFCC 維度
    TIME_STEPS = 200     # 語音長度
    EMBED_DIM = 256

    # 1. 建立模型與 Loss
    model = JFENetwork(NUM_SPEAKERS, NUM_AGE_CLASSES, INPUT_DIM, EMBED_DIM)
    criterion = JFELoss(lambda_entropy=0.1, lambda_mapc=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 2. 產生假數據
    dummy_input = torch.randn(BATCH_SIZE, INPUT_DIM, TIME_STEPS) # (32, 30, 200)
    dummy_target_spkr = torch.randint(0, NUM_SPEAKERS, (BATCH_SIZE,))
    dummy_target_age = torch.randint(0, NUM_AGE_CLASSES, (BATCH_SIZE,))

    # 3. 訓練步驟範例
    model.train()
    
    # Forward
    outputs = model(dummy_input)
    
    # Calculate Loss
    loss, loss_dict = criterion(outputs, dummy_target_spkr, dummy_target_age)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 4. 輸出結果檢查
    print(f"Total Loss: {loss.item():.4f}")
    print("詳細 Loss 組成:")
    print(f"  - Speaker CE (Main): {loss_dict['loss_spkr']:.4f} (需下降)")
    print(f"  - Age CE (Main):     {loss_dict['loss_age']:.4f} (需下降)")
    print(f"  - MAPC:              {loss_dict['mapc']:.4f} (需下降)")
    print(f"  - Entropy (Age Sub): {loss_dict['entropy_age']:.4f} (需上升)")
    
    # 檢查 Embedding 形狀
    print(f"\nEmbedding Shape w_spkr: {outputs['w_spkr'].shape}")
    print(f"Embedding Shape w_age: {outputs['w_age'].shape}")
    print("測試完成。")