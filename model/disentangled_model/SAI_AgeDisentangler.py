# the model structure is modified based on the paper: https://arxiv.org/abs/1911.00940 
import torch
import torch.nn as nn
import torch.nn.functional as F

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class Disentangler(nn.Module):
    def __init__(self, input_dim=192, spk_dim=128, age_dim=64, num_spks=1000, num_age_groups=6):
        super(Disentangler, self).__init__()
        
        # --- 1. Encoder (論文中的 Enc) ---
        # 負責將原始 Embedding 轉換並壓縮到潛在空間
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 160),
            nn.BatchNorm1d(160),
            nn.ReLU(),
            nn.Linear(160, spk_dim + age_dim) # 輸出層被切分為 h1 和 h2
        )
        self.spk_dim = spk_dim
        self.age_dim = age_dim
        
        # --- 2. Decoder (論文中的 Dec) ---
        # 負責重建原始 Embedding，確保 h1+h2 保留了完整資訊
        self.decoder = nn.Sequential(
            nn.Linear(spk_dim + age_dim, 160),
            nn.ReLU(),
            nn.Linear(160, input_dim)
        )
        
        # --- 3. Speaker Classifier (主任務) ---
        # 確保 h1 (h_spk) 還是能認出是誰
        self.spk_classifier = nn.Linear(spk_dim, num_spks)
        
        # --- 4. Age Classifier (輔助任務) ---
        # 確保 h2 (h_age) 能準確預測年齡
        self.age_classifier = nn.Linear(age_dim, num_age_groups)
        
        # --- 5. Age Adversary (對抗任務) ---
        # 試圖從 h1 (h_spk) 中猜出年齡 (這是我們要欺騙的對象)
        self.age_adversary = nn.Sequential(
            nn.Linear(spk_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_age_groups)
        )

    def forward(self, x, alpha=1.0):
        # x: [Batch, 192] (ECAPA-TDNN 的原始 Embedding)
        
        # Step A: Encode & Split
        latent = self.encoder(x)
        
        # 根據維度切分向量
        # h1: 說話者特徵 (我們希望它不含年齡)
        # h2: 年齡特徵 (我們希望它只含年齡)
        h_spk = latent[:, :self.spk_dim] 
        h_age = latent[:, self.spk_dim:] 
        
        # Step B: Reconstruction (自我監督)
        # 確保 h_spk + h_age 能還原回 x
        # 這保證了資訊沒有憑空消失，只是被搬運了
        x_recon = self.decoder(latent)
        
        # Step C: Downstream Tasks
        # 1. 預測說話者 (用 h_spk)
        spk_pred = self.spk_classifier(h_spk)
        
        # 2. 預測年齡 (用 h_age) -> 強迫 h_age 吸收年齡資訊
        age_pred_from_h2 = self.age_classifier(h_age)
        
        # Step D: Adversarial Attack (用 h_spk)
        # 對 h_spk 應用梯度反轉，然後嘗試預測年齡
        # 如果這個分類器訓練失敗，代表 h_spk 成功移除了年齡
        h_spk_reversed = GradientReversal.apply(h_spk, alpha)
        age_pred_from_h1 = self.age_adversary(h_spk_reversed)
        
        return x_recon, spk_pred, age_pred_from_h2, age_pred_from_h1, h_spk, h_age
    
    def correlation_loss(self, h1, h2):
        """
        h1: [Batch, D1] (e.g., Speaker features, 128)
        h2: [Batch, D2] (e.g., Age features, 64)
        """
        # 1. 批次歸一化 (Batch Normalization logic) - 減去平均值
        h1_centered = h1 - h1.mean(dim=0, keepdim=True)
        h2_centered = h2 - h2.mean(dim=0, keepdim=True)
        
        # 2. L2 正規化 (讓標準差為 1)
        h1_norm = F.normalize(h1_centered, p=2, dim=0)
        h2_norm = F.normalize(h2_centered, p=2, dim=0)
        
        # 3. 計算互相關矩陣 (Cross-Correlation Matrix)
        # 矩陣乘法: [D1, Batch] x [Batch, D2] -> [D1, D2]
        corr_matrix = torch.mm(h1_norm.t(), h2_norm)
        
        # 4. Loss = 所有相關係數的平方平均 (Square makes it stronger than abs)
        loss = torch.mean(corr_matrix ** 2)
        
        return loss