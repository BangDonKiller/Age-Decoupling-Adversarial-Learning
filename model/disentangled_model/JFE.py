# JFE.py 完整代碼
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
        
        # 2. 全連接神經層 (更新：輸入維度 + 1 以容納性別特徵)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + 1, 512), 
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

    def forward(self, x, gender, mode):
        with torch.no_grad():
            feature = self.backbone(x)
            
        # --- 性別特徵化 (Gender as Feature) ---
        # 將性別資訊與原始特徵拼接，餵入 encoder
        # gender shape: (Batch), 需轉換為 (Batch, 1) 後拼接
        feature_with_gender = torch.cat((feature, gender.unsqueeze(1).float()), dim=1)
            
        latent = self.encoder(feature_with_gender)
            
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
    def __init__(self, lambda_entropy=0.1, lambda_mapc=0.0, lambda_recon=1.0, lambda_ortho=0.1):
        super(JFELoss, self).__init__()
        self.lambda_entropy = lambda_entropy
        self.lambda_mapc = lambda_mapc
        self.lambda_recon = lambda_recon
        self.lambda_ortho = lambda_ortho
        self.ce_loss_spkr = nn.CrossEntropyLoss()
        self.ce_loss_age = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def compute_entropy(self, logits):
        probs = F.softmax(logits, dim=1); log_probs = F.log_softmax(logits, dim=1)
        return -torch.sum(probs * log_probs, dim=1).mean()

    def compute_mapc(self, v1, v2):
        v1_m = v1 - v1.mean(dim=0, keepdim=True); v2_m = v2 - v2.mean(dim=0, keepdim=True)
        v1_s = v1.std(dim=0, keepdim=True) + 1e-8; v2_s = v2.std(dim=0, keepdim=True) + 1e-8
        corr = (v1_m * v2_m).mean(dim=0) / (v1_s.squeeze() * v2_s.squeeze())
        return torch.abs(corr).mean()

    def compute_orthogonal_loss(self, z_id, z_age):
        """實作教授建議的正交損失"""
        z_id_n = F.normalize(z_id, p=2, dim=1); z_age_n = F.normalize(z_age, p=2, dim=1)
        return torch.pow(torch.sum(z_id_n * z_age_n, dim=1), 2).mean()

    def forward(self, outputs, target_spkr, target_age):
        loss_spkr_main = self.ce_loss_spkr(outputs['logits_spkr_main'], target_spkr)
        loss_age_main = self.ce_loss_age(outputs['logits_age_main'], target_age)
        entropy_age_sub = self.compute_entropy(outputs['logits_age_sub'])
        entropy_spkr_sub = self.compute_entropy(outputs['logits_spkr_sub'])
        loss_mapc = self.compute_mapc(outputs['w_spkr'], outputs['w_age'])
        loss_recon = self.mse_loss(outputs['x_recon'], outputs['spkr_emb'])
        loss_ortho = self.compute_orthogonal_loss(outputs['w_spkr'], outputs['w_age'])
        
        total_loss = (loss_spkr_main + loss_age_main) \
                     + (self.lambda_mapc * loss_mapc) \
                     - (self.lambda_entropy * (entropy_age_sub + entropy_spkr_sub)) \
                     + (self.lambda_recon * loss_recon) \
                     + (self.lambda_ortho * loss_ortho)
                     
        return total_loss, {
            "loss_spkr": loss_spkr_main.item(), "loss_age": loss_age_main.item(),
            "entropy_age": entropy_age_sub.item(), "entropy_spkr": entropy_spkr_sub.item(),
            "mapc": loss_mapc.item(), "loss_recon": loss_recon.item(), "loss_ortho": loss_ortho.item()
        }