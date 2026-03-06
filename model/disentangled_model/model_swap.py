import torch
import torch.nn as nn
import torch.nn.functional as F
from ..feature_extractor.speechbrain_model import SpeakerEmbeddingExtractor

class JFENetworkSwap(nn.Module):
    def __init__(self, MODEL_ID, input_dim=192, spk_dim=256, age_dim=256, num_speakers=5990, num_age_groups=7):
        super(JFENetworkSwap, self).__init__()
        
        self.spk_dim = spk_dim
        self.age_dim = age_dim

        self.backbone = SpeakerEmbeddingExtractor(
            model_id=MODEL_ID,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.backbone.eval()
        self.backbone.requires_grad_(False)
        
        # Encoder: (Feature + Gender) -> Latent
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + 1, 512), 
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, spk_dim + age_dim)
        )
        
        # Decoder: Latent -> Feature Reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(spk_dim + age_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )
        
        self.classifier_spkr = nn.Sequential(
            nn.Linear(spk_dim, spk_dim//2),
            nn.ReLU(),
            nn.Linear(spk_dim//2, num_speakers)
        )
        self.classifier_age = nn.Sequential(
            nn.Linear(age_dim, age_dim//2),
            nn.ReLU(),
            nn.Linear(age_dim//2, num_age_groups)
        )

    def forward_encoder(self, feature, gender):
        """輔助函式：讓 feature 重新進入 encoder (用於 Swap 流程)"""
        feat_gen = torch.cat((feature, gender.unsqueeze(1).float()), dim=1)
        latent = self.encoder(feat_gen)
        return latent[:, :self.spk_dim], latent[:, self.spk_dim:]

    def forward(self, x, gender, mode="train"):
        with torch.no_grad():
            feature = self.backbone(x)
        
        h_spk, h_age = self.forward_encoder(feature, gender)
        
        if mode != "train":
            return {"spkr_emb": feature, "w_spkr": h_spk, "w_age": h_age} 
        
        # 基本分類
        logits_spkr_main = self.classifier_spkr(h_spk)
        logits_age_main = self.classifier_age(h_age)
        
        # 洩漏檢測 (對抗/熵用)
        logits_age_sub = self.classifier_age(h_spk)
        logits_spkr_sub = self.classifier_spkr(h_age)
        
        # 原始重構
        x_recon = self.decoder(torch.cat((h_spk, h_age), dim=1))
        
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

class JFELossSwap(nn.Module):
    def __init__(self, lambda_entropy=0.1, lambda_recon=1.0, lambda_ortho=0.1, lambda_swap=0.5, lambda_emb_recon=0.5):
        super(JFELossSwap, self).__init__()
        self.lambda_entropy = lambda_entropy
        self.lambda_recon = lambda_recon
        self.lambda_ortho = lambda_ortho
        self.lambda_swap = lambda_swap
        self.lambda_emb_recon = lambda_emb_recon
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def compute_entropy(self, logits):
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        return -torch.sum(probs * log_probs, dim=1).mean()

    def compute_orthogonal_loss(self, z_id, z_age):
        z_id_n = F.normalize(z_id, p=2, dim=1)
        z_age_n = F.normalize(z_age, p=2, dim=1)
        return torch.pow(torch.sum(z_id_n * z_age_n, dim=1), 2).mean()

    def forward(self, outputs, target_spkr, target_age, swap_results=None, target_age_swapped=None, 
                h_spk_reswap=None, h_age_reswap=None, h_spk_orig=None, h_age_orig=None):
        # 1. 基礎任務 Loss
        loss_spkr = self.ce_loss(outputs['logits_spkr_main'], target_spkr)
        loss_age = self.ce_loss(outputs['logits_age_main'], target_age)
        
        # 2. 洩漏懲罰 (Entropy 越大代表洩漏越少)
        ent_age = self.compute_entropy(outputs['logits_age_sub'])
        ent_spk = self.compute_entropy(outputs['logits_spkr_sub'])
        
        # 3. 幾何約束
        # loss_ortho = self.compute_orthogonal_loss(outputs['w_spkr'], outputs['w_age'])
        loss_ortho = 0.0
        loss_recon = self.mse_loss(outputs['x_recon'], outputs['spkr_emb'])
        
        # 4. Embedding 重建損失 (特徵交換後的循環一致性)
        loss_emb_spk = 0
        loss_emb_age = 0
        if h_spk_reswap is not None and h_spk_orig is not None:
            # h_spk 經過交換和重建後應保持相似 (spk 維度不變)
            loss_emb_spk = self.mse_loss(h_spk_reswap, h_spk_orig)
        
        if h_age_reswap is not None and h_age_orig is not None:
            # h_age 經過交換和重建後可能有偏差，但應該相对接近
            loss_emb_age = self.mse_loss(h_age_reswap, h_age_orig)
        
        loss_emb_recon = loss_emb_spk + loss_emb_age
        
        # 5. 核心創新：Swap 循環一致性 Loss
        loss_swap = 0
        if swap_results is not None:
            # 重構出的特徵必須保持原來的身份
            l_swap_spk = self.ce_loss(swap_results['logits_spkr'], target_spkr)
            # 重構出的特徵必須符合新換過來的年齡
            l_swap_age = self.ce_loss(swap_results['logits_age'], target_age_swapped)
            loss_swap = l_swap_spk + l_swap_age

        total_loss = (loss_spkr + loss_age) \
                     - (self.lambda_entropy * (ent_age + ent_spk)) \
                     + (self.lambda_recon * loss_recon) \
                     + (self.lambda_ortho * loss_ortho) \
                     + (self.lambda_swap * loss_swap) \
                     + (self.lambda_emb_recon * loss_emb_recon)
                     
        return total_loss, {
            "loss_spkr": loss_spkr.item(), 
            "loss_age": loss_age.item(), 
            "loss_entropy": (ent_age + ent_spk).item(),
            "loss_recon": loss_recon.item(), 
            "loss_ortho": loss_ortho.item(),
            "loss_swap": loss_swap.item() if isinstance(loss_swap, torch.Tensor) else 0,
            "loss_emb_spk": loss_emb_spk.item() if isinstance(loss_emb_spk, torch.Tensor) else 0,
            "loss_emb_age": loss_emb_age.item() if isinstance(loss_emb_age, torch.Tensor) else 0,
            "loss_emb_recon": loss_emb_recon.item() if isinstance(loss_emb_recon, torch.Tensor) else 0
        }