import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
from feature_extractor.ecapa_tdnn import SpeakerEmbeddingExtractor


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self, lambd=1.0):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambd)

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        # x shape: (Batch, Channels, 1) for Conv1d logic
        batch_size, channels, length = x.size()
        channels_per_group = channels // self.groups
        x = x.view(batch_size, self.groups, channels_per_group, length)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batch_size, -1, length)
        return x

class RFRBEncoder(nn.Module):
    """
    殘差特徵細化塊 (Shared Encoder)
    """
    def __init__(self, input_dim=192, hidden_dim=256, groups=8):
        super().__init__()
        self.ln1 = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.silu = nn.SiLU()
        
        # 使用 Conv1d 來實作 Grouped Linear 以節省參數並優化硬體效率
        self.grouped_linear = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, groups=groups)
        self.shuffle = ChannelShuffle(groups)
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # 1. 標準化與投影
        res = self.proj(self.ln1(x)) # (B, 256)
        x = self.silu(res)
        
        # 2. 轉換為 1D 卷積格式進行分組處理 (B, C, L=1)
        x = x.unsqueeze(-1)
        x = self.grouped_linear(x)
        x = self.shuffle(x)
        x = x.squeeze(-1)
        
        # 3. 殘差連接與最終標準化
        x = self.ln2(x + res)
        return x

class GenderPredictor(nn.Module):
    """
    性別感知器：從 ECAPA Embedding 預測性別
    """
    def __init__(self, input_dim=192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2), # 輸出男女機率 (Softmax)
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)

class HyperGating(nn.Module):
    """
    超網路門控 (FiLM): 根據性別動態生成 Z_bio 的調製參數
    """
    def __init__(self, gender_dim=2, feature_dim=256):
        super().__init__()
        # 超網路：極小，產生 Gamma 與 Beta
        self.hyper = nn.Sequential(
            nn.Linear(gender_dim, 16),
            nn.ReLU(),
            nn.Linear(16, feature_dim * 2) 
        )
        self.feature_dim = feature_dim

    def forward(self, gender_prob, x):
        params = self.hyper(gender_prob) # (B, 512)
        gamma, beta = torch.split(params, self.feature_dim, dim=-1)
        # FiLM 調製公式: h' = h * γ + β
        return x * gamma + beta

class G_AIDA(nn.Module):
    def __init__(self, model_id, input_dim=192, zid_dim=128, zbio_dim=64, num_speakers=5990, num_age_groups=7):
        super().__init__()
        
        self.speaker_model = SpeakerEmbeddingExtractor(
            model_id=model_id,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        self.speaker_model.eval()  # 固定 ECAPA-TDNN 的權重，不參與訓練
        self.speaker_model.requires_grad_(False)
        
        # 1. Shared Encoder (RFRB)
        self.shared_encoder = RFRBEncoder(input_dim=input_dim, hidden_dim=256)
        self.shared_dim = 256
        self.half_shared_dim = self.shared_dim // 2
        
        # 2. Gender Self-Sensing
        self.gender_predictor = GenderPredictor(input_dim=input_dim)
        
        # 3. Hyper-Gating
        self.hyper_gating = HyperGating(gender_dim=2, feature_dim=self.half_shared_dim)
        
        # 4. VAE Heads (輸出兩組均值與標準差)
        # Z_id: 身分特徵 (不受性別調製)
        self.fc_mu_id = nn.Linear(self.half_shared_dim, zid_dim)
        self.fc_logvar_id = nn.Linear(self.half_shared_dim, zid_dim)
        
        # Z_bio: 生物特徵 (受性別調製)
        self.fc_mu_bio = nn.Linear(self.half_shared_dim, zbio_dim)
        self.fc_logvar_bio = nn.Linear(self.half_shared_dim, zbio_dim)
        
        # 5. Placeholder Decoder (暫時替代)
        self.decoder = nn.Sequential(
            nn.Linear(zid_dim + zbio_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

        self.classifier_id = nn.Sequential(
            nn.Linear(zid_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_speakers)
        )

        self.classifier_age = nn.Sequential(
            nn.Linear(zbio_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_age_groups)
        )

        self.grl_id_to_age = GradientReversal(lambd=1.0)
        self.grl_age_to_id = GradientReversal(lambd=1.0)

        self.adv_classifier_age_from_id = nn.Sequential(
            nn.Linear(zid_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_age_groups)
        )

        self.adv_classifier_id_from_age = nn.Sequential(
            nn.Linear(zbio_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_speakers)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.speaker_model(x) # 提取 ECAPA-TDNN 說話者嵌入 (B, 192)
        
        # A. 性別自感知
        gender_prob = self.gender_predictor(x)
        
        # B. Shared Encoding
        h_shared = self.shared_encoder(x)
        h_id, h_age = torch.chunk(h_shared, 2, dim=-1)
        
        # C. 門控調製 (只針對 Z_bio 的路徑)
        h_bio = self.hyper_gating(gender_prob, h_age)
        
        # D. 生成 Z_id 分佈
        mu_id = self.fc_mu_id(h_id)
        logvar_id = self.fc_logvar_id(h_id)
        z_id = self.reparameterize(mu_id, logvar_id)
        
        # E. 生成 Z_bio 分佈
        mu_bio = self.fc_mu_bio(h_bio)
        logvar_bio = self.fc_logvar_bio(h_bio)
        z_bio = self.reparameterize(mu_bio, logvar_bio)
        
        # F. 解碼還原 (Placeholder)
        z_combined = torch.cat([z_id, z_bio], dim=-1)
        recon_x = self.decoder(z_combined)

        logits_id = self.classifier_id(z_id)
        logits_age = self.classifier_age(z_bio)

        logits_adv_age_from_id = self.adv_classifier_age_from_id(self.grl_id_to_age(z_id))
        logits_adv_id_from_age = self.adv_classifier_id_from_age(self.grl_age_to_id(z_bio))
        
        return {
            "recon_x": recon_x,
            "spkr_emb": x,
            "mu_id": mu_id, "logvar_id": logvar_id,
            "mu_bio": mu_bio, "logvar_bio": logvar_bio,
            "gender_prob": gender_prob,
            "z_id": z_id,
            "z_bio": z_bio,
            "logits_id": logits_id,
            "logits_age": logits_age,
            "logits_adv_age_from_id": logits_adv_age_from_id,
            "logits_adv_id_from_age": logits_adv_id_from_age
        }

class G_AIDA_Loss(nn.Module):
    def __init__(
        self,
        lambda_recon=1.0,
        lambda_kl_id=1.0,
        lambda_kl_bio=1.0,
        lambda_mi=0.0,
        lambda_gender=0.1,
        lambda_spk=1.0,
        lambda_age=1.0,
        lambda_adv_age=0.5,
        lambda_adv_spk=0.5,
    ):
        super().__init__()
        self.lambda_recon = lambda_recon
        self.base_lambda_kl_id = lambda_kl_id
        self.base_lambda_kl_bio = lambda_kl_bio
        self.base_lambda_mi = lambda_mi
        self.lambda_kl_id = lambda_kl_id
        self.lambda_kl_bio = lambda_kl_bio
        self.lambda_mi = lambda_mi
        self.lambda_gender = lambda_gender
        self.lambda_spk = lambda_spk
        self.lambda_age = lambda_age
        self.lambda_adv_age = lambda_adv_age
        self.lambda_adv_spk = lambda_adv_spk

        self.mse_loss = nn.MSELoss()
        self.nll_loss = nn.NLLLoss()
        self.ce_loss = nn.CrossEntropyLoss()

    # def set_kl_mi_warmup_factor(self, warmup_factor):
    #     warmup_factor = float(max(0.0, min(1.0, warmup_factor)))
    #     self.lambda_kl_id = self.base_lambda_kl_id * warmup_factor
    #     self.lambda_kl_bio = self.base_lambda_kl_bio * warmup_factor
    #     self.lambda_mi = self.base_lambda_mi * warmup_factor

    def compute_kl(self, mu, logvar, prior_mu=None):
        """
        KL(q(z|x)||p(z))，其中 p(z)=N(prior_mu, I)
        若 prior_mu=None，則預設 prior_mu=0。
        """
        if prior_mu is None:
            prior_mu = torch.zeros_like(mu)

        return 0.5 * torch.mean(
            torch.sum((mu - prior_mu).pow(2) + logvar.exp() - 1.0 - logvar, dim=1)
        )

    def compute_mutual_information_gaussian(self, z1, z2, eps=1e-6):
        """
        以 joint Gaussian 近似估計 I(z1; z2):
        I = 0.5 * log( |Σ11| |Σ22| / |Σ_joint| )
        z1, z2 shape: (B, D1), (B, D2)
        """
        if z1.dim() != 2 or z2.dim() != 2:
            raise ValueError("z1 與 z2 必須為 2D tensor，形狀為 (Batch, Dim)")

        if z1.size(0) != z2.size(0):
            raise ValueError("z1 與 z2 的 batch size 必須相同")

        batch_size = z1.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=z1.device, dtype=z1.dtype)

        z = torch.cat([z1, z2], dim=1)
        z = z - z.mean(dim=0, keepdim=True)

        cov = (z.T @ z) / (batch_size - 1)
        d1 = z1.size(1)
        d2 = z2.size(1)

        eye1 = torch.eye(d1, device=z1.device, dtype=z1.dtype)
        eye2 = torch.eye(d2, device=z1.device, dtype=z1.dtype)
        eye_joint = torch.eye(d1 + d2, device=z1.device, dtype=z1.dtype)

        cov11 = cov[:d1, :d1] + eps * eye1
        cov22 = cov[d1:, d1:] + eps * eye2
        cov_joint = cov + eps * eye_joint

        _, logdet11 = torch.linalg.slogdet(cov11)
        _, logdet22 = torch.linalg.slogdet(cov22)
        _, logdet_joint = torch.linalg.slogdet(cov_joint)

        mi = 0.5 * (logdet11 + logdet22 - logdet_joint)
        return torch.clamp(mi, min=0.0)

    def forward(self, outputs, target_spk=None, target_age=None, target_gender=None):
        loss_recon = self.mse_loss(outputs["recon_x"], outputs["spkr_emb"])

        loss_kl_id = self.compute_kl(outputs["mu_id"], outputs["logvar_id"])

        target_spk_cls = None
        if target_spk is not None:
            target_spk_cls = target_spk.long().view(-1)

        target_gender_cls = None
        if target_gender is not None:
            target_gender_cls = target_gender.long().view(-1)

        target_age_cls = None
        if target_age is not None:
            target_age_cls = target_age.long().view(-1)
        loss_kl_bio = self.compute_kl(outputs["mu_bio"], outputs["logvar_bio"])

        loss_mi = 0.0
        # loss_mi = self.compute_mutual_information_gaussian(outputs["z_id"], outputs["z_bio"])

        loss_gender = torch.tensor(0.0, device=loss_recon.device, dtype=loss_recon.dtype)
        if target_gender_cls is not None:
            gender_prob = torch.clamp(outputs["gender_prob"], min=1e-8)
            loss_gender = self.nll_loss(torch.log(gender_prob), target_gender_cls)

        loss_spk = torch.tensor(0.0, device=loss_recon.device, dtype=loss_recon.dtype)
        if target_spk_cls is not None:
            loss_spk = self.ce_loss(outputs["logits_id"], target_spk_cls)

        loss_age = torch.tensor(0.0, device=loss_recon.device, dtype=loss_recon.dtype)
        if target_age_cls is not None:
            loss_age = self.ce_loss(outputs["logits_age"], target_age_cls)

        loss_adv_age = torch.tensor(0.0, device=loss_recon.device, dtype=loss_recon.dtype)
        if target_age_cls is not None:
            loss_adv_age = self.ce_loss(outputs["logits_adv_age_from_id"], target_age_cls)

        loss_adv_spk = torch.tensor(0.0, device=loss_recon.device, dtype=loss_recon.dtype)
        if target_spk_cls is not None:
            loss_adv_spk = self.ce_loss(outputs["logits_adv_id_from_age"], target_spk_cls)

        total_loss = (self.lambda_recon * loss_recon) \
                     + (self.lambda_kl_id * loss_kl_id) \
                     + (self.lambda_kl_bio * loss_kl_bio) \
                     + (self.lambda_mi * loss_mi) \
                     + (self.lambda_gender * loss_gender) \
                     + (self.lambda_spk * loss_spk) \
                     + (self.lambda_age * loss_age) \
                     + (self.lambda_adv_age * loss_adv_age) \
                     + (self.lambda_adv_spk * loss_adv_spk)

        return total_loss, {
            "loss_recon": loss_recon.item(),
            "loss_kl_id": loss_kl_id.item(),
            "loss_kl_bio": loss_kl_bio.item(),
            "loss_mi": loss_mi,
            # "loss_mi": loss_mi.item(),
            "loss_gender": loss_gender.item() if isinstance(loss_gender, torch.Tensor) else 0.0,
            "loss_spk": loss_spk.item() if isinstance(loss_spk, torch.Tensor) else 0.0,
            "loss_age": loss_age.item() if isinstance(loss_age, torch.Tensor) else 0.0,
            "loss_adv_age": loss_adv_age.item() if isinstance(loss_adv_age, torch.Tensor) else 0.0,
            "loss_adv_spk": loss_adv_spk.item() if isinstance(loss_adv_spk, torch.Tensor) else 0.0,
        }

# --- 測試模型 ---
if __name__ == "__main__":
    # 模擬 8 個樣本的 ECAPA Embedding (192維)
    mock_input = torch.randn(8, 2000)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = G_AIDA(model_id="speechbrain/spkrec-ecapa-voxceleb").to(device)
    mock_input = mock_input.to(device)
    outputs = model(mock_input)
    
    # implement loss
    criterion = G_AIDA_Loss(lambda_recon=1.0, lambda_kl_id=1.0, lambda_kl_bio=0.1, lambda_mi=0.1, lambda_gender=0.0).to(device)
    total_loss, loss_dict = criterion(
        outputs,
        target_spk=torch.randint(0, 5990, (8,), device=device),
        target_age=torch.randint(0, 7, (8,), device=device),
        target_gender=torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], device=device)
    )
    
    print(f"輸入尺寸: {mock_input.shape}")
    print(f"還原尺寸: {outputs['recon_x'].shape}")
    print(f"身分 Embedding (Z_id) 尺寸: {outputs['z_id'].shape}")
    print(f"身分分類 logits 尺寸: {outputs['logits_id'].shape}")
    print(f"年齡分類 logits 尺寸: {outputs['logits_age'].shape}")
    print(f"預測性別機率樣本: \n{outputs['gender_prob'][0]}")
    
    print(f"總損失: {total_loss.item()}")
    print(f"損失細項: {loss_dict}")