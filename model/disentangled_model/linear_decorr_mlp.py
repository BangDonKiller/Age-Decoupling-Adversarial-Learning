import torch
import torch.nn as nn
import torch.nn.functional as F

from loss.arcface import ArcMarginProduct


class LinearDecorrMLP(nn.Module):
    """
    ECAPA speaker embedding 後接 MLP，最後輸出線性空間向量 z。
    z 的前 age_dim 維給年齡分類，其餘維度給說話者分類。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims,
        output_dim: int,
        num_speakers: int,
        num_age_groups: int,
        dropout: float = 0.1,
        age_dim: int = 2,
    ):
        super().__init__()

        if age_dim < 1:
            raise ValueError("age_dim 必須 >= 1")
        if output_dim <= age_dim:
            raise ValueError("output_dim 必須 > age_dim，至少要有年齡維度 + 1 維說話者")

        self.age_dim = age_dim

        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        elif hidden_dims is None:
            hidden_dims = []

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.projector = nn.Sequential(*layers)

        self.age_head = nn.Linear(age_dim, num_age_groups)
        self.spk_head = nn.Linear(output_dim - age_dim, num_speakers)
        
        # 非線性分類頭 (暫時不使用)
        # age_hidden_dim = max(age_dim * 4, 16)
        # self.age_head_nonlinear = nn.Sequential(
        #     nn.Linear(age_dim, age_hidden_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(age_hidden_dim, num_age_groups),
        # )
        
        # spk_hidden_dim = max((output_dim - age_dim) * 2, 128)
        # self.spk_head_nonlinear = nn.Sequential(
        #     nn.Linear(output_dim - age_dim, spk_hidden_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(spk_hidden_dim, num_speakers),
        # )

    def forward(self, speaker_emb: torch.Tensor):
        z = self.projector(speaker_emb)
        z_age = z[:, : self.age_dim]
        z_id = z[:, self.age_dim :]

        logits_age = self.age_head(z_age)
        logits_spk = self.spk_head(z_id)
        # logits_age = self.age_head_nonlinear(z_age)
        # logits_spk = self.spk_head_nonlinear(z_id)

        return {
            "z": z,
            "z_age": z_age,
            "z_id": z_id,
            "logits_age": logits_age,
            "logits_spk": logits_spk,
        }

def linear_correlation_loss(z_age: torch.Tensor, z_id: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    只最小化 z_age 與 z_id 之間的線性相關性，不干涉 z_id 內部的相關性。
    """
    batch_size = z_age.shape[0]
    
    # 標準化 z_age
    z_age_centered = z_age - z_age.mean(dim=0, keepdim=True)
    z_age_norm = z_age_centered / z_age_centered.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
    
    # 標準化 z_id
    z_id_centered = z_id - z_id.mean(dim=0, keepdim=True)
    z_id_norm = z_id_centered / z_id_centered.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
    
    # 計算 z_age 與 z_id 之間的交叉相關矩陣 (shape: [dim_age, dim_id])
    cross_corr = (z_age_norm.transpose(0, 1) @ z_id_norm) / float(batch_size)
    
    # 最小化交叉相關矩陣的平方和
    return cross_corr.pow(2).mean()


def compute_age_neuron_correlations(z: torch.Tensor, age_dim: int = 2, eps: float = 1e-6):
    """
    計算前 age_dim 個年齡神經元與其他神經元的相關係數。
    返回相關係數矩陣，其中前 age_dim 個欄位對應年齡神經元自身。
    """
    if z.dim() != 2:
        raise ValueError(f"z 需為 [batch, dim]，目前 shape={tuple(z.shape)}")

    batch_size, dim = z.shape
    if batch_size < 2 or dim < age_dim:
        return None

    z_centered = z - z.mean(dim=0, keepdim=True)
    z_std = z_centered.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
    z_norm = z_centered / z_std

    # 計算相關係數矩陣
    corr = (z_norm.transpose(0, 1) @ z_norm) / float(batch_size)
    # 提取年齡神經元與所有神經元的相關係數
    # 先 detach，避免在 requires_grad=True 的圖中直接轉 numpy 造成錯誤
    age_corr = corr[:age_dim, :].detach().cpu().numpy()
    return age_corr


class LinearDecorrLoss(nn.Module):
    def __init__(
        self,
        latent_id_dim: int,
        num_speakers: int,
        lambda_spk: float = 1.0,
        lambda_age: float = 1.0,
        lambda_decorr: float = 1.0,
        arcface_s: float = 30.0,
        arcface_m: float = 0.35,
        arcface_easy_margin: bool = False,
    ):
        super().__init__()
        self.arcface = ArcMarginProduct(
            in_features=latent_id_dim,
            out_features=num_speakers,
            s=arcface_s,
            m=arcface_m,
            easy_margin=arcface_easy_margin,
        )
        self.lambda_spk = lambda_spk
        self.lambda_age = lambda_age
        self.lambda_decorr = lambda_decorr

    def forward(self, outputs, target_spk: torch.Tensor, target_age: torch.Tensor):
        arcface_logits = self.arcface(outputs["z_id"], target_spk)
        loss_spk = F.cross_entropy(arcface_logits, target_spk)
        loss_age = F.cross_entropy(outputs["logits_age"], target_age)
        loss_decorr = linear_correlation_loss(outputs["z_age"], outputs["z_id"])

        total_loss = (
            self.lambda_spk * loss_spk
            + self.lambda_age * loss_age
            + self.lambda_decorr * loss_decorr
        )
        
        loss_dict = {
            "loss_spk": loss_spk.detach(),
            "loss_age": loss_age.detach(),
            "loss_decorr": loss_decorr.detach(),
            "arcface_logits": arcface_logits.detach(),
        }
        return total_loss, loss_dict
