import torch
import torch.nn as nn
import torch.nn.functional as F

from model.disentangled_model.arcface import ArcMarginProduct


class LinearDecorrMLP(nn.Module):
    """
    ECAPA speaker embedding 後接 MLP，最後輸出線性空間向量 z。
    z 的第 0 維給年齡分類，其餘維度給說話者分類。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims,
        output_dim: int,
        num_speakers: int,
        num_age_groups: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        if output_dim < 2:
            raise ValueError("output_dim 必須 >= 2，至少要有 1 維年齡 + 1 維說話者")

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

        self.age_head = nn.Linear(1, num_age_groups)
        self.spk_head = nn.Linear(output_dim - 1, num_speakers)

    def forward(self, speaker_emb: torch.Tensor):
        z = self.projector(speaker_emb)
        z_age = z[:, :1]
        z_id = z[:, 1:]

        logits_age = self.age_head(z_age)
        logits_spk = self.spk_head(z_id)

        return {
            "z": z,
            "z_age": z_age,
            "z_id": z_id,
            "logits_age": logits_age,
            "logits_spk": logits_spk,
        }


def linear_correlation_loss(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    對 z 的每個維度計算線性相關性懲罰：
    1) 先做零均值 + 標準差正規化
    2) 估計相關係數矩陣
    3) 最小化非對角線元素平方平均
    """
    if z.dim() != 2:
        raise ValueError(f"z 需為 [batch, dim]，目前 shape={tuple(z.shape)}")

    batch_size, dim = z.shape
    if batch_size < 2 or dim < 2:
        return z.new_zeros(())

    z_centered = z - z.mean(dim=0, keepdim=True)
    z_std = z_centered.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
    z_norm = z_centered / z_std

    corr = (z_norm.transpose(0, 1) @ z_norm) / float(batch_size)
    off_diag = corr - torch.diag(torch.diag(corr))
    return off_diag.pow(2).mean()


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
        loss_decorr = linear_correlation_loss(outputs["z"])

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
