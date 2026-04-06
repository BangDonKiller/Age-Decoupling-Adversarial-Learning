import torch
import torch.nn as nn
import torch.nn.functional as F

from model.disentangled_model.arcface import ArcMarginProduct
from model.disentangled_model.dual_path_vae import DualPathVAELoss


class DualPathVAELossArcFace(DualPathVAELoss):
    """
    在不修改原始 DualPathVAE 架構的前提下，新增 ArcFace 的 loss 版本。

    設計重點:
    1) 不改動原本 model/disentangled_model/dual_path_vae.py 的網路定義。
    2) 保留原本的重建、KL、年齡分類損失。
    3) 僅把 speaker 分類損失由一般 CE(logits_spk) 改為 ArcFace CE。
    4) 新增 z_id 的年齡對抗分類損失（GRL 在 model 內）。

    使用方式:
    - 訓練時改用這個 criterion 類別。
    - optimizer 需同時包含 model 參數 + criterion 參數（ArcFace head 權重在 criterion 內）。
    """

    def __init__(
        self,
        latent_id_dim: int,
        num_speakers: int,
        lambda_recon: float = 1.0,
        lambda_kl_age: float = 1.0,
        lambda_kl_id: float = 1.0,
        lambda_cls_spk: float = 1.0,
        lambda_cls_age: float = 1.0,
        lambda_adv_age_id: float = 0.0,
        lambda_cosine_disentangle: float = 0.0,
        arcface_s: float = 30.0,
        arcface_m: float = 0.35,
        arcface_easy_margin: bool = False,
        speaker_label_smoothing: float = 0.0,
    ):
        super().__init__(
            lambda_recon=lambda_recon,
            lambda_kl_age=lambda_kl_age,
            lambda_kl_id=lambda_kl_id,
            lambda_cls_spk=lambda_cls_spk,
            lambda_cls_age=lambda_cls_age,
            lambda_adv_age_id=lambda_adv_age_id,
            lambda_cosine_disentangle=lambda_cosine_disentangle,
        )

        # ArcFace 分類頭：輸入 mu_id，輸出 speaker logits。
        self.arcface_head = ArcMarginProduct(
            in_features=latent_id_dim,
            out_features=num_speakers,
            s=arcface_s,
            m=arcface_m,
            easy_margin=arcface_easy_margin,
        )

        # speaker 分類可選 label smoothing，通常 0.0~0.1。
        self.spk_ce_loss = nn.CrossEntropyLoss(label_smoothing=speaker_label_smoothing)

    def forward(
        self,
        outputs: dict,
        speaker_emb_target: torch.Tensor,
        target_spk: torch.Tensor,
        target_age: torch.Tensor,
    ):
        """
        與原始 DualPathVAELoss.forward 同介面，方便直接替換。

        差異只有一點:
        - cls_spk 由 ArcFace logits 計算，而不是用 outputs["logits_spk"]。
        """
        recon_speaker_emb = outputs["recon_speaker_emb"]

        mu_age = outputs["mu_age"]
        logvar_age = outputs["logvar_age"]
        mu_id = outputs["mu_id"]
        logvar_id = outputs["logvar_id"]
        mu_p = outputs["mu_p"]
        logvar_p = outputs["logvar_p"]
        logits_age = outputs["logits_age"]
        logits_age_adv = outputs.get("logits_age_adv", None)

        # 1) Reconstruction Loss
        recon_loss = F.mse_loss(recon_speaker_emb, speaker_emb_target, reduction="mean")

        # 2) KL_Age
        kl_age = self.kl_gaussian_to_gaussian(mu_age, logvar_age, mu_p, logvar_p)

        # 3) KL_ID
        kl_id = self.kl_gaussian_to_standard_normal(mu_id, logvar_id)

        # 4) ArcFace speaker classification
        arcface_logits = self.arcface_head(mu_id, target_spk.long())
        cls_spk = self.spk_ce_loss(arcface_logits, target_spk.long())

        # 5) Age classification（保留原本 head）
        cls_age = self.ce_loss(logits_age, target_age.long())

        # 6) z_id 年齡對抗損失（GRL 已在模型端反轉梯度）
        if logits_age_adv is None:
            adv_age_id = torch.zeros((), device=mu_id.device, dtype=mu_id.dtype)
        else:
            adv_age_id = self.ce_loss(logits_age_adv, target_age.long())

        # 7) 總損失
        total_loss = (
            self.lambda_recon * recon_loss
            + self.lambda_kl_age * kl_age
            + self.lambda_kl_id * kl_id
            + self.lambda_cls_spk * cls_spk
            + self.lambda_cls_age * cls_age
            + self.lambda_adv_age_id * adv_age_id
        )

        loss_dict = {
            "total_loss": total_loss.detach(),
            "recon_loss": recon_loss.detach(),
            "kl_age": kl_age.detach(),
            "kl_id": kl_id.detach(),
            "cls_spk": cls_spk.detach(),
            "cls_age": cls_age.detach(),
            "adv_age_id": adv_age_id.detach(),
            # 額外把 ArcFace logits 帶出，方便訓練腳本算 speaker acc。
            "arcface_logits": arcface_logits.detach(),
        }

        return total_loss, loss_dict
