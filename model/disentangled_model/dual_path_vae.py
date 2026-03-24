import torch
import torch.nn as nn
import torch.nn.functional as F


class AgePriorNet(nn.Module):
    """
    AgePriorNet
    ------------
    這個模組負責把「聲學全域向量 P」映射成 Age 潛在空間的先驗分佈參數。

    輸入:
        acoustic_vec: [B, ACOUSTIC_DIM]

    輸出:
        mu_p:     [B, LATENT_AGE_DIM]
        logvar_p: [B, LATENT_AGE_DIM]

    說明:
        依據你的需求，Age 路徑的 KL 不再對齊標準常態 N(0, I)，
        而是對齊由聲學特徵推得的先驗 N(mu_p, var_p)。
    """

    def __init__(self, acoustic_dim: int, latent_age_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(acoustic_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, latent_age_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_age_dim)

    def forward(self, acoustic_vec: torch.Tensor):
        h = self.net(acoustic_vec)
        mu_p = self.mu_head(h)
        logvar_p = self.logvar_head(h)
        return mu_p, logvar_p


class SharedEncoder(nn.Module):
    """
    SharedEncoder
    -------------
    這是雙路徑編碼器的共同骨幹，先把 frozen speaker embedding S
    映射到較高語意的中間表示，再分支到 Age / ID 兩個 head。
    """

    def __init__(self, speaker_emb_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(speaker_emb_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, speaker_emb: torch.Tensor):
        return self.net(speaker_emb)


class DualPathVAE(nn.Module):
    """
    Dual-Path VAE
    =============
    目標:
        將輸入的 Speaker Embedding S 解耦成兩個潛在空間：
        1) z_age: 與年齡相關的特徵
        2) z_id:  與核心身份相關的特徵

    核心流程:
        1. 用 AgePriorNet(P) 產生 Age 先驗分佈參數 (mu_p, logvar_p)
        2. 用 SharedEncoder(S) 取得共享表示 h
        3. 用 AgeEncoderHead 從 h 估計 q(z_age|S) 的 (mu_age, logvar_age)
        4. 用 IDEncoderHead  從 h 估計 q(z_id|S)  的 (mu_id, logvar_id)
        5. 對兩條路徑都使用 Reparameterization Trick 取樣 z
        6. 將 [z_age, z_id] 串接後送入 Decoder 重建 S_hat
    """

    def __init__(
        self,
        speaker_emb_dim: int = 256,
        acoustic_dim: int = 8,
        latent_age_dim: int = 16,
        latent_id_dim: int = 64,
        num_speakers: int = 5990,
        num_age_groups: int = 7,
        encoder_hidden_dim: int = 256,
        decoder_hidden_dim: int = 256,
        prior_hidden_dim: int = 64,
    ):
        super().__init__()

        self.speaker_emb_dim = speaker_emb_dim
        self.acoustic_dim = acoustic_dim
        self.latent_age_dim = latent_age_dim
        self.latent_id_dim = latent_id_dim
        self.num_speakers = num_speakers
        self.num_age_groups = num_age_groups

        # 1) 年齡先驗網路：P -> (mu_p, logvar_p)
        self.age_prior_net = AgePriorNet(
            acoustic_dim=acoustic_dim,
            latent_age_dim=latent_age_dim,
            hidden_dim=prior_hidden_dim,
        )

        # 2) 共享編碼器：S -> h
        self.shared_encoder = SharedEncoder(
            speaker_emb_dim=speaker_emb_dim,
            hidden_dim=encoder_hidden_dim,
        )

        # 3) Age 分支 head：h -> (mu_age, logvar_age)
        self.age_mu_head = nn.Linear(encoder_hidden_dim, latent_age_dim)
        self.age_logvar_head = nn.Linear(encoder_hidden_dim, latent_age_dim)

        # 4) ID 分支 head：h -> (mu_id, logvar_id)
        self.id_mu_head = nn.Linear(encoder_hidden_dim, latent_id_dim)
        self.id_logvar_head = nn.Linear(encoder_hidden_dim, latent_id_dim)

        # 5) 解碼器：[z_age, z_id] -> S_hat
        decoder_in_dim = latent_age_dim + latent_id_dim
        self.decoder = nn.Sequential(
            nn.Linear(decoder_in_dim, decoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(decoder_hidden_dim, decoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(decoder_hidden_dim, speaker_emb_dim),
        )

        # 6) 解耦後分類頭（均為三層 MLP）
        # - speaker classifier: 使用 mu_id 檢測身份資訊保留程度
        self.speaker_classifier = nn.Sequential(
            nn.Linear(latent_id_dim, latent_id_dim),
            nn.ReLU(),
            nn.Linear(latent_id_dim, latent_id_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_id_dim // 2, num_speakers),
        )

        # - age classifier: 使用 mu_age 檢測年齡資訊分離程度
        hidden_age = max(8, latent_age_dim)
        self.age_classifier = nn.Sequential(
            nn.Linear(latent_age_dim, hidden_age),
            nn.ReLU(),
            nn.Linear(hidden_age, max(4, hidden_age // 2)),
            nn.ReLU(),
            nn.Linear(max(4, hidden_age // 2), num_age_groups),
        )

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor):
        """
        Reparameterization Trick
        -----------------------
        給定高斯分佈參數 (mu, logvar)，透過
            z = mu + sigma * eps, eps ~ N(0, I)
        來取得可微分的隨機取樣，讓梯度能回傳到 mu / logvar。
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, speaker_emb: torch.Tensor):
        """
        僅執行編碼階段，輸出兩條路徑的 posterior 參數。
        """
        h = self.shared_encoder(speaker_emb)

        mu_age = self.age_mu_head(h)
        logvar_age = self.age_logvar_head(h)

        mu_id = self.id_mu_head(h)
        logvar_id = self.id_logvar_head(h)

        return mu_age, logvar_age, mu_id, logvar_id

    def decode(self, z_age: torch.Tensor, z_id: torch.Tensor):
        """
        將兩個潛在向量串接後重建回原始 speaker embedding 空間。
        """
        z = torch.cat([z_age, z_id], dim=1)
        return self.decoder(z)

    def forward(self, speaker_emb: torch.Tensor, acoustic_vec: torch.Tensor):
        """
        主要前向流程。

        參數:
            speaker_emb: [B, SPEAKER_EMB_DIM]，預先抽取且可視情況凍結的說話者向量 S
            acoustic_vec: [B, ACOUSTIC_DIM]，外部預先提取好的聲學向量 P

        回傳:
            dict，包含重建結果、posterior/prior 參數與取樣後 latent。
        """
        # 讓單筆輸入可直接使用：
        # - speaker_emb: [D] -> [1, D]
        # - acoustic_vec: [A] -> [1, A]
        if speaker_emb.dim() == 1:
            speaker_emb = speaker_emb.unsqueeze(0)
        if acoustic_vec.dim() == 1:
            acoustic_vec = acoustic_vec.unsqueeze(0)

        if speaker_emb.dim() != 2 or acoustic_vec.dim() != 2:
            raise ValueError(
                "speaker_emb 與 acoustic_vec 必須是 2D tensor，"
                "形狀應為 [B, SPEAKER_EMB_DIM] 與 [B, ACOUSTIC_DIM]。"
            )

        if acoustic_vec.device != speaker_emb.device:
            acoustic_vec = acoustic_vec.to(speaker_emb.device)
        if acoustic_vec.dtype != speaker_emb.dtype:
            acoustic_vec = acoustic_vec.to(speaker_emb.dtype)

        # 若其中一邊是單筆而另一邊是 batch，允許自動 broadcast 到相同 batch 大小
        if speaker_emb.size(0) != acoustic_vec.size(0):
            if acoustic_vec.size(0) == 1:
                acoustic_vec = acoustic_vec.expand(speaker_emb.size(0), -1)
            elif speaker_emb.size(0) == 1:
                speaker_emb = speaker_emb.expand(acoustic_vec.size(0), -1)
            else:
                raise ValueError(
                    "speaker_emb 與 acoustic_vec 的 batch 大小不一致，"
                    f"收到 {speaker_emb.size(0)} 與 {acoustic_vec.size(0)}。"
                )

        if acoustic_vec.size(1) != self.acoustic_dim:
            raise ValueError(
                f"抽取後 acoustic_vec 維度不符，預期 {self.acoustic_dim}，"
                f"實際 {acoustic_vec.size(1)}。"
            )

        # A) 由聲學向量產生 Age 路徑先驗分佈參數
        mu_p, logvar_p = self.age_prior_net(acoustic_vec)

        # B) 由 speaker embedding 估計兩條 posterior 分佈參數
        mu_age, logvar_age, mu_id, logvar_id = self.encode(speaker_emb)

        # C) 兩條路徑都使用重參數化取樣
        z_age = self.reparameterize(mu_age, logvar_age)
        z_id = self.reparameterize(mu_id, logvar_id)

        # D) 解碼重建 speaker embedding
        recon_speaker_emb = self.decode(z_age, z_id)

        # E) 分類檢測（使用 posterior mean，避免取樣噪聲影響分類穩定性）
        logits_spk = self.speaker_classifier(mu_id)
        logits_age = self.age_classifier(mu_age)

        return {
            "recon_speaker_emb": recon_speaker_emb,
            "z_age": z_age,
            "z_id": z_id,
            "mu_age": mu_age,
            "logvar_age": logvar_age,
            "mu_id": mu_id,
            "logvar_id": logvar_id,
            "mu_p": mu_p,
            "logvar_p": logvar_p,
            "logits_spk": logits_spk,
            "logits_age": logits_age,
        }


class DualPathVAELoss(nn.Module):
    """
    DualPathVAELoss
    ---------------
    依照需求實作：
        Total Loss = Reconstruction Loss (MSE) + KL_Age + KL_ID

    其中：
        - KL_Age = KL[ q(z_age|S) || p(z_age|P) ]
            q 為編碼器估計的 N(mu_age, var_age)
            p 為 AgePriorNet 估計的 N(mu_p, var_p)

        - KL_ID = KL[ q(z_id|S) || N(0, I) ]
            讓 ID 潛在空間符合標準常態先驗
    """

    def __init__(
        self,
        lambda_recon: float = 1.0,
        lambda_kl_age: float = 1.0,
        lambda_kl_id: float = 1.0,
        lambda_cls_spk: float = 1.0,
        lambda_cls_age: float = 1.0,
        lambda_cosine_disentangle: float = 0.1,
    ):
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_kl_age = lambda_kl_age
        self.lambda_kl_id = lambda_kl_id
        self.lambda_cls_spk = lambda_cls_spk
        self.lambda_cls_age = lambda_cls_age
        self.lambda_cosine_disentangle = lambda_cosine_disentangle
        self.ce_loss = nn.CrossEntropyLoss()

    @staticmethod
    def kl_gaussian_to_gaussian(
        mu_q: torch.Tensor,
        logvar_q: torch.Tensor,
        mu_p: torch.Tensor,
        logvar_p: torch.Tensor,
    ):
        """
        計算 KL( q || p )，其中 q 與 p 皆為對角高斯分佈。

        公式(逐維):
            KL = 0.5 * [ log(var_p/var_q)
                        + (var_q + (mu_q - mu_p)^2) / var_p
                        - 1 ]

        實作上使用 logvar 避免數值不穩定。
        最後會先對 latent 維度加總，再對 batch 平均。
        """
        var_q = torch.exp(logvar_q)
        var_p = torch.exp(logvar_p)

        kl_per_dim = 0.5 * (
            (logvar_p - logvar_q)
            + (var_q + (mu_q - mu_p).pow(2)) / (var_p + 1e-8)
            - 1.0
        )
        kl_per_sample = kl_per_dim.sum(dim=1)
        return kl_per_sample.mean()

    @staticmethod
    def kl_gaussian_to_standard_normal(mu: torch.Tensor, logvar: torch.Tensor):
        """
        計算 KL( N(mu, var) || N(0, I) ) 的 closed-form。
        """
        kl_per_dim = -0.5 * (1.0 + logvar - mu.pow(2) - torch.exp(logvar))
        kl_per_sample = kl_per_dim.sum(dim=1)
        return kl_per_sample.mean()

    def forward(
        self,
        outputs: dict,
        speaker_emb_target: torch.Tensor,
        target_spk: torch.Tensor,
        target_age: torch.Tensor,
    ):
        """
        參數:
            outputs: 由 DualPathVAE.forward 回傳的字典
            speaker_emb_target: 原始 speaker embedding S，作為重建目標

        回傳:
            total_loss: 可直接 backward 的總損失
            loss_dict: 便於 logger 記錄的各分項損失
        """
        recon_speaker_emb = outputs["recon_speaker_emb"]

        mu_age = outputs["mu_age"]
        logvar_age = outputs["logvar_age"]
        mu_id = outputs["mu_id"]
        logvar_id = outputs["logvar_id"]
        mu_p = outputs["mu_p"]
        logvar_p = outputs["logvar_p"]
        logits_spk = outputs["logits_spk"]
        logits_age = outputs["logits_age"]

        # 1) Reconstruction Loss：確保 z_age + z_id 能重建原始 S
        recon_loss = F.mse_loss(recon_speaker_emb, speaker_emb_target, reduction="mean")

        # 2) KL_Age：posterior q(z_age|S) 對齊 acoustic-guided prior p(z_age|P)
        kl_age = self.kl_gaussian_to_gaussian(mu_age, logvar_age, mu_p, logvar_p)

        # 3) KL_ID：posterior q(z_id|S) 對齊標準常態 N(0, I)
        kl_id = self.kl_gaussian_to_standard_normal(mu_id, logvar_id)

        # 4) 分類損失：用於檢測解耦後的可辨識性
        cls_spk = self.ce_loss(logits_spk, target_spk.long())
        cls_age = self.ce_loss(logits_age, target_age.long())

        # 5) 餘弦相似度損失：鼓勵 mu_id 與 mu_age 在共同子空間上更不相似
        # 由於 latent 維度可能不同，使用前 min_dim 維做 cosine。
        min_dim = min(mu_id.size(1), mu_age.size(1))
        mu_id_align = F.normalize(mu_id[:, :min_dim], p=2, dim=1)
        mu_age_align = F.normalize(mu_age[:, :min_dim], p=2, dim=1)
        cosine_disentangle = torch.abs((mu_id_align * mu_age_align).sum(dim=1)).mean()

        # 6) 總損失加權組合
        total_loss = (
            self.lambda_recon * recon_loss
            + self.lambda_kl_age * kl_age
            + self.lambda_kl_id * kl_id
            + self.lambda_cls_spk * cls_spk
            + self.lambda_cls_age * cls_age
            + self.lambda_cosine_disentangle * cosine_disentangle
        )

        loss_dict = {
            "total_loss": total_loss.detach(),
            "recon_loss": recon_loss.detach(),
            "kl_age": kl_age.detach(),
            "kl_id": kl_id.detach(),
            "cls_spk": cls_spk.detach(),
            "cls_age": cls_age.detach(),
            "cosine_disentangle": cosine_disentangle.detach(),
        }

        return total_loss, loss_dict
