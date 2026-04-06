import torch
import torch.nn as nn
import torch.nn.functional as F

from model.disentangled_model.arcface import ArcMarginProduct


class GradientReversalFunction(torch.autograd.Function):
	"""
	Gradient Reversal Layer (GRL)
	-----------------------------
	前向傳遞時是 identity；反向傳遞時把梯度乘上 -lambda。
	常用在對抗式解耦：
	- 分類器想從 z_age 判別 speaker
	- 編碼器則被迫讓 z_age 不帶 speaker 資訊
	"""

	@staticmethod
	def forward(ctx, x: torch.Tensor, lambd: float):
		ctx.lambd = lambd
		return x.view_as(x)

	@staticmethod
	def backward(ctx, grad_output: torch.Tensor):
		return -ctx.lambd * grad_output, None


class GradientReversalLayer(nn.Module):
	def __init__(self, lambd: float = 1.0):
		super().__init__()
		self.lambd = lambd

	def forward(self, x: torch.Tensor):
		return GradientReversalFunction.apply(x, self.lambd)


class SharedEncoder(nn.Module):
	"""
	SharedEncoder
	-------------
	把 frozen speaker embedding 先映射到共享語意空間 h，
	後續再分支成 ID / Age 兩條路徑。
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


class AgeCodebookVAE(nn.Module):
	"""
	Age Codebook VAE
	================
	核心設計：
	1) 年齡路徑不再對齊單一高斯 prior。
	2) 改用「離散年齡 token codebook + 連續殘差」建構 z_age。
	3) acoustic feature 直接影響 token 分佈，讓外部年齡線索可介入。

	主要流程：
	- S -> SharedEncoder -> h
	- h -> ID head -> (mu_id, logvar_id) -> z_id
	- h -> Age continuous head -> z_age_cont
	- h -> token logits_x -> pi_x
	- acoustic -> token logits_a -> pi_a
	- pi_x @ codebook -> z_age_code
	- z_age = z_age_cont + z_age_code
	- [z_age, z_id] -> Decoder -> recon_speaker_emb
	"""

	def __init__(
		self,
		speaker_emb_dim: int = 256,
		acoustic_dim: int = 8,
		latent_age_dim: int = 24,
		latent_id_dim: int = 128,
		num_speakers: int = 5990,
		num_age_groups: int = 7,
		num_age_tokens: int = 32,
		encoder_hidden_dim: int = 256,
		decoder_hidden_dim: int = 256,
		token_temperature: float = 1.0,
		use_gumbel_softmax: bool = False,
		grl_lambda: float = 1.0,
	):
		super().__init__()

		self.speaker_emb_dim = speaker_emb_dim
		self.acoustic_dim = acoustic_dim
		self.latent_age_dim = latent_age_dim
		self.latent_id_dim = latent_id_dim
		self.num_speakers = num_speakers
		self.num_age_groups = num_age_groups
		self.num_age_tokens = num_age_tokens
		self.token_temperature = token_temperature
		self.use_gumbel_softmax = use_gumbel_softmax

		# 1) 共享編碼器
		self.shared_encoder = SharedEncoder(
			speaker_emb_dim=speaker_emb_dim,
			hidden_dim=encoder_hidden_dim,
		)

		# 2) ID 路徑（保留 VAE 形式）
		self.id_mu_head = nn.Linear(encoder_hidden_dim, latent_id_dim)
		self.id_logvar_head = nn.Linear(encoder_hidden_dim, latent_id_dim)

		# 3) Age 連續殘差路徑
		self.age_cont_head = nn.Sequential(
			nn.Linear(encoder_hidden_dim, latent_age_dim),
			nn.LayerNorm(latent_age_dim),
			nn.Tanh(),
		)

		# 4) 由共享特徵 h 預測 token 分佈 pi_x
		self.token_selector_x = nn.Linear(encoder_hidden_dim, num_age_tokens)

		# 5) 由 acoustic feature 預測 token 分佈 pi_a（外部年齡線索）
		self.token_selector_a = nn.Sequential(
			nn.Linear(acoustic_dim, 64),
			nn.ReLU(),
			nn.Linear(64, num_age_tokens),
		)

		# 6) 年齡 codebook：每列是一個年齡原型 token
		self.age_codebook = nn.Parameter(torch.randn(num_age_tokens, latent_age_dim) * 0.02)

		# 7) acoustic -> age embedding（供 codebook 對齊使用）
		self.acoustic_age_adapter = nn.Sequential(
			nn.Linear(acoustic_dim, 64),
			nn.ReLU(),
			nn.Linear(64, latent_age_dim),
		)

		# 8) 解碼器
		decoder_in_dim = latent_age_dim + latent_id_dim
		self.decoder = nn.Sequential(
			nn.Linear(decoder_in_dim, decoder_hidden_dim),
			nn.ReLU(),
			nn.Linear(decoder_hidden_dim, decoder_hidden_dim),
			nn.ReLU(),
			nn.Linear(decoder_hidden_dim, speaker_emb_dim),
		)

		# 9) Age classifier（監督 z_age 是否具年齡語意）
		hidden_age = max(8, latent_age_dim)
		self.age_classifier = nn.Sequential(
			nn.Linear(latent_age_dim, hidden_age),
			nn.ReLU(),
			nn.Linear(hidden_age, max(4, hidden_age // 2)),
			nn.ReLU(),
			nn.Linear(max(4, hidden_age // 2), num_age_groups),
		)

		# 10) speaker classifier（僅供觀測，不作主要 speaker loss）
		self.speaker_classifier = nn.Sequential(
			nn.Linear(latent_id_dim, latent_id_dim),
			nn.ReLU(),
			nn.Linear(latent_id_dim, latent_id_dim // 2),
			nn.ReLU(),
			nn.Linear(latent_id_dim // 2, num_speakers),
		)

		# 11) z_age 的 speaker 對抗頭（GRL）
		self.grl = GradientReversalLayer(lambd=grl_lambda)
		self.spk_adv_head = nn.Sequential(
			nn.Linear(latent_age_dim, latent_age_dim),
			nn.ReLU(),
			nn.Linear(latent_age_dim, num_speakers),
		)

	@staticmethod
	def reparameterize(mu: torch.Tensor, logvar: torch.Tensor):
		std = torch.exp(0.5 * logvar)
		eps = torch.randn_like(std)
		return mu + eps * std

	def _safe_prepare_inputs(self, speaker_emb: torch.Tensor, acoustic_vec: torch.Tensor):
		# 允許單筆向量輸入
		if speaker_emb.dim() == 1:
			speaker_emb = speaker_emb.unsqueeze(0)
		if acoustic_vec.dim() == 1:
			acoustic_vec = acoustic_vec.unsqueeze(0)

		if speaker_emb.dim() != 2 or acoustic_vec.dim() != 2:
			raise ValueError(
				"speaker_emb 與 acoustic_vec 必須為 2D tensor，"
				"形狀應為 [B, SPEAKER_EMB_DIM] 與 [B, ACOUSTIC_DIM]。"
			)

		if acoustic_vec.device != speaker_emb.device:
			acoustic_vec = acoustic_vec.to(speaker_emb.device)
		if acoustic_vec.dtype != speaker_emb.dtype:
			acoustic_vec = acoustic_vec.to(speaker_emb.dtype)

		# 允許一邊是單筆，另一邊是 batch 時自動展開
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
				f"acoustic_vec 維度不符，預期 {self.acoustic_dim}，實際 {acoustic_vec.size(1)}。"
			)

		return speaker_emb, acoustic_vec

	def _token_prob(self, logits: torch.Tensor):
		if self.use_gumbel_softmax and self.training:
			return F.gumbel_softmax(logits, tau=self.token_temperature, hard=False, dim=1)
		return F.softmax(logits / max(1e-6, self.token_temperature), dim=1)

	def encode(self, speaker_emb: torch.Tensor, acoustic_vec: torch.Tensor):
		h = self.shared_encoder(speaker_emb)

		mu_id = self.id_mu_head(h)
		logvar_id = self.id_logvar_head(h)
		z_id = self.reparameterize(mu_id, logvar_id)

		z_age_cont = self.age_cont_head(h)

		logits_token_x = self.token_selector_x(h)
		logits_token_a = self.token_selector_a(acoustic_vec)

		pi_x = self._token_prob(logits_token_x)
		pi_a = self._token_prob(logits_token_a)

		# 使用軟加權組合 codebook，不做硬指派，提升訓練穩定性
		z_age_code = torch.matmul(pi_x, self.age_codebook)
		z_age = z_age_cont + z_age_code

		acoustic_age_embed = self.acoustic_age_adapter(acoustic_vec)

		return {
			"h": h,
			"mu_id": mu_id,
			"logvar_id": logvar_id,
			"z_id": z_id,
			"z_age_cont": z_age_cont,
			"z_age_code": z_age_code,
			"z_age": z_age,
			"logits_token_x": logits_token_x,
			"logits_token_a": logits_token_a,
			"pi_x": pi_x,
			"pi_a": pi_a,
			"acoustic_age_embed": acoustic_age_embed,
		}

	def decode(self, z_age: torch.Tensor, z_id: torch.Tensor):
		z = torch.cat([z_age, z_id], dim=1)
		return self.decoder(z)

	def forward(self, speaker_emb: torch.Tensor, acoustic_vec: torch.Tensor):
		speaker_emb, acoustic_vec = self._safe_prepare_inputs(speaker_emb, acoustic_vec)

		enc = self.encode(speaker_emb=speaker_emb, acoustic_vec=acoustic_vec)

		recon_speaker_emb = self.decode(enc["z_age"], enc["z_id"])

		logits_spk = self.speaker_classifier(enc["mu_id"])
		logits_age = self.age_classifier(enc["z_age"])

		# GRL 對抗：分類器盡量辨識 speaker，編碼器被迫移除 z_age 中的 speaker 訊息
		logits_spk_adv = self.spk_adv_head(self.grl(enc["z_age"]))

		outputs = {
			"recon_speaker_emb": recon_speaker_emb,
			"z_id": enc["z_id"],
			"z_age": enc["z_age"],
			"z_age_cont": enc["z_age_cont"],
			"z_age_code": enc["z_age_code"],
			"mu_id": enc["mu_id"],
			"logvar_id": enc["logvar_id"],
			"pi_x": enc["pi_x"],
			"pi_a": enc["pi_a"],
			"logits_token_x": enc["logits_token_x"],
			"logits_token_a": enc["logits_token_a"],
			"acoustic_age_embed": enc["acoustic_age_embed"],
			"logits_spk": logits_spk,
			"logits_age": logits_age,
			"logits_spk_adv": logits_spk_adv,
		}
		return outputs


class AgeCodebookVAELossArcFace(nn.Module):
	"""
	AgeCodebookVAE 專用 loss（ArcFace 版本）
	-------------------------------------
	與原 dual_path_vae_arcface 的設計兼容，但把 Age KL 改成：
	1) token 對齊損失（pi_x 對齊 pi_a）
	2) codebook 對齊損失（z_age_code 對齊 acoustic-age embedding）
	3) token 使用熵正則（降低 token collapse 風險）
	"""

	def __init__(
		self,
		latent_id_dim: int,
		num_speakers: int,
		lambda_recon: float = 1.0,
		lambda_kl_id: float = 1.0,
		lambda_cls_spk: float = 1.0,
		lambda_cls_age: float = 1.0,
		lambda_token_align: float = 0.5,
		lambda_codebook_align: float = 0.3,
		lambda_token_entropy: float = 0.02,
		lambda_spk_adv: float = 0.2,
		lambda_cosine_disentangle: float = 0.1,
		arcface_s: float = 30.0,
		arcface_m: float = 0.35,
		arcface_easy_margin: bool = False,
		speaker_label_smoothing: float = 0.0,
	):
		super().__init__()

		self.lambda_recon = lambda_recon
		self.lambda_kl_id = lambda_kl_id
		self.lambda_cls_spk = lambda_cls_spk
		self.lambda_cls_age = lambda_cls_age
		self.lambda_token_align = lambda_token_align
		self.lambda_codebook_align = lambda_codebook_align
		self.lambda_token_entropy = lambda_token_entropy
		self.lambda_spk_adv = lambda_spk_adv
		self.lambda_cosine_disentangle = lambda_cosine_disentangle

		self.ce_loss = nn.CrossEntropyLoss()
		self.spk_ce_loss = nn.CrossEntropyLoss(label_smoothing=speaker_label_smoothing)

		self.arcface_head = ArcMarginProduct(
			in_features=latent_id_dim,
			out_features=num_speakers,
			s=arcface_s,
			m=arcface_m,
			easy_margin=arcface_easy_margin,
		)

	@staticmethod
	def kl_gaussian_to_standard_normal(mu: torch.Tensor, logvar: torch.Tensor):
		kl_per_dim = -0.5 * (1.0 + logvar - mu.pow(2) - torch.exp(logvar))
		kl_per_sample = kl_per_dim.sum(dim=1)
		return kl_per_sample.mean()

	@staticmethod
	def orthogonal_disentangle_loss(z_id: torch.Tensor, z_age: torch.Tensor, eps: float = 1e-8):
		z_id = z_id - z_id.mean(dim=0, keepdim=True)
		z_age = z_age - z_age.mean(dim=0, keepdim=True)

		z_id = z_id / (z_id.norm(p=2, dim=1, keepdim=True) + eps)
		z_age = z_age / (z_age.norm(p=2, dim=1, keepdim=True) + eps)

		batch_size = z_id.size(0)
		cross_corr = torch.matmul(z_id.transpose(0, 1), z_age) / max(1, batch_size)
		return (cross_corr ** 2).mean()

	@staticmethod
	def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8):
		# 使用 Jensen-Shannon divergence 讓 token 分佈雙向對齊。
		p = torch.clamp(p, min=eps)
		q = torch.clamp(q, min=eps)
		m = 0.5 * (p + q)
		js = 0.5 * (p * (p.log() - m.log())).sum(dim=1) + 0.5 * (q * (q.log() - m.log())).sum(dim=1)
		return js.mean()

	@staticmethod
	def batch_token_entropy(pi: torch.Tensor, eps: float = 1e-8):
		# 熵值越高表示 token 使用越平均；用於抑制 collapse。
		pi = torch.clamp(pi, min=eps)
		sample_entropy = -(pi * pi.log()).sum(dim=1).mean()
		mean_pi = pi.mean(dim=0)
		mean_entropy = -(mean_pi * torch.clamp(mean_pi, min=eps).log()).sum()
		return sample_entropy, mean_entropy

	def forward(
		self,
		outputs: dict,
		speaker_emb_target: torch.Tensor,
		target_spk: torch.Tensor,
		target_age: torch.Tensor,
	):
		recon_speaker_emb = outputs["recon_speaker_emb"]
		mu_id = outputs["mu_id"]
		logvar_id = outputs["logvar_id"]
		z_id = outputs["z_id"]
		z_age = outputs["z_age"]
		z_age_code = outputs["z_age_code"]
		acoustic_age_embed = outputs["acoustic_age_embed"]
		logits_age = outputs["logits_age"]
		logits_spk_adv = outputs["logits_spk_adv"]
		pi_x = outputs["pi_x"]
		pi_a = outputs["pi_a"]

		# 1) 重建
		recon_loss = F.mse_loss(recon_speaker_emb, speaker_emb_target, reduction="mean")

		# 2) ID KL（保留 VAE 正則）
		kl_id = self.kl_gaussian_to_standard_normal(mu_id, logvar_id)

		# 3) speaker ArcFace（主要身份監督）
		arcface_logits = self.arcface_head(mu_id, target_spk.long())
		cls_spk = self.spk_ce_loss(arcface_logits, target_spk.long())

		# 4) age CE（要求 z_age 保持年齡辨識能力）
		cls_age = self.ce_loss(logits_age, target_age.long())

		# 5) token 分佈對齊（embedding 路徑 vs acoustic 路徑）
		token_align = self.js_divergence(pi_x, pi_a)

		# 6) codebook 對齊（讓 codebook 真的承載 acoustic 年齡線索）
		codebook_align = F.mse_loss(z_age_code, acoustic_age_embed, reduction="mean")

		# 7) token 熵正則（避免 token collapse）
		sample_entropy, mean_entropy = self.batch_token_entropy(pi_x)
		token_entropy_reg = -mean_entropy

		# 8) 對抗損失：z_age 不應含太多 speaker 訊息
		spk_adv_loss = self.ce_loss(logits_spk_adv, target_spk.long())

		# 9) 正交解耦（z_id 與 z_age）
		cosine_disentangle = self.orthogonal_disentangle_loss(z_id, z_age)

		total_loss = (
			self.lambda_recon * recon_loss
			+ self.lambda_kl_id * kl_id
			+ self.lambda_cls_spk * cls_spk
			+ self.lambda_cls_age * cls_age
			+ self.lambda_token_align * token_align
			+ self.lambda_codebook_align * codebook_align
			+ self.lambda_token_entropy * token_entropy_reg
			+ self.lambda_spk_adv * spk_adv_loss
			+ self.lambda_cosine_disentangle * cosine_disentangle
		)

		loss_dict = {
			"total_loss": total_loss.detach(),
			"recon_loss": recon_loss.detach(),
			"kl_id": kl_id.detach(),
			"cls_spk": cls_spk.detach(),
			"cls_age": cls_age.detach(),
			"token_align": token_align.detach(),
			"codebook_align": codebook_align.detach(),
			"token_sample_entropy": sample_entropy.detach(),
			"token_mean_entropy": mean_entropy.detach(),
			"token_entropy_reg": token_entropy_reg.detach(),
			"spk_adv_loss": spk_adv_loss.detach(),
			"cosine_disentangle": cosine_disentangle.detach(),
			"arcface_logits": arcface_logits.detach(),
		}
		return total_loss, loss_dict

