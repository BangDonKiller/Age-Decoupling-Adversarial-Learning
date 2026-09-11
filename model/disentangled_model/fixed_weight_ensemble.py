"""
Three-expert MoE variant with dynamic router training and fixed router weights for inference.

Key design:
1. Load three ECAPA experts from full-finetune checkpoints.
2. Freeze all expert parameters.
3. Train the router only during training.
4. After training, save one fixed 3-way weight vector and reuse it for all test pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.feature_extractor.ecapa_tdnn import ECAPA_TDNN


@dataclass(frozen=True)
class FixedRouterExpertCheckpointPaths:
	small: str = "checkpoints/siamese_full_finetune/full_ft_lr1e4_small_seed42/siamese_best.pt"
	medium: str = "checkpoints/siamese_full_finetune/full_ft_lr1e4_medium_seed42/siamese_best.pt"
	large: str = "checkpoints/siamese_full_finetune/full_ft_lr1e4_large_seed42/siamese_best.pt"


def _extract_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
	if isinstance(checkpoint, dict):
		for key in ("model_state_dict", "state_dict", "model"):
			if key in checkpoint and isinstance(checkpoint[key], dict):
				return checkpoint[key]
	if isinstance(checkpoint, dict):
		return checkpoint
	raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")


def _load_flexible_weights(module: nn.Module, path: str) -> None:
	checkpoint = torch.load(path, map_location="cpu")
	loaded_state = _extract_state_dict(checkpoint)
	current_state = module.state_dict()
	loaded_weights = []
	skipped_weights = []
	unmatched_weights = []

	for name, param in loaded_state.items():
		if name in ("speaker_loss.weight", "speaker_loss.bias"):
			skipped_weights.append(name)
			continue

		candidate_names = [name]
		if name.startswith("module."):
			candidate_names.append(name.replace("module.", "", 1))
		if name.startswith("encoder."):
			candidate_names.append(name.replace("encoder.", "", 1))
		if name.startswith("module.encoder."):
			candidate_names.append(name.replace("module.encoder.", "", 1))
		if name.startswith("speaker_encoder."):
			candidate_names.append(name.replace("speaker_encoder.", "", 1))
		if name.startswith("module.speaker_encoder."):
			candidate_names.append(name.replace("module.speaker_encoder.", "", 1))

		matched_name = None
		for candidate in candidate_names:
			if candidate in current_state and current_state[candidate].shape == param.shape:
				matched_name = candidate
				break

		if matched_name is None:
			unmatched_weights.append(name)
			continue

		current_state[matched_name].copy_(param)
		loaded_weights.append((name, matched_name))

	total_target_params = len(current_state)
	print("\n" + "=" * 66)
	print(f"Expert checkpoint load summary: {path}")
	print("-" * 66)
	print(f"{'loaded_params':<20}: {len(loaded_weights)} / {total_target_params}")
	print(f"{'skipped_params':<20}: {len(skipped_weights)}")
	print(f"{'unmatched_params':<20}: {len(unmatched_weights)}")
	if loaded_weights:
		print(f"{'first_loaded':<20}: {loaded_weights[0][0]} -> {loaded_weights[0][1]}")
	if unmatched_weights:
		print(f"{'first_unmatched':<20}: {unmatched_weights[0]}")
	if len(loaded_weights) == 0:
		print("[Warn] No checkpoint parameters matched the expert encoder. The expert remains randomly initialized.")
	print("=" * 66)


class BaseExpertModel(nn.Module):
	def __init__(self, C: int = 1024):
		super().__init__()
		self.encoder = ECAPA_TDNN(C=C)

	def forward(self, x: torch.Tensor, spec_aug: bool = False) -> torch.Tensor:
		return self.encoder(x, aug=spec_aug)


class FixedWeightRouter(nn.Module):
	"""Train dynamically, but support switching to one frozen global weight vector."""

	def __init__(
		self,
		feature_dim: int = 192,
		hidden_dim: int = 256,
		dropout: float = 0.1,
		input_mode: str = "embedding_diff",
	):
		super().__init__()
		if input_mode not in {"embedding_diff", "cosine_distance"}:
			raise ValueError(f"Unsupported router input_mode: {input_mode}")
		self.input_mode = input_mode

		input_dim = feature_dim * 3 if input_mode == "embedding_diff" else 3
		self.mlp = nn.Sequential(
			nn.Linear(input_dim, hidden_dim),
			nn.ReLU(inplace=True),
			nn.BatchNorm1d(hidden_dim),
			nn.Dropout(dropout),
			nn.Linear(hidden_dim, 3),
		)
		self.register_buffer("fixed_weights", torch.tensor([1 / 3, 1 / 3, 1 / 3], dtype=torch.float32))
		self.use_fixed_weights = False

	def set_fixed_weights(self, weights: torch.Tensor) -> None:
		weights = weights.detach().float().flatten()
		if weights.numel() != 3:
			raise ValueError(f"fixed router weights must have 3 elements, got shape {tuple(weights.shape)}")
		weights = torch.clamp(weights, min=0.0)
		weight_sum = torch.sum(weights)
		if torch.isclose(weight_sum, torch.tensor(0.0, device=weights.device)):
			raise ValueError("fixed router weights sum to zero")
		weights = weights / weight_sum
		self.fixed_weights.copy_(weights.to(self.fixed_weights.device))

	def enable_fixed_weights(self, enabled: bool = True) -> None:
		self.use_fixed_weights = enabled

	def forward(
		self,
		diff_small: torch.Tensor,
		diff_medium: torch.Tensor,
		diff_large: torch.Tensor,
		return_logits: bool = False,
	) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor | None]:
		if self.use_fixed_weights:
			batch_size = diff_small.size(0)
			weights = self.fixed_weights.unsqueeze(0).expand(batch_size, -1)
			if return_logits:
				return weights, None
			return weights

		if self.input_mode == "cosine_distance":
			fused_diff = torch.stack([diff_small, diff_medium, diff_large], dim=-1)
		else:
			fused_diff = torch.cat([diff_small, diff_medium, diff_large], dim=-1)
		logits = self.mlp(fused_diff)
		weights = F.softmax(logits, dim=-1)
		if return_logits:
			return weights, logits
		return weights


class CrossGapFixedRouterEnsemble(nn.Module):
	"""Three-expert fixed-weight ensemble without score calibrators and with optional fixed routing at inference."""

	def __init__(
		self,
		C: int = 1024,
		feature_dim: int = 192,
		router_hidden_dim: int = 256,
		router_dropout: float = 0.1,
		router_input_mode: str = "embedding_diff",
		expert_ckpt_paths: FixedRouterExpertCheckpointPaths | None = None,
	):
		super().__init__()

		paths = expert_ckpt_paths or FixedRouterExpertCheckpointPaths()

		self.expert_small = BaseExpertModel(C=C)
		self.expert_medium = BaseExpertModel(C=C)
		self.expert_large = BaseExpertModel(C=C)

		_load_flexible_weights(self.expert_small.encoder, paths.small)
		_load_flexible_weights(self.expert_medium.encoder, paths.medium)
		_load_flexible_weights(self.expert_large.encoder, paths.large)

		self._freeze_experts()

		self.router = FixedWeightRouter(
			feature_dim=feature_dim,
			hidden_dim=router_hidden_dim,
			dropout=router_dropout,
			input_mode=router_input_mode,
		)

	def _freeze_experts(self) -> None:
		for model in (self.expert_small, self.expert_medium, self.expert_large):
			model.requires_grad_(False)
			model.eval()

	def set_fixed_router_weights(self, weights: torch.Tensor) -> None:
		self.router.set_fixed_weights(weights)

	def enable_fixed_router(self, enabled: bool = True) -> None:
		self.router.enable_fixed_weights(enabled)

	def train(self, mode: bool = True):
		super().train(mode)
		self.expert_small.eval()
		self.expert_medium.eval()
		self.expert_large.eval()
		return self

	def forward(
		self,
		x1: torch.Tensor,
		x2: torch.Tensor,
		spec_aug: bool = False,
		return_details: bool = False,
	) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
		with torch.no_grad():
			f1_s = self.expert_small(x1, spec_aug=spec_aug)
			f2_s = self.expert_small(x2, spec_aug=spec_aug)
			f1_m = self.expert_medium(x1, spec_aug=spec_aug)
			f2_m = self.expert_medium(x2, spec_aug=spec_aug)
			f1_l = self.expert_large(x1, spec_aug=spec_aug)
			f2_l = self.expert_large(x2, spec_aug=spec_aug)

			f1_s = F.normalize(f1_s, p=2, dim=1)
			f2_s = F.normalize(f2_s, p=2, dim=1)
			f1_m = F.normalize(f1_m, p=2, dim=1)
			f2_m = F.normalize(f2_m, p=2, dim=1)
			f1_l = F.normalize(f1_l, p=2, dim=1)
			f2_l = F.normalize(f2_l, p=2, dim=1)

			score_s = torch.sum(f1_s * f2_s, dim=-1)
			score_m = torch.sum(f1_m * f2_m, dim=-1)
			score_l = torch.sum(f1_l * f2_l, dim=-1)

			diff_s = torch.abs(f1_s - f2_s)
			diff_m = torch.abs(f1_m - f2_m)
			diff_l = torch.abs(f1_l - f2_l)

			dist_s = 1 - score_s
			dist_m = 1 - score_m
			dist_l = 1 - score_l

		if self.router.input_mode == "cosine_distance":
			router_in_small, router_in_medium, router_in_large = dist_s, dist_m, dist_l
		else:
			router_in_small, router_in_medium, router_in_large = diff_s, diff_m, diff_l

		weights, router_logits = self.router(
			router_in_small,
			router_in_medium,
			router_in_large,
			return_logits=True,
		)

		stacked_scores = torch.stack([score_s, score_m, score_l], dim=1)
		score_final = torch.sum(weights * stacked_scores, dim=1)

		if not return_details:
			return score_final, weights

		details = {
			"score_small": score_s,
			"score_medium": score_m,
			"score_large": score_l,
			"diff_small": diff_s,
			"diff_medium": diff_m,
			"diff_large": diff_l,
			"dist_small": dist_s,
			"dist_medium": dist_m,
			"dist_large": dist_l,
			"fixed_router_weights": self.router.fixed_weights.clone(),
		}
		if router_logits is not None:
			details["router_logits"] = router_logits
		return score_final, weights, details