"""
Three-expert full-finetune MoE model with dynamic pairwise router.

Key design:
1. Load three ECAPA experts from full-finetune checkpoints.
2. Freeze all expert parameters.
3. Add one trainable score calibrator (scale + bias) per expert.
4. Route each pair dynamically by expert-specific feature differences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.feature_extractor.ecapa_tdnn import ECAPA_TDNN


@dataclass(frozen=True)
class ExpertCheckpointPaths:
	small: str = "checkpoints/siamese_full_finetune/full_ft_lr1e4_small_seed42/siamese_best.pt"
	medium: str = "checkpoints/siamese_full_finetune/full_ft_lr1e4_medium_seed42/siamese_best.pt"
	large: str = "checkpoints/siamese_full_finetune/full_ft_lr1e4_large_seed42/siamese_best.pt"


def _extract_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
	"""Extract state dict from different checkpoint wrappers."""
	if isinstance(checkpoint, dict):
		for key in ("model_state_dict", "state_dict", "model"):
			if key in checkpoint and isinstance(checkpoint[key], dict):
				return checkpoint[key]
	if isinstance(checkpoint, dict):
		return checkpoint
	raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")


def _load_flexible_weights(module: nn.Module, path: str) -> None:
	"""Load weights with prefix compatibility (module./speaker_encoder.)."""
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
	"""ECAPA expert backbone wrapper."""

	def __init__(self, C: int = 1024):
		super().__init__()
		self.encoder = ECAPA_TDNN(C=C)

	def forward(self, x: torch.Tensor, spec_aug: bool = False) -> torch.Tensor:
		return self.encoder(x, aug=spec_aug)


class ScoreCalibrator(nn.Module):
	"""Per-expert trainable affine calibration for cosine score."""

	def __init__(self):
		super().__init__()
		self.alpha = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
		self.beta = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))

	def forward(self, score: torch.Tensor) -> torch.Tensor:
		return self.alpha * score + self.beta


class PairwiseRouter(nn.Module):
	"""Dynamic router based on concatenated expert-wise pair differences."""

	def __init__(self, feature_dim: int = 192, hidden_dim: int = 256, dropout: float = 0.1):
		super().__init__()
		input_dim = feature_dim * 3
		self.mlp = nn.Sequential(
			nn.Linear(input_dim, hidden_dim),
			nn.ReLU(inplace=True),
			nn.BatchNorm1d(hidden_dim),
			nn.Dropout(dropout),
			nn.Linear(hidden_dim, 3),
		)

	def forward(
		self,
		diff_small: torch.Tensor,
		diff_medium: torch.Tensor,
		diff_large: torch.Tensor,
		return_logits: bool = False,
	) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
		fused_diff = torch.cat([diff_small, diff_medium, diff_large], dim=-1)
		logits = self.mlp(fused_diff)
		weights = F.softmax(logits, dim=-1)
		if return_logits:
			return weights, logits
		return weights


class CrossGapMoE(nn.Module):
	"""
	Three-expert MoE with frozen experts and trainable router/calibrators.

	Forward input:
	- x1, x2: pair waveform tensors.

	Forward output:
	- score_final: [B] fused similarity score.
	- weights: [B, 3] router weights for (small, medium, large).
	"""

	def __init__(
		self,
		C: int = 1024,
		feature_dim: int = 192,
		router_hidden_dim: int = 256,
		router_dropout: float = 0.1,
		expert_ckpt_paths: ExpertCheckpointPaths | None = None,
	):
		super().__init__()

		paths = expert_ckpt_paths or ExpertCheckpointPaths()

		self.expert_small = BaseExpertModel(C=C)
		self.expert_medium = BaseExpertModel(C=C)
		self.expert_large = BaseExpertModel(C=C)

		_load_flexible_weights(self.expert_small.encoder, paths.small)
		_load_flexible_weights(self.expert_medium.encoder, paths.medium)
		_load_flexible_weights(self.expert_large.encoder, paths.large)

		self._freeze_experts()

		self.calib_small = ScoreCalibrator()
		self.calib_medium = ScoreCalibrator()
		self.calib_large = ScoreCalibrator()

		self.router = PairwiseRouter(
			feature_dim=feature_dim,
			hidden_dim=router_hidden_dim,
			dropout=router_dropout,
		)

	def _freeze_experts(self) -> None:
		for model in (self.expert_small, self.expert_medium, self.expert_large):
			model.requires_grad_(False)
			model.eval()

	def train(self, mode: bool = True):
		super().train(mode)
		# Keep frozen experts in eval mode to lock BN/Dropout behavior.
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

		weights, router_logits = self.router(
			diff_s,
			diff_m,
			diff_l,
			return_logits=True,
		)

		score_s_calib = self.calib_small(score_s)
		score_m_calib = self.calib_medium(score_m)
		score_l_calib = self.calib_large(score_l)

		stacked_scores = torch.stack([score_s_calib, score_m_calib, score_l_calib], dim=1)
		score_final = torch.sum(weights.detach() * stacked_scores, dim=1)

		if not return_details:
			return score_final, weights

		details = {
			"router_logits": router_logits,
			"score_small": score_s,
			"score_medium": score_m,
			"score_large": score_l,
			"score_small_calib": score_s_calib,
			"score_medium_calib": score_m_calib,
			"score_large_calib": score_l_calib,
			"diff_small": diff_s,
			"diff_medium": diff_m,
			"diff_large": diff_l,
		}
		return score_final, weights, details

