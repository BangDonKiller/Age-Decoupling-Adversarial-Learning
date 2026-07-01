"""CA20 small/large expert embedding SVD analysis.

This script:
1. Loads the Vox-CA20 test split.
2. Extracts speaker-level embeddings with the small and large experts.
3. Centers each embedding matrix and runs SVD.
4. Plots the cumulative variance explained by the top-K singular values.
"""

from __future__ import annotations

import argparse
import os
import warnings
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm

from data.vox1_loader import PairwiseDataset
from model.disentangled_model.siamese_network import SiameseNetwork
from params.param import DATASET_INFO, DEVICE

warnings.filterwarnings(
    "ignore",
    message=r".*torchaudio\.load_with_torchcodec.*",
    category=UserWarning,
)
warnings.filterwarnings(
	"ignore",
	message=r".*StreamingMediaDecoder has been deprecated.*",
	category=UserWarning,
)

TEST_DATASET_NAME = "VoxCeleb1"
TEST_DATASET_VARIANT = "Vox-CA20"

DEFAULT_SMALL_CKPT = "checkpoints/siamese_full_finetune/full_ft_lr1e4_small_seed42/siamese_best.pt"
DEFAULT_LARGE_CKPT = "checkpoints/siamese_full_finetune/full_ft_lr1e4_large_seed42/siamese_best.pt"
DEFAULT_OUTPUT_DIR = Path("logs") / "svd_analysis" / "ca20"


def build_test_dataset() -> PairwiseDataset:
	test_dataset = PairwiseDataset(
		audio_dir=DATASET_INFO[TEST_DATASET_NAME][TEST_DATASET_VARIANT]["AUDIO_DIR"],
		audio_meta_dir=DATASET_INFO[TEST_DATASET_NAME][TEST_DATASET_VARIANT]["AUDIO_DATALIST"],
		audio_meta_csv_path=DATASET_INFO[TEST_DATASET_NAME]["AUDIO_META_DIR"],
	)
	return test_dataset


def _unwrap_checkpoint_state(checkpoint: object) -> Dict[str, torch.Tensor]:
	if isinstance(checkpoint, dict):
		for key in ("model_state_dict", "state_dict", "model"):
			state_dict = checkpoint.get(key)
			if isinstance(state_dict, dict):
				return state_dict
		if all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
			return checkpoint  # type: ignore[return-value]
	raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")


def _normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
	normalized: Dict[str, torch.Tensor] = {}
	for key, value in state_dict.items():
		new_key = key
		if new_key.startswith("module."):
			new_key = new_key.replace("module.", "", 1)
		if new_key.startswith("speaker_encoder."):
			new_key = new_key.replace("speaker_encoder.", "encoder.", 1)
		normalized[new_key] = value
	return normalized


def load_siamese_model(checkpoint_path: str) -> SiameseNetwork:
	if not os.path.exists(checkpoint_path):
		raise FileNotFoundError(f"Cannot find checkpoint: {checkpoint_path}")

	model = SiameseNetwork().to(DEVICE)
	checkpoint = torch.load(checkpoint_path, map_location="cpu")
	state_dict = _normalize_state_dict_keys(_unwrap_checkpoint_state(checkpoint))
	model.load_state_dict(state_dict, strict=False)
	model.eval()
	return model


def load_audio_tensor(wav_path: str) -> torch.Tensor:
	signal, sr = torchaudio.load(wav_path)
	if sr != 16000:
		resampler = T.Resample(orig_freq=sr, new_freq=16000)
		signal = resampler(signal)
	if signal.shape[0] > 1:
		signal = signal.mean(dim=0, keepdim=True)
	return signal.squeeze(0)


def extract_embedding(model: SiameseNetwork, waveform: torch.Tensor) -> torch.Tensor:
	with torch.no_grad():
		embedding = model.encoder(waveform, aug=False)
		embedding = F.normalize(embedding, p=2, dim=1)
	return embedding.squeeze(0).detach().cpu()


def collect_speaker_embeddings(
	dataset: PairwiseDataset,
	small_model: SiameseNetwork,
	large_model: SiameseNetwork,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
	audio_to_speaker: Dict[str, str] = {}
	for _, spk1_id, spk2_id, spk1_path, spk2_path, _, _ in dataset.datalist:
		audio_to_speaker[spk1_path] = spk1_id
		audio_to_speaker[spk2_path] = spk2_id

	ordered_audio_paths = sorted(audio_to_speaker.keys())
	print(f"Found {len(ordered_audio_paths)} unique utterances in {TEST_DATASET_VARIANT}.")

	speaker_embeddings_small: DefaultDict[str, List[torch.Tensor]] = defaultdict(list)
	speaker_embeddings_large: DefaultDict[str, List[torch.Tensor]] = defaultdict(list)

	for audio_path in tqdm(ordered_audio_paths, desc="Extracting embeddings", dynamic_ncols=True):
		wav = load_audio_tensor(audio_path).to(DEVICE)
		wav = wav.unsqueeze(0)
		speaker_id = audio_to_speaker[audio_path]

		emb_small = extract_embedding(small_model, wav)
		emb_large = extract_embedding(large_model, wav)

		speaker_embeddings_small[speaker_id].append(emb_small)
		speaker_embeddings_large[speaker_id].append(emb_large)

	speaker_ids = sorted(speaker_embeddings_small.keys())
	if speaker_ids != sorted(speaker_embeddings_large.keys()):
		raise RuntimeError("Small and large expert speaker sets do not match.")

	def _stack_and_average(embeddings_by_speaker: DefaultDict[str, List[torch.Tensor]]) -> np.ndarray:
		rows: List[np.ndarray] = []
		for speaker_id in speaker_ids:
			stacked = torch.stack(embeddings_by_speaker[speaker_id], dim=0)
			mean_embedding = stacked.mean(dim=0)
			mean_embedding = F.normalize(mean_embedding.unsqueeze(0), p=2, dim=1).squeeze(0)
			rows.append(mean_embedding.cpu().numpy())
		return np.stack(rows, axis=0)

	return speaker_ids, _stack_and_average(speaker_embeddings_small), _stack_and_average(speaker_embeddings_large)


def compute_cumulative_variance_explained(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	if matrix.ndim != 2:
		raise ValueError(f"Expected a 2D matrix, got shape={matrix.shape}")
	if matrix.shape[0] < 2:
		raise ValueError("Need at least two rows to run SVD.")

	centered = matrix.astype(np.float64, copy=False) - matrix.mean(axis=0, keepdims=True)
	singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
	variance = singular_values**2
	total_variance = float(variance.sum())
	if total_variance <= 0.0:
		raise RuntimeError("Total variance is zero, cannot compute explained variance ratio.")

	explained_variance_ratio = variance / total_variance
	cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
	return singular_values, explained_variance_ratio, cumulative_variance_ratio


def plot_singular_value_decay(
	small_matrix: np.ndarray,
	large_matrix: np.ndarray,
	out_path: Path,
) -> None:
	def _compute_svd_stats(embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
		centered = embeddings.astype(np.float64, copy=False) - embeddings.mean(axis=0, keepdims=True)
		_, S, _ = np.linalg.svd(centered, full_matrices=False)
		eigenvalues = S ** 2
		explained = eigenvalues / eigenvalues.sum()
		return explained, np.cumsum(explained)

	var_small, cum_var_small = _compute_svd_stats(small_matrix)
	var_large, cum_var_large = _compute_svd_stats(large_matrix)

	dim_90_small = int(np.argmax(cum_var_small >= 0.90)) + 1
	dim_90_large = int(np.argmax(cum_var_large >= 0.90)) + 1

	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
	x_axis = np.arange(1, len(var_small) + 1)

	ax1.plot(x_axis, var_small, color="#440154", linewidth=2.5, label="Small Expert")
	ax1.plot(x_axis, var_large, color="#fde725", linewidth=2.5, label="Large Expert")
	ax1.set_yscale("log")
	ax1.set_xlabel("Singular Value Rank (Dimension Index)", fontsize=12)
	ax1.set_ylabel("Explained Variance Ratio (Log Scale)", fontsize=12)
	ax1.set_title("Singular Value Decay (Feature Collapse)", fontsize=14, pad=10)
	ax1.grid(True, linestyle=":", alpha=0.6)
	ax1.legend(fontsize=11)

	ax2.plot(x_axis, cum_var_small, color="#440154", linewidth=2.5, label=f"Small Expert (90% at dim {dim_90_small})")
	ax2.plot(x_axis, cum_var_large, color="#fde725", linewidth=2.5, label=f"Large Expert (90% at dim {dim_90_large})")
	ax2.axhline(y=0.90, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
	ax2.text(x_axis[-1] * 0.8, 0.92, "90% Variance", color="red", fontsize=11)
	ax2.axvline(x=dim_90_small, color="#440154", linestyle=":", linewidth=1.5, alpha=0.6)
	ax2.axvline(x=dim_90_large, color="#fde725", linestyle=":", linewidth=1.5, alpha=0.6)
	ax2.set_xlabel("Number of Dimensions Used", fontsize=12)
	ax2.set_ylabel("Cumulative Explained Variance", fontsize=12)
	ax2.set_title("Effective Dimensionality Analysis", fontsize=14, pad=10)
	ax2.grid(True, linestyle=":", alpha=0.6)
	ax2.legend(fontsize=11, loc="lower right")

	plt.tight_layout()
	out_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(out_path, dpi=200, bbox_inches="tight")
	plt.close(fig)



def plot_cumulative_variance(
	results: Sequence[Tuple[str, np.ndarray]],
	out_path: Path,
	k: int,
) -> None:
	plt.figure(figsize=(8.8, 6.6))
	colors = ["#1f77b4", "#d62728"]

	for idx, (label, cumulative_variance) in enumerate(results):
		max_k = min(k, len(cumulative_variance))
		x = np.arange(1, max_k + 1)
		plt.plot(
			x,
			cumulative_variance[:max_k],
			label=label,
			linewidth=2.4,
			color=colors[idx % len(colors)],
		)
		plt.scatter([max_k], [cumulative_variance[max_k - 1]], s=28, color=colors[idx % len(colors)])

	plt.axhline(0.9, linestyle="--", linewidth=1.0, color="#666666", alpha=0.7)
	plt.xlabel("Top-K Singular Values")
	plt.ylabel("Cumulative Variance Explained")
	plt.title(f"CA20 Speaker Embeddings: Cumulative Variance Explained (K={k})")
	plt.ylim(0.0, 1.02)
	plt.xlim(1, max(2, k))
	plt.grid(True, linestyle=":", alpha=0.4)
	plt.legend(loc="lower right")
	plt.tight_layout()
	out_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(out_path, dpi=200)
	plt.close()


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="SVD analysis for CA20 small/large expert embeddings.")
	parser.add_argument("--k", type=int, default=None, help="Optional manual cap for K. Defaults to the maximum available SVD dimension.")
	parser.add_argument("--small-checkpoint", type=str, default=DEFAULT_SMALL_CKPT, help="Path to the small expert checkpoint.")
	parser.add_argument("--large-checkpoint", type=str, default=DEFAULT_LARGE_CKPT, help="Path to the large expert checkpoint.")
	parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Directory used to save figures and arrays.")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	print("Loading CA20 test split...")
	dataset = build_test_dataset()

	print("Loading experts...")
	small_model = load_siamese_model(args.small_checkpoint)
	large_model = load_siamese_model(args.large_checkpoint)

	print("Collecting speaker-level embeddings...")
	speaker_ids, small_matrix, large_matrix = collect_speaker_embeddings(dataset, small_model, large_model)

	print(f"Speaker count: {len(speaker_ids)}")
	print(f"Small matrix shape: {small_matrix.shape}")
	print(f"Large matrix shape: {large_matrix.shape}")

	print("Running SVD...")
	small_singular_values, small_explained, small_cumulative = compute_cumulative_variance_explained(small_matrix)
	large_singular_values, large_explained, large_cumulative = compute_cumulative_variance_explained(large_matrix)

	max_k = min(len(small_cumulative), len(large_cumulative))
	plot_k = min(args.k, max_k) if args.k is not None else max_k
	plot_path = output_dir / f"{TEST_DATASET_VARIANT}_small_vs_large_cumulative_variance_k{plot_k}.png"
	plot_cumulative_variance(
		[("Small Expert", small_cumulative), ("Large Expert", large_cumulative)],
		plot_path,
		plot_k,
	)

	np.savez_compressed(
		output_dir / f"{TEST_DATASET_VARIANT}_svd_stats.npz",
		speaker_ids=np.array(speaker_ids),
		small_matrix=small_matrix,
		large_matrix=large_matrix,
		small_singular_values=small_singular_values,
		large_singular_values=large_singular_values,
		small_explained_variance=small_explained,
		large_explained_variance=large_explained,
		small_cumulative_variance=small_cumulative,
		large_cumulative_variance=large_cumulative,
	)

	decay_path = output_dir / f"{TEST_DATASET_VARIANT}_singular_value_decay.png"
	plot_singular_value_decay(small_matrix, large_matrix, decay_path)

	print(f"Saved plot: {plot_path}")
	print(f"Saved decay plot: {decay_path}")
	print(f"Saved stats: {output_dir / f'{TEST_DATASET_VARIANT}_svd_stats.npz'}")


if __name__ == "__main__":
	main()
