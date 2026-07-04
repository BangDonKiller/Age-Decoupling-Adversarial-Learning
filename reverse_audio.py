"""Reverse-audio verification analysis for Vox-O and Vox-CA20.

This script evaluates same-speaker pairs with the Small and Large experts,
computes cosine similarity on the original waveform and on the time-reversed
waveform, prints a summary report, and saves a grouped bar chart.
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.vox1_loader import PairwiseDataset
from model.disentangled_model.siamese_network import SiameseNetwork
from params.param import BATCH_SIZE, DATASET_INFO, DEVICE, NUM_WORKERS


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
DEFAULT_DATASET_VARIANTS = ("Vox-O", "Vox-CA20")
DEFAULT_SMALL_CKPT = "checkpoints/siamese_full_finetune/full_ft_lr1e4_small_seed42/siamese_best.pt"
DEFAULT_LARGE_CKPT = "checkpoints/siamese_full_finetune/full_ft_lr1e4_large_seed42/siamese_best.pt"
DEFAULT_OUTPUT_DIR = Path("logs") / "reverse_audio_analysis"


def load_audio_tensor(wav_path: str) -> torch.Tensor:
	"""Load a waveform and convert it to mono 16 kHz."""
	signal, sample_rate = torchaudio.load(wav_path)
	if sample_rate != 16000:
		resampler = T.Resample(orig_freq=sample_rate, new_freq=16000)
		signal = resampler(signal)
	if signal.shape[0] > 1:
		signal = signal.mean(dim=0, keepdim=True)
	return signal.squeeze(0)


def reverse_waveform(waveform: torch.Tensor) -> torch.Tensor:
	"""Reverse the waveform along the time axis."""
	return torch.flip(waveform, dims=[-1]).contiguous()


def get_dataset_info() -> Dict[str, Dict[str, object]]:
	return DATASET_INFO


def build_pair_dataset(dataset_name: str, dataset_variant: str) -> PairwiseDataset:
	dataset_info = get_dataset_info()
	return PairwiseDataset(
		audio_dir=dataset_info[dataset_name][dataset_variant]["AUDIO_DIR"],
		audio_meta_dir=dataset_info[dataset_name][dataset_variant]["AUDIO_DATALIST"],
		audio_meta_csv_path=dataset_info[dataset_name]["AUDIO_META_DIR"],
	)


def build_pair_loader(dataset_name: str, dataset_variant: str, batch_size: int) -> DataLoader:
	dataset = build_pair_dataset(dataset_name, dataset_variant)
	return DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=NUM_WORKERS,
		pin_memory=torch.cuda.is_available(),
	)


def _unwrap_checkpoint_state(checkpoint: object) -> Dict[str, torch.Tensor]:
	if isinstance(checkpoint, dict):
		for key in ("model_state_dict", "state_dict", "model"):
			state_dict = checkpoint.get(key)
			if isinstance(state_dict, dict):
				return state_dict
		if all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
			return checkpoint
	raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")


def _normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
	normalized: Dict[str, torch.Tensor] = {}
	for key, value in state_dict.items():
		new_key = key
		if new_key.startswith("module."):
			new_key = new_key.replace("module.", "", 1)
		if new_key.startswith("speaker_encoder."):
			new_key = new_key.replace("speaker_encoder.", "", 1)
		normalized[new_key] = value
	return normalized


def load_expert_model(checkpoint_path: str, device: str) -> SiameseNetwork:
	if not os.path.exists(checkpoint_path):
		raise FileNotFoundError(f"Cannot find checkpoint: {checkpoint_path}")

	model = SiameseNetwork().to(device)
	checkpoint = torch.load(checkpoint_path, map_location="cpu")
	state_dict = _normalize_state_dict_keys(_unwrap_checkpoint_state(checkpoint))
	model.load_state_dict(state_dict, strict=False)
	model.eval()
	return model


def _format_float(value: float, digits: int = 4) -> str:
	if value is None or not np.isfinite(value):
		return "nan"
	return f"{value:.{digits}f}"


def _aggregate_results(results_df: pd.DataFrame) -> pd.DataFrame:
	agg = (
		results_df.groupby(["dataset_variant", "expert"], as_index=False)
		.agg(
			mean_normal_score=("normal_score", "mean"),
			mean_reversed_score=("reversed_score", "mean"),
			mean_drop=("drop", "mean"),
			count=("drop", "size"),
		)
		.sort_values(["dataset_variant", "expert"], kind="stable")
	)
	return agg


def print_summary_report(summary_df: pd.DataFrame) -> None:
	columns = ["Dataset", "Expert", "Count", "Normal", "Reversed", "Drop"]
	rows: List[List[str]] = []
	for _, row in summary_df.iterrows():
		rows.append(
			[
				str(row["dataset_variant"]),
				str(row["expert"]),
				str(int(row["count"])),
				_format_float(float(row["mean_normal_score"])),
				_format_float(float(row["mean_reversed_score"])),
				_format_float(float(row["mean_drop"])),
			]
		)

	widths = [len(col) for col in columns]
	for row in rows:
		for idx, value in enumerate(row):
			widths[idx] = max(widths[idx], len(value))

	print("\n=== Reversal Summary ===")
	print("  ".join(col.ljust(widths[idx]) for idx, col in enumerate(columns)))
	print("  ".join("-" * widths[idx] for idx in range(len(columns))))
	for row in rows:
		print("  ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)))
	print()


def collect_reversal_results(
	models: Dict[str, SiameseNetwork],
	loader: DataLoader,
	dataset_variant: str,
	device: str,
	max_pairs: int | None = None,
) -> pd.DataFrame:
	rows: List[Dict[str, object]] = []
	processed_pairs = 0

	with torch.no_grad():
		for batch in tqdm(loader, desc=f"Evaluating {dataset_variant}", dynamic_ncols=True):
			pair_label, spk1_id, spk2_id, wav1, wav2, spk1_age, spk2_age = batch
			label_mask = pair_label.long() == 1
			if label_mask.sum().item() == 0:
				processed_pairs += pair_label.size(0)
				continue

			wav1 = wav1.to(device, non_blocking=True)
			wav2 = wav2.to(device, non_blocking=True)
			reversed_wav1 = reverse_waveform(wav1)
			reversed_wav2 = reverse_waveform(wav2)

			for model_name, model in models.items():
				_, _, normal_score = model(wav1, wav2, spec_aug=False)
				_, _, reversed_score = model(reversed_wav1, reversed_wav2, spec_aug=False)

				normal_score_cpu = normal_score.detach().cpu().numpy()
				reversed_score_cpu = reversed_score.detach().cpu().numpy()

				for idx in torch.nonzero(label_mask, as_tuple=False).flatten().tolist():
					rows.append(
						{
							"dataset_variant": dataset_variant,
							"expert": model_name,
							"pair_index": processed_pairs + idx,
							"spk1_id": str(spk1_id[idx]),
							"spk2_id": str(spk2_id[idx]),
							"spk1_age": int(spk1_age[idx].item()),
							"spk2_age": int(spk2_age[idx].item()),
							"normal_score": float(normal_score_cpu[idx]),
							"reversed_score": float(reversed_score_cpu[idx]),
							"drop": float(normal_score_cpu[idx] - reversed_score_cpu[idx]),
						}
					)

			processed_pairs += pair_label.size(0)
			if max_pairs is not None and processed_pairs >= max_pairs:
				break

	results_df = pd.DataFrame(rows)
	if results_df.empty:
		raise RuntimeError(f"No same-speaker pairs were collected for {dataset_variant}.")
	return results_df


def plot_reversal_results(results_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
	"""Plot grouped bar chart for normal/reversed scores and return the summary table."""
	output_path.parent.mkdir(parents=True, exist_ok=True)
	summary_df = _aggregate_results(results_df)

	palette = {
		("Small", "normal"): "#1f77b4",
		("Small", "reversed"): "#9ecae1",
		("Large", "normal"): "#d62728",
		("Large", "reversed"): "#f4a6a6",
	}

	dataset_order = [dataset for dataset in DEFAULT_DATASET_VARIANTS if dataset in set(summary_df["dataset_variant"])]
	if not dataset_order:
		dataset_order = list(summary_df["dataset_variant"].drop_duplicates())
	bar_labels = [("Small", "normal"), ("Small", "reversed"), ("Large", "normal"), ("Large", "reversed")]
	bar_width = 0.18
	offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * bar_width
	x_positions = np.arange(len(dataset_order), dtype=np.float32)
	all_score_values = results_df[["normal_score", "reversed_score"]].to_numpy().astype(np.float32)
	value_span = float(np.nanmax(all_score_values) - np.nanmin(all_score_values)) if all_score_values.size else 1.0
	annot_gap = max(0.015 * max(1.0, value_span), 1e-3)

	fig, ax = plt.subplots(figsize=(11.5, 6.5))

	for dataset_idx, dataset_variant in enumerate(dataset_order):
		for bar_idx, (expert, condition) in enumerate(bar_labels):
			row = summary_df[(summary_df["dataset_variant"] == dataset_variant) & (summary_df["expert"] == expert)]
			if row.empty:
				continue
			value = float(row.iloc[0]["mean_normal_score"] if condition == "normal" else row.iloc[0]["mean_reversed_score"])
			x = x_positions[dataset_idx] + offsets[bar_idx]
			label = f"{expert} {condition.capitalize()}" if dataset_idx == 0 else None
			ax.bar(
				x,
				value,
				width=bar_width,
				color=palette[(expert, condition)],
				edgecolor="#1f2937",
				linewidth=0.8,
				label=label,
			)

	for dataset_idx, dataset_variant in enumerate(dataset_order):
		for expert_idx, expert in enumerate(("Small", "Large")):
			row = summary_df[(summary_df["dataset_variant"] == dataset_variant) & (summary_df["expert"] == expert)]
			if row.empty:
				continue
			row = row.iloc[0]
			normal_value = float(row["mean_normal_score"])
			reversed_value = float(row["mean_reversed_score"])
			drop_value = float(row["mean_drop"])
			x_center = x_positions[dataset_idx] + np.mean(offsets[expert_idx * 2 : expert_idx * 2 + 2])
			y_top = max(normal_value, reversed_value)
			ax.text(
				x_center,
				y_top + annot_gap,
				f"Δ {drop_value:.3f}",
				ha="center",
				va="bottom",
				fontsize=9,
				fontweight="bold",
				color="#111827",
			)

	ax.set_xticks(x_positions)
	ax.set_xticklabels(dataset_order, fontsize=12)
	ax.set_ylabel("Cosine Similarity", fontsize=12)
	ax.set_title("Reverse-Audio Verification on Vox-O and Vox-CA20", fontsize=15, fontweight="bold")
	ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.4)
	ax.set_axisbelow(True)

	legend = ax.legend(ncol=2, frameon=True, fontsize=10, loc="upper right")
	legend.get_frame().set_alpha(0.95)

	fig.tight_layout()
	fig.savefig(output_path, dpi=200, bbox_inches="tight")
	plt.close(fig)
	return summary_df


def save_results_csv(results_df: pd.DataFrame, output_path: Path) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	results_df.to_csv(output_path, index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Evaluate reverse-audio robustness on Vox-O and Vox-CA20.")
	parser.add_argument(
		"--datasets",
		nargs="+",
		default=list(DEFAULT_DATASET_VARIANTS),
		help="Dataset variants to evaluate.",
	)
	parser.add_argument("--small-ckpt", type=str, default=DEFAULT_SMALL_CKPT, help="Path to the small expert checkpoint.")
	parser.add_argument("--large-ckpt", type=str, default=DEFAULT_LARGE_CKPT, help="Path to the large expert checkpoint.")
	parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Evaluation batch size.")
	parser.add_argument("--max-pairs", type=int, default=None, help="Optional cap on same-speaker pairs per dataset.")
	parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Directory for CSV and figure outputs.")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	models = {
		"Small": load_expert_model(args.small_ckpt, DEVICE),
		"Large": load_expert_model(args.large_ckpt, DEVICE),
	}

	all_results: List[pd.DataFrame] = []
	for dataset_variant in args.datasets:
		loader = build_pair_loader(TEST_DATASET_NAME, dataset_variant, batch_size=1)
		results_df = collect_reversal_results(
			models=models,
			loader=loader,
			dataset_variant=dataset_variant,
			device=DEVICE,
			max_pairs=args.max_pairs,
		)
		all_results.append(results_df)

	combined_df = pd.concat(all_results, ignore_index=True)
	results_csv = output_dir / "reverse_audio_results.csv"
	plot_path = output_dir / "reverse_audio_grouped_bar.png"
	summary_df = plot_reversal_results(combined_df, plot_path)
	save_results_csv(combined_df, results_csv)
	print_summary_report(summary_df)
	print(f"Saved results CSV: {results_csv}")
	print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
	main()
