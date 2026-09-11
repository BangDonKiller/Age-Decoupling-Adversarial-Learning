"""Pre-normalized embedding norm analysis for full-finetune ECAPA experts.

This script reuses the existing pairwise dataset metadata as the source of
utterance paths, randomly picks one side from each pair, keeps 1,000 unique
audio paths per split, extracts raw embeddings from the encoder, and plots the
L2-norm KDE for Vox-O and Vox-CA20 on the same figure.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import warnings
from pathlib import Path
from typing import Dict, List, Sequence
from xml.sax.saxutils import escape

import numpy as np

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
TEST_DATASET_VARIANTS = ("Vox-O", "Vox-CA20")
DEFAULT_NUM_SAMPLES = 1000
DEFAULT_SEED = 42
DEFAULT_SMALL_CKPT = "checkpoints/siamese_full_finetune/full_ft_lr1e4_small_seed42/siamese_best.pt"
DEFAULT_LARGE_CKPT = "checkpoints/siamese_full_finetune/full_ft_lr1e4_large_seed42/siamese_best.pt"
DEFAULT_OUTPUT_DIR = Path("logs") / "pre_normalize_analysis"


def get_dataset_info() -> Dict[str, Dict[str, object]]:
	from params.param import DATASET_INFO

	return DATASET_INFO


def get_device() -> str:
	from params.param import DEVICE

	return str(DEVICE)


def get_pairwise_dataset_class():
	from data.vox1_loader import PairwiseDataset

	return PairwiseDataset


def get_siamese_network_class():
	from model.disentangled_model.siamese_network import SiameseNetwork

	return SiameseNetwork


def build_pairwise_dataset(dataset_name: str, dataset_variant: str):
	dataset_info = get_dataset_info()
	PairwiseDataset = get_pairwise_dataset_class()
	return PairwiseDataset(
		audio_dir=dataset_info[dataset_name][dataset_variant]["AUDIO_DIR"],
		audio_meta_dir=dataset_info[dataset_name][dataset_variant]["AUDIO_DATALIST"],
		audio_meta_csv_path=dataset_info[dataset_name]["AUDIO_META_DIR"],
	)


def _unwrap_checkpoint_state(checkpoint: object) -> Dict[str, object]:
	import torch

	if isinstance(checkpoint, dict):
		for key in ("model_state_dict", "state_dict", "model"):
			state_dict = checkpoint.get(key)
			if isinstance(state_dict, dict):
				return state_dict
		if all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
			return checkpoint
	raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")


def _normalize_state_dict_keys(state_dict: Dict[str, object]) -> Dict[str, object]:
	normalized: Dict[str, object] = {}
	for key, value in state_dict.items():
		new_key = key
		if new_key.startswith("module."):
			new_key = new_key.replace("module.", "", 1)
		if new_key.startswith("speaker_encoder."):
			new_key = new_key.replace("speaker_encoder.", "encoder.", 1)
		normalized[new_key] = value
	return normalized


def load_expert_model(checkpoint_path: str, device: str):
	import torch

	if not os.path.exists(checkpoint_path):
		raise FileNotFoundError(f"Cannot find checkpoint: {checkpoint_path}")

	SiameseNetwork = get_siamese_network_class()
	model = SiameseNetwork().to(device)
	checkpoint = torch.load(checkpoint_path, map_location="cpu")
	state_dict = _normalize_state_dict_keys(_unwrap_checkpoint_state(checkpoint))
	model.load_state_dict(state_dict, strict=False)
	model.eval()
	return model


def load_audio_tensor(wav_path: str):
	import torch
	import torchaudio
	import torchaudio.transforms as T

	signal, sample_rate = torchaudio.load(wav_path)
	if sample_rate != 16000:
		resampler = T.Resample(orig_freq=sample_rate, new_freq=16000)
		signal = resampler(signal)
	if signal.shape[0] > 1:
		signal = signal.mean(dim=0, keepdim=True)
	return signal.squeeze(0)


def collect_unique_pointwise_paths(
	dataset,
	num_samples: int,
	seed: int,
	variant_name: str,
) -> List[str]:
	rng = random.Random(seed)
	unique_paths: List[str] = []
	seen = set()

	for _, _, _, speaker1_path, speaker2_path, _, _ in dataset.datalist:
		chosen_path = str(rng.choice((speaker1_path, speaker2_path)))
		if chosen_path in seen:
			continue
		seen.add(chosen_path)
		unique_paths.append(chosen_path)

	if len(unique_paths) < num_samples:
		warnings.warn(
			f"{variant_name}: only {len(unique_paths)} unique audio paths available; "
			f"using all of them instead of the requested {num_samples}.",
			RuntimeWarning,
		)
		rng.shuffle(unique_paths)
		return unique_paths

	rng.shuffle(unique_paths)
	return unique_paths[:num_samples]


def extract_raw_embedding(model, waveform):
	import torch

	with torch.inference_mode():
		raw_embedding = model.encoder(waveform, aug=False)
	return raw_embedding.detach().cpu()


def compute_l2_norms(
	model,
	audio_paths: Sequence[str],
	variant_name: str,
	device: str,
) -> List[Dict[str, object]]:
	import torch

	rows: List[Dict[str, object]] = []

	for audio_path in audio_paths:
		waveform = load_audio_tensor(audio_path).to(device)
		raw_embedding = extract_raw_embedding(model, waveform.unsqueeze(0))
		l2_norm = float(torch.linalg.norm(raw_embedding.squeeze(0), ord=2).item())
		rows.append(
			{
				"dataset_variant": variant_name,
				"audio_path": audio_path,
				"l2_norm": l2_norm,
			}
		)

	return rows


def summarize_norms(rows: Sequence[Dict[str, object]]) -> Dict[str, float]:
	values = np.asarray([float(row["l2_norm"]) for row in rows], dtype=np.float64)
	if values.size == 0:
		return {"count": 0.0, "mean": float("nan"), "std": float("nan"), "median": float("nan")}
	return {
		"count": float(values.size),
		"mean": float(values.mean()),
		"std": float(values.std()),
		"median": float(np.median(values)),
	}


def save_norms_csv(output_path: Path, rows: Sequence[Dict[str, object]]) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with open(output_path, mode="w", newline="", encoding="utf-8") as file_obj:
		writer = csv.writer(file_obj)
		writer.writerow(["expert", "dataset_variant", "audio_path", "l2_norm"])
		for row in rows:
			writer.writerow([
				row["expert"],
				row["dataset_variant"],
				row["audio_path"],
				row["l2_norm"],
			])


def gaussian_kde(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
	values = np.asarray(values, dtype=np.float64)
	if values.size < 2:
		raise ValueError("KDE needs at least two samples.")

	std = float(values.std(ddof=1))
	if not math.isfinite(std) or std <= 0.0:
		std = 1.0

	bandwidth = 1.06 * std * (values.size ** (-1.0 / 5.0))
	bandwidth = max(bandwidth, 1e-3)

	diff = (grid[:, None] - values[None, :]) / bandwidth
	density = np.exp(-0.5 * diff * diff).sum(axis=1)
	density /= values.size * bandwidth * math.sqrt(2.0 * math.pi)
	return density


def _map_x(x: float, left: float, right: float, width: float, margin_left: float) -> float:
	if right <= left:
		return margin_left
	return margin_left + (x - left) / (right - left) * width


def _map_y(y: float, bottom: float, top: float, height: float, margin_top: float) -> float:
	if top <= bottom:
		return margin_top + height
	return margin_top + height - (y - bottom) / (top - bottom) * height


def build_svg_kde_plot(rows: Sequence[Dict[str, object]], output_path: Path) -> None:
	palette = {
		("Small", "Vox-O"): "#1f77b4",
		("Small", "Vox-CA20"): "#1f77b4",
		("Large", "Vox-O"): "#d62728",
		("Large", "Vox-CA20"): "#d62728",
	}
	linestyles = {
		"Vox-O": None,
		"Vox-CA20": "8,5",
	}

	series = []
	for expert_name in ("Small", "Large"):
		for dataset_variant in TEST_DATASET_VARIANTS:
			values = np.asarray(
				[
					float(row["l2_norm"])
					for row in rows
					if row["expert"] == expert_name and row["dataset_variant"] == dataset_variant
				],
				dtype=np.float64,
			)
			if values.size < 2:
				continue
			series.append((expert_name, dataset_variant, values))

	if not series:
		raise RuntimeError("No valid norm series available for plotting.")

	all_values = np.concatenate([values for _, _, values in series])
	x_min = float(all_values.min())
	x_max = float(all_values.max())
	x_pad = max((x_max - x_min) * 0.08, 1e-3)
	x_left = x_min - x_pad
	x_right = x_max + x_pad

	grid = np.linspace(x_left, x_right, 400)
	densities = []
	for expert_name, dataset_variant, values in series:
		density = gaussian_kde(values, grid)
		densities.append((expert_name, dataset_variant, values, density))

	y_max = max(float(density.max()) for _, _, _, density in densities)
	y_max = max(y_max * 1.12, 1e-6)
	y_min = 0.0

	width = 1200.0
	height = 760.0
	margin_left = 110.0
	margin_right = 50.0
	margin_top = 90.0
	margin_bottom = 110.0
	plot_width = width - margin_left - margin_right
	plot_height = height - margin_top - margin_bottom

	lines = []
	lines.append(
		f'<svg xmlns="http://www.w3.org/2000/svg" width="{int(width)}" height="{int(height)}" viewBox="0 0 {int(width)} {int(height)}">'
	)
	lines.append("<rect x='0' y='0' width='100%' height='100%' fill='#ffffff'/>")
	lines.append("<rect x='0' y='0' width='100%' height='100%' fill='none' stroke='#f0f0f0' stroke-width='1'/>")

	lines.append(
		f"<text x='{width / 2:.1f}' y='42' text-anchor='middle' font-family='Arial, sans-serif' font-size='26' font-weight='700' fill='#111827'>Pre-normalized Embedding L2 Norm KDE</text>"
	)
	lines.append(
		f"<text x='{width / 2:.1f}' y='68' text-anchor='middle' font-family='Arial, sans-serif' font-size='14' fill='#4b5563'>Small vs Large experts on Vox-O and Vox-CA20</text>"
	)

	for tick_idx in range(6):
		x_value = x_left + (x_right - x_left) * tick_idx / 5.0
		x_pos = _map_x(x_value, x_left, x_right, plot_width, margin_left)
		lines.append(
			f"<line x1='{x_pos:.2f}' y1='{margin_top:.2f}' x2='{x_pos:.2f}' y2='{margin_top + plot_height:.2f}' stroke='#e5e7eb' stroke-width='1'/>"
		)
		lines.append(
			f"<text x='{x_pos:.2f}' y='{margin_top + plot_height + 28:.2f}' text-anchor='middle' font-family='Arial, sans-serif' font-size='12' fill='#4b5563'>{x_value:.2f}</text>"
		)

	for tick_idx in range(6):
		y_value = y_min + (y_max - y_min) * tick_idx / 5.0
		y_pos = _map_y(y_value, y_min, y_max, plot_height, margin_top)
		lines.append(
			f"<line x1='{margin_left:.2f}' y1='{y_pos:.2f}' x2='{margin_left + plot_width:.2f}' y2='{y_pos:.2f}' stroke='#e5e7eb' stroke-width='1'/>"
		)
		lines.append(
			f"<text x='{margin_left - 12:.2f}' y='{y_pos + 4:.2f}' text-anchor='end' font-family='Arial, sans-serif' font-size='12' fill='#4b5563'>{y_value:.4f}</text>"
		)

	lines.append(
		f"<line x1='{margin_left:.2f}' y1='{margin_top + plot_height:.2f}' x2='{margin_left + plot_width:.2f}' y2='{margin_top + plot_height:.2f}' stroke='#111827' stroke-width='1.4'/>"
	)
	lines.append(
		f"<line x1='{margin_left:.2f}' y1='{margin_top:.2f}' x2='{margin_left:.2f}' y2='{margin_top + plot_height:.2f}' stroke='#111827' stroke-width='1.4'/>"
	)

	lines.append(
		f"<text x='{width / 2:.1f}' y='{height - 32:.2f}' text-anchor='middle' font-family='Arial, sans-serif' font-size='15' fill='#111827'>L2 Norm of Raw Embedding</text>"
	)
	lines.append(
		f"<text x='34' y='{height / 2:.1f}' transform='rotate(-90 34 {height / 2:.1f})' text-anchor='middle' font-family='Arial, sans-serif' font-size='15' fill='#111827'>Density</text>"
	)

	for expert_name, dataset_variant, values, density in densities:
		points = []
		for x_value, y_value in zip(grid, density):
			x_pos = _map_x(float(x_value), x_left, x_right, plot_width, margin_left)
			y_pos = _map_y(float(y_value), y_min, y_max, plot_height, margin_top)
			points.append(f"{x_pos:.2f},{y_pos:.2f}")
		dash = linestyles[dataset_variant]
		stroke = palette[(expert_name, dataset_variant)]
		style = f"stroke:{stroke};stroke-width:3.0;fill:none;"
		if dash:
			style += f"stroke-dasharray:{dash};"
		lines.append(f"<polyline points='{' '.join(points)}' style='{style}' />")

	legend_x = width - 365.0
	legend_y = 110.0
	legend_w = 315.0
	legend_h = 150.0
	lines.append(
		f"<rect x='{legend_x:.2f}' y='{legend_y:.2f}' width='{legend_w:.2f}' height='{legend_h:.2f}' rx='14' fill='#ffffff' stroke='#d1d5db' stroke-width='1.2'/>"
	)
	lines.append(
		f"<text x='{legend_x + 18:.2f}' y='{legend_y + 28:.2f}' font-family='Arial, sans-serif' font-size='14' font-weight='700' fill='#111827'>Legend</text>"
	)

	legend_rows = [
		("Small / Vox-O", palette[("Small", "Vox-O")], None),
		("Small / Vox-CA20", palette[("Small", "Vox-CA20")], linestyles["Vox-CA20"]),
		("Large / Vox-O", palette[("Large", "Vox-O")], None),
		("Large / Vox-CA20", palette[("Large", "Vox-CA20")], linestyles["Vox-CA20"]),
	]
	for idx, (label, color, dash) in enumerate(legend_rows):
		y = legend_y + 55 + idx * 22
		style = f"stroke:{color};stroke-width:3.2;fill:none;"
		if dash:
			style += f"stroke-dasharray:{dash};"
		lines.append(f"<line x1='{legend_x + 18:.2f}' y1='{y:.2f}' x2='{legend_x + 66:.2f}' y2='{y:.2f}' style='{style}' />")
		lines.append(
			f"<text x='{legend_x + 78:.2f}' y='{y + 4:.2f}' font-family='Arial, sans-serif' font-size='12.5' fill='#111827'>{escape(label)}</text>"
		)

	lines.append("</svg>")
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text("\n".join(lines), encoding="utf-8")


def build_argument_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Analyze pre-normalized embedding norms for full-finetune experts.")
	parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES, help="Number of unique utterances per dataset variant.")
	parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for selecting pointwise utterances.")
	parser.add_argument("--small-ckpt", type=str, default=DEFAULT_SMALL_CKPT, help="Path to the small expert checkpoint.")
	parser.add_argument("--large-ckpt", type=str, default=DEFAULT_LARGE_CKPT, help="Path to the large expert checkpoint.")
	parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for CSV and figure outputs.")
	return parser


def main() -> None:
	args = build_argument_parser().parse_args()
	device = get_device()
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	print("建立 pairwise 資料集...")
	datasets = {
		variant: build_pairwise_dataset(TEST_DATASET_NAME, variant)
		for variant in TEST_DATASET_VARIANTS
	}

	print("載入小專家與大專家權重...")
	models = {
		"Small": load_expert_model(args.small_ckpt, device),
		"Large": load_expert_model(args.large_ckpt, device),
	}

	all_rows: List[Dict[str, object]] = []

	for dataset_variant in TEST_DATASET_VARIANTS:
		print(f"抽樣 {dataset_variant} 的 {args.num_samples} 筆不重複音檔路徑...")
		sampled_paths = collect_unique_pointwise_paths(
			datasets[dataset_variant],
			num_samples=args.num_samples,
			seed=args.seed,
			variant_name=dataset_variant,
		)

		variant_rows: List[Dict[str, object]] = []
		for expert_name, model in models.items():
			print(f"- {expert_name} on {dataset_variant}")
			rows = compute_l2_norms(model, sampled_paths, dataset_variant, device)
			for row in rows:
				row["expert"] = expert_name
				variant_rows.append(row)
				all_rows.append(row)

		summary = summarize_norms(variant_rows)
		print(
			f"  {dataset_variant}: count={int(summary['count'])}, mean={summary['mean']:.4f}, "
			f"std={summary['std']:.4f}, median={summary['median']:.4f}"
		)

	csv_path = output_dir / "pre_normalized_embedding_norms.csv"
	fig_path = output_dir / "pre_normalized_embedding_norm_kde.svg"

	save_norms_csv(csv_path, all_rows)
	build_svg_kde_plot(all_rows, fig_path)

	print("\n=== 輸出檔案 ===")
	print(f"CSV: {csv_path}")
	print(f"Figure: {fig_path}")


if __name__ == "__main__":
	main()