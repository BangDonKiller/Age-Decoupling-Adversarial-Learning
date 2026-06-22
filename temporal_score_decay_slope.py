"""
Temporal Score Decay Slope (Beta) 分析腳本。

功能：
1) 載入 Siamese full-finetune 的 small / medium / large 模型（固定同一 seed）。
2) 針對 Vox-O 與 Vox-CA20 計算每個 pair 的 cosine 分數與 age gap。
3) 嚴格篩選 label == 1（同人正樣本）後，對三個專家分別進行一維線性迴歸。
4) 在 console 輸出 slope(beta)、intercept、R-squared。
5) 輸出散佈 + 線性擬合趨勢圖。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import linregress
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.vox1_loader import PairwiseDataset
from model.disentangled_model.siamese_network import SiameseNetwork
from params.param import DATASET_INFO, DEVICE, NUM_WORKERS


DEFAULT_DATASETS = ["Vox-O", "Vox-CA20"]
DEFAULT_EXPERTS = ["small", "medium", "large"]


def build_ckpt_path(expert_name: str, seed: int) -> Path:
	return (
		Path("checkpoints")
		/ "siamese_full_finetune"
		/ f"full_ft_lr1e4_{expert_name}_seed{seed}"
		/ "siamese_best.pt"
	)


def build_test_loader(dataset_variant: str) -> DataLoader:
	ds_cfg = DATASET_INFO["VoxCeleb1"][dataset_variant]
	meta_csv = DATASET_INFO["VoxCeleb1"]["AUDIO_META_DIR"]

	dataset = PairwiseDataset(
		audio_dir=ds_cfg["AUDIO_DIR"],
		audio_meta_dir=ds_cfg["AUDIO_DATALIST"],
		audio_meta_csv_path=meta_csv,
	)
	return DataLoader(
		dataset,
		batch_size=1,
		shuffle=False,
		num_workers=NUM_WORKERS,
		pin_memory=torch.cuda.is_available(),
	)


def load_model(checkpoint_path: Path) -> SiameseNetwork:
	if not checkpoint_path.exists():
		raise FileNotFoundError(f"找不到 checkpoint: {checkpoint_path}")

	model = SiameseNetwork().to(DEVICE)
	state_dict = torch.load(str(checkpoint_path), map_location="cpu")
	model.load_state_dict(state_dict)
	model.eval()
	return model


@torch.no_grad()
def infer_scores_for_dataset(
	dataset_variant: str,
	expert_models: Dict[str, SiameseNetwork],
) -> pd.DataFrame:
	loader = build_test_loader(dataset_variant)

	rows: List[Dict[str, float]] = []
	for batch in tqdm(loader, desc=f"Scoring {dataset_variant}", unit="pair"):
		label, _, _, wav1, wav2, age1, age2 = batch

		label_value = int(label.item())
		age1_value = int(age1.item())
		age2_value = int(age2.item())
		if age1_value < 0 or age2_value < 0:
			continue

		wav1 = wav1.to(DEVICE, non_blocking=True)
		wav2 = wav2.to(DEVICE, non_blocking=True)
		age_gap = abs(age1_value - age2_value)

		_, _, score_small = expert_models["small"](wav1, wav2, spec_aug=False)
		_, _, score_medium = expert_models["medium"](wav1, wav2, spec_aug=False)
		_, _, score_large = expert_models["large"](wav1, wav2, spec_aug=False)

		rows.append(
			{
				"dataset": dataset_variant,
				"label": label_value,
				"age_gap": float(age_gap),
				"score_small": float(score_small.item()),
				"score_medium": float(score_medium.item()),
				"score_large": float(score_large.item()),
			}
		)

	if not rows:
		return pd.DataFrame(columns=["dataset", "label", "age_gap", "score_small", "score_medium", "score_large"])
	return pd.DataFrame(rows)


def compute_temporal_decay_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
	positive_df = df[df["label"] == 1].copy()
	if positive_df.empty:
		raise ValueError("篩選 label == 1 後沒有資料，無法計算 Temporal Score Decay Slope。")

	stats: Dict[str, Dict[str, float]] = {}
	for expert_name, score_col in [
		("small", "score_small"),
		("medium", "score_medium"),
		("large", "score_large"),
	]:
		regression = linregress(positive_df["age_gap"], positive_df[score_col])
		stats[expert_name] = {
			"slope": float(regression.slope),
			"intercept": float(regression.intercept),
			"r2": float(regression.rvalue ** 2),
		}

	return stats


def print_stats(dataset_variant: str, stats: Dict[str, Dict[str, float]]) -> None:
	print("\n" + "=" * 80)
	print(f"Temporal Score Decay Slope (Beta) - {dataset_variant} (label == 1)")
	print("=" * 80)
	for expert_name in ["small", "medium", "large"]:
		item = stats[expert_name]
		print(
			f"{expert_name.capitalize():>6} Expert | "
			f"Slope (Beta): {item['slope']:.6f} | "
			f"Intercept: {item['intercept']:.6f} | "
			f"R-squared: {item['r2']:.6f}"
		)


def plot_decay_trend(
	positive_df: pd.DataFrame,
	stats: Dict[str, Dict[str, float]],
	dataset_variant: str,
	output_dir: Path,
	scatter_alpha: float,
	scatter_enabled: bool,
) -> Path:
	sns.set_theme(style="whitegrid")
	fig, ax = plt.subplots(figsize=(10, 6))

	x_line = pd.Series(sorted(positive_df["age_gap"].unique()), dtype=float)
	if x_line.empty:
		raise ValueError("沒有可用的 age_gap 值可繪圖。")

	expert_styles = [
		("small", "score_small", "#1f77b4", "Small Expert"),
		("medium", "score_medium", "#ff7f0e", "Medium Expert"),
		("large", "score_large", "#2ca02c", "Large Expert"),
	]

	for expert_name, score_col, color, display_name in expert_styles:
		if scatter_enabled:
			sns.scatterplot(
				data=positive_df,
				x="age_gap",
				y=score_col,
				s=10,
				alpha=scatter_alpha,
				color=color,
				linewidth=0,
				ax=ax,
				legend=False,
			)

		slope = stats[expert_name]["slope"]
		intercept = stats[expert_name]["intercept"]
		y_line = slope * x_line + intercept
		ax.plot(
			x_line,
			y_line,
			color=color,
			linewidth=2.5,
			label=f"{display_name} (Slope: {slope:.3f})",
		)

	ax.set_xlabel("Age Gap (Years)")
	ax.set_ylabel("Cosine Similarity Score (Positive Pairs)")
	ax.set_title("Temporal Score Decay Analysis across Different Experts")
	ax.legend(loc="best")
	ax.grid(True, linestyle="--", alpha=0.35)
	fig.tight_layout()

	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_dir / f"temporal_decay_{dataset_variant.lower().replace('-', '_')}.png"
	fig.savefig(output_path, dpi=200)
	plt.close(fig)
	return output_path


def run(args: argparse.Namespace) -> None:
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	print("載入三個專家模型（small / medium / large）...")
	expert_models: Dict[str, SiameseNetwork] = {}
	for expert_name in DEFAULT_EXPERTS:
		ckpt_path = build_ckpt_path(expert_name, args.seed)
		print(f"  - {expert_name}: {ckpt_path}")
		expert_models[expert_name] = load_model(ckpt_path)

	all_dataset_frames: List[pd.DataFrame] = []

	for dataset_variant in DEFAULT_DATASETS:
		dataset_df = infer_scores_for_dataset(dataset_variant, expert_models)
		if dataset_df.empty:
			print(f"[警告] {dataset_variant} 沒有可用資料，略過。")
			continue

		dataset_csv = output_dir / f"temporal_scores_{dataset_variant.lower().replace('-', '_')}.csv"
		dataset_df.to_csv(dataset_csv, index=False)
		print(f"已輸出逐筆分數 CSV: {dataset_csv}")

		positive_df = dataset_df[dataset_df["label"] == 1].copy()
		stats = compute_temporal_decay_stats(dataset_df)
		print_stats(dataset_variant, stats)

		figure_path = plot_decay_trend(
			positive_df=positive_df,
			stats=stats,
			dataset_variant=dataset_variant,
			output_dir=output_dir,
			scatter_alpha=args.scatter_alpha,
			scatter_enabled=not args.only_regression_line,
		)
		print(f"已輸出趨勢圖: {figure_path}")

		stats_rows = []
		for expert_name in DEFAULT_EXPERTS:
			item = stats[expert_name]
			stats_rows.append(
				{
					"dataset": dataset_variant,
					"expert": expert_name,
					"slope_beta": item["slope"],
					"intercept": item["intercept"],
					"r_squared": item["r2"],
					"positive_count": int(len(positive_df)),
				}
			)
		stats_df = pd.DataFrame(stats_rows)
		stats_csv = output_dir / f"temporal_decay_stats_{dataset_variant.lower().replace('-', '_')}.csv"
		stats_df.to_csv(stats_csv, index=False)
		print(f"已輸出迴歸統計 CSV: {stats_csv}")

		all_dataset_frames.append(dataset_df)

	if all_dataset_frames:
		merged_df = pd.concat(all_dataset_frames, axis=0, ignore_index=True)
		merged_csv = output_dir / "temporal_scores_vox_o_vox_ca20_merged.csv"
		merged_df.to_csv(merged_csv, index=False)
		print(f"\n已輸出合併 CSV: {merged_csv}")
	else:
		print("\n沒有任何資料完成分析。")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Temporal Score Decay Slope (Beta) analysis")
	parser.add_argument("--seed", type=int, default=42, help="固定同一組 small/medium/large 權重 seed")
	parser.add_argument(
		"--output-dir",
		type=str,
		default="logs/siamese_full_finetune/temporal_decay",
		help="輸出資料夾",
	)
	parser.add_argument(
		"--scatter-alpha",
		type=float,
		default=0.1,
		help="散佈點透明度（預設 0.1）",
	)
	parser.add_argument(
		"--only-regression-line",
		action="store_true",
		help="只畫三條回歸線，不畫散佈點",
	)
	return parser.parse_args()


if __name__ == "__main__":
	run(parse_args())
