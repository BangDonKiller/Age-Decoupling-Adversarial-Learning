"""
三個 full-finetune 專家（Small/Medium/Large）在 zero-shot 測試集上的分數分佈分析。

輸出：
1) Target（同人，label=1）分數 KDE 圖
2) Imposter（異人，label=0）分數 KDE 圖
3) 每個專家的 EER / minDCF 指標摘要 CSV

每張圖都會標註三個專家的：
- EER threshold（虛線）
- minDCF threshold（點線）
"""

import csv
import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.vox1_loader import PairwiseDataset
from model.disentangled_model.siamese_network import SiameseNetwork
from params.param import DATASET_INFO, DEVICE, NUM_WORKERS
from tool.EER import ComputeErrorRates, ComputeMinDcf, compute_eer

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


# ==========================================
# 1) 基本設定
# ==========================================
TEST_DATASET_NAME = "VoxCeleb1"
TEST_DATASET_VARIANT = "Vox-CA20"

EXPERT_WEIGHTS: List[Tuple[str, str]] = [
	("Small", "checkpoints/siamese_full_finetune/full_ft_lr1e4_small_seed42/siamese_best.pt"),
	("Medium", "checkpoints/siamese_full_finetune/full_ft_lr1e4_medium_seed42/siamese_best.pt"),
	("Large", "checkpoints/siamese_full_finetune/full_ft_lr1e4_large_seed42/siamese_best.pt"),
]

OUTPUT_DIR = Path("logs") / "score_distribution"
SCORES_CSV = OUTPUT_DIR / f"{TEST_DATASET_VARIANT}_three_expert_scores.csv"
SUMMARY_CSV = OUTPUT_DIR / f"{TEST_DATASET_VARIANT}_three_expert_threshold_summary.csv"
TARGET_FIG_PATH = OUTPUT_DIR / f"{TEST_DATASET_VARIANT}_target_score_kde.png"
IMPOSTER_FIG_PATH = OUTPUT_DIR / f"{TEST_DATASET_VARIANT}_imposter_score_kde.png"


def build_test_loader(dataset_name: str, dataset_variant: str) -> DataLoader:
	test_dataset = PairwiseDataset(
		audio_dir=DATASET_INFO[dataset_name][dataset_variant]["AUDIO_DIR"],
		audio_meta_dir=DATASET_INFO[dataset_name][dataset_variant]["AUDIO_DATALIST"],
		audio_meta_csv_path=DATASET_INFO[dataset_name]["AUDIO_META_DIR"],
	)
	return DataLoader(
		test_dataset,
		batch_size=1,
		shuffle=False,
		num_workers=NUM_WORKERS,
		pin_memory=torch.cuda.is_available(),
	)


def build_siamese_model() -> SiameseNetwork:
	model = SiameseNetwork().to(DEVICE)
	model.eval()
	return model


def _load_full_checkpoint(model: SiameseNetwork, checkpoint_path: str) -> None:
	checkpoint = torch.load(checkpoint_path, map_location="cpu")
	if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
		state_dict = checkpoint["model_state_dict"]
	else:
		state_dict = checkpoint
	model.load_state_dict(state_dict, strict=False)


def load_expert_model(weight_path: str) -> SiameseNetwork:
	if not os.path.exists(weight_path):
		raise FileNotFoundError(f"找不到權重: {weight_path}")

	model = build_siamese_model()
	_load_full_checkpoint(model, weight_path)
	model = model.to(DEVICE)
	model.eval()
	return model


def run_zero_shot_three_experts(
	experts: List[Tuple[str, SiameseNetwork]],
	loader: DataLoader,
	score_csv_path: Path,
) -> Dict[str, Dict[str, np.ndarray]]:
	score_csv_path.parent.mkdir(parents=True, exist_ok=True)

	stats: Dict[str, Dict[str, List[float]]] = {
		expert_name: {"scores": [], "labels": []} for expert_name, _ in experts
	}

	with open(score_csv_path, mode="w", newline="", encoding="utf-8") as file_obj:
		writer = csv.writer(file_obj)
		writer.writerow(["pair_index", "label", *[f"{name}_score" for name, _ in experts]])

		for pair_index, batch in enumerate(
			tqdm(loader, desc="Zero-shot inference", dynamic_ncols=True),
			start=1,
		):
			pair_label, _, _, wav1, wav2, _, _ = batch

			label_value = int(pair_label.item())
			wav1 = wav1.to(DEVICE, non_blocking=True)
			wav2 = wav2.to(DEVICE, non_blocking=True)

			row = [pair_index, label_value]

			with torch.no_grad():
				for expert_name, model in experts:
					_, _, cosine_score = model(wav1, wav2, spec_aug=False)
					score_value = float(cosine_score.detach().cpu().item())
					stats[expert_name]["scores"].append(score_value)
					stats[expert_name]["labels"].append(label_value)
					row.append(score_value)

			writer.writerow(row)

	packed: Dict[str, Dict[str, np.ndarray]] = {}
	for expert_name in stats:
		packed[expert_name] = {
			"scores": np.asarray(stats[expert_name]["scores"], dtype=np.float32),
			"labels": np.asarray(stats[expert_name]["labels"], dtype=np.int32),
		}
	return packed


def summarize_metrics(scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
	num_pos = int(np.sum(labels == 1))
	num_neg = int(np.sum(labels == 0))

	if len(scores) == 0 or num_pos == 0 or num_neg == 0:
		return {
			"eer": float("nan"),
			"eer_threshold": float("nan"),
			"min_dcf": float("nan"),
			"min_dcf_threshold": float("nan"),
			"num_pos": num_pos,
			"num_neg": num_neg,
		}

	eer, eer_threshold = compute_eer(scores, labels)
	fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
	min_dcf, min_dcf_threshold = ComputeMinDcf(fnrs, fprs, thresholds, p_target=0.01, c_miss=1, c_fa=1)

	return {
		"eer": float(eer),
		"eer_threshold": float(eer_threshold),
		"min_dcf": float(min_dcf),
		"min_dcf_threshold": float(min_dcf_threshold),
		"num_pos": num_pos,
		"num_neg": num_neg,
	}


def save_summary_csv(summary_path: Path, summary_rows: List[Dict[str, float]]) -> None:
	with open(summary_path, mode="w", newline="", encoding="utf-8") as file_obj:
		writer = csv.writer(file_obj)
		writer.writerow(
			[
				"expert",
				"eer",
				"eer_threshold",
				"min_dcf",
				"min_dcf_threshold",
				"num_pos",
				"num_neg",
			]
		)
		for row in summary_rows:
			writer.writerow(
				[
					row["expert"],
					row["eer"],
					row["eer_threshold"],
					row["min_dcf"],
					row["min_dcf_threshold"],
					row["num_pos"],
					row["num_neg"],
				]
			)


def plot_kde_by_label(
	result_map: Dict[str, Dict[str, np.ndarray]],
	metric_map: Dict[str, Dict[str, float]],
	label_value: int,
	output_path: Path,
) -> None:
	if label_value not in (0, 1):
		raise ValueError("label_value 必須是 0 或 1")

	score_type = "Target (同人)" if label_value == 1 else "Imposter (異人)"
	palette = {
		"Small": "#1f77b4",
		"Medium": "#2ca02c",
		"Large": "#d62728",
	}

	plt.figure(figsize=(10, 6.5))
	ax = plt.gca()

	for expert_name, arrays in result_map.items():
		scores = arrays["scores"]
		labels = arrays["labels"]
		selected_scores = scores[labels == label_value]

		if len(selected_scores) < 2:
			continue

		color = palette.get(expert_name, None)
		sns.kdeplot(
			selected_scores,
			bw_adjust=0.85,
			linewidth=2,
			fill=False,
			common_norm=False,
			color=color,
			label=f"{expert_name} density (n={len(selected_scores)})",
			ax=ax,
		)

		eer_threshold = metric_map[expert_name]["eer_threshold"]
		min_dcf_threshold = metric_map[expert_name]["min_dcf_threshold"]

		ax.axvline(
			eer_threshold,
			color=color,
			linestyle="--",
			linewidth=1.8,
			alpha=0.9,
			label=f"{expert_name} EER th={eer_threshold:.4f}",
		)
		ax.axvline(
			min_dcf_threshold,
			color=color,
			linestyle=":",
			linewidth=2.0,
			alpha=0.9,
			label=f"{expert_name} minDCF th={min_dcf_threshold:.4f}",
		)

	ax.set_title(f"{score_type} Score KDE ({TEST_DATASET_NAME}/{TEST_DATASET_VARIANT})")
	ax.set_xlabel("Cosine Similarity Score")
	ax.set_ylabel("Density")
	ax.grid(True, linestyle="--", alpha=0.35)
	ax.legend(loc="best", fontsize=9)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	plt.tight_layout()
	plt.savefig(output_path, dpi=300)
	plt.close()


def main() -> None:
	if len(EXPERT_WEIGHTS) != 3:
		raise ValueError("EXPERT_WEIGHTS 必須剛好有三個 (Small/Medium/Large)")

	print("建立 zero-shot 測試資料集...")
	test_loader = build_test_loader(TEST_DATASET_NAME, TEST_DATASET_VARIANT)

	print("載入三個專家權重...")
	experts: List[Tuple[str, SiameseNetwork]] = []
	for expert_name, weight_path in EXPERT_WEIGHTS:
		print(f"- {expert_name}: {weight_path}")
		experts.append((expert_name, load_expert_model(weight_path)))

	print("開始 zero-shot 推論並記錄分數...")
	result_map = run_zero_shot_three_experts(experts, test_loader, SCORES_CSV)

	metric_map: Dict[str, Dict[str, float]] = {}
	summary_rows: List[Dict[str, float]] = []

	print("\n=== 三專家門檻統計 ===")
	for expert_name in result_map:
		metrics = summarize_metrics(result_map[expert_name]["scores"], result_map[expert_name]["labels"])
		metric_map[expert_name] = metrics
		summary_rows.append({"expert": expert_name, **metrics})
		print(
			f"{expert_name}: "
			f"EER={metrics['eer']:.4f}, "
			f"EER_threshold={metrics['eer_threshold']:.6f}, "
			f"minDCF={metrics['min_dcf']:.4f}, "
			f"minDCF_threshold={metrics['min_dcf_threshold']:.6f}, "
			f"pos={metrics['num_pos']}, neg={metrics['num_neg']}"
		)

	save_summary_csv(SUMMARY_CSV, summary_rows)
	plot_kde_by_label(result_map, metric_map, label_value=1, output_path=TARGET_FIG_PATH)
	plot_kde_by_label(result_map, metric_map, label_value=0, output_path=IMPOSTER_FIG_PATH)

	print("\n=== 輸出檔案 ===")
	print(f"逐筆分數: {SCORES_CSV}")
	print(f"門檻摘要: {SUMMARY_CSV}")
	print(f"正對分佈圖: {TARGET_FIG_PATH}")
	print(f"負對分佈圖: {IMPOSTER_FIG_PATH}")


if __name__ == "__main__":
	main()
