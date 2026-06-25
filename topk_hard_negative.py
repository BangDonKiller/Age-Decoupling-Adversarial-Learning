"""
Top-K hard negative analysis for three full-finetune experts on Vox-CA20.

Goal:
1) Collect all negative pairs (label=0, imposters) from Vox-CA20.
2) Score each pair with Small / Medium / Large experts.
3) Sort negative pairs by cosine similarity in descending order.
4) Export the Top-K hardest negative pairs to support the claim that
   the Large expert suppresses extreme false alarms better.

Outputs:
1) Full negative-pair score CSV
2) Top-K hard-negative CSV for each expert (long format)
3) Per-expert summary CSV
4) Top-K rank curve figure
"""

import csv
import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.vox1_loader import PairwiseDataset
from model.disentangled_model.siamese_network import SiameseNetwork
from params.param import DATASET_INFO, DEVICE, NUM_WORKERS

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
TOP_K = 100

EXPERT_WEIGHTS: List[Tuple[str, str]] = [
	("Small", "checkpoints/siamese_full_finetune/full_ft_lr1e4_small_seed42/siamese_best.pt"),
	("Medium", "checkpoints/siamese_full_finetune/full_ft_lr1e4_medium_seed42/siamese_best.pt"),
	("Large", "checkpoints/siamese_full_finetune/full_ft_lr1e4_large_seed42/siamese_best.pt"),
]

OUTPUT_DIR = Path("logs") / "topk_hard_negative"
NEGATIVE_SCORES_CSV = OUTPUT_DIR / f"{TEST_DATASET_VARIANT}_negative_scores.csv"
TOPK_CSV = OUTPUT_DIR / f"{TEST_DATASET_VARIANT}_top{TOP_K}_hard_negatives.csv"
SUMMARY_CSV = OUTPUT_DIR / f"{TEST_DATASET_VARIANT}_top{TOP_K}_summary.csv"
RANK_CURVE_FIG = OUTPUT_DIR / f"{TEST_DATASET_VARIANT}_top{TOP_K}_rank_curve.png"


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


def _unwrap_singleton(value):
	if isinstance(value, (list, tuple)) and len(value) == 1:
		return value[0]
	return value


def score_negative_pairs(
	experts: List[Tuple[str, SiameseNetwork]],
	loader: DataLoader,
) -> List[Dict[str, float]]:
	records: List[Dict[str, float]] = []

	for pair_index, batch in enumerate(
		tqdm(loader, desc="Scoring negative pairs", dynamic_ncols=True),
		start=1,
	):
		pair_label, spk1_id, spk2_id, wav1, wav2, spk1_age, spk2_age = batch
		label_value = int(pair_label.item())

		if label_value != 0:
			continue

		wav1 = wav1.to(DEVICE, non_blocking=True)
		wav2 = wav2.to(DEVICE, non_blocking=True)

		record: Dict[str, float] = {
			"pair_index": pair_index,
			"label": label_value,
			"spk1_id": str(_unwrap_singleton(spk1_id)),
			"spk2_id": str(_unwrap_singleton(spk2_id)),
			"spk1_age": int(spk1_age.item()),
			"spk2_age": int(spk2_age.item()),
		}

		with torch.no_grad():
			for expert_name, model in experts:
				_, _, cosine_score = model(wav1, wav2, spec_aug=False)
				record[f"{expert_name}_score"] = float(cosine_score.detach().cpu().item())

		records.append(record)

	return records


def sort_topk_hard_negatives(
	records: List[Dict[str, float]],
	expert_name: str,
	top_k: int,
) -> List[Dict[str, float]]:
	return sorted(
		records,
		key=lambda row: row[f"{expert_name}_score"],
		reverse=True,
	)[: min(top_k, len(records))]


def save_negative_scores_csv(
	output_path: Path,
	records: List[Dict[str, float]],
	expert_names: List[str],
) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with open(output_path, mode="w", newline="", encoding="utf-8") as file_obj:
		writer = csv.writer(file_obj)
		writer.writerow(
			[
				"pair_index",
				"label",
				"spk1_id",
				"spk2_id",
				"spk1_age",
				"spk2_age",
				*[f"{expert_name}_score" for expert_name in expert_names],
			]
		)

		for row in records:
			writer.writerow(
				[
					row["pair_index"],
					row["label"],
					row["spk1_id"],
					row["spk2_id"],
					row["spk1_age"],
					row["spk2_age"],
					*[row[f"{expert_name}_score"] for expert_name in expert_names],
				]
			)


def save_topk_csv(
	output_path: Path,
	records: List[Dict[str, float]],
	expert_names: List[str],
	top_k: int,
) -> None:
	with open(output_path, mode="w", newline="", encoding="utf-8") as file_obj:
		writer = csv.writer(file_obj)
		writer.writerow(
			[
				"anchor_expert",
				"rank",
				"pair_index",
				"spk1_id",
				"spk2_id",
				"spk1_age",
				"spk2_age",
				*[f"{expert_name}_score" for expert_name in expert_names],
			]
		)

		for expert_name in expert_names:
			topk_records = sort_topk_hard_negatives(records, expert_name, top_k)
			for rank, row in enumerate(topk_records, start=1):
				writer.writerow(
					[
						expert_name,
						rank,
						row["pair_index"],
						row["spk1_id"],
						row["spk2_id"],
						row["spk1_age"],
						row["spk2_age"],
						*[row[f"{expert_name}_score"] for expert_name in expert_names],
					]
				)


def build_summary_rows(
	records: List[Dict[str, float]],
	expert_names: List[str],
	top_k: int,
) -> List[Dict[str, float]]:
	summary_rows: List[Dict[str, float]] = []

	for expert_name in expert_names:
		all_scores = np.asarray([row[f"{expert_name}_score"] for row in records], dtype=np.float32)
		topk_records = sort_topk_hard_negatives(records, expert_name, top_k)
		topk_scores = np.asarray([row[f"{expert_name}_score"] for row in topk_records], dtype=np.float32)

		if len(topk_scores) == 0:
			summary_rows.append(
				{
					"expert": expert_name,
					"num_negative_pairs": len(records),
					"hardest_score": float("nan"),
					"topk_mean": float("nan"),
					"topk_median": float("nan"),
					"topk_min": float("nan"),
					"topk_max": float("nan"),
				}
			)
			continue

		summary_rows.append(
			{
				"expert": expert_name,
				"num_negative_pairs": len(records),
				"hardest_score": float(np.max(all_scores)),
				"topk_mean": float(np.mean(topk_scores)),
				"topk_median": float(np.median(topk_scores)),
				"topk_min": float(np.min(topk_scores)),
				"topk_max": float(np.max(topk_scores)),
			}
		)

	return summary_rows


def save_summary_csv(output_path: Path, summary_rows: List[Dict[str, float]]) -> None:
	with open(output_path, mode="w", newline="", encoding="utf-8") as file_obj:
		writer = csv.writer(file_obj)
		writer.writerow(
			[
				"expert",
				"num_negative_pairs",
				"hardest_score",
				"topk_mean",
				"topk_median",
				"topk_min",
				"topk_max",
			]
		)
		for row in summary_rows:
			writer.writerow(
				[
					row["expert"],
					row["num_negative_pairs"],
					row["hardest_score"],
					row["topk_mean"],
					row["topk_median"],
					row["topk_min"],
					row["topk_max"],
				]
			)


def print_summary_table(summary_rows: List[Dict[str, float]]) -> None:
	expert_label_map = {
		"Small": "小專家",
		"Medium": "中專家",
		"Large": "大專家",
	}

	header_title = f"=== Top-{TOP_K} 困難負樣本統計 ==="
	header_line = (
		f"{'專家類型':>10}"
		f"{'Top-' + str(TOP_K) + ' 最高分 (Max)':>24}"
		f"{'Top-' + str(TOP_K) + ' 平均分 (Mean)':>25}"
		f"{'Top-' + str(TOP_K) + ' 最低分 (Min)':>24}"
	)

	print(f"\n{header_title}")
	print(header_line)
	for row in summary_rows:
		expert_label = expert_label_map.get(row["expert"], row["expert"])
		print(
			f"{expert_label:>10}"
			f"{row['topk_max']:>24.6f}"
			f"{row['topk_mean']:>25.6f}"
			f"{row['topk_min']:>24.6f}"
		)


def plot_topk_rank_curve(
	output_path: Path,
	records: List[Dict[str, float]],
	expert_names: List[str],
	top_k: int,
) -> None:
	palette = {
		"Small": "#1f77b4",
		"Medium": "#2ca02c",
		"Large": "#d62728",
	}

	plt.figure(figsize=(9.5, 6))
	for expert_name in expert_names:
		topk_records = sort_topk_hard_negatives(records, expert_name, top_k)
		topk_scores = [row[f"{expert_name}_score"] for row in topk_records]
		ranks = np.arange(1, len(topk_scores) + 1)
		plt.plot(
			ranks,
			topk_scores,
			marker="o",
			markersize=3.5,
			linewidth=2,
			color=palette.get(expert_name),
			label=f"{expert_name} Top-{len(topk_scores)}",
		)

	plt.title(f"Top-{top_k} Hard Negative Rank Curve ({TEST_DATASET_NAME}/{TEST_DATASET_VARIANT})")
	plt.xlabel("Rank among negative pairs (higher score = harder)")
	plt.ylabel("Cosine Similarity Score")
	plt.grid(True, linestyle="--", alpha=0.3)
	plt.legend(loc="best")
	plt.tight_layout()
	plt.savefig(output_path, dpi=300)
	plt.close()


def main() -> None:
	if len(EXPERT_WEIGHTS) != 3:
		raise ValueError("EXPERT_WEIGHTS 必須剛好有三個專家")

	print("建立 Vox-CA20 測試資料集...")
	test_loader = build_test_loader(TEST_DATASET_NAME, TEST_DATASET_VARIANT)

	print("載入三個 full-finetune 專家權重...")
	experts: List[Tuple[str, SiameseNetwork]] = []
	for expert_name, weight_path in EXPERT_WEIGHTS:
		print(f"- {expert_name}: {weight_path}")
		experts.append((expert_name, load_expert_model(weight_path)))

	print("開始蒐集負樣本分數...")
	records = score_negative_pairs(experts, test_loader)
	if not records:
		raise RuntimeError("在測試資料集中找不到任何負樣本，無法進行 hard negative 分析。")

	expert_names = [expert_name for expert_name, _ in experts]
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

	save_negative_scores_csv(NEGATIVE_SCORES_CSV, records, expert_names)
	save_topk_csv(TOPK_CSV, records, expert_names, TOP_K)
	summary_rows = build_summary_rows(records, expert_names, TOP_K)
	save_summary_csv(SUMMARY_CSV, summary_rows)
	plot_topk_rank_curve(RANK_CURVE_FIG, records, expert_names, TOP_K)

	print_summary_table(summary_rows)

	print("\n=== 輸出檔案 ===")
	print(f"完整負樣本分數: {NEGATIVE_SCORES_CSV}")
	print(f"Top-{TOP_K} 負樣本排名: {TOPK_CSV}")
	print(f"Top-{TOP_K} 摘要: {SUMMARY_CSV}")
	print(f"Top-{TOP_K} rank curve: {RANK_CURVE_FIG}")


if __name__ == "__main__":
	main()
