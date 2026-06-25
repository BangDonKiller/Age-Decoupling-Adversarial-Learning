"""
三個專家（大/中/小）zero-shot 測試並繪製同圖 DET 曲線。

參考 `ECAPA-TDNN_siamese_full_finetune_train.py` 的推論流程：
1. 使用 PairwiseDataset 建立 zero-shot 測試資料
2. 逐筆計算 cosine score
3. 統計 EER / minDCF
4. 將三個專家的 DET 曲線畫在同一張圖
"""

import csv
import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import DetCurveDisplay
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

# 這裡可直接替換成你自己的三個 expert 權重路徑。
# 預設命名：Small / Medium / Large
EXPERT_WEIGHTS: List[Tuple[str, str]] = [
	("Small", "checkpoints/siamese_full_finetune/full_ft_lr1e4_small_seed42/siamese_best.pt"),
	("Medium", "checkpoints/siamese_full_finetune/full_ft_lr1e4_medium_seed42/siamese_best.pt"),
	("Large", "checkpoints/siamese_full_finetune/full_ft_lr1e4_large_seed42/siamese_best.pt"),
]

OUTPUT_DIR = Path("logs") / "det_curve"
SCORES_CSV = OUTPUT_DIR / f"{TEST_DATASET_VARIANT}_three_expert_scores.csv"
SUMMARY_CSV = OUTPUT_DIR / f"{TEST_DATASET_VARIANT}_three_expert_det_summary.csv"
DET_FIG_PATH = OUTPUT_DIR / f"{TEST_DATASET_VARIANT}_three_expert_det_curve.png"


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
	"""載入 full-finetune 的 Siamese checkpoint（.pt/.pth）。"""
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
	if len(scores) == 0:
		return {
			"eer": float("nan"),
			"threshold": float("nan"),
			"min_dcf": float("nan"),
			"min_dcf_threshold": float("nan"),
			"min_dcf_fpr": float("nan"),
			"min_dcf_fnr": float("nan"),
			"num_pos": 0,
			"num_neg": 0,
		}

	num_pos = int(np.sum(labels == 1))
	num_neg = int(np.sum(labels == 0))
	if num_pos == 0 or num_neg == 0:
		return {
			"eer": float("nan"),
			"threshold": float("nan"),
			"min_dcf": float("nan"),
			"min_dcf_threshold": float("nan"),
			"min_dcf_fpr": float("nan"),
			"min_dcf_fnr": float("nan"),
			"num_pos": num_pos,
			"num_neg": num_neg,
		}

	eer, threshold = compute_eer(scores, labels)
	fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
	min_dcf, min_dcf_threshold = ComputeMinDcf(fnrs, fprs, thresholds, p_target=0.01, c_miss=1, c_fa=1)

	min_dcf_idx = 0
	for idx, t in enumerate(thresholds):
		if np.isclose(float(t), float(min_dcf_threshold)):
			min_dcf_idx = idx
			break

	min_dcf_fnr = float(fnrs[min_dcf_idx])
	min_dcf_fpr = float(fprs[min_dcf_idx])

	return {
		"eer": float(eer),
		"threshold": float(threshold),
		"min_dcf": float(min_dcf),
		"min_dcf_threshold": float(min_dcf_threshold),
		"min_dcf_fpr": min_dcf_fpr,
		"min_dcf_fnr": min_dcf_fnr,
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
				"threshold",
				"min_dcf",
				"min_dcf_threshold",
				"min_dcf_fpr",
				"min_dcf_fnr",
				"num_pos",
				"num_neg",
			]
		)
		for row in summary_rows:
			writer.writerow(
				[
					row["expert"],
					row["eer"],
					row["threshold"],
					row["min_dcf"],
					row["min_dcf_threshold"],
					row["min_dcf_fpr"],
					row["min_dcf_fnr"],
					row["num_pos"],
					row["num_neg"],
				]
			)


def plot_three_expert_det(
	result_map: Dict[str, Dict[str, np.ndarray]],
	metric_map: Dict[str, Dict[str, float]],
	output_path: Path,
) -> None:
	plt.figure(figsize=(9, 7))
	ax = plt.gca()

	for expert_name, arrays in result_map.items():
		labels = arrays["labels"]
		scores = arrays["scores"]
		eer_value = metric_map[expert_name]["eer"] * 100.0
		curve_name = f"{expert_name} (EER={eer_value:.2f}%)"
		DetCurveDisplay.from_predictions(labels, scores, name=curve_name, ax=ax)

	ax.set_title(f"DET Curve ({TEST_DATASET_NAME}/{TEST_DATASET_VARIANT})")
	ax.grid(True, linestyle="--", alpha=0.35)
	ax.legend(loc="best")

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

	print("開始 zero-shot 推論並記錄三專家分數...")
	result_map = run_zero_shot_three_experts(experts, test_loader, SCORES_CSV)

	metric_map: Dict[str, Dict[str, float]] = {}
	summary_rows: List[Dict[str, float]] = []
	print("\n=== 三專家統計 ===")
	for expert_name in result_map:
		metrics = summarize_metrics(result_map[expert_name]["scores"], result_map[expert_name]["labels"])
		metric_map[expert_name] = metrics
		summary_rows.append({"expert": expert_name, **metrics})
		print(
			f"{expert_name}: "
			f"EER={metrics['eer']:.4f}, "
			f"EER_threshold={metrics['threshold']:.6f}, "
			f"minDCF={metrics['min_dcf']:.4f}, "
			f"minDCF_threshold={metrics['min_dcf_threshold']:.6f}, "
			f"minDCF_FPR={metrics['min_dcf_fpr']:.6f}, "
			f"minDCF_FNR={metrics['min_dcf_fnr']:.6f}, "
			f"pos={metrics['num_pos']}, neg={metrics['num_neg']}"
		)

	save_summary_csv(SUMMARY_CSV, summary_rows)
	plot_three_expert_det(result_map, metric_map, DET_FIG_PATH)

	print("\n=== 輸出檔案 ===")
	print(f"逐筆分數: {SCORES_CSV}")
	print(f"統計摘要: {SUMMARY_CSV}")
	print(f"DET 曲線圖: {DET_FIG_PATH}")


if __name__ == "__main__":
	main()
