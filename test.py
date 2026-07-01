"""
Vox-CA20 正樣本散點圖分析。

作法：
1) 讀取 Vox-CA20 測試集的所有 pair。
2) 只保留 label = 1 的同人樣本。
3) 分別使用小專家與大專家計算 cosine similarity score。
4) 畫出 2D scatter plot：X 軸為小專家分數，Y 軸為大專家分數。

輸出：
1) 散點圖 PNG
2) 正樣本分數 CSV
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

EXPERT_WEIGHTS: List[Tuple[str, str]] = [
	("Small", "checkpoints/siamese_full_finetune/full_ft_lr1e4_small_seed42/siamese_best.pt"),
	("Large", "checkpoints/siamese_full_finetune/full_ft_lr1e4_large_seed42/siamese_best.pt"),
]

OUTPUT_DIR = Path("logs") / "positive_pair_scatter"
SCORES_CSV = OUTPUT_DIR / f"{TEST_DATASET_VARIANT}_positive_pair_scores.csv"
SCATTER_FIG = OUTPUT_DIR / f"{TEST_DATASET_VARIANT}_small_vs_large_scatter.png"


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


def collect_positive_pair_scores(
	experts: List[Tuple[str, SiameseNetwork]],
	loader: DataLoader,
) -> List[Dict[str, float]]:
	records: List[Dict[str, float]] = []

	for pair_index, batch in enumerate(
		tqdm(loader, desc="Scoring positive pairs", dynamic_ncols=True),
		start=1,
	):
		pair_label, spk1_id, spk2_id, wav1, wav2, spk1_age, spk2_age = batch
		label_value = int(pair_label.item())

		if label_value != 1:
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


def save_scores_csv(output_path: Path, records: List[Dict[str, float]], expert_names: List[str]) -> None:
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


def plot_positive_pair_scatter(records: List[Dict[str, float]], output_path: Path) -> None:
	if not records:
		raise RuntimeError("沒有找到任何 label=1 的樣本，無法繪圖。")

	x_scores = np.asarray([row["Small_score"] for row in records], dtype=np.float32)
	y_scores = np.asarray([row["Large_score"] for row in records], dtype=np.float32)
	age_gap = np.asarray([abs(int(row["spk1_age"]) - int(row["spk2_age"])) for row in records], dtype=np.float32)

	plt.figure(figsize=(8.5, 7.0))
	scatter = plt.scatter(
		x_scores,
		y_scores,
		c=age_gap,
		cmap="viridis",
		s=28,
		alpha=0.75,
		edgecolors="none",
	)
	min_score = float(min(x_scores.min(), y_scores.min()))
	max_score = float(max(x_scores.max(), y_scores.max()))
	plt.plot([min_score, max_score], [min_score, max_score], linestyle="--", color="#666666", linewidth=1.2, label="y = x")
	plt.colorbar(scatter, label="Age Gap")
	plt.xlabel("Small Expert Cosine Similarity")
	plt.ylabel("Large Expert Cosine Similarity")
	plt.title(f"Vox-CA20 Positive Pairs Scatter Plot\nN = {len(records)}")
	plt.legend(loc="best")
	plt.tight_layout()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(str(output_path), dpi=180)
	plt.close()


def main() -> None:
	loader = build_test_loader(TEST_DATASET_NAME, TEST_DATASET_VARIANT)
	experts = [(expert_name, load_expert_model(weight_path)) for expert_name, weight_path in EXPERT_WEIGHTS]

	records = collect_positive_pair_scores(experts, loader)
	save_scores_csv(SCORES_CSV, records, [expert_name for expert_name, _ in EXPERT_WEIGHTS])
	plot_positive_pair_scatter(records, SCATTER_FIG)

	print(f"已輸出正樣本分數 CSV: {SCORES_CSV}")
	print(f"已輸出散點圖: {SCATTER_FIG}")


if __name__ == "__main__":
	main()
