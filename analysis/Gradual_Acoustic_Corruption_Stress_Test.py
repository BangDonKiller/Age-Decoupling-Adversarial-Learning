"""
Gradual acoustic corruption stress test on Vox-O.

流程：
1) 使用 VoxCeleb1 / Vox-O 測試集。
2) 對每個 SNR 條件（Clean, 20, 10, 0）逐筆加白噪音。
3) 分別用 Small / Medium / Large 三個 full-finetune 專家計算 EER。
4) 繪製 SNR vs EER(%) 折線圖。
"""

import csv
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.vox1_loader import PairwiseDataset
from model.disentangled_model.siamese_network import SiameseNetwork
from params.param import DATASET_INFO, DEVICE, NUM_WORKERS
from tool.EER import compute_eer

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
TEST_DATASET_VARIANT = "Vox-O"

# 依需求：SNR 由高到低（越小越吵），並加入 Clean 對照。
SNR_LEVELS = [20, 15, 10, 5, 0]

# 固定噪音亂數種子，確保每次執行結果可重現。
NOISE_BASE_SEED = 42

EXPERT_WEIGHTS: List[Tuple[str, str]] = [
	("Small", "checkpoints/siamese_full_finetune/full_ft_lr1e4_small_seed42/siamese_best.pt"),
	("Medium", "checkpoints/siamese_full_finetune/full_ft_lr1e4_medium_seed42/siamese_best.pt"),
	("Large", "checkpoints/siamese_full_finetune/full_ft_lr1e4_large_seed42/siamese_best.pt"),
]

OUTPUT_DIR = Path("logs") / "gradual_acoustic_corruption"
EER_SUMMARY_CSV = OUTPUT_DIR / f"{TEST_DATASET_VARIANT}_snr_eer_summary.csv"
PLOT_PATH = OUTPUT_DIR / f"{TEST_DATASET_VARIANT}_snr_eer_curve.png"


def add_white_noise(waveform: torch.Tensor, snr_db: float, seed: int) -> torch.Tensor:
	"""對 waveform 注入指定 SNR(dB) 的高斯白噪音。"""
	eps = 1e-12
	generator = torch.Generator(device=waveform.device)
	generator.manual_seed(int(seed))

	signal_power = waveform.pow(2).mean(dim=-1, keepdim=True).clamp_min(eps)
	noise = torch.randn(
		waveform.shape,
		dtype=waveform.dtype,
		device=waveform.device,
		generator=generator,
	)
	noise_power = noise.pow(2).mean(dim=-1, keepdim=True).clamp_min(eps)

	factor = torch.sqrt(signal_power / (noise_power * (10 ** (snr_db / 10.0))))
	noisy_waveform = waveform + factor * noise
	return torch.clamp(noisy_waveform, -1.0, 1.0)


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


def compute_expert_eer(scores: List[float], labels: List[int]) -> Dict[str, float]:
	scores_np = np.asarray(scores, dtype=np.float32)
	labels_np = np.asarray(labels, dtype=np.int32)

	num_pos = int(np.sum(labels_np == 1))
	num_neg = int(np.sum(labels_np == 0))

	if len(scores_np) == 0 or num_pos == 0 or num_neg == 0:
		return {
			"eer": float("nan"),
			"eer_percent": float("nan"),
			"num_pos": num_pos,
			"num_neg": num_neg,
		}

	eer, _ = compute_eer(scores_np, labels_np)
	return {
		"eer": float(eer),
		"eer_percent": float(eer * 100.0),
		"num_pos": num_pos,
		"num_neg": num_neg,
	}


def evaluate_snr_condition(
	loader: DataLoader,
	experts: List[Tuple[str, SiameseNetwork]],
	snr_db: Optional[float],
) -> Dict[str, Dict[str, float]]:
	stats: Dict[str, Dict[str, List[float]]] = {
		expert_name: {"scores": [], "labels": []} for expert_name, _ in experts
	}

	condition_name = "Clean" if snr_db is None else f"{snr_db} dB"
	for pair_index, batch in enumerate(
		tqdm(loader, desc=f"Evaluating {condition_name}", dynamic_ncols=True),
		start=1,
	):
		pair_label, _, _, wav1, wav2, _, _ = batch

		label_value = int(pair_label.item())
		wav1 = wav1.to(DEVICE, non_blocking=True)
		wav2 = wav2.to(DEVICE, non_blocking=True)

		if snr_db is not None:
			snr_tag = int(float(snr_db) * 10)
			seed_wav1 = NOISE_BASE_SEED + snr_tag * 1_000_003 + pair_index * 2
			seed_wav2 = seed_wav1 + 1
			wav1 = add_white_noise(wav1, snr_db, seed_wav1)
			wav2 = add_white_noise(wav2, snr_db, seed_wav2)

		with torch.no_grad():
			for expert_name, model in experts:
				_, _, cosine_score = model(wav1, wav2, spec_aug=False)
				score_value = float(cosine_score.detach().cpu().item())
				stats[expert_name]["scores"].append(score_value)
				stats[expert_name]["labels"].append(label_value)

	metrics: Dict[str, Dict[str, float]] = {}
	for expert_name in stats:
		metrics[expert_name] = compute_expert_eer(
			stats[expert_name]["scores"],
			stats[expert_name]["labels"],
		)
	return metrics


def save_summary_csv(rows: List[Dict[str, float]], output_path: Path) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with open(output_path, mode="w", newline="", encoding="utf-8") as file_obj:
		writer = csv.writer(file_obj)
		writer.writerow(["snr", "expert", "eer", "eer_percent", "num_pos", "num_neg"])
		for row in rows:
			writer.writerow(
				[
					row["snr"],
					row["expert"],
					row["eer"],
					row["eer_percent"],
					row["num_pos"],
					row["num_neg"],
				]
			)


def plot_eer_curve(
	x_labels: List[str],
	expert_to_eers: Dict[str, List[float]],
	output_path: Path,
) -> None:
	x = np.arange(len(x_labels), dtype=np.int32)

	plt.figure(figsize=(9, 6))
	for expert_name, eers in expert_to_eers.items():
		plt.plot(x, eers, marker="o", linewidth=2, label=expert_name)

	plt.xticks(x, x_labels)
	plt.xlabel("SNR (dB)")
	plt.ylabel("EER (%)")
	plt.title(f"Vox-O Gradual Acoustic Corruption Stress Test ({TEST_DATASET_VARIANT})")
	plt.grid(True, linestyle="--", alpha=0.35)
	plt.legend(loc="best")
	plt.tight_layout()

	output_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(output_path, dpi=300)
	plt.close()


def main() -> None:
	print(f"測試集: {TEST_DATASET_NAME}/{TEST_DATASET_VARIANT}")
	print(f"SNR levels: {SNR_LEVELS}")

	print("建立 Vox-O 測試資料集...")
	test_loader = build_test_loader(TEST_DATASET_NAME, TEST_DATASET_VARIANT)

	print("載入三個專家權重...")
	experts: List[Tuple[str, SiameseNetwork]] = []
	for expert_name, weight_path in EXPERT_WEIGHTS:
		print(f"- {expert_name}: {weight_path}")
		experts.append((expert_name, load_expert_model(weight_path)))

	snr_sweep: List[Optional[float]] = [None] + SNR_LEVELS
	x_labels = ["Clean"] + [str(snr) for snr in SNR_LEVELS]

	expert_to_eers: Dict[str, List[float]] = {name: [] for name, _ in experts}
	summary_rows: List[Dict[str, float]] = []

	print("\n開始逐 SNR 條件評估...")
	for snr in snr_sweep:
		snr_label = "Clean" if snr is None else str(int(snr))
		metrics = evaluate_snr_condition(test_loader, experts, snr)

		print(f"\n[{snr_label}] EER")
		for expert_name, _ in experts:
			m = metrics[expert_name]
			expert_to_eers[expert_name].append(m["eer_percent"])
			summary_rows.append(
				{
					"snr": snr_label,
					"expert": expert_name,
					"eer": m["eer"],
					"eer_percent": m["eer_percent"],
					"num_pos": m["num_pos"],
					"num_neg": m["num_neg"],
				}
			)
			print(
				f"{expert_name}: EER={m['eer_percent']:.2f}% "
				f"(pos={m['num_pos']}, neg={m['num_neg']})"
			)

	save_summary_csv(summary_rows, EER_SUMMARY_CSV)
	plot_eer_curve(x_labels, expert_to_eers, PLOT_PATH)

	print("\n=== 輸出檔案 ===")
	print(f"EER 摘要: {EER_SUMMARY_CSV}")
	print(f"折線圖: {PLOT_PATH}")


if __name__ == "__main__":
	main()
