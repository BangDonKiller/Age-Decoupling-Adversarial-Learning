"""
Singular value entropy analysis for three full-finetune experts.

流程：
1) 合併 VoxCeleb1 的 Vox-O 與 Vox-CA20 測試 pair 清單。
2) 從 pair 兩側蒐集單邊 utterance，直接使用全部語音。
3) 分別用 Small / Medium / Large expert 提取 embedding 矩陣 E in [N, D]。
4) 對每個 expert 的 embedding 矩陣做 SVD，取得 singular values。
5) 將 singular values 正規化為機率分布後計算 entropy。
6) 繪製三個 expert 的 singular value spectrum。
"""

import argparse
import csv
import os
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
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
TEST_DATASET_VARIANTS = ("Vox-O", "Vox-CA20")

DEFAULT_BATCH_SIZE = 16
DEFAULT_SEED = 42

EXPERT_WEIGHTS: List[Tuple[str, str]] = [
	("Small", "checkpoints/siamese_full_finetune/full_ft_lr1e4_small_seed42/siamese_best.pt"),
	("Medium", "checkpoints/siamese_full_finetune/full_ft_lr1e4_medium_seed42/siamese_best.pt"),
	("Large", "checkpoints/siamese_full_finetune/full_ft_lr1e4_large_seed42/siamese_best.pt"),
]

OUTPUT_DIR = Path("logs") / "singular_value_entropy"
SUMMARY_CSV = OUTPUT_DIR / "vox_o_vox_ca20_entropy_summary.csv"
SINGULAR_VALUES_CSV = OUTPUT_DIR / "vox_o_vox_ca20_singular_values.csv"
SPECTRUM_FIG = OUTPUT_DIR / "vox_o_vox_ca20_singular_value_spectrum.png"


class UtteranceDataset(Dataset):
	def __init__(self, records: Sequence[Dict[str, object]], target_sample_rate: int = 16000):
		self.records = list(records)
		self.target_sample_rate = target_sample_rate

	def __len__(self) -> int:
		return len(self.records)

	def _load_waveform(self, wav_path: str) -> torch.Tensor:
		signal, sample_rate = torchaudio.load(wav_path)
		if sample_rate != self.target_sample_rate:
			resampler = T.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
			signal = resampler(signal)
		if signal.shape[0] > 1:
			signal = signal.mean(dim=0, keepdim=True)
		return signal.squeeze(0)

	def __getitem__(self, index: int):
		record = self.records[index]
		waveform = self._load_waveform(str(record["audio_path"]))
		return {
			"dataset_variant": str(record["dataset_variant"]),
			"speaker_id": str(record["speaker_id"]),
			"age": int(record["age"]),
			"audio_path": str(record["audio_path"]),
			"waveform": waveform,
		}


def collate_utterances(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
	waveforms = [item["waveform"] for item in batch]
	padded_waveforms = pad_sequence(waveforms, batch_first=True)
	lengths = torch.tensor([waveform.shape[0] for waveform in waveforms], dtype=torch.long)
	return {
		"dataset_variant": [str(item["dataset_variant"]) for item in batch],
		"speaker_id": [str(item["speaker_id"]) for item in batch],
		"age": torch.tensor([int(item["age"]) for item in batch], dtype=torch.long),
		"audio_path": [str(item["audio_path"]) for item in batch],
		"lengths": lengths,
		"waveform": padded_waveforms,
	}


def build_pairwise_dataset(dataset_name: str, dataset_variant: str) -> PairwiseDataset:
	return PairwiseDataset(
		audio_dir=DATASET_INFO[dataset_name][dataset_variant]["AUDIO_DIR"],
		audio_meta_dir=DATASET_INFO[dataset_name][dataset_variant]["AUDIO_DATALIST"],
		audio_meta_csv_path=DATASET_INFO[dataset_name]["AUDIO_META_DIR"],
	)


def collect_unique_utterances(
	dataset_name: str,
	dataset_variants: Iterable[str],
) -> List[Dict[str, object]]:
	unique_records: Dict[str, Dict[str, object]] = {}

	for dataset_variant in dataset_variants:
		dataset = build_pairwise_dataset(dataset_name, dataset_variant)
		for _, spk1_id, spk2_id, spk1_path, spk2_path, spk1_age, spk2_age in dataset.datalist:
			if spk1_path not in unique_records:
				unique_records[spk1_path] = {
					"dataset_variant": dataset_variant,
					"speaker_id": spk1_id,
					"age": int(spk1_age),
					"audio_path": spk1_path,
				}
			if spk2_path not in unique_records:
				unique_records[spk2_path] = {
					"dataset_variant": dataset_variant,
					"speaker_id": spk2_id,
					"age": int(spk2_age),
					"audio_path": spk2_path,
				}

	records = list(unique_records.values())
	print(f"合併後唯一 utterance 數量: {len(records)}")
	return records


def build_utterance_loader(
	utterance_records: Sequence[Dict[str, object]],
	batch_size: int,
) -> DataLoader:
	dataset = UtteranceDataset(utterance_records)
	return DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=NUM_WORKERS,
		pin_memory=torch.cuda.is_available(),
		collate_fn=collate_utterances,
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


def extract_embeddings(
	model: SiameseNetwork,
	loader: DataLoader,
	expert_name: str,
) -> torch.Tensor:
	embedding_batches: List[torch.Tensor] = []

	with torch.inference_mode():
		for batch in tqdm(loader, desc=f"Extracting {expert_name} embeddings", dynamic_ncols=True):
			waveforms = batch["waveform"].to(DEVICE, non_blocking=True)
			embeddings = model.encoder(waveforms, aug=False)
			embedding_batches.append(embeddings.detach().cpu())

	if not embedding_batches:
		raise RuntimeError(f"{expert_name} 沒有成功提取任何 embedding。")

	return torch.cat(embedding_batches, dim=0)


def compute_singular_values(embedding_matrix: torch.Tensor) -> torch.Tensor:
	if hasattr(torch, "linalg") and hasattr(torch.linalg, "svd"):
		_, singular_values, _ = torch.linalg.svd(embedding_matrix, full_matrices=False)
		return singular_values

	_, singular_values, _ = torch.svd(embedding_matrix)
	return singular_values


def compute_singular_value_entropy(embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
	embedding_matrix = embeddings.float()
	singular_values = compute_singular_values(embedding_matrix)
	probabilities = singular_values / singular_values.sum().clamp_min(1e-12)
	entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum().item()
	return singular_values, probabilities, float(entropy)


def save_summary_csv(rows: Sequence[Dict[str, object]], output_path: Path) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with open(output_path, mode="w", newline="", encoding="utf-8") as file_obj:
		writer = csv.writer(file_obj)
		writer.writerow(["expert", "num_samples", "embedding_dim", "entropy"])
		for row in rows:
			writer.writerow(
				[
					row["expert"],
					row["num_samples"],
					row["embedding_dim"],
					row["entropy"],
				]
			)


def save_singular_values_csv(
	expert_to_singular_values: Dict[str, torch.Tensor],
	expert_to_probabilities: Dict[str, torch.Tensor],
	output_path: Path,
) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with open(output_path, mode="w", newline="", encoding="utf-8") as file_obj:
		writer = csv.writer(file_obj)
		writer.writerow(["expert", "rank", "singular_value", "probability"])
		for expert_name, singular_values in expert_to_singular_values.items():
			probabilities = expert_to_probabilities[expert_name]
			for rank, (singular_value, probability) in enumerate(
				zip(singular_values.tolist(), probabilities.tolist()),
				start=1,
			):
				writer.writerow([expert_name, rank, singular_value, probability])


def plot_singular_value_spectrum(
	expert_to_singular_values: Dict[str, torch.Tensor],
	output_path: Path,
) -> None:
	plt.figure(figsize=(9, 6))
	for expert_name, singular_values in expert_to_singular_values.items():
		x = np.arange(1, len(singular_values) + 1, dtype=np.int32)
		plt.plot(x, singular_values.numpy(), linewidth=2, label=expert_name)

	plt.yscale("log")
	plt.xlabel("Rank")
	plt.ylabel("Singular Value")
	plt.title("Vox-O + Vox-CA20 Singular Value Spectrum")
	plt.grid(True, linestyle="--", alpha=0.35)
	plt.legend(loc="best")
	plt.tight_layout()

	output_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(output_path, dpi=300)
	plt.close()


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Three-expert singular value entropy analysis")
	parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
	parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	print(f"資料集: {TEST_DATASET_NAME} / {', '.join(TEST_DATASET_VARIANTS)}")
	print(f"裝置: {DEVICE}")

	utterance_records = collect_unique_utterances(TEST_DATASET_NAME, TEST_DATASET_VARIANTS)
	print(f"使用全部 utterance 數量: {len(utterance_records)}")
	utterance_loader = build_utterance_loader(utterance_records, args.batch_size)

	experts: List[Tuple[str, SiameseNetwork]] = []
	print("載入三個專家權重...")
	for expert_name, weight_path in EXPERT_WEIGHTS:
		print(f"- {expert_name}: {weight_path}")
		experts.append((expert_name, load_expert_model(weight_path)))

	summary_rows: List[Dict[str, object]] = []
	expert_to_singular_values: Dict[str, torch.Tensor] = {}
	expert_to_probabilities: Dict[str, torch.Tensor] = {}

	for expert_name, model in experts:
		embeddings = extract_embeddings(model, utterance_loader, expert_name)
		singular_values, probabilities, entropy = compute_singular_value_entropy(embeddings)

		expert_to_singular_values[expert_name] = singular_values
		expert_to_probabilities[expert_name] = probabilities
		summary_rows.append(
			{
				"expert": expert_name,
				"num_samples": int(embeddings.shape[0]),
				"embedding_dim": int(embeddings.shape[1]),
				"entropy": entropy,
			}
		)

		print(
			f"{expert_name}: E.shape={tuple(embeddings.shape)}, "
			f"entropy={entropy:.6f}, top-5 singular values={singular_values[:5].tolist()}"
		)

	save_summary_csv(summary_rows, SUMMARY_CSV)
	save_singular_values_csv(expert_to_singular_values, expert_to_probabilities, SINGULAR_VALUES_CSV)
	plot_singular_value_spectrum(expert_to_singular_values, SPECTRUM_FIG)

	print("\n=== 輸出檔案 ===")
	print(f"Entropy 摘要: {SUMMARY_CSV}")
	print(f"Singular values: {SINGULAR_VALUES_CSV}")
	print(f"Spectrum figure: {SPECTRUM_FIG}")


if __name__ == "__main__":
	main()
