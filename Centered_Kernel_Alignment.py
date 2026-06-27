"""
Linear CKA analysis for three full-finetune experts.

流程：
1) 分別在 Vox-O 與 Vox-CA20 蒐集同一組測試 utterances。
2) 使用 Small / Medium / Large 三個 full-finetune expert 提取 embedding。
3) 計算三者兩兩之間的 linear CKA similarity。
4) 為每個資料集輸出 3x3 heatmap，並額外輸出一張雙子圖方便比較。
"""

import argparse
import csv
import os
import warnings
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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

EXPERT_WEIGHTS: List[Tuple[str, str]] = [
	("Small", "checkpoints/siamese_full_finetune/full_ft_lr1e4_small_seed42/siamese_best.pt"),
	("Medium", "checkpoints/siamese_full_finetune/full_ft_lr1e4_medium_seed42/siamese_best.pt"),
	("Large", "checkpoints/siamese_full_finetune/full_ft_lr1e4_large_seed42/siamese_best.pt"),
]

OUTPUT_DIR = Path("logs") / "centered_kernel_alignment"
SUMMARY_CSV = OUTPUT_DIR / "three_expert_linear_cka_summary.csv"
COMBINED_HEATMAP = OUTPUT_DIR / "vox_o_vox_ca20_linear_cka_heatmaps.png"


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

	def __getitem__(self, index: int) -> Dict[str, object]:
		record = self.records[index]
		waveform = self._load_waveform(str(record["audio_path"]))
		return {
			"speaker_id": str(record["speaker_id"]),
			"age": int(record["age"]),
			"audio_path": str(record["audio_path"]),
			"waveform": waveform,
		}


def collate_utterances(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
	waveforms = [item["waveform"] for item in batch]
	padded_waveforms = pad_sequence(waveforms, batch_first=True)
	return {
		"speaker_id": [str(item["speaker_id"]) for item in batch],
		"age": torch.tensor([int(item["age"]) for item in batch], dtype=torch.long),
		"audio_path": [str(item["audio_path"]) for item in batch],
		"waveform": padded_waveforms,
	}


def build_pairwise_dataset(dataset_name: str, dataset_variant: str) -> PairwiseDataset:
	return PairwiseDataset(
		audio_dir=DATASET_INFO[dataset_name][dataset_variant]["AUDIO_DIR"],
		audio_meta_dir=DATASET_INFO[dataset_name][dataset_variant]["AUDIO_DATALIST"],
		audio_meta_csv_path=DATASET_INFO[dataset_name]["AUDIO_META_DIR"],
	)


def collect_unique_utterances(dataset_name: str, dataset_variant: str) -> List[Dict[str, object]]:
	dataset = build_pairwise_dataset(dataset_name, dataset_variant)
	unique_records: Dict[str, Dict[str, object]] = {}

	for _, spk1_id, spk2_id, spk1_path, spk2_path, spk1_age, spk2_age in dataset.datalist:
		if spk1_path not in unique_records:
			unique_records[spk1_path] = {
				"speaker_id": spk1_id,
				"age": int(spk1_age),
				"audio_path": spk1_path,
			}
		if spk2_path not in unique_records:
			unique_records[spk2_path] = {
				"speaker_id": spk2_id,
				"age": int(spk2_age),
				"audio_path": spk2_path,
			}

	records = list(unique_records.values())
	records.sort(key=lambda item: str(item["audio_path"]))
	return records


def build_utterance_loader(utterance_records: Sequence[Dict[str, object]], batch_size: int) -> DataLoader:
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


def extract_embeddings(model: SiameseNetwork, loader: DataLoader, expert_name: str) -> torch.Tensor:
	embedding_batches: List[torch.Tensor] = []

	with torch.inference_mode():
		for batch in tqdm(loader, desc=f"Extracting {expert_name} embeddings", dynamic_ncols=True):
			waveforms = batch["waveform"].to(DEVICE, non_blocking=True)
			embeddings = model.encoder(waveforms, aug=False)
			embedding_batches.append(embeddings.detach().cpu().float())

	if not embedding_batches:
		raise RuntimeError(f"{expert_name} 沒有成功提取任何 embedding。")

	return torch.cat(embedding_batches, dim=0)


def compute_linear_cka(embeddings_a: torch.Tensor, embeddings_b: torch.Tensor) -> float:
	if embeddings_a.ndim != 2 or embeddings_b.ndim != 2:
		raise ValueError("linear CKA 需要 2D embedding matrix。")
	if embeddings_a.shape[0] != embeddings_b.shape[0]:
		raise ValueError("兩個 embedding matrix 的樣本數必須一致。")

	x = embeddings_a - embeddings_a.mean(dim=0, keepdim=True)
	y = embeddings_b - embeddings_b.mean(dim=0, keepdim=True)

	cross_covariance = torch.matmul(x.T, y)
	x_covariance = torch.matmul(x.T, x)
	y_covariance = torch.matmul(y.T, y)

	numerator = torch.sum(cross_covariance * cross_covariance)
	denominator = torch.linalg.norm(x_covariance, ord="fro") * torch.linalg.norm(y_covariance, ord="fro")
	if torch.isclose(denominator, torch.tensor(0.0, dtype=denominator.dtype)):
		return float("nan")

	return float((numerator / denominator).item())


def build_similarity_matrix(expert_to_embeddings: Dict[str, torch.Tensor], expert_names: Sequence[str]) -> np.ndarray:
	matrix = np.zeros((len(expert_names), len(expert_names)), dtype=np.float32)

	for row_idx, expert_a in enumerate(expert_names):
		for col_idx, expert_b in enumerate(expert_names):
			matrix[row_idx, col_idx] = compute_linear_cka(
				expert_to_embeddings[expert_a],
				expert_to_embeddings[expert_b],
			)

	return matrix


def save_summary_csv(rows: Sequence[Dict[str, object]], output_path: Path) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with open(output_path, mode="w", newline="", encoding="utf-8") as file_obj:
		writer = csv.writer(file_obj)
		writer.writerow(["dataset", "expert_a", "expert_b", "linear_cka"])
		for row in rows:
			writer.writerow([row["dataset"], row["expert_a"], row["expert_b"], row["linear_cka"]])


def plot_heatmap(matrix: np.ndarray, labels: Sequence[str], title: str, output_path: Path) -> None:
	fig, ax = plt.subplots(figsize=(6.5, 5.5))
	image = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)

	ax.set_xticks(np.arange(len(labels)))
	ax.set_yticks(np.arange(len(labels)))
	ax.set_xticklabels(labels)
	ax.set_yticklabels(labels)
	ax.set_title(title)

	for row_idx in range(matrix.shape[0]):
		for col_idx in range(matrix.shape[1]):
			value = matrix[row_idx, col_idx]
			ax.text(
				col_idx,
				row_idx,
				f"{value:.4f}",
				ha="center",
				va="center",
				color="white" if value < 0.6 else "black",
				fontsize=10,
			)

	color_bar = fig.colorbar(image, ax=ax)
	color_bar.set_label("Linear CKA")
	fig.tight_layout()

	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, dpi=300)
	plt.close(fig)


def plot_combined_heatmaps(
	dataset_to_matrix: Dict[str, np.ndarray],
	labels: Sequence[str],
	output_path: Path,
) -> None:
	fig, axes = plt.subplots(1, len(dataset_to_matrix), figsize=(12.5, 5.2))
	if len(dataset_to_matrix) == 1:
		axes = [axes]

	last_image = None
	for axis, (dataset_variant, matrix) in zip(axes, dataset_to_matrix.items()):
		last_image = axis.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)
		axis.set_xticks(np.arange(len(labels)))
		axis.set_yticks(np.arange(len(labels)))
		axis.set_xticklabels(labels)
		axis.set_yticklabels(labels)
		axis.set_title(dataset_variant)

		for row_idx in range(matrix.shape[0]):
			for col_idx in range(matrix.shape[1]):
				value = matrix[row_idx, col_idx]
				axis.text(
					col_idx,
					row_idx,
					f"{value:.4f}",
					ha="center",
					va="center",
					color="white" if value < 0.6 else "black",
					fontsize=10,
				)

	if last_image is not None:
		color_bar = fig.colorbar(last_image, ax=axes, shrink=0.92)
		color_bar.set_label("Linear CKA")

	fig.suptitle("Three-Expert Linear CKA Heatmaps")
	fig.tight_layout()
	fig.savefig(output_path, dpi=300)
	plt.close(fig)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Three-expert linear CKA analysis")
	parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
	parser.add_argument(
		"--max-utterances",
		type=int,
		default=None,
		help="若指定則只取每個資料集前 N 個 unique utterances。",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	print(f"資料集: {TEST_DATASET_NAME} / {', '.join(TEST_DATASET_VARIANTS)}")
	print(f"裝置: {DEVICE}")

	print("載入三個專家權重...")
	experts: List[Tuple[str, SiameseNetwork]] = []
	for expert_name, weight_path in EXPERT_WEIGHTS:
		print(f"- {expert_name}: {weight_path}")
		experts.append((expert_name, load_expert_model(weight_path)))

	expert_names = [expert_name for expert_name, _ in experts]
	summary_rows: List[Dict[str, object]] = []
	dataset_to_matrix: Dict[str, np.ndarray] = {}

	for dataset_variant in TEST_DATASET_VARIANTS:
		print(f"\n=== {dataset_variant} ===")
		utterance_records = collect_unique_utterances(TEST_DATASET_NAME, dataset_variant)
		if args.max_utterances is not None:
			utterance_records = utterance_records[: args.max_utterances]
		if not utterance_records:
			raise RuntimeError(f"{dataset_variant} 沒有可用 utterances。")

		print(f"使用 utterance 數量: {len(utterance_records)}")
		utterance_loader = build_utterance_loader(utterance_records, args.batch_size)

		expert_to_embeddings: Dict[str, torch.Tensor] = {}
		for expert_name, model in experts:
			expert_to_embeddings[expert_name] = extract_embeddings(model, utterance_loader, expert_name)

		similarity_matrix = build_similarity_matrix(expert_to_embeddings, expert_names)
		dataset_to_matrix[dataset_variant] = similarity_matrix

		for row_idx, expert_a in enumerate(expert_names):
			for col_idx, expert_b in enumerate(expert_names):
				summary_rows.append(
					{
						"dataset": dataset_variant,
						"expert_a": expert_a,
						"expert_b": expert_b,
						"linear_cka": float(similarity_matrix[row_idx, col_idx]),
					}
				)

		heatmap_path = OUTPUT_DIR / f"{dataset_variant.lower().replace('-', '_')}_linear_cka_heatmap.png"
		plot_heatmap(
			similarity_matrix,
			expert_names,
			title=f"{dataset_variant} Three-Expert Linear CKA",
			output_path=heatmap_path,
		)

		print("Linear CKA matrix:")
		for row_idx, expert_a in enumerate(expert_names):
			row_values = ", ".join(
				f"{expert_b}={similarity_matrix[row_idx, col_idx]:.4f}"
				for col_idx, expert_b in enumerate(expert_names)
			)
			print(f"  {expert_a}: {row_values}")
		print(f"Heatmap: {heatmap_path}")

	save_summary_csv(summary_rows, SUMMARY_CSV)
	plot_combined_heatmaps(dataset_to_matrix, expert_names, COMBINED_HEATMAP)

	print("\n=== 輸出檔案 ===")
	print(f"CKA 摘要: {SUMMARY_CSV}")
	print(f"雙資料集比較圖: {COMBINED_HEATMAP}")


if __name__ == "__main__":
	main()
