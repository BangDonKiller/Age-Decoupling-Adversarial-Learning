"""
Score-CAM style analysis (using Integrated Gradients) for three full-finetune experts.

需求對應：
1) 從 Vox-CA20 與 Vox1-H.S 各挑一對同人 pair（label=1）
2) 計算三位專家的 cosine similarity
3) 對「cosine 分數」對「輸入 spectrogram」做 attribution
4) 比較三專家熱力圖
"""

import os
import random
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import IntegratedGradients

from data.vox1_loader import PairwiseDataset
from model.disentangled_model.siamese_network import SiameseNetwork
from params.param import DATASET_INFO, DEVICE

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


SEED = 42
TEST_DATASET_NAME = "VoxCeleb1"
DATASET_VARIANTS = {
	"LargeGap": "Vox-CA20",
	"ShortGap": "Vox1-H.S",
}

EXPERT_WEIGHTS: List[Tuple[str, str]] = [
	("Small", "checkpoints/siamese_full_finetune/full_ft_lr1e4_small_seed42/siamese_best.pt"),
	("Medium", "checkpoints/siamese_full_finetune/full_ft_lr1e4_medium_seed42/siamese_best.pt"),
	("Large", "checkpoints/siamese_full_finetune/full_ft_lr1e4_large_seed42/siamese_best.pt"),
]

OUTPUT_DIR = Path("logs") / "score_cam"
SUMMARY_TXT = OUTPUT_DIR / "score_cam_pair_summary.txt"
FIG_PATH = OUTPUT_DIR / "score_cam_three_experts_ig.png"


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	os.environ["PYTHONHASHSEED"] = str(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def build_dataset(variant: str) -> PairwiseDataset:
	return PairwiseDataset(
		audio_dir=DATASET_INFO[TEST_DATASET_NAME][variant]["AUDIO_DIR"],
		audio_meta_dir=DATASET_INFO[TEST_DATASET_NAME][variant]["AUDIO_DATALIST"],
		audio_meta_csv_path=DATASET_INFO[TEST_DATASET_NAME]["AUDIO_META_DIR"],
	)


def build_siamese_model() -> SiameseNetwork:
	model = SiameseNetwork().to(DEVICE)
	model.eval()
	return model


def load_expert_model(weight_path: str) -> SiameseNetwork:
	if not os.path.exists(weight_path):
		raise FileNotFoundError(f"找不到權重: {weight_path}")

	model = build_siamese_model()
	checkpoint = torch.load(weight_path, map_location="cpu")
	if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
		state_dict = checkpoint["model_state_dict"]
	else:
		state_dict = checkpoint
	model.load_state_dict(state_dict, strict=False)
	model = model.to(DEVICE)
	model.eval()
	return model


def select_positive_pair(dataset: PairwiseDataset, seed: int):
	positive_indices = []
	for idx, item in enumerate(dataset.datalist):
		is_same, _, _, _, _, age1, age2 = item
		if int(is_same) == 1 and age1 != -1 and age2 != -1:
			positive_indices.append(idx)

	if not positive_indices:
		raise RuntimeError("找不到可用的同人 pair（label=1 且有有效年齡標籤）。")

	rng = random.Random(seed)
	selected_idx = rng.choice(positive_indices)

	label, spk1, spk2, wav1, wav2, age1, age2 = dataset[selected_idx]
	if wav1.dim() == 1:
		wav1 = wav1.unsqueeze(0)
	if wav2.dim() == 1:
		wav2 = wav2.unsqueeze(0)

	return {
		"index": selected_idx,
		"label": int(label.item()),
		"spk1": spk1,
		"spk2": spk2,
		"age1": int(age1.item()),
		"age2": int(age2.item()),
		"wav1": wav1.to(DEVICE),
		"wav2": wav2.to(DEVICE),
	}


class CosineFromSpec(nn.Module):
	"""固定一端 embedding，對另一端 spectrogram 輸出 cosine。"""

	def __init__(self, encoder: nn.Module, fixed_embedding: torch.Tensor):
		super().__init__()
		self.encoder = encoder
		self.register_buffer("fixed_embedding", fixed_embedding)

	def forward(self, spec_input: torch.Tensor) -> torch.Tensor:
		emb = self.encoder.forward_from_spectrogram(spec_input)
		emb = F.normalize(emb, p=2, dim=1)
		fixed = F.normalize(self.fixed_embedding, p=2, dim=1)
		score = F.cosine_similarity(emb, fixed, dim=1)
		return score


def normalize_map(attn: torch.Tensor) -> np.ndarray:
	arr = attn.detach().cpu().numpy()
	arr = np.abs(arr)
	min_v = float(arr.min())
	max_v = float(arr.max())
	if max_v - min_v < 1e-12:
		return np.zeros_like(arr, dtype=np.float32)
	return ((arr - min_v) / (max_v - min_v)).astype(np.float32)


def resize_to_width(spec_2d: torch.Tensor, target_width: int) -> torch.Tensor:
	src = spec_2d.unsqueeze(0).unsqueeze(0)
	out = F.interpolate(src, size=(spec_2d.shape[0], target_width), mode="bilinear", align_corners=False)
	return out.squeeze(0).squeeze(0)


def get_pair_attribution(
	model: SiameseNetwork,
	wav1: torch.Tensor,
	wav2: torch.Tensor,
	n_steps: int = 32,
) -> Dict[str, np.ndarray]:
	encoder = model.encoder
	encoder.eval()

	with torch.no_grad():
		spec1 = encoder.get_spectrogram(wav1).detach()
		spec2 = encoder.get_spectrogram(wav2).detach()

		emb1 = encoder.forward_from_spectrogram(spec1)
		emb2 = encoder.forward_from_spectrogram(spec2)
		emb1 = F.normalize(emb1, p=2, dim=1)
		emb2 = F.normalize(emb2, p=2, dim=1)
		cosine_value = float(F.cosine_similarity(emb1, emb2, dim=1).item())

	spec1_for_attr = spec1.clone().detach().requires_grad_(True)
	spec2_for_attr = spec2.clone().detach().requires_grad_(True)

	ig_1 = IntegratedGradients(CosineFromSpec(encoder, emb2.detach()))
	ig_2 = IntegratedGradients(CosineFromSpec(encoder, emb1.detach()))

	attr1 = ig_1.attribute(
		inputs=spec1_for_attr,
		baselines=torch.zeros_like(spec1_for_attr),
		n_steps=n_steps,
	)
	attr2 = ig_2.attribute(
		inputs=spec2_for_attr,
		baselines=torch.zeros_like(spec2_for_attr),
		n_steps=n_steps,
	)

	heat1 = attr1.squeeze(0)
	heat2 = attr2.squeeze(0)
	spec1_2d = spec1.squeeze(0)
	spec2_2d = spec2.squeeze(0)

	target_width = max(int(spec1_2d.shape[1]), int(spec2_2d.shape[1]))
	heat1 = resize_to_width(heat1, target_width)
	heat2 = resize_to_width(heat2, target_width)
	spec1_2d = resize_to_width(spec1_2d, target_width)
	spec2_2d = resize_to_width(spec2_2d, target_width)

	pair_heat = 0.5 * (heat1 + heat2)
	pair_spec = 0.5 * (spec1_2d + spec2_2d)

	return {
		"cosine": cosine_value,
		"heatmap": normalize_map(pair_heat),
		"spec": pair_spec.detach().cpu().numpy(),
	}


def plot_comparison(results: Dict[str, Dict[str, Dict[str, np.ndarray]]], fig_path: Path) -> None:
	fig, axes = plt.subplots(2, 4, figsize=(20, 9), constrained_layout=True)

	row_order = ["LargeGap", "ShortGap"]
	col_order = ["Small", "Medium", "Large"]

	for row_idx, row_key in enumerate(row_order):
		# 第一欄: 平均 spectrogram
		spec = results[row_key]["Small"]["spec"]
		axes[row_idx, 0].imshow(spec, aspect="auto", origin="lower", cmap="magma")
		axes[row_idx, 0].set_title(f"{row_key}: Avg Spectrogram")
		axes[row_idx, 0].set_xlabel("Time")
		axes[row_idx, 0].set_ylabel("Mel Bin")

		for col_idx, expert_name in enumerate(col_order, start=1):
			heat = results[row_key][expert_name]["heatmap"]
			cosine = results[row_key][expert_name]["cosine"]
			ax = axes[row_idx, col_idx]
			im = ax.imshow(heat, aspect="auto", origin="lower", cmap="jet", vmin=0.0, vmax=1.0)
			ax.set_title(f"{row_key} | {expert_name} | Cos={cosine:.4f}")
			ax.set_xlabel("Time")
			ax.set_ylabel("Mel Bin")
			fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

	fig_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(fig_path, dpi=300)
	plt.close(fig)


def write_summary(
	pair_meta: Dict[str, Dict[str, object]],
	result_map: Dict[str, Dict[str, Dict[str, np.ndarray]]],
	output_path: Path,
) -> None:
	lines: List[str] = []
	for pair_key in ["LargeGap", "ShortGap"]:
		meta = pair_meta[pair_key]
		lines.append(f"[{pair_key}]")
		lines.append(f"dataset_variant={meta['dataset_variant']}")
		lines.append(f"pair_index={meta['index']}")
		lines.append(f"label={meta['label']}")
		lines.append(f"speaker1={meta['spk1']}, age1={meta['age1']}")
		lines.append(f"speaker2={meta['spk2']}, age2={meta['age2']}")
		lines.append(f"age_gap={abs(meta['age1'] - meta['age2'])}")
		for expert_name in ["Small", "Medium", "Large"]:
			lines.append(f"{expert_name}_cosine={result_map[pair_key][expert_name]['cosine']:.6f}")
		lines.append("")

	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
	set_seed(SEED)
	print(f"使用固定隨機種子: {SEED}")

	print("建立兩個測試 split 的資料集...")
	datasets = {
		key: build_dataset(variant)
		for key, variant in DATASET_VARIANTS.items()
	}

	print("從每個 split 挑選一對同人 pair...")
	pair_meta: Dict[str, Dict[str, object]] = {}
	for key, dataset in datasets.items():
		chosen = select_positive_pair(dataset, seed=SEED)
		chosen["dataset_variant"] = DATASET_VARIANTS[key]
		pair_meta[key] = chosen

	print("載入三個專家模型...")
	experts: List[Tuple[str, SiameseNetwork]] = []
	for expert_name, weight_path in EXPERT_WEIGHTS:
		print(f"  - {expert_name}: {weight_path}")
		experts.append((expert_name, load_expert_model(weight_path)))

	print("計算 cosine 與 Integrated Gradients 熱力圖...")
	result_map: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {
		"LargeGap": {},
		"ShortGap": {},
	}

	for pair_key in ["LargeGap", "ShortGap"]:
		wav1 = pair_meta[pair_key]["wav1"]
		wav2 = pair_meta[pair_key]["wav2"]

		for expert_name, model in experts:
			result_map[pair_key][expert_name] = get_pair_attribution(
				model=model,
				wav1=wav1,
				wav2=wav2,
				n_steps=32,
			)

	plot_comparison(result_map, FIG_PATH)
	write_summary(pair_meta, result_map, SUMMARY_TXT)

	print(f"完成。圖檔輸出: {FIG_PATH}")
	print(f"完成。摘要輸出: {SUMMARY_TXT}")


if __name__ == "__main__":
	main()
