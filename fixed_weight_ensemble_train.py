"""
Train script for CrossGapFixedRouterEnsemble (3 frozen experts + dynamic router during training).

Loss design:
1) Router guidance loss:
   loss_router = CrossEntropyLoss(weights_logits, true_gap_labels)
2) Final metric loss:
   loss_circle = CircleLoss(S_final, same_person_labels)
3) Joint optimization:
   total_loss = loss_circle + beta * loss_router

Inference behavior:
- No score calibrators.
- The router is converted to one fixed 3-way weight vector from the saved checkpoint.
- Test-time routing is not sample-adaptive.
"""

from __future__ import annotations

import csv
import os
import random
import re
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.vox1_loader import PairwiseDataset as Vox1PairDataset
from data.vox2_loader import Vox2PairDataset
from loss.circleloss import CircleLoss
from model.disentangled_model import CrossGapFixedRouterEnsemble, FixedRouterExpertCheckpointPaths
from params.param import DEVICE, NUM_WORKERS, BATCH_SIZE, DATASET_INFO
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

SEEDS = [42, 1, 2026]
INFERENCE_SEEDS = [42, 1, 2026]
EPOCHS = 10
LR = 1e-3
WEIGHT_DECAY = 1e-5
BETA = 1.0

SAME_LABEL_IS_ONE = True

TRAIN_SPEC_AUG = False

MODE = "train"

TRAIN_DATASET_NAME = "VoxCeleb2"
TRAIN_DATASET_VARIANT = "ensemble"

INFERENCE_DATASETS: list[tuple[str, str]] = [
	("VoxCeleb1", "Vox-O"),
	("VoxCeleb1", "Vox1-H.S"),
	("VoxCeleb1", "Vox-CA5"),
	("VoxCeleb1", "Vox-CA10"),
	("VoxCeleb1", "Vox-CA15"),
	("VoxCeleb1", "Vox-CA20"),
]

INFERENCE_CKPT_PATHS: list[str] = []

TRAIN_AUDIO_DIR = DATASET_INFO[TRAIN_DATASET_NAME][TRAIN_DATASET_VARIANT]["train"]["AUDIO_DIR"]
TRAIN_META_CSV = DATASET_INFO[TRAIN_DATASET_NAME][TRAIN_DATASET_VARIANT]["train"]["AUDIO_META_DIR"]
VAL_AUDIO_DIR = DATASET_INFO[TRAIN_DATASET_NAME][TRAIN_DATASET_VARIANT]["val"]["AUDIO_DIR"]
VAL_META_CSV = DATASET_INFO[TRAIN_DATASET_NAME][TRAIN_DATASET_VARIANT]["val"]["AUDIO_META_DIR"]

SAVE_ROOT = Path("checkpoints/cross_gap_ensemble_fixed_router")
LOG_ROOT = Path("logs/cross_gap_ensemble_fixed_router")
RUN_NAME = "cross_gap_ensemble_fixed_router_no_calibrator"


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def _to_same_labels(pair_labels: torch.Tensor) -> torch.Tensor:
	pair_labels = pair_labels.long()
	if SAME_LABEL_IS_ONE:
		return pair_labels
	return 1 - pair_labels


def build_vox2_loader(audio_dir: str, meta_csv: str, shuffle: bool, batch_size: int) -> DataLoader:
	dataset = Vox2PairDataset(audio_dir=audio_dir, audio_meta_dir=meta_csv)
	return DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=NUM_WORKERS,
		pin_memory=torch.cuda.is_available(),
		drop_last=False,
	)


def build_test_loader(dataset_name: str, dataset_variant: str) -> DataLoader:
	test_audio_dir = DATASET_INFO[dataset_name][dataset_variant]["AUDIO_DIR"]
	test_pair_meta = DATASET_INFO[dataset_name][dataset_variant]["AUDIO_DATALIST"]
	test_pair_meta_csv = DATASET_INFO[dataset_name]["AUDIO_META_DIR"]

	test_dataset = Vox1PairDataset(
		audio_dir=test_audio_dir,
		audio_meta_dir=test_pair_meta,
		audio_meta_csv_path=test_pair_meta_csv,
	)

	return DataLoader(
		test_dataset,
		batch_size=1,
		shuffle=False,
		num_workers=0,
		pin_memory=torch.cuda.is_available(),
		drop_last=False,
	)


def _fmt_metric(value: float, digits: int = 4, suffix: str = "") -> str:
	if isinstance(value, (float, np.floating)) and np.isnan(value):
		return "N/A"
	return f"{value:.{digits}f}{suffix}"


def _sanitize_filename(text: str) -> str:
	return re.sub(r"[^A-Za-z0-9._-]+", "_", text)


def _weights_to_tuple(weights: torch.Tensor) -> tuple[float, float, float]:
	weights = weights.detach().cpu().flatten()
	return float(weights[0].item()), float(weights[1].item()), float(weights[2].item())


def save_router_weights_csv(
	csv_path: Path,
	rows: list[tuple[int, int, float, float, float, float, float, float, float, float]],
) -> None:
	csv_path.parent.mkdir(parents=True, exist_ok=True)
	with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow([
			"sample_index",
			"same_label",
			"score_final",
			"router_mode",
			"weight_small",
			"weight_medium",
			"weight_large",
			"fixed_weight_small",
			"fixed_weight_medium",
			"fixed_weight_large",
			"max_abs_diff_from_fixed",
		])
		writer.writerows(rows)


def build_run_name(seed: int) -> str:
	return f"{RUN_NAME}_seed{seed}"


def build_model() -> CrossGapFixedRouterEnsemble:
	return CrossGapFixedRouterEnsemble(
		C=1024,
		feature_dim=192,
		router_hidden_dim=256,
		router_dropout=0.1,
		router_input_mode="embedding_diff",
		expert_ckpt_paths=FixedRouterExpertCheckpointPaths(),
	).to(DEVICE)


def resolve_inference_ckpt_paths() -> list[str]:
	paths: list[str] = []

	if INFERENCE_CKPT_PATHS:
		paths.extend(INFERENCE_CKPT_PATHS)
	else:
		for seed in INFERENCE_SEEDS:
			run_name = build_run_name(seed)
			paths.append(str(SAVE_ROOT / run_name / "best.pt"))

	unique_paths: list[str] = []
	seen = set()
	for path in paths:
		normalized = str(Path(path))
		if normalized not in seen:
			unique_paths.append(normalized)
			seen.add(normalized)

	return unique_paths


def load_ensemble_checkpoint(model: CrossGapFixedRouterEnsemble, checkpoint_path: Path, device: str) -> dict:
	checkpoint = torch.load(str(checkpoint_path), map_location=device)
	model_state_dict = checkpoint.get("model_state_dict")
	if model_state_dict is None:
		raise KeyError(f"Checkpoint missing model_state_dict: {checkpoint_path}")

	load_result = model.load_state_dict(model_state_dict, strict=True)
	checkpoint_fixed_weights = checkpoint.get("fixed_router_weights")
	if checkpoint_fixed_weights is None:
		router_weights_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_router_weights.pt")
		if router_weights_path.exists():
			router_weights_checkpoint = torch.load(str(router_weights_path), map_location=device)
			checkpoint_fixed_weights = router_weights_checkpoint.get("fixed_router_weights")
	if checkpoint_fixed_weights is not None:
		model.set_fixed_router_weights(checkpoint_fixed_weights.to(device))
	else:
		print("[Warn] Checkpoint missing fixed_router_weights. Falling back to uniform weights.")
	model.enable_fixed_router(True)

	fixed_router_weights = model.router.fixed_weights.detach().cpu()
	print("\n" + "=" * 66)
	print(f"Loaded ensemble checkpoint: {checkpoint_path}")
	print("-" * 66)
	print(f"{'missing_keys':<16}: {len(load_result.missing_keys)}")
	print(f"{'unexpected_keys':<16}: {len(load_result.unexpected_keys)}")
	print(f"{'fixed_weights':<16}: {_weights_to_tuple(fixed_router_weights)}")
	print("=" * 66)
	return checkpoint


def run_inference_for_multiple_ckpts_and_datasets() -> None:
	ckpt_paths = resolve_inference_ckpt_paths()
	if not ckpt_paths:
		raise ValueError("找不到可用的推論 checkpoint，請檢查 INFERENCE_CKPT_PATHS 或 INFERENCE_SEEDS 設定")

	available_ckpts = [path for path in ckpt_paths if os.path.exists(path)]
	missing_ckpts = [path for path in ckpt_paths if not os.path.exists(path)]

	for missing in missing_ckpts:
		print(f"[警告] 找不到 checkpoint，將略過: {missing}")

	if not available_ckpts:
		raise FileNotFoundError("所有推論 checkpoint 都不存在，無法執行推論")

	print("\n推論設定：")
	print(f"- 資料集數量: {len(INFERENCE_DATASETS)}")
	print(f"- 權重數量: {len(available_ckpts)}")

	model = build_model()
	overall_eers = []
	overall_min_dcfs = []

	for dataset_name, dataset_variant in INFERENCE_DATASETS:
		dataset_key = f"{dataset_name}/{dataset_variant}"
		print(f"\n{'=' * 72}")
		print(f"推論資料集: {dataset_key}")
		print(f"{'=' * 72}")

		test_loader = build_test_loader(dataset_name, dataset_variant)
		dataset_eers = []
		dataset_min_dcfs = []

		for ckpt_path in available_ckpts:
			print(f"載入 checkpoint: {ckpt_path}")
			load_ensemble_checkpoint(model, Path(ckpt_path), DEVICE)

			infer_metrics = run_zeroshot_inference_epoch(
				model=model,
				loader=test_loader,
				device=DEVICE,
				collect_router_weights=True,
			)

			dataset_eers.append(infer_metrics["eer"])
			dataset_min_dcfs.append(infer_metrics["min_dcf"])
			overall_eers.append(infer_metrics["eer"])
			overall_min_dcfs.append(infer_metrics["min_dcf"])

			print_zeroshot_metrics_block(
				title=f"[Inference][{dataset_key}][{Path(ckpt_path).parent.name}]",
				metrics=infer_metrics,
			)

			weights_dir = LOG_ROOT / "inference_router_weights"
			dataset_tag = _sanitize_filename(dataset_key)
			ckpt_tag = _sanitize_filename(Path(ckpt_path).parent.name)
			weights_csv_path = weights_dir / f"{dataset_tag}__{ckpt_tag}.csv"
			save_router_weights_csv(weights_csv_path, infer_metrics["router_weight_rows"])
			print(f"[Router Weights] CSV saved: {weights_csv_path}")
			print(
				"[Router Weights Summary] "
				f"mode={infer_metrics['router_mode']} | "
				f"fixed_weights={infer_metrics['fixed_router_weights']} | "
				f"max_abs_diff={infer_metrics['max_abs_diff_from_fixed']:.8f}"
			)

		eer_mean = float(np.mean(dataset_eers)) if dataset_eers else float("nan")
		eer_std = float(np.std(dataset_eers)) if dataset_eers else float("nan")
		min_dcf_mean = float(np.mean(dataset_min_dcfs)) if dataset_min_dcfs else float("nan")
		min_dcf_std = float(np.std(dataset_min_dcfs)) if dataset_min_dcfs else float("nan")

		print(
			f"[資料集統計] {dataset_key} | "
			f"EER mean/std: {eer_mean:.4f}/{eer_std:.4f} | "
			f"minDCF mean/std: {min_dcf_mean:.4f}/{min_dcf_std:.4f}"
		)

	overall_eer_mean = float(np.mean(overall_eers)) if overall_eers else float("nan")
	overall_eer_std = float(np.std(overall_eers)) if overall_eers else float("nan")
	overall_min_dcf_mean = float(np.mean(overall_min_dcfs)) if overall_min_dcfs else float("nan")
	overall_min_dcf_std = float(np.std(overall_min_dcfs)) if overall_min_dcfs else float("nan")

	print("\n" + "=" * 72)
	print("整體統計（所有資料集 x 所有權重）")
	print("=" * 72)
	print(f"EER mean/std: {overall_eer_mean:.4f}/{overall_eer_std:.4f}")
	print(f"minDCF mean/std: {overall_min_dcf_mean:.4f}/{overall_min_dcf_std:.4f}")


def print_metrics_block(title: str, metrics: dict, show_det_metrics: bool) -> None:
	print("\n" + "=" * 66)
	print(f"{title}")
	print("-" * 66)
	print(f"{'total_loss':<16}: {_fmt_metric(metrics['total_loss'])}")
	print(f"{'circle_loss':<16}: {_fmt_metric(metrics['circle_loss'])}")
	print(f"{'router_loss':<16}: {_fmt_metric(metrics['router_loss'])}")
	print(f"{'router_acc':<16}: {_fmt_metric(metrics['router_acc'], digits=2, suffix='%')}")
	print(f"{'pair_acc':<16}: {_fmt_metric(metrics['pair_acc'], digits=2, suffix='%')}")
	if "mean_router_weights" in metrics:
		mean_weights = metrics["mean_router_weights"]
		print(f"{'router_weights':<16}: {tuple(round(x, 4) for x in mean_weights)}")
	if show_det_metrics:
		print(f"{'eer':<16}: {_fmt_metric(metrics['eer'])}")
		print(f"{'min_dcf':<16}: {_fmt_metric(metrics['min_dcf'])}")
		print(f"{'eer_threshold':<16}: {_fmt_metric(metrics['eer_threshold'])}")
	print("=" * 66)


def print_zeroshot_metrics_block(title: str, metrics: dict) -> None:
	print("\n" + "=" * 66)
	print(f"{title}")
	print("-" * 66)
	print(f"{'eer':<16}: {_fmt_metric(metrics['eer'])}")
	print(f"{'min_dcf':<16}: {_fmt_metric(metrics['min_dcf'])}")
	print(f"{'eer_threshold':<16}: {_fmt_metric(metrics['eer_threshold'])}")
	print("=" * 66)


def print_trainable_parameters(model: CrossGapFixedRouterEnsemble) -> None:
	trainable_named_params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
	total_trainable = sum(param.numel() for _, param in trainable_named_params)

	print("\n" + "=" * 66)
	print("Trainable Parameters")
	print("-" * 66)
	print(f"{'param_name':<48} {'shape':<12} {'count':>12}")
	print("-" * 66)
	for name, param in trainable_named_params:
		shape_str = str(tuple(param.shape))
		print(f"{name:<48} {shape_str:<12} {param.numel():>12}")
	print("-" * 66)
	print(f"{'total_trainable':<48} {'':<12} {total_trainable:>12}")
	print("=" * 66)


def run_epoch(
	model: CrossGapFixedRouterEnsemble,
	loader: DataLoader,
	optimizer: torch.optim.Optimizer | None,
	ce_loss_fn: nn.Module,
	circle_loss_fn: nn.Module,
	beta: float,
	device: str,
	epoch_idx: int,
	total_epochs: int,
	dataset_mode: str,
	compute_det_metrics: bool = False,
	collect_router_weights: bool = False,
):
	is_train = optimizer is not None
	if is_train:
		model.enable_fixed_router(False)
	model.train(is_train)

	total_total_loss = 0.0
	total_circle_loss = 0.0
	total_router_loss = 0.0
	total_samples = 0
	total_router_weight_sum = torch.zeros(3, dtype=torch.float64)

	total_router_correct = 0
	total_pair_correct = 0
	all_scores: list[torch.Tensor] = []
	all_labels: list[torch.Tensor] = []
	router_weight_rows: list[tuple[int, int, float, float, float, float, float, float, float, float]] = []
	sample_index = 0

	pbar = tqdm(
		loader,
		desc=f"{'Train' if is_train else 'Val'} {epoch_idx + 1}/{total_epochs}",
		dynamic_ncols=True,
		leave=False,
	)

	for batch in pbar:
		if dataset_mode == "vox2":
			pair_labels, wav1, wav2, _, true_gap_labels = batch
		elif dataset_mode == "vox1":
			pair_labels, _, _, wav1, wav2, _, _ = batch
			true_gap_labels = None
		else:
			raise ValueError(f"Unsupported dataset_mode: {dataset_mode}")

		same_labels = _to_same_labels(pair_labels)

		wav1 = wav1.to(device, non_blocking=True)
		wav2 = wav2.to(device, non_blocking=True)
		same_labels = same_labels.to(device, non_blocking=True)
		if true_gap_labels is not None:
			true_gap_labels = true_gap_labels.to(device, non_blocking=True)

		if is_train:
			optimizer.zero_grad(set_to_none=True)

		s_final, weights, details = model(
			wav1,
			wav2,
			spec_aug=TRAIN_SPEC_AUG if is_train else False,
			return_details=True,
		)
		weights_logits = details.get("router_logits")
		loss_circle = circle_loss_fn(s_final, same_labels)

		if true_gap_labels is not None and weights_logits is not None:
			loss_router = ce_loss_fn(weights_logits, true_gap_labels)
		else:
			loss_router = torch.zeros_like(loss_circle)

		total_loss = loss_circle + beta * loss_router

		if is_train:
			total_loss.backward()
			optimizer.step()

		with torch.no_grad():
			bs = same_labels.size(0)
			total_samples += bs

			total_total_loss += float(total_loss.detach().cpu()) * bs
			total_circle_loss += float(loss_circle.detach().cpu()) * bs
			total_router_loss += float(loss_router.detach().cpu()) * bs
			total_router_weight_sum += weights.detach().cpu().to(torch.float64).sum(dim=0)

			if true_gap_labels is not None and weights_logits is not None:
				router_pred = torch.argmax(weights_logits, dim=1)
				total_router_correct += int((router_pred == true_gap_labels).sum().item())

			pair_pred = (s_final >= 0.0).long()
			total_pair_correct += int((pair_pred == same_labels).sum().item())

			if compute_det_metrics:
				all_scores.append(s_final.detach().cpu())
				all_labels.append(same_labels.detach().cpu())

			if collect_router_weights:
				weights_cpu = weights.detach().cpu()
				scores_cpu = s_final.detach().cpu()
				labels_cpu = same_labels.detach().cpu()
				for i in range(bs):
					row = (
						sample_index,
						int(labels_cpu[i].item()),
						float(scores_cpu[i].item()),
						float(weights_cpu[i, 0].item()),
						float(weights_cpu[i, 1].item()),
						float(weights_cpu[i, 2].item()),
					)
					router_weight_rows.append(row)
					sample_index += 1

	denom = max(1, total_samples)
	eer = float("nan")
	min_dcf = float("nan")
	eer_threshold = float("nan")

	if compute_det_metrics and all_scores:
		scores_np = torch.cat(all_scores).numpy()
		labels_np = torch.cat(all_labels).numpy()
		if np.unique(labels_np).size >= 2:
			eer, eer_threshold = compute_eer(scores_np, labels_np)
			fnrs, fprs, thresholds = ComputeErrorRates(scores_np, labels_np)
			min_dcf, _ = ComputeMinDcf(fnrs, fprs, thresholds, p_target=0.01, c_miss=1, c_fa=1)
		else:
			print("[Warn] DET metrics skipped: validation labels contain only one class.")

	mean_router_weights_tensor = (total_router_weight_sum / denom).to(torch.float32)
	router_acc = 100.0 * total_router_correct / denom if dataset_mode == "vox2" else float("nan")
	return {
		"total_loss": total_total_loss / denom,
		"circle_loss": total_circle_loss / denom,
		"router_loss": total_router_loss / denom,
		"router_acc": router_acc,
		"pair_acc": 100.0 * total_pair_correct / denom,
		"eer": eer,
		"min_dcf": min_dcf,
		"eer_threshold": eer_threshold,
		"router_weight_rows": router_weight_rows,
		"mean_router_weights_tensor": mean_router_weights_tensor,
		"mean_router_weights": _weights_to_tuple(mean_router_weights_tensor),
	}


def run_zeroshot_inference_epoch(
	model: CrossGapFixedRouterEnsemble,
	loader: DataLoader,
	device: str,
	collect_router_weights: bool = False,
) -> dict:
	model.enable_fixed_router(True)
	model.eval()

	all_scores: list[torch.Tensor] = []
	all_labels: list[torch.Tensor] = []
	router_weight_rows: list[tuple[int, int, float, float, float, float, float, float, float, float]] = []
	sample_index = 0
	fixed_router_weights = model.router.fixed_weights.detach().cpu().flatten()
	router_mode = "fixed" if model.router.use_fixed_weights else "dynamic"
	max_abs_diff_from_fixed = 0.0

	pbar = tqdm(loader, desc="Inference", dynamic_ncols=True, leave=False)

	with torch.no_grad():
		for batch in pbar:
			pair_labels, _, _, wav1, wav2, _, _ = batch
			same_labels = _to_same_labels(pair_labels)

			wav1 = wav1.to(device, non_blocking=True)
			wav2 = wav2.to(device, non_blocking=True)
			same_labels = same_labels.to(device, non_blocking=True)

			s_final, weights, _ = model(
				wav1,
				wav2,
				spec_aug=False,
				return_details=True,
			)

			all_scores.append(s_final.detach().cpu())
			all_labels.append(same_labels.detach().cpu())

			if collect_router_weights:
				bs = same_labels.size(0)
				weights_cpu = weights.detach().cpu()
				scores_cpu = s_final.detach().cpu()
				labels_cpu = same_labels.detach().cpu()
				for i in range(bs):
					row_weights = weights_cpu[i]
					max_abs_diff = float(torch.max(torch.abs(row_weights - fixed_router_weights)).item())
					max_abs_diff_from_fixed = max(max_abs_diff_from_fixed, max_abs_diff)
					row = (
						sample_index,
						int(labels_cpu[i].item()),
						float(scores_cpu[i].item()),
						router_mode,
						float(row_weights[0].item()),
						float(row_weights[1].item()),
						float(row_weights[2].item()),
						float(fixed_router_weights[0].item()),
						float(fixed_router_weights[1].item()),
						float(fixed_router_weights[2].item()),
						max_abs_diff,
					)
					router_weight_rows.append(row)
					sample_index += 1

	eer = float("nan")
	min_dcf = float("nan")
	eer_threshold = float("nan")

	if all_scores:
		scores_np = torch.cat(all_scores).numpy()
		labels_np = torch.cat(all_labels).numpy()
		if np.unique(labels_np).size >= 2:
			eer, eer_threshold = compute_eer(scores_np, labels_np)
			fnrs, fprs, thresholds = ComputeErrorRates(scores_np, labels_np)
			min_dcf, _ = ComputeMinDcf(fnrs, fprs, thresholds, p_target=0.01, c_miss=1, c_fa=1)
		else:
			print("[Warn] DET metrics skipped: inference labels contain only one class.")

	return {
		"eer": eer,
		"min_dcf": min_dcf,
		"eer_threshold": eer_threshold,
		"router_weight_rows": router_weight_rows,
		"router_mode": router_mode,
		"fixed_router_weights": _weights_to_tuple(fixed_router_weights),
		"max_abs_diff_from_fixed": max_abs_diff_from_fixed,
	}


def save_checkpoint(
	model: CrossGapFixedRouterEnsemble,
	optimizer: torch.optim.Optimizer,
	epoch: int,
	path: Path,
	fixed_router_weights: torch.Tensor,
) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	router_weights_path = path.with_name(f"{path.stem}_router_weights.pt")
	fixed_router_weights_cpu = fixed_router_weights.detach().cpu()
	torch.save(
		{
			"epoch": epoch,
			"model_state_dict": model.state_dict(),
			"optimizer_state_dict": optimizer.state_dict(),
			"beta": BETA,
			"same_label_is_one": SAME_LABEL_IS_ONE,
			"train_audio_dir": TRAIN_AUDIO_DIR,
			"train_meta_csv": TRAIN_META_CSV,
			"fixed_router_weights": fixed_router_weights_cpu,
		},
		str(path),
	)
	torch.save(
		{
			"epoch": epoch,
			"checkpoint_path": str(path),
			"fixed_router_weights": fixed_router_weights_cpu,
		},
		str(router_weights_path),
	)


def main() -> None:
	if MODE not in {"train", "inference"}:
		raise ValueError(f"MODE must be 'train' or 'inference', got: {MODE}")

	is_train_mode = MODE == "train"

	if is_train_mode:
		if not os.path.exists(TRAIN_META_CSV):
			raise FileNotFoundError(f"Train metadata csv not found: {TRAIN_META_CSV}")
	else:
		run_inference_for_multiple_ckpts_and_datasets()
		return

	print(f"\nRunning multi-seed training: {SEEDS}")
	seed_summaries = []

	for seed in SEEDS:
		set_seed(seed)

		train_loader = build_vox2_loader(TRAIN_AUDIO_DIR, TRAIN_META_CSV, shuffle=True, batch_size=BATCH_SIZE)
		val_loader = None
		if VAL_AUDIO_DIR and VAL_META_CSV:
			if os.path.exists(VAL_META_CSV):
				val_loader = build_vox2_loader(VAL_AUDIO_DIR, VAL_META_CSV, shuffle=False, batch_size=1)
			else:
				print(f"[Warn] Skip val: metadata not found at {VAL_META_CSV}")

		model = build_model()

		trainable_params = [p for p in model.parameters() if p.requires_grad]
		print_trainable_parameters(model)

		optimizer = torch.optim.Adam(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
		ce_loss_fn = nn.CrossEntropyLoss()
		circle_loss_fn = CircleLoss(m=0.25, gamma=256.0)

		run_name = build_run_name(seed)
		run_dir = SAVE_ROOT / run_name
		log_dir = LOG_ROOT / run_name
		run_dir.mkdir(parents=True, exist_ok=True)
		log_dir.mkdir(parents=True, exist_ok=True)
		csv_path = log_dir / "train_log.csv"

		print(f"\n{'=' * 66}")
		print(f"[Seed {seed}] Training start | run_name={run_name}")
		print(f"{'=' * 66}")

		best_train_loss = float("inf")
		best_val_eer = float("inf")
		best_eer_ckpt_path = run_dir / "best.pt"
		last_epoch_ckpt_path = run_dir / "last.pt"

		with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
			writer = csv.writer(f)
			writer.writerow(
				[
					"epoch",
					"train_total_loss",
					"train_circle_loss",
					"train_router_loss",
					"train_router_acc",
					"train_pair_acc",
					"train_weight_small",
					"train_weight_medium",
					"train_weight_large",
					"val_total_loss",
					"val_circle_loss",
					"val_router_loss",
					"val_router_acc",
					"val_pair_acc",
					"val_eer",
					"val_min_dcf",
					"val_eer_threshold",
				]
			)

			for epoch in range(EPOCHS):
				train_metrics = run_epoch(
					model=model,
					loader=train_loader,
					optimizer=optimizer,
					ce_loss_fn=ce_loss_fn,
					circle_loss_fn=circle_loss_fn,
					beta=BETA,
					device=DEVICE,
					epoch_idx=epoch,
					total_epochs=EPOCHS,
					dataset_mode="vox2",
					compute_det_metrics=False,
				)

				print_metrics_block(
					title=f"[Train][Seed {seed}][Epoch {epoch + 1}/{EPOCHS}]",
					metrics=train_metrics,
					show_det_metrics=False,
				)

				if train_metrics["total_loss"] < best_train_loss:
					best_train_loss = train_metrics["total_loss"]

				val_metrics = {
					"total_loss": float("nan"),
					"circle_loss": float("nan"),
					"router_loss": float("nan"),
					"router_acc": float("nan"),
					"pair_acc": float("nan"),
					"eer": float("nan"),
					"min_dcf": float("nan"),
					"eer_threshold": float("nan"),
				}

				if val_loader is not None:
					with torch.no_grad():
						val_metrics = run_epoch(
							model=model,
							loader=val_loader,
							optimizer=None,
							ce_loss_fn=ce_loss_fn,
							circle_loss_fn=circle_loss_fn,
							beta=BETA,
							device=DEVICE,
							epoch_idx=epoch,
							total_epochs=EPOCHS,
							dataset_mode="vox2",
							compute_det_metrics=True,
						)

					print_metrics_block(
						title=f"[Val][Seed {seed}][Epoch {epoch + 1}/{EPOCHS}]",
						metrics=val_metrics,
						show_det_metrics=True,
					)

					if not np.isnan(val_metrics["eer"]):
						if val_metrics["eer"] < best_val_eer:
							best_val_eer = val_metrics["eer"]
							save_checkpoint(
								model,
								optimizer,
								epoch + 1,
								best_eer_ckpt_path,
								train_metrics["mean_router_weights_tensor"],
							)

				writer.writerow(
					[
						epoch + 1,
						train_metrics["total_loss"],
						train_metrics["circle_loss"],
						train_metrics["router_loss"],
						train_metrics["router_acc"],
						train_metrics["pair_acc"],
						*train_metrics["mean_router_weights"],
						val_metrics["total_loss"],
						val_metrics["circle_loss"],
						val_metrics["router_loss"],
						val_metrics["router_acc"],
						val_metrics["pair_acc"],
						val_metrics["eer"],
						val_metrics["min_dcf"],
						val_metrics["eer_threshold"],
					]
				)
				f.flush()

				save_checkpoint(
					model,
					optimizer,
					epoch + 1,
					last_epoch_ckpt_path,
					train_metrics["mean_router_weights_tensor"],
				)

		seed_summaries.append(
			{
				"seed": seed,
				"best_train_loss": best_train_loss,
				"best_val_eer": best_val_eer,
				"run_dir": str(run_dir),
				"csv_path": str(csv_path),
			}
		)

		print(f"[Seed {seed}] Training completed. Best train loss: {best_train_loss:.4f}")
		if np.isnan(best_val_eer) or np.isinf(best_val_eer):
			print(f"[Seed {seed}] Best EER checkpoint: N/A (validation EER unavailable)")
		else:
			print(f"[Seed {seed}] Best EER checkpoint: {best_eer_ckpt_path} (EER={best_val_eer:.4f})")
		print(f"[Seed {seed}] Last epoch checkpoint: {last_epoch_ckpt_path}")
		print(f"[Seed {seed}] Checkpoints: {run_dir}")
		print(f"[Seed {seed}] Log file: {csv_path}")

	print("\n" + "=" * 66)
	print("Multi-Seed Summary")
	print("-" * 66)
	for summary in seed_summaries:
		best_val_eer_str = "N/A" if np.isnan(summary["best_val_eer"]) else f"{summary['best_val_eer']:.4f}"
		print(
			f"seed={summary['seed']} | best_train_loss={summary['best_train_loss']:.4f} | "
			f"best_val_eer={best_val_eer_str}"
		)
	print("=" * 66)


if __name__ == "__main__":
	main()