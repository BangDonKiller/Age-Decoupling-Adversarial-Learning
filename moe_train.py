"""
Train script for CrossGapMoE (3 frozen experts + dynamic router + per-expert calibrators).

Loss design (as requested):
1) Router guidance loss:
   loss_router = CrossEntropyLoss(weights_logits, true_gap_labels)
2) Final metric loss:
   loss_circle = CircleLoss(S_final, same_person_labels)
3) Joint optimization:
   total_loss = loss_circle + beta * loss_router
"""

from __future__ import annotations

import csv
import os
import random
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
from model.disentangled_model import CrossGapMoE, ExpertCheckpointPaths
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

# ==========================================
# 1) Config
# ==========================================
SEEDS = [42, 1, 2026]
EPOCHS = 1
LR = 1e-3
WEIGHT_DECAY = 1e-5
BETA = 0.1  # total_loss = loss_circle + beta * loss_router

# If your csv uses 1 for same-speaker and 0 for different-speaker, keep True.
# If your csv uses 0 for same-speaker and 1 for different-speaker, set False.
SAME_LABEL_IS_ONE = True

TRAIN_SPEC_AUG = False

MODE = "train"  # "train" or "inference"

TRAIN_DATASET_NAME = "VoxCeleb2"
TRAIN_DATASET_VARIANT = "moe"

INFER_DATASET_NAME = "VoxCeleb1"
INFER_DATASET_VARIANT = "Vox-O"

TRAIN_AUDIO_DIR = DATASET_INFO[TRAIN_DATASET_NAME][TRAIN_DATASET_VARIANT]["train"]["AUDIO_DIR"]
TRAIN_META_CSV = DATASET_INFO[TRAIN_DATASET_NAME][TRAIN_DATASET_VARIANT]["train"]["AUDIO_META_DIR"]
VAL_AUDIO_DIR = DATASET_INFO[TRAIN_DATASET_NAME][TRAIN_DATASET_VARIANT]["val"]["AUDIO_DIR"]
VAL_META_CSV = DATASET_INFO[TRAIN_DATASET_NAME][TRAIN_DATASET_VARIANT]["val"]["AUDIO_META_DIR"]

INFER_AUDIO_DIR = DATASET_INFO[INFER_DATASET_NAME][INFER_DATASET_VARIANT]["AUDIO_DIR"]
INFER_META_LIST = DATASET_INFO[INFER_DATASET_NAME][INFER_DATASET_VARIANT]["AUDIO_DATALIST"]
INFER_META_CSV = DATASET_INFO[INFER_DATASET_NAME]["AUDIO_META_DIR"]

SAVE_ROOT = Path("checkpoints/cross_gap_moe")
LOG_ROOT = Path("logs/cross_gap_moe")
RUN_NAME = "cross_gap_moe_router_circle"


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


def build_vox1_loader(audio_dir: list[str], pair_list: str, age_meta_csv: str) -> DataLoader:
	dataset = Vox1PairDataset(
		audio_dir=audio_dir,
		audio_meta_dir=pair_list,
		audio_meta_csv_path=age_meta_csv,
	)
	return DataLoader(
		dataset,
		batch_size=1,
		shuffle=False,
		num_workers=NUM_WORKERS,
		pin_memory=torch.cuda.is_available(),
		drop_last=False,
	)


def _fmt_metric(value: float, digits: int = 4, suffix: str = "") -> str:
	if isinstance(value, (float, np.floating)) and np.isnan(value):
		return "N/A"
	return f"{value:.{digits}f}{suffix}"


def print_metrics_block(title: str, metrics: dict, show_det_metrics: bool) -> None:
	print("\n" + "=" * 66)
	print(f"{title}")
	print("-" * 66)
	print(f"{'total_loss':<16}: {_fmt_metric(metrics['total_loss'])}")
	print(f"{'circle_loss':<16}: {_fmt_metric(metrics['circle_loss'])}")
	print(f"{'router_loss':<16}: {_fmt_metric(metrics['router_loss'])}")
	print(f"{'router_acc':<16}: {_fmt_metric(metrics['router_acc'], digits=2, suffix='%')}")
	print(f"{'pair_acc':<16}: {_fmt_metric(metrics['pair_acc'], digits=2, suffix='%')}")
	if show_det_metrics:
		print(f"{'eer':<16}: {_fmt_metric(metrics['eer'])}")
		print(f"{'min_dcf':<16}: {_fmt_metric(metrics['min_dcf'])}")
		print(f"{'eer_threshold':<16}: {_fmt_metric(metrics['eer_threshold'])}")
	print("=" * 66)


def print_trainable_parameters(model: CrossGapMoE) -> None:
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
	model: CrossGapMoE,
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
):
	is_train = optimizer is not None
	model.train(is_train)

	total_total_loss = 0.0
	total_circle_loss = 0.0
	total_router_loss = 0.0
	total_samples = 0

	total_router_correct = 0
	total_pair_correct = 0
	all_scores: list[torch.Tensor] = []
	all_labels: list[torch.Tensor] = []

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
		weights_logits = details["router_logits"]
		loss_circle = circle_loss_fn(s_final, same_labels)

		# 1) Router guidance loss
		if true_gap_labels is not None:
			loss_router = ce_loss_fn(weights_logits, true_gap_labels)
		else:
			loss_router = torch.zeros_like(loss_circle)

		# 2) Final metric loss

		# 3) Joint optimization
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

			if true_gap_labels is not None:
				router_pred = torch.argmax(weights_logits, dim=1)
				total_router_correct += int((router_pred == true_gap_labels).sum().item())

			pair_pred = (s_final >= 0.0).long()
			total_pair_correct += int((pair_pred == same_labels).sum().item())

			if compute_det_metrics:
				all_scores.append(s_final.detach().cpu())
				all_labels.append(same_labels.detach().cpu())

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
	}


def save_checkpoint(model: CrossGapMoE, optimizer: torch.optim.Optimizer, epoch: int, path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	torch.save(
		{
			"epoch": epoch,
			"model_state_dict": model.state_dict(),
			"optimizer_state_dict": optimizer.state_dict(),
			"beta": BETA,
			"same_label_is_one": SAME_LABEL_IS_ONE,
			"train_audio_dir": TRAIN_AUDIO_DIR,
			"train_meta_csv": TRAIN_META_CSV,
		},
		str(path),
	)


def main() -> None:
	if MODE not in {"train", "inference"}:
		raise ValueError(f"MODE must be 'train' or 'inference', got: {MODE}")

	is_train_mode = MODE == "train"

	if is_train_mode:
		if not os.path.exists(TRAIN_META_CSV):
			raise FileNotFoundError(f"Train metadata csv not found: {TRAIN_META_CSV}")
	else:
		set_seed(SEEDS[0])
		if not os.path.exists(INFER_META_LIST):
			raise FileNotFoundError(f"Inference pair list not found: {INFER_META_LIST}")
		if not os.path.exists(INFER_META_CSV):
			raise FileNotFoundError(f"Inference metadata csv not found: {INFER_META_CSV}")
		infer_loader = build_vox1_loader(INFER_AUDIO_DIR, INFER_META_LIST, INFER_META_CSV)

	if not is_train_mode:
		model = CrossGapMoE(
			C=1024,
			feature_dim=192,
			router_hidden_dim=256,
			router_dropout=0.1,
			expert_ckpt_paths=ExpertCheckpointPaths(),
		).to(DEVICE)

		ce_loss_fn = nn.CrossEntropyLoss()
		circle_loss_fn = CircleLoss(m=0.25, gamma=256.0)

		with torch.no_grad():
			infer_metrics = run_epoch(
				model=model,
				loader=infer_loader,
				optimizer=None,
				ce_loss_fn=ce_loss_fn,
				circle_loss_fn=circle_loss_fn,
				beta=BETA,
				device=DEVICE,
				epoch_idx=0,
				total_epochs=1,
				dataset_mode="vox1",
				compute_det_metrics=True,
			)

		print_metrics_block(
			title=f"[Inference][{INFER_DATASET_NAME}-{INFER_DATASET_VARIANT}]",
			metrics=infer_metrics,
			show_det_metrics=True,
		)
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

		model = CrossGapMoE(
			C=1024,
			feature_dim=192,
			router_hidden_dim=256,
			router_dropout=0.1,
			expert_ckpt_paths=ExpertCheckpointPaths(),
		).to(DEVICE)

		trainable_params = [p for p in model.parameters() if p.requires_grad]
		print_trainable_parameters(model)

		optimizer = torch.optim.Adam(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
		ce_loss_fn = nn.CrossEntropyLoss()
		circle_loss_fn = CircleLoss(m=0.25, gamma=256.0)

		run_name = f"{RUN_NAME}_seed{seed}"
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
		best_eer_ckpt_path = run_dir / f"best_eer_seed{seed}.pt"
		last_epoch_ckpt_path = run_dir / f"last_epoch_seed{seed}.pt"

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
							save_checkpoint(model, optimizer, epoch + 1, best_eer_ckpt_path)

				writer.writerow(
					[
						epoch + 1,
						train_metrics["total_loss"],
						train_metrics["circle_loss"],
						train_metrics["router_loss"],
						train_metrics["router_acc"],
						train_metrics["pair_acc"],
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

				save_checkpoint(model, optimizer, epoch + 1, run_dir / f"epoch_{epoch + 1:03d}.pt")
				save_checkpoint(model, optimizer, epoch + 1, last_epoch_ckpt_path)

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
