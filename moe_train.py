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
SEED = 42
EPOCHS = 1
LR = 1e-4
WEIGHT_DECAY = 1e-5
BETA = 0.1  # total_loss = loss_circle + beta * loss_router

# If your csv uses 1 for same-speaker and 0 for different-speaker, keep True.
# If your csv uses 0 for same-speaker and 1 for different-speaker, set False.
SAME_LABEL_IS_ONE = True

TRAIN_SPEC_AUG = False
PRINT_EVERY_STEPS = 20

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
):
	is_train = optimizer is not None
	model.train(is_train)

	total_total_loss = 0.0
	total_circle_loss = 0.0
	total_router_loss = 0.0
	total_samples = 0

	total_router_correct = 0
	total_pair_correct = 0

	pbar = tqdm(
		loader,
		desc=f"{'Train' if is_train else 'Val'} {epoch_idx + 1}/{total_epochs}",
		dynamic_ncols=True,
		leave=False,
	)

	for step, batch in enumerate(pbar, start=1):
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

			if step % PRINT_EVERY_STEPS == 0 or step == len(loader):
				pbar.set_postfix(
					total=f"{float(total_loss.detach().cpu()):.4f}",
					circle=f"{float(loss_circle.detach().cpu()):.4f}",
					router=f"{float(loss_router.detach().cpu()):.4f}",
				)

	denom = max(1, total_samples)
	router_acc = 100.0 * total_router_correct / denom if dataset_mode == "vox2" else float("nan")
	return {
		"total_loss": total_total_loss / denom,
		"circle_loss": total_circle_loss / denom,
		"router_loss": total_router_loss / denom,
		"router_acc": router_acc,
		"pair_acc": 100.0 * total_pair_correct / denom,
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
	set_seed(SEED)
	if MODE not in {"train", "inference"}:
		raise ValueError(f"MODE must be 'train' or 'inference', got: {MODE}")

	is_train_mode = MODE == "train"

	if is_train_mode:
		if not os.path.exists(TRAIN_META_CSV):
			raise FileNotFoundError(f"Train metadata csv not found: {TRAIN_META_CSV}")

		train_loader = build_vox2_loader(TRAIN_AUDIO_DIR, TRAIN_META_CSV, shuffle=True, batch_size=BATCH_SIZE)
		val_loader = None
		if VAL_AUDIO_DIR and VAL_META_CSV:
			if os.path.exists(VAL_META_CSV):
				val_loader = build_vox2_loader(VAL_AUDIO_DIR, VAL_META_CSV, shuffle=False, batch_size=1)
			else:
				print(f"[Warn] Skip val: metadata not found at {VAL_META_CSV}")
	else:
		if not os.path.exists(INFER_META_LIST):
			raise FileNotFoundError(f"Inference pair list not found: {INFER_META_LIST}")
		if not os.path.exists(INFER_META_CSV):
			raise FileNotFoundError(f"Inference metadata csv not found: {INFER_META_CSV}")
		infer_loader = build_vox1_loader(INFER_AUDIO_DIR, INFER_META_LIST, INFER_META_CSV)

	model = CrossGapMoE(
		C=1024,
		feature_dim=192,
		router_hidden_dim=256,
		router_dropout=0.1,
		expert_ckpt_paths=ExpertCheckpointPaths(),
	).to(DEVICE)

	trainable_params = [p for p in model.parameters() if p.requires_grad]
	print(f"Trainable parameter count: {sum(p.numel() for p in trainable_params)}")

	optimizer = torch.optim.Adam(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
	ce_loss_fn = nn.CrossEntropyLoss()
	circle_loss_fn = CircleLoss(m=0.25, gamma=256.0)

	run_dir = SAVE_ROOT / RUN_NAME
	log_dir = LOG_ROOT / RUN_NAME
	run_dir.mkdir(parents=True, exist_ok=True)
	log_dir.mkdir(parents=True, exist_ok=True)
	csv_path = log_dir / "train_log.csv"

	best_train_loss = float("inf")

	if not is_train_mode:
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
			)

		print(
			f"[Inference][{INFER_DATASET_NAME}-{INFER_DATASET_VARIANT}] "
			f"total={infer_metrics['total_loss']:.4f}, "
			f"circle={infer_metrics['circle_loss']:.4f}, "
			f"pair_acc={infer_metrics['pair_acc']:.2f}%"
		)
		return

	with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(
			[
				"epoch",
				"split",
				"total_loss",
				"circle_loss",
				"router_loss",
				"router_acc",
				"pair_acc",
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
			)

			print(
				f"[Train][Epoch {epoch + 1}/{EPOCHS}] "
				f"total={train_metrics['total_loss']:.4f}, "
				f"circle={train_metrics['circle_loss']:.4f}, "
				f"router={train_metrics['router_loss']:.4f}, "
				f"router_acc={train_metrics['router_acc']:.2f}%, "
				f"pair_acc={train_metrics['pair_acc']:.2f}%"
			)

			writer.writerow(
				[
					epoch + 1,
					"train",
					train_metrics["total_loss"],
					train_metrics["circle_loss"],
					train_metrics["router_loss"],
					train_metrics["router_acc"],
					train_metrics["pair_acc"],
				]
			)
			f.flush()

			if train_metrics["total_loss"] < best_train_loss:
				best_train_loss = train_metrics["total_loss"]
				save_checkpoint(model, optimizer, epoch + 1, run_dir / "best_train.pt")

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
					)

				print(
					f"[Val][Epoch {epoch + 1}/{EPOCHS}] "
					f"total={val_metrics['total_loss']:.4f}, "
					f"circle={val_metrics['circle_loss']:.4f}, "
					f"router={val_metrics['router_loss']:.4f}, "
					f"router_acc={val_metrics['router_acc']:.2f}%, "
					f"pair_acc={val_metrics['pair_acc']:.2f}%"
				)

				writer.writerow(
					[
						epoch + 1,
						"val",
						val_metrics["total_loss"],
						val_metrics["circle_loss"],
						val_metrics["router_loss"],
						val_metrics["router_acc"],
						val_metrics["pair_acc"],
					]
				)
				f.flush()

			save_checkpoint(model, optimizer, epoch + 1, run_dir / f"epoch_{epoch + 1:03d}.pt")

	print(f"Training completed. Best train loss: {best_train_loss:.4f}")
	print(f"Checkpoints: {run_dir}")
	print(f"Log file: {csv_path}")


if __name__ == "__main__":
	main()
