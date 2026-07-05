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
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.vox2_loader import Vox2PairDataset
from loss.circleloss import CircleLoss
from model.disentangled_model import CrossGapMoE, ExpertCheckpointPaths
from params.param import DEVICE, NUM_WORKERS, BATCH_SIZE, DATASET_INFO


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

TRAIN_DATASET_NAME = "VoxCeleb2"
TRAIN_DATASET_VARIANT = "moe"

TRAIN_AUDIO_DIR = DATASET_INFO[TRAIN_DATASET_NAME][TRAIN_DATASET_VARIANT]["train"]["AUDIO_DIR"]
TRAIN_META_CSV = DATASET_INFO[TRAIN_DATASET_NAME][TRAIN_DATASET_VARIANT]["train"]["AUDIO_META_DIR"]
VAL_AUDIO_DIR = DATASET_INFO[TRAIN_DATASET_NAME][TRAIN_DATASET_VARIANT]["val"]["AUDIO_DIR"]
VAL_META_CSV = DATASET_INFO[TRAIN_DATASET_NAME][TRAIN_DATASET_VARIANT]["val"]["AUDIO_META_DIR"]

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


def build_loader(audio_dir: str, meta_csv: str, shuffle: bool) -> DataLoader:
	dataset = Vox2PairDataset(audio_dir=audio_dir, audio_meta_dir=meta_csv)
	return DataLoader(
		dataset,
		batch_size=BATCH_SIZE,
		shuffle=shuffle,
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
		pair_labels, wav1, wav2, _, true_gap_labels = batch

		same_labels = _to_same_labels(pair_labels)

		wav1 = wav1.to(device, non_blocking=True)
		wav2 = wav2.to(device, non_blocking=True)
		same_labels = same_labels.to(device, non_blocking=True)
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

		# 1) Router guidance loss
		loss_router = ce_loss_fn(weights_logits, true_gap_labels)

		# 2) Final metric loss
		loss_circle = circle_loss_fn(s_final, same_labels)

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
	return {
		"total_loss": total_total_loss / denom,
		"circle_loss": total_circle_loss / denom,
		"router_loss": total_router_loss / denom,
		"router_acc": 100.0 * total_router_correct / denom,
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

	if not os.path.exists(TRAIN_META_CSV):
		raise FileNotFoundError(f"Train metadata csv not found: {TRAIN_META_CSV}")

	train_loader = build_loader(TRAIN_AUDIO_DIR, TRAIN_META_CSV, shuffle=True)
	val_loader = None
	if VAL_AUDIO_DIR and VAL_META_CSV:
		if os.path.exists(VAL_META_CSV):
			val_loader = build_loader(VAL_AUDIO_DIR, VAL_META_CSV, shuffle=False)
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
