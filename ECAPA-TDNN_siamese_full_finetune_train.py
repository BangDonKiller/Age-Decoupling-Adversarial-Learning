"""
使用 Vox2PairDataset 訓練/驗證 Siamese ECAPA-TDNN，採用全參數微調（不使用 LoRA）。

模式切換：
  - RUN_MODE = "train"：訓練 + 每個 epoch 驗證
  - RUN_MODE = "inference"：載入完整模型 checkpoint，跑 zero-shot 評估
"""

import csv
import os
import random
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.vox2_loader import Vox2PairDataset
from data.vox1_loader import PairwiseDataset
from loss.cosineloss import SmoothCosineLoss
from model.disentangled_model.siamese_network import SiameseNetwork
from params.param import BATCH_SIZE, DATASET_INFO, DEVICE, NUM_WORKERS
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
RUN_MODE = "train"  # "train" 或 "inference"

TRAIN_DATASET_NAME = "VoxCeleb2"
TRAIN_DATASET_VARIANT = "small"
TEST_DATASET_NAME = "VoxCeleb1"
TEST_DATASET_VARIANT = "Vox-O"

# train / val 都使用 Vox2PairDataset。測試時使用 PairwiseDataset 做 zero-shot 評估
TRAIN_AUDIO_DIR = DATASET_INFO[TRAIN_DATASET_NAME][TRAIN_DATASET_VARIANT]["train"]["AUDIO_DIR"]
TRAIN_PAIR_META = DATASET_INFO[TRAIN_DATASET_NAME][TRAIN_DATASET_VARIANT]["train"]["AUDIO_META_DIR"]
VAL_AUDIO_DIR = DATASET_INFO[TRAIN_DATASET_NAME][TRAIN_DATASET_VARIANT]["val"]["AUDIO_DIR"]
VAL_PAIR_META = DATASET_INFO[TRAIN_DATASET_NAME][TRAIN_DATASET_VARIANT]["val"]["AUDIO_META_DIR"]

TEST_AUDIO_DIR = DATASET_INFO[TEST_DATASET_NAME][TEST_DATASET_VARIANT]["AUDIO_DIR"]
TEST_PAIR_META = DATASET_INFO[TEST_DATASET_NAME][TEST_DATASET_VARIANT]["AUDIO_DATALIST"]
TEST_PAIR_META_CSV = DATASET_INFO[TEST_DATASET_NAME]["AUDIO_META_DIR"]

EPOCHS = 10
START_LR = 1e-4
END_LR = 1e-6
WEIGHT_DECAY = 1e-5
SPLIT_SEED = 42

CHECKPOINT_ROOT = "checkpoints/siamese_full_finetune"
LOG_ROOT = "logs/siamese_full_finetune"
RUN_NAME = f"full_ft_lr1e4_{TRAIN_DATASET_VARIANT}"
INFERENCE_CKPT_PATH = f"{CHECKPOINT_ROOT}/{RUN_NAME}/siamese_best.pt"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_pair_loader(audio_dir: str, meta_csv: str, shuffle: bool, batch_size: int = BATCH_SIZE) -> DataLoader:
    dataset = Vox2PairDataset(
        audio_dir=audio_dir,
        audio_meta_dir=meta_csv,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
    )


def run_epoch(model, loader, optimizer=None, compute_eer_metrics: bool = False):
    is_train = optimizer is not None
    model.train(is_train)
    criterion = SmoothCosineLoss()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    all_scores = []
    all_labels = []

    desc = "Train" if is_train else "Val"

    for pair_label, wav1, wav2, _, _ in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True):
        pair_label = pair_label.to(DEVICE, non_blocking=True)
        wav1 = wav1.to(DEVICE, non_blocking=True)
        wav2 = wav2.to(DEVICE, non_blocking=True)

        same_label = pair_label.float()

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        feat1, feat2, cosine_score = model(wav1, wav2, spec_aug=is_train)
        loss = criterion(feat1, feat2, same_label)

        if is_train:
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            pred_same = (cosine_score >= 0.0).long()
            correct = (pred_same == same_label.long()).sum().item()

            batch_size = pair_label.size(0)
            total_count += batch_size
            total_correct += correct
            total_loss += loss.item() * batch_size

            if compute_eer_metrics:
                all_scores.append(cosine_score.detach().cpu())
                all_labels.append(same_label.detach().cpu().long())

    if total_count == 0:
        return {
            "loss": float("nan"),
            "acc": float("nan"),
            "eer": float("nan"),
            "threshold": float("nan"),
            "min_dcf": float("nan"),
            "pos_mean": float("nan"),
            "pos_std": float("nan"),
            "neg_mean": float("nan"),
            "neg_std": float("nan"),
        }

    eer = float("nan")
    threshold = float("nan")
    min_dcf = float("nan")
    pos_mean = float("nan")
    pos_std = float("nan")
    neg_mean = float("nan")
    neg_std = float("nan")

    if compute_eer_metrics and all_scores:
        scores_np = torch.cat(all_scores).numpy()
        labels_np = torch.cat(all_labels).numpy()

        eer, threshold = compute_eer(scores_np, labels_np)
        fnrs, fprs, thresholds = ComputeErrorRates(scores_np, labels_np)
        min_dcf, _ = ComputeMinDcf(fnrs, fprs, thresholds, p_target=0.01, c_miss=1, c_fa=1)

        pos_scores = scores_np[labels_np == 1]
        neg_scores = scores_np[labels_np == 0]

        if pos_scores.size > 0:
            pos_mean = float(np.mean(pos_scores))
            pos_std = float(np.std(pos_scores))
        if neg_scores.size > 0:
            neg_mean = float(np.mean(neg_scores))
            neg_std = float(np.std(neg_scores))

    return {
        "loss": total_loss / total_count,
        "acc": 100.0 * total_correct / total_count,
        "eer": eer,
        "threshold": threshold,
        "min_dcf": min_dcf,
        "pos_mean": pos_mean,
        "pos_std": pos_std,
        "neg_mean": neg_mean,
        "neg_std": neg_std,
    }


def evaluate_zero_shot(model, loader):
    model.eval()
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for pair_label, _, _, wav1, wav2, _, _ in tqdm(loader, desc="Inference (zero-shot)", leave=False, dynamic_ncols=True):
            pair_label = pair_label.to(DEVICE, non_blocking=True)
            wav1 = wav1.to(DEVICE, non_blocking=True)
            wav2 = wav2.to(DEVICE, non_blocking=True)

            _, _, cosine_score = model(wav1, wav2, spec_aug=False)

            all_scores.append(cosine_score.detach().cpu())
            all_labels.append(pair_label.detach().cpu())

    if not all_scores:
        return float("nan"), float("nan"), float("nan")

    scores_np = torch.cat(all_scores).numpy()
    labels_np = torch.cat(all_labels).numpy()

    eer, threshold = compute_eer(scores_np, labels_np)
    fnrs, fprs, thresholds = ComputeErrorRates(scores_np, labels_np)
    min_dcf, _ = ComputeMinDcf(fnrs, fprs, thresholds, p_target=0.01, c_miss=1, c_fa=1)

    return eer, threshold, min_dcf


def build_model() -> SiameseNetwork:
    model = SiameseNetwork().to(DEVICE)
    for param in model.parameters():
        param.requires_grad = True
    return model


def save_csv_header(csv_path: str) -> None:
    with open(csv_path, mode="w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(
            [
                "epoch",
                "lr",
                "train_loss",
                "train_acc",
                "val_loss",
                "val_acc",
                "val_eer",
                "val_min_dcf",
            ]
        )


def append_csv_row(csv_path: str, row) -> None:
    with open(csv_path, mode="a", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(row)


def print_parameter_summary(model) -> None:
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    trainable_ratio = 100.0 * trainable_params / total_params if total_params > 0 else 0.0

    print(f"總參數量: {total_params:,}")
    print(f"可訓練參數量: {trainable_params:,}")
    print(f"可訓練參數占比: {trainable_ratio:.4f}%")


def main():
    set_seed(SPLIT_SEED)

    if RUN_MODE not in {"train", "inference"}:
        raise ValueError(f"RUN_MODE 只能是 train 或 inference，目前是: {RUN_MODE}")

    print("建立資料集...")
    train_loader, val_loader = None, None
    if RUN_MODE == "train":
        train_loader = build_pair_loader(TRAIN_AUDIO_DIR, TRAIN_PAIR_META, shuffle=True)
        val_loader = build_pair_loader(VAL_AUDIO_DIR, VAL_PAIR_META, shuffle=False, batch_size=1)

    print("建立 Siamese 模型（全參數微調）...")
    model = build_model()
    print_parameter_summary(model)

    run_dir = Path(CHECKPOINT_ROOT) / RUN_NAME
    log_dir = Path(LOG_ROOT) / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    if RUN_MODE == "inference":
        if not os.path.exists(INFERENCE_CKPT_PATH):
            raise FileNotFoundError(f"找不到 checkpoint: {INFERENCE_CKPT_PATH}")

        test_dataset = PairwiseDataset(
            audio_dir=TEST_AUDIO_DIR,
            audio_meta_dir=TEST_PAIR_META,
            audio_meta_csv_path=TEST_PAIR_META_CSV,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

        print(f"載入 checkpoint: {INFERENCE_CKPT_PATH}")
        state_dict = torch.load(INFERENCE_CKPT_PATH, map_location="cpu")
        model.load_state_dict(state_dict)
        model = model.to(DEVICE)
        model.eval()

        eer, threshold, min_dcf = evaluate_zero_shot(model, test_loader)
        print(f"Zero-shot EER: {eer:.4f}, threshold: {threshold:.6f}, minDCF: {min_dcf:.4f}")
        return

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=START_LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=END_LR,
    )

    csv_path = str(log_dir / "train_log.csv")
    save_csv_header(csv_path)

    best_val_eer = float("inf")

    print(f"開始訓練，總共 {EPOCHS} epochs")
    for epoch in range(EPOCHS):
        current_lr = optimizer.param_groups[0]["lr"]

        train_stats = run_epoch(model, train_loader, optimizer=optimizer, compute_eer_metrics=False)
        val_stats = run_epoch(model, val_loader, optimizer=None, compute_eer_metrics=True)

        print(
            f"Epoch [{epoch + 1:02d}/{EPOCHS}] | "
            f"LR {current_lr:.2e} | "
            f"Train loss {train_stats['loss']:.4f}, acc {train_stats['acc']:.2f}% | "
            f"Val loss {val_stats['loss']:.4f}, acc {val_stats['acc']:.2f}%, eer {val_stats['eer']:.4f}, minDCF {val_stats['min_dcf']:.4f}"
        )

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"[Val Score Stats] Epoch {epoch + 1:02d} | "
                f"Pos mean/std: {val_stats['pos_mean']:.4f}/{val_stats['pos_std']:.4f} | "
                f"Neg mean/std: {val_stats['neg_mean']:.4f}/{val_stats['neg_std']:.4f}"
            )

        append_csv_row(
            csv_path,
            [
                epoch + 1,
                current_lr,
                train_stats["loss"],
                train_stats["acc"],
                val_stats["loss"],
                val_stats["acc"],
                val_stats["eer"],
                val_stats["min_dcf"],
            ],
        )

        last_ckpt = run_dir / "siamese_last.pt"
        torch.save(model.state_dict(), str(last_ckpt))

        if val_stats["eer"] < best_val_eer:
            best_val_eer = val_stats["eer"]
            best_ckpt = run_dir / "siamese_best.pt"
            torch.save(model.state_dict(), str(best_ckpt))

        scheduler.step()

    print(f"訓練完成，最佳驗證 EER: {best_val_eer:.4f}")
    print(f"最佳模型: {run_dir / 'siamese_best.pt'}")
    print(f"訓練紀錄: {csv_path}")


if __name__ == "__main__":
    main()