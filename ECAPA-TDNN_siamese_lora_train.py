"""
使用 Vox2PairDataset 訓練/驗證 Siamese LoRA，並支援 zero-shot 推論。

模式切換：
  - RUN_MODE = "train"：訓練 + 每個 epoch 驗證
  - RUN_MODE = "inference"：只載入 LoRA adapter，跑 zero-shot 評估
"""

import csv
import os
import random
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.vox2_loader import Vox2PairDataset
from data.vox1_loader import PairwiseDataset
from model.disentangled_model.siamese_network import SiameseNetwork
from params.param import BATCH_SIZE, DATASET_INFO, DEVICE, LEARNING_RATE, NUM_WORKERS
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

# train / val 都使用 Vox2PairDataset。測試時使用 PairwiseDataset 以驗證 LoRA adapter 的 zero-shot 能力
TRAIN_AUDIO_DIR = DATASET_INFO[TRAIN_DATASET_NAME][TRAIN_DATASET_VARIANT]["train"]["AUDIO_DIR"]
TRAIN_PAIR_META = DATASET_INFO[TRAIN_DATASET_NAME][TRAIN_DATASET_VARIANT]["train"]["AUDIO_META_DIR"]
VAL_AUDIO_DIR = DATASET_INFO[TRAIN_DATASET_NAME][TRAIN_DATASET_VARIANT]["val"]["AUDIO_DIR"]
VAL_PAIR_META = DATASET_INFO[TRAIN_DATASET_NAME][TRAIN_DATASET_VARIANT]["val"]["AUDIO_META_DIR"]

TEST_AUDIO_DIR = DATASET_INFO[TRAIN_DATASET_NAME][TRAIN_DATASET_VARIANT]["AUDIO_DIR"]
TEST_PAIR_META = DATASET_INFO[TRAIN_DATASET_NAME][TRAIN_DATASET_VARIANT]["AUDIO_DATALIST"]
TEST_PAIR_META_CSV = DATASET_INFO[TRAIN_DATASET_NAME][TRAIN_DATASET_VARIANT]["AUDIO_META_DIR"]

PRETRAINED_PATH = "pretrained_models/pretrain.model"

# LoRA target_modules 參考 ECAPA-TDNN_train.py
MODULES = {
    1: ["attention.0", "attention.4"],
    2: ["layer4"],
    3: ["fc6"],
    4: ["attention.0", "attention.4", "layer4", "fc6"],
}
MODULE_ID = 4
LORA_R = 4
LORA_ALPHA = 8
LORA_DROPOUT = 0.05

EPOCHS = 5
WEIGHT_DECAY = 1e-5
MIN_LR = 0.0
SPLIT_SEED = 42
COSINE_MARGIN = 0.2

CHECKPOINT_ROOT = "checkpoints/siamese_expert_lora"
LOG_ROOT = "logs/siamese_expert_lora"
RUN_NAME = f"m{MODULE_ID}_r{LORA_R}_a{LORA_ALPHA}_{TRAIN_DATASET_VARIANT}"
INFERENCE_ADAPTER_PATH = f"{CHECKPOINT_ROOT}/{RUN_NAME}/lora_adapter_best"

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

def _to_cosine_target(same_label: torch.Tensor) -> torch.Tensor:
    # 原始標籤: 0=不同人, 1=同人 -> 轉成 CosineEmbeddingLoss 需要的 -1/1
    return same_label * 2 - 1

def run_epoch(model, loader, optimizer=None, compute_eer_metrics: bool = False):
    is_train = optimizer is not None
    model.train(is_train)
    criterion = torch.nn.CosineEmbeddingLoss(margin=COSINE_MARGIN)

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

        target = _to_cosine_target(same_label).to(feat1.dtype)
        loss = criterion(feat1, feat2, target)

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
        }

    eer = float("nan")
    threshold = float("nan")
    min_dcf = float("nan")
    if compute_eer_metrics and all_scores:
        scores_np = torch.cat(all_scores).numpy()
        labels_np = torch.cat(all_labels).numpy()

        eer, threshold = compute_eer(scores_np, labels_np)
        fnrs, fprs, thresholds = ComputeErrorRates(scores_np, labels_np)
        min_dcf, _ = ComputeMinDcf(fnrs, fprs, thresholds, p_target=0.01, c_miss=1, c_fa=1)

    return {
        "loss": total_loss / total_count,
        "acc": 100.0 * total_correct / total_count,
        "eer": eer,
        "threshold": threshold,
        "min_dcf": min_dcf,
    }

def evaluate_zero_shot(model, loader):
    model.eval()
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for pair_label, wav1, wav2, _, _ in tqdm(loader, desc="Inference (zero-shot)", leave=False, dynamic_ncols=True):
            pair_label = pair_label.to(DEVICE, non_blocking=True)
            wav1 = wav1.to(DEVICE, non_blocking=True)
            wav2 = wav2.to(DEVICE, non_blocking=True)

            same_label = pair_label
            _, _, cosine_score = model(wav1, wav2, spec_aug=False)

            all_scores.append(cosine_score.detach().cpu())
            all_labels.append(same_label.detach().cpu())

    if not all_scores:
        return float("nan"), float("nan"), float("nan")

    scores_np = torch.cat(all_scores).numpy()
    labels_np = torch.cat(all_labels).numpy()

    eer, threshold = compute_eer(scores_np, labels_np)
    fnrs, fprs, thresholds = ComputeErrorRates(scores_np, labels_np)
    min_dcf, _ = ComputeMinDcf(fnrs, fprs, thresholds, p_target=0.01, c_miss=1, c_fa=1)

    return eer, threshold, min_dcf

def build_model(apply_lora: bool = True) -> SiameseNetwork:
    model = SiameseNetwork().to(DEVICE)

    if apply_lora:
        if MODULE_ID not in MODULES:
            raise ValueError(f"MODULE_ID 必須在 {list(MODULES.keys())}，目前是 {MODULE_ID}")

        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=MODULES[MODULE_ID],
            lora_dropout=LORA_DROPOUT,
            bias="none",
        )

        model.encoder = get_peft_model(model.encoder, lora_config)
    return model


def save_csv_header(csv_path: str) -> None:
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
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
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def main():
    set_seed(SPLIT_SEED)

    if RUN_MODE not in {"train", "inference"}:
        raise ValueError(f"RUN_MODE 只能是 train 或 inference，目前是: {RUN_MODE}")

    print("建立資料集...")
    train_loader,val_loader = None, None
    if RUN_MODE == "train":
        train_loader = build_pair_loader(TRAIN_AUDIO_DIR, TRAIN_PAIR_META, shuffle=True)
        val_loader = build_pair_loader(VAL_AUDIO_DIR, VAL_PAIR_META, shuffle=False, batch_size=1)

    print("建立 Siamese 模型...")
    model = build_model(apply_lora=(RUN_MODE == "train"))
    if RUN_MODE == "train":
        model.encoder.print_trainable_parameters()

    run_dir = Path(CHECKPOINT_ROOT) / RUN_NAME
    log_dir = Path(LOG_ROOT) / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    if RUN_MODE == "inference":
        if not os.path.exists(INFERENCE_ADAPTER_PATH):
            raise FileNotFoundError(f"找不到 adapter: {INFERENCE_ADAPTER_PATH}")
        
        test_dataset = PairwiseDataset(
            audio_dir=TEST_AUDIO_DIR,
            audio_meta_dir=TEST_PAIR_META,
            audio_meta_csv_path=DATASET_INFO[TRAIN_DATASET_NAME][TRAIN_DATASET_VARIANT]["PAIR_META_CSV"],
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

        print(f"載入 LoRA adapter: {INFERENCE_ADAPTER_PATH}")
        model.encoder = PeftModel.from_pretrained(model.encoder, INFERENCE_ADAPTER_PATH)
        model = model.to(DEVICE)
        model.eval()

        eer, threshold, min_dcf = evaluate_zero_shot(model, test_loader)
        print(f"Zero-shot EER: {eer:.4f}, threshold: {threshold:.6f}, minDCF: {min_dcf:.4f}")
        return

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, EPOCHS), eta_min=MIN_LR)

    csv_path = str(log_dir / "train_log.csv")
    save_csv_header(csv_path)

    best_val_eer = float("inf")

    print(f"開始訓練，總共 {EPOCHS} epochs")
    for epoch in range(EPOCHS):
        train_stats = run_epoch(model, train_loader, optimizer=optimizer, compute_eer_metrics=False)
        val_stats = run_epoch(model, val_loader, optimizer=None, compute_eer_metrics=True)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch + 1:02d}/{EPOCHS}] | "
            f"LR {current_lr:.2e} | "
            f"Train loss {train_stats['loss']:.4f}, acc {train_stats['acc']:.2f}% | "
            f"Val loss {val_stats['loss']:.4f}, acc {val_stats['acc']:.2f}%, eer {val_stats['eer']:.4f}, minDCF {val_stats['min_dcf']:.4f}"
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

        adapter_dir = run_dir / "lora_adapter_last"
        model.encoder.save_pretrained(str(adapter_dir))

        if val_stats["eer"] < best_val_eer:
            best_val_eer = val_stats["eer"]
            best_dir = run_dir / "lora_adapter_best"
            model.encoder.save_pretrained(str(best_dir))
        
        if epoch == EPOCHS - 1:
            last_dir = run_dir / "lora_adapter_last"
            model.encoder.save_pretrained(str(last_dir))

        scheduler.step()

    print(f"訓練完成，最佳驗證 EER: {best_val_eer:.4f}")
    print(f"訓練紀錄: {csv_path}")


if __name__ == "__main__":
    main()
