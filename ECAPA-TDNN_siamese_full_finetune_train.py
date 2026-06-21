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
from typing import List, Tuple

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
TRAIN_DATASET_VARIANT = "large"

TRAIN_SEEDS = [42, 1, 2026]
INFERENCE_SEEDS = [42, 1, 2026]

# 推論可同時跑多個測試資料集（dataset_name, dataset_variant）
INFERENCE_DATASETS: List[Tuple[str, str]] = [
    ("VoxCeleb1", "Vox-O"),
    ("VoxCeleb1", "Vox1-H.S"),
    ("VoxCeleb1", "Vox-CA10"),
    ("VoxCeleb1", "Vox-CA20"),
]

# 若有指定，就優先使用這些權重；留空則會由 INFERENCE_SEEDS 自動組合路徑
INFERENCE_CKPT_PATHS: List[str] = []

# train / val 都使用 Vox2PairDataset。測試時使用 PairwiseDataset 做 zero-shot 評估
TRAIN_AUDIO_DIR = DATASET_INFO[TRAIN_DATASET_NAME][TRAIN_DATASET_VARIANT]["train"]["AUDIO_DIR"]
TRAIN_PAIR_META = DATASET_INFO[TRAIN_DATASET_NAME][TRAIN_DATASET_VARIANT]["train"]["AUDIO_META_DIR"]
VAL_AUDIO_DIR = DATASET_INFO[TRAIN_DATASET_NAME][TRAIN_DATASET_VARIANT]["val"]["AUDIO_DIR"]
VAL_PAIR_META = DATASET_INFO[TRAIN_DATASET_NAME][TRAIN_DATASET_VARIANT]["val"]["AUDIO_META_DIR"]

EPOCHS = 10
START_LR = 1e-5
END_LR = 1e-6
WEIGHT_DECAY = 1e-5

CHECKPOINT_ROOT = "checkpoints/siamese_full_finetune"
LOG_ROOT = "logs/siamese_full_finetune"
RUN_NAME_BASE = f"full_ft_lr1e4_{TRAIN_DATASET_VARIANT}"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_run_name(seed: int) -> str:
    return f"{RUN_NAME_BASE}_seed{seed}"


def resolve_inference_ckpt_paths() -> List[str]:
    paths: List[str] = []

    if INFERENCE_CKPT_PATHS:
        paths.extend(INFERENCE_CKPT_PATHS)
    else:
        for seed in INFERENCE_SEEDS:
            run_name = build_run_name(seed)
            paths.append(str(Path(CHECKPOINT_ROOT) / run_name / "siamese_best.pt"))

    unique_paths: List[str] = []
    seen = set()
    for path in paths:
        normalized = str(Path(path))
        if normalized not in seen:
            unique_paths.append(normalized)
            seen.add(normalized)

    return unique_paths


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


def build_test_loader(dataset_name: str, dataset_variant: str) -> DataLoader:
    test_audio_dir = DATASET_INFO[dataset_name][dataset_variant]["AUDIO_DIR"]
    test_pair_meta = DATASET_INFO[dataset_name][dataset_variant]["AUDIO_DATALIST"]
    test_pair_meta_csv = DATASET_INFO[dataset_name]["AUDIO_META_DIR"]

    test_dataset = PairwiseDataset(
        audio_dir=test_audio_dir,
        audio_meta_dir=test_pair_meta,
        audio_meta_csv_path=test_pair_meta_csv,
    )

    return DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )


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


def train_single_seed(seed: int):
    set_seed(seed)
    run_name = build_run_name(seed)

    print(f"\n{'=' * 72}")
    print(f"開始訓練 seed={seed} | run_name={run_name}")
    print(f"{'=' * 72}")

    print("建立資料集...")
    train_loader = build_pair_loader(TRAIN_AUDIO_DIR, TRAIN_PAIR_META, shuffle=True)
    val_loader = build_pair_loader(VAL_AUDIO_DIR, VAL_PAIR_META, shuffle=False, batch_size=1)

    print("建立 Siamese 模型（全參數微調）...")
    model = build_model()
    print_parameter_summary(model)

    run_dir = Path(CHECKPOINT_ROOT) / run_name
    log_dir = Path(LOG_ROOT) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

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
    best_ckpt_path = run_dir / "siamese_best.pt"

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
            torch.save(model.state_dict(), str(best_ckpt_path))

        scheduler.step()

    print(f"訓練完成，seed={seed}，最佳驗證 EER: {best_val_eer:.4f}")
    print(f"最佳模型: {best_ckpt_path}")
    print(f"訓練紀錄: {csv_path}")

    return {
        "seed": seed,
        "best_val_eer": best_val_eer,
        "best_ckpt": str(best_ckpt_path),
        "log_csv": csv_path,
    }


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
            state_dict = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(state_dict)
            model = model.to(DEVICE)
            model.eval()

            eer, threshold, min_dcf = evaluate_zero_shot(model, test_loader)
            dataset_eers.append(eer)
            dataset_min_dcfs.append(min_dcf)
            overall_eers.append(eer)
            overall_min_dcfs.append(min_dcf)

            print(
                f"[{dataset_key}] {Path(ckpt_path).parent.name} | "
                f"EER: {eer:.4f}, threshold: {threshold:.6f}, minDCF: {min_dcf:.4f}"
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


def main():
    if RUN_MODE not in {"train", "inference"}:
        raise ValueError(f"RUN_MODE 只能是 train 或 inference，目前是: {RUN_MODE}")

    if RUN_MODE == "inference":
        run_inference_for_multiple_ckpts_and_datasets()
        return

    seed_train_results = []
    for seed in TRAIN_SEEDS:
        result = train_single_seed(seed)
        seed_train_results.append(result)

    best_eers = [result["best_val_eer"] for result in seed_train_results]
    mean_best_eer = float(np.mean(best_eers)) if best_eers else float("nan")
    std_best_eer = float(np.std(best_eers)) if best_eers else float("nan")

    print("\n" + "=" * 72)
    print("多 seed 訓練完成")
    print("=" * 72)
    for result in seed_train_results:
        print(
            f"seed={result['seed']} | "
            f"best_val_eer={result['best_val_eer']:.4f} | "
            f"best_ckpt={result['best_ckpt']}"
        )
    print(f"Best Val EER mean/std: {mean_best_eer:.4f}/{std_best_eer:.4f}")


if __name__ == "__main__":
    main()