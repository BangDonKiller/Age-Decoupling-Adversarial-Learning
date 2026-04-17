import csv
import os
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data.vox2_loader import Vox2Dataset
from model.feature_extractor.ecapa_tdnn import SpeakerEmbeddingExtractor
from params.param import BATCH_SIZE, DATASET_INFO, MODEL_ID

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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_DATASET_NAME = "VoxCeleb2"

NUM_WORKERS = 0
EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
SPLIT_SEED = 42
TRAIN_UTTS_PER_SPK = 8
VAL_UTTS_PER_SPK = 2

SPEAKER_EMB_DIM = 192
MLP_HIDDEN_DIM = 128
MLP_DROPOUT = 0.1


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_train_val_indices_by_speaker(
    dataset,
    train_utts_per_spk: int = 8,
    val_utts_per_spk: int = 2,
    seed: int = 42,
):
    """以 speaker 為單位做固定數量切分（與 ECAPA-TDNN_linear_decorr_mlp.py 一致）。"""
    if train_utts_per_spk <= 0 or val_utts_per_spk <= 0:
        raise ValueError("train_utts_per_spk 與 val_utts_per_spk 必須 > 0")

    required = train_utts_per_spk + val_utts_per_spk

    speaker_to_indices = {}
    for idx, (_, speaker_id, _, _) in enumerate(dataset.datalist):
        speaker_to_indices.setdefault(speaker_id, []).append(idx)

    rng = np.random.default_rng(seed)
    train_indices = []
    val_indices = []
    used_speakers = []

    for speaker_id, indices in speaker_to_indices.items():
        if len(indices) < required:
            raise RuntimeError(
                f"Speaker {speaker_id} 的語句數不足 {required}，"
                "請確認 Vox2Dataset 的 min_utts_per_speaker / 取樣設定。"
            )
        permuted = rng.permutation(indices)
        train_indices.extend(permuted[:train_utts_per_spk].tolist())
        val_indices.extend(permuted[train_utts_per_spk:required].tolist())
        used_speakers.append(speaker_id)

    return train_indices, val_indices, used_speakers


def compute_age_auc(age_logits: torch.Tensor, age_targets: torch.Tensor, num_age_groups: int) -> float:
    """多分類使用 macro OVR AUC，二分類使用正類 AUC。"""
    if age_logits is None or age_targets is None:
        return float("nan")
    if age_logits.numel() == 0 or age_targets.numel() == 0:
        return float("nan")

    y_true = age_targets.detach().cpu().numpy().astype(int)
    y_score = torch.softmax(age_logits.detach(), dim=1).cpu().numpy()

    try:
        if num_age_groups <= 2:
            if y_score.shape[1] < 2:
                return float("nan")
            return float(roc_auc_score(y_true, y_score[:, 1]))

        return float(roc_auc_score(y_true, y_score, multi_class="ovr", average="macro"))
    except ValueError:
        return float("nan")


class AgeLinearClassifier(nn.Module):
    def __init__(self, input_dim: int, num_age_groups: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_age_groups)

    def forward(self, x):
        return self.fc(x)


class AgeSimpleMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_age_groups: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_age_groups),
        )

    def forward(self, x):
        return self.net(x)


def run_one_epoch(model, speaker_extractor, dataloader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    logits_all = []
    targets_all = []

    for waveform, _, _, age_label in tqdm(dataloader, desc="Train" if is_train else "Val"):
        waveform = waveform.to(DEVICE, non_blocking=True)
        age_label = age_label.to(DEVICE, non_blocking=True).long()

        with torch.no_grad():
            speaker_emb = speaker_extractor(waveform)

        if is_train:
            optimizer.zero_grad()

        logits = model(speaker_emb)
        loss = criterion(logits, age_label)

        if is_train:
            loss.backward()
            optimizer.step()

        pred = logits.argmax(dim=1)
        total_loss += float(loss.item())
        total_correct += (pred == age_label).sum().item()
        total_samples += age_label.size(0)
        logits_all.append(logits.detach().cpu())
        targets_all.append(age_label.detach().cpu())

    avg_loss = total_loss / max(1, len(dataloader))
    acc = 100.0 * total_correct / max(1, total_samples)
    all_logits = torch.cat(logits_all, dim=0) if logits_all else torch.tensor([])
    all_targets = torch.cat(targets_all, dim=0) if targets_all else torch.tensor([], dtype=torch.long)

    return avg_loss, acc, all_logits, all_targets


def train_and_validate(model_name, model, speaker_extractor, train_loader, val_loader, num_age_groups, csv_writer):
    print(f"\n===== {model_name} =====")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best = {
        "epoch": -1,
        "val_acc": -1.0,
        "val_auc": float("nan"),
    }

    for epoch in range(EPOCHS):
        train_loss, train_acc, _, _ = run_one_epoch(
            model=model,
            speaker_extractor=speaker_extractor,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
        )

        with torch.no_grad():
            val_loss, val_acc, val_logits, val_targets = run_one_epoch(
                model=model,
                speaker_extractor=speaker_extractor,
                dataloader=val_loader,
                criterion=criterion,
                optimizer=None,
            )
        val_auc = compute_age_auc(val_logits, val_targets, num_age_groups=num_age_groups)

        print(
            f"Epoch [{epoch + 1:2d}/{EPOCHS}] | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val AUC: {val_auc:.6f}"
        )

        csv_writer.writerow(
            [
                model_name,
                epoch + 1,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                val_auc,
            ]
        )

        if val_acc > best["val_acc"]:
            best["epoch"] = epoch + 1
            best["val_acc"] = val_acc
            best["val_auc"] = val_auc

    return best


def main():
    set_seed(SPLIT_SEED)

    print(f"Building dataset on {DEVICE}...")
    dataset = Vox2Dataset(
        audio_dir=DATASET_INFO[TRAIN_DATASET_NAME]["AUDIO_DIR"],
        audio_meta_dir=DATASET_INFO[TRAIN_DATASET_NAME]["AUDIO_META_DIR"],
        musan_path=DATASET_INFO["MUSAN"]["AUDIO_DIR"],
        rir_path=DATASET_INFO["RIR"]["AUDIO_DIR"],
        suffix=DATASET_INFO[TRAIN_DATASET_NAME]["audio_suffix"],
        age_target_mode="group",
        use_acoustic_features=False,
    )

    train_indices, val_indices, used_speakers = build_train_val_indices_by_speaker(
        dataset,
        train_utts_per_spk=TRAIN_UTTS_PER_SPK,
        val_utts_per_spk=VAL_UTTS_PER_SPK,
        seed=SPLIT_SEED,
    )
    if len(used_speakers) == 0:
        raise RuntimeError(
            "沒有任何 speaker 能滿足 8 train + 2 val 的切分條件，"
            "請增加每位 speaker 可用語句或調整切分比例。"
        )

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_speaker_ids = sorted({dataset.datalist[idx][1] for idx in train_indices})
    val_speaker_ids = sorted({dataset.datalist[idx][1] for idx in val_indices})

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    print(
        f"Split summary | Eligible speakers: {len(used_speakers)}"
    )
    print(
        f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)} | "
        f"num_age_groups: {dataset.num_age_classes}"
    )
    print(
        f"Train speaker classes: {len(train_speaker_ids)} | "
        f"Val speaker classes: {len(val_speaker_ids)}"
    )
    print(
        f"Train/Val speaker class overlap: {len(set(train_speaker_ids) & set(val_speaker_ids))}"
    )

    print("Loading speaker extractor...")
    speaker_extractor = SpeakerEmbeddingExtractor(model_id=MODEL_ID, device=str(DEVICE))
    speaker_extractor.eval()
    speaker_extractor.requires_grad_(False)

    linear_model = AgeLinearClassifier(
        input_dim=SPEAKER_EMB_DIM,
        num_age_groups=dataset.num_age_classes,
    ).to(DEVICE)

    mlp_model = AgeSimpleMLP(
        input_dim=SPEAKER_EMB_DIM,
        hidden_dim=MLP_HIDDEN_DIM,
        num_age_groups=dataset.num_age_classes,
        dropout=MLP_DROPOUT,
    ).to(DEVICE)

    log_dir = "logs/age_prediction_baselines"
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, "train_val_metrics.csv")

    csv_file = open(csv_path, mode="w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        [
            "model",
            "epoch",
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc",
            "val_auc",
        ]
    )

    linear_best = train_and_validate(
        model_name="Linear Age Classifier",
        model=linear_model,
        speaker_extractor=speaker_extractor,
        train_loader=train_loader,
        val_loader=val_loader,
        num_age_groups=dataset.num_age_classes,
        csv_writer=csv_writer,
    )

    mlp_best = train_and_validate(
        model_name="Simple MLP Age Classifier",
        model=mlp_model,
        speaker_extractor=speaker_extractor,
        train_loader=train_loader,
        val_loader=val_loader,
        num_age_groups=dataset.num_age_classes,
        csv_writer=csv_writer,
    )

    csv_writer.writerow([])
    csv_writer.writerow(["summary"])
    csv_writer.writerow(
        [
            "Linear Age Classifier",
            linear_best["epoch"],
            "",
            "",
            "",
            linear_best["val_acc"],
            linear_best["val_auc"],
        ]
    )
    csv_writer.writerow(
        [
            "Simple MLP Age Classifier",
            mlp_best["epoch"],
            "",
            "",
            "",
            mlp_best["val_acc"],
            mlp_best["val_auc"],
        ]
    )
    csv_file.close()

    print("\n" + "=" * 90)
    print("Age Prediction Baseline Summary")
    print(
        f"Linear | best epoch: {linear_best['epoch']} | "
        f"best val acc: {linear_best['val_acc']:.2f}% | best val AUC: {linear_best['val_auc']:.6f}"
    )
    print(
        f"MLP    | best epoch: {mlp_best['epoch']} | "
        f"best val acc: {mlp_best['val_acc']:.2f}% | best val AUC: {mlp_best['val_auc']:.6f}"
    )
    print(f"CSV saved to: {csv_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()