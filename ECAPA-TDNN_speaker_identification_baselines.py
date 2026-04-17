import csv
import os
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data.vox1_loader import PairwiseDataset
from data.vox2_loader import Vox2Dataset
from model.disentangled_model.arcface import ArcMarginProduct
from model.feature_extractor.ecapa_tdnn import SpeakerEmbeddingExtractor
from params.param import BATCH_SIZE, DATASET_INFO, MODEL_ID
from tool.EER import compute_eer

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
TEST_DATASET_NAME = "Vox-CA20"

NUM_WORKERS = 0
EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
SPLIT_SEED = 42
TRAIN_UTTS_PER_SPK = 8
VAL_UTTS_PER_SPK = 2

SPEAKER_EMB_DIM = 192
MLP_HIDDEN_DIM = 192
MLP_DROPOUT = 0.1

ARCFACE_S = 64
ARCFACE_M = 0.2
ARCFACE_EASY_MARGIN = False


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


def compute_speaker_id_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """計算說話者身分分類準確度。"""
    if logits is None or targets is None:
        return float("nan")
    if logits.numel() == 0 or targets.numel() == 0:
        return float("nan")

    pred = logits.argmax(dim=1)
    return float((pred == targets).sum().item()) / targets.numel()


class SpeakerSimpleMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, feature_dim: int = MLP_HIDDEN_DIM, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), 
            nn.PReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, x):
        return self.net(x)


class SpeakerDirectArcFace(nn.Module):
    """直接使用 ECAPA 特徵輸出，不經過中間層，直接接 ArcFace 分類頭。"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def eval_vox1_eer(model, speaker_extractor, eval_dataloader):
    """使用 Vox1 pairwise 測試集計算 EER。"""
    model.eval()
    speaker_extractor.eval()

    scores = []
    labels = []

    with torch.no_grad():
        for is_same, _, _, wav1, wav2, _, _ in tqdm(eval_dataloader, desc="Test"):
            wav1 = wav1.to(DEVICE, non_blocking=True)
            wav2 = wav2.to(DEVICE, non_blocking=True)

            spk_emb1 = speaker_extractor(wav1)
            spk_emb2 = speaker_extractor(wav2)
            feat1 = model(spk_emb1)
            feat2 = model(spk_emb2)

            norm_feat1 = F.normalize(feat1, p=2, dim=1)
            norm_feat2 = F.normalize(feat2, p=2, dim=1)
            pair_scores = F.cosine_similarity(norm_feat1, norm_feat2).cpu()

            scores.append(pair_scores)
            labels.extend(is_same.cpu().tolist())

    if not scores:
        return float("nan")

    final_scores = torch.cat(scores, dim=0).numpy()
    final_labels = np.array(labels)
    return float(compute_eer(final_scores, final_labels))


def run_one_epoch(model, speaker_extractor, dataloader, arcface_head, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    arcface_head.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    logits_all = []
    targets_all = []

    for waveform, speaker_label, _, _ in tqdm(dataloader, desc="Train" if is_train else "Val"):
        waveform = waveform.to(DEVICE, non_blocking=True)
        speaker_label = speaker_label.to(DEVICE, non_blocking=True).long()

        with torch.no_grad():
            speaker_emb = speaker_extractor(waveform)

        if is_train:
            optimizer.zero_grad()

        speaker_feat = model(speaker_emb)
        logits = arcface_head(speaker_feat, speaker_label)
        loss = criterion(logits, speaker_label)

        if is_train:
            loss.backward()
            optimizer.step()

        pred = logits.argmax(dim=1)
        total_loss += float(loss.item())
        total_correct += (pred == speaker_label).sum().item()
        total_samples += speaker_label.size(0)
        logits_all.append(logits.detach().cpu())
        targets_all.append(speaker_label.detach().cpu())

    avg_loss = total_loss / max(1, len(dataloader))
    acc = 100.0 * total_correct / max(1, total_samples)
    all_logits = torch.cat(logits_all, dim=0) if logits_all else torch.tensor([])
    all_targets = torch.cat(targets_all, dim=0) if targets_all else torch.tensor([], dtype=torch.long)

    return avg_loss, acc, all_logits, all_targets


def train_and_validate(
    model_name,
    model,
    speaker_extractor,
    train_loader,
    val_loader,
    test_loader,
    num_speakers,
    csv_writer,
    arcface_feature_dim=None,
):
    print(f"\n===== {model_name} =====")
    
    # 如果沒有指定 arcface_feature_dim，則使用 MLP_HIDDEN_DIM；如果是 Direct ArcFace，使用 SPEAKER_EMB_DIM
    if arcface_feature_dim is None:
        arcface_feature_dim = MLP_HIDDEN_DIM
    
    arcface_head = ArcMarginProduct(
        in_features=arcface_feature_dim,
        out_features=num_speakers,
        s=ARCFACE_S,
        m=ARCFACE_M,
        easy_margin=ARCFACE_EASY_MARGIN,
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(arcface_head.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best = {
        "epoch": -1,
        "val_acc": -1.0,
        "test_eer": float("inf"),
    }

    for epoch in range(EPOCHS):
        train_loss, train_acc, _, _ = run_one_epoch(
            model=model,
            speaker_extractor=speaker_extractor,
            dataloader=train_loader,
            arcface_head=arcface_head,
            criterion=criterion,
            optimizer=optimizer,
        )

        with torch.no_grad():
            val_loss, val_acc, val_logits, val_targets = run_one_epoch(
                model=model,
                speaker_extractor=speaker_extractor,
                dataloader=val_loader,
                arcface_head=arcface_head,
                criterion=criterion,
                optimizer=None,
            )

        print(
            f"Epoch [{epoch + 1:2d}/{EPOCHS}] | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

        test_eer = eval_vox1_eer(
            model=model,
            speaker_extractor=speaker_extractor,
            eval_dataloader=test_loader,
        )
        print(f"Test EER (Vox1 {TEST_DATASET_NAME}): {test_eer:.4f}")

        csv_writer.writerow(
            [
                model_name,
                epoch + 1,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                test_eer,
            ]
        )

        if val_acc > best["val_acc"]:
            best["epoch"] = epoch + 1
            best["val_acc"] = val_acc

        if test_eer < best["test_eer"]:
            best["test_eer"] = test_eer

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
    test_dataset = PairwiseDataset(
        audio_dir=DATASET_INFO["VoxCeleb1"][TEST_DATASET_NAME]["AUDIO_DIR"],
        audio_meta_dir=DATASET_INFO["VoxCeleb1"][TEST_DATASET_NAME]["AUDIO_DATALIST"],
        audio_meta_csv_path=DATASET_INFO["VoxCeleb1"]["AUDIO_META_DIR"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    num_speakers = len(dataset.speaker2idx)

    print(
        f"Split summary | Eligible speakers: {len(used_speakers)}"
    )
    print(
        f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)} | "
        f"Test pairs: {len(test_dataset)} | num_speakers: {num_speakers}"
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

    mlp_model = SpeakerSimpleMLP(
        input_dim=SPEAKER_EMB_DIM,
        hidden_dim=256,
        feature_dim=MLP_HIDDEN_DIM,
        dropout=MLP_DROPOUT,
    ).to(DEVICE)

    direct_arcface_model = SpeakerDirectArcFace().to(DEVICE)

    log_dir = "logs/speaker_identification_baselines"
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
            "test_eer",
        ]
    )

    mlp_best = train_and_validate(
        model_name="Simple MLP Speaker Classifier",
        model=mlp_model,
        speaker_extractor=speaker_extractor,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_speakers=num_speakers,
        csv_writer=csv_writer,
    )

    direct_arcface_best = train_and_validate(
        model_name="Direct ArcFace (ECAPA → ArcFace)",
        model=direct_arcface_model,
        speaker_extractor=speaker_extractor,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_speakers=num_speakers,
        csv_writer=csv_writer,
        arcface_feature_dim=SPEAKER_EMB_DIM,
    )

    csv_writer.writerow([])
    csv_writer.writerow(["summary"])
    csv_writer.writerow(
        [
            "Simple MLP Speaker Classifier",
            mlp_best["epoch"],
            "",
            "",
            "",
            mlp_best["val_acc"],
            mlp_best["test_eer"],
        ]
    )
    csv_writer.writerow(
        [
            "Direct ArcFace (ECAPA → ArcFace)",
            direct_arcface_best["epoch"],
            "",
            "",
            "",
            direct_arcface_best["val_acc"],
            direct_arcface_best["test_eer"],
        ]
    )
    csv_file.close()

    print("\n" + "=" * 90)
    print("Speaker Identification Baseline Summary")
    print(
        f"MLP    | best epoch: {mlp_best['epoch']} | "
        f"best val acc: {mlp_best['val_acc']:.2f}% | "
        f"best test EER: {mlp_best['test_eer']:.4f}"
    )
    print(
        f"Direct ArcFace | best epoch: {direct_arcface_best['epoch']} | "
        f"best val acc: {direct_arcface_best['val_acc']:.2f}% | "
        f"best test EER: {direct_arcface_best['test_eer']:.4f}"
    )
    print(f"CSV saved to: {csv_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()
