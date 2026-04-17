import argparse
import csv
import random
import re
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Subset, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.vox2_loader import Vox2Dataset
from model.disentangled_model.arcface import ArcMarginProduct
from model.disentangled_model.linear_decorr_mlp import LinearDecorrMLP
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


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_DATASET_NAME = "VoxCeleb2"
DEFAULT_CHECKPOINT_ROOT = Path("checkpoints/linear_decorr_mlp")
DEFAULT_LOG_ROOT = Path("logs/linear_decorr_identity_mlp")
DEFAULT_OUTPUT_ROOT = Path("checkpoints/linear_decorr_identity_mlp")

SPEAKER_EMB_DIM = 192
AGE_DIM = 1
MLP_HIDDEN_DIMS = [256]
MLP_OUTPUT_DIM = 129
ARCFACE_S = 64
ARCFACE_M = 0.2
ARCFACE_EASY_MARGIN = False

NUM_WORKERS = 0
EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
SPLIT_SEED = 42
TRAIN_UTTS_PER_SPK = 8
VAL_UTTS_PER_SPK = 2
TOPK = 5


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_next_exp_dir(base_dir: Path, prefix: str = "exp"):
    base_dir.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    max_idx = 0

    for child in base_dir.iterdir():
        if not child.is_dir():
            continue
        match = pattern.match(child.name)
        if match:
            max_idx = max(max_idx, int(match.group(1)))

    exp_name = f"{prefix}{max_idx + 1}"
    exp_path = base_dir / exp_name
    exp_path.mkdir(parents=True, exist_ok=False)
    return exp_path, exp_name


def find_latest_backbone_checkpoint(checkpoint_root: Path) -> Path:
    if not checkpoint_root.exists():
        raise FileNotFoundError(f"找不到 checkpoint 根目錄: {checkpoint_root}")

    pattern = re.compile(r"^exp(\d+)$")
    candidates = []
    for child in checkpoint_root.iterdir():
        if not child.is_dir():
            continue
        match = pattern.match(child.name)
        if not match:
            continue

        exp_idx = int(match.group(1))
        best_ckpt = child / "best_model.pth"
        last_ckpt = child / "last_model.pth"

        if best_ckpt.exists():
            candidates.append((exp_idx, best_ckpt))
        elif last_ckpt.exists():
            candidates.append((exp_idx, last_ckpt))

    if not candidates:
        raise FileNotFoundError(
            f"在 {checkpoint_root} 找不到任何 exp*/best_model.pth 或 exp*/last_model.pth"
        )

    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def build_train_val_indices_by_speaker(
    dataset,
    train_utts_per_spk: int = 8,
    val_utts_per_spk: int = 2,
    seed: int = 42,
):
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
                f"Speaker {speaker_id} 的語句數不足 {required}，請確認 Vox2Dataset 的取樣設定。"
            )

        permuted = rng.permutation(indices)
        train_indices.extend(permuted[:train_utts_per_spk].tolist())
        val_indices.extend(permuted[train_utts_per_spk:required].tolist())
        used_speakers.append(speaker_id)

    return train_indices, val_indices, used_speakers


def compute_topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
    if logits is None or targets is None:
        return float("nan")
    if logits.numel() == 0 or targets.numel() == 0:
        return float("nan")

    k = min(k, logits.shape[1])
    topk = logits.topk(k, dim=1).indices
    correct = topk.eq(targets.view(-1, 1)).any(dim=1).float().mean().item()
    return float(correct) * 100.0


class SpeakerSimpleMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, feature_dim: int = 192, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


def load_backbone_checkpoint(model: nn.Module, checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    state_dict = checkpoint.get("model", checkpoint)
    cleaned_state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)

    print(f"已載入 backbone checkpoint: {checkpoint_path}")
    if missing:
        print(f"  Missing keys: {missing}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected}")


def extract_identity_features(
    backbone: nn.Module,
    speaker_extractor: nn.Module,
    dataloader: DataLoader,
):
    backbone.eval()
    speaker_extractor.eval()

    feature_chunks = []
    label_chunks = []

    with torch.no_grad():
        for waveform, speaker_label, _, _ in tqdm(dataloader, desc="Extracting features"):
            waveform = waveform.to(DEVICE, non_blocking=True)
            speaker_label = speaker_label.to(DEVICE, non_blocking=True).long()

            speaker_emb = speaker_extractor(waveform)
            outputs = backbone(speaker_emb)
            feature_chunks.append(outputs["z"].detach().cpu())
            label_chunks.append(speaker_label.detach().cpu())

    features = torch.cat(feature_chunks, dim=0) if feature_chunks else torch.empty(0)
    labels = torch.cat(label_chunks, dim=0) if label_chunks else torch.empty(0, dtype=torch.long)
    return features, labels


def build_feature_loader(features: torch.Tensor, labels: torch.Tensor, batch_size: int, shuffle: bool):
    dataset = TensorDataset(features, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def run_one_epoch(model: nn.Module, arcface_head: nn.Module, dataloader: DataLoader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    arcface_head.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    logits_all = []
    targets_all = []

    for features, labels in tqdm(dataloader, desc="Train" if is_train else "Val"):
        features = features.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True).long()

        if is_train:
            optimizer.zero_grad()

        speaker_feat = model(features)
        logits = arcface_head(speaker_feat, labels)
        loss = criterion(logits, labels)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)
        logits_all.append(logits.detach().cpu())
        targets_all.append(labels.detach().cpu())

    avg_loss = total_loss / max(1, len(dataloader))
    acc = 100.0 * total_correct / max(1, total_samples)
    all_logits = torch.cat(logits_all, dim=0) if logits_all else torch.empty(0)
    all_targets = torch.cat(targets_all, dim=0) if targets_all else torch.empty(0, dtype=torch.long)

    if all_logits.numel() > 0 and all_targets.numel() > 0:
        preds = all_logits.argmax(dim=1).numpy()
        target_np = all_targets.numpy()
        macro_f1 = float(f1_score(target_np, preds, average="macro"))
        topk_acc = compute_topk_accuracy(all_logits, all_targets, k=TOPK)
    else:
        macro_f1 = float("nan")
        topk_acc = float("nan")

    return avg_loss, acc, macro_f1, topk_acc, all_logits, all_targets


def main():
    parser = argparse.ArgumentParser(description="用 linear_decorr_mlp 的完整 z 特徵訓練 speaker 身分 MLP")
    parser.add_argument("--backbone-checkpoint", type=str, default="", help="已訓練好的 backbone 權重路徑")
    parser.add_argument("--checkpoint-root", type=str, default=str(DEFAULT_CHECKPOINT_ROOT), help="backbone checkpoint 根目錄")
    parser.add_argument("--log-root", type=str, default=str(DEFAULT_LOG_ROOT), help="TensorBoard / CSV 輸出根目錄")
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT), help="分類器 checkpoint 輸出根目錄")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--feature-dim", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--train-utts-per-spk", type=int, default=TRAIN_UTTS_PER_SPK)
    parser.add_argument("--val-utts-per-spk", type=int, default=VAL_UTTS_PER_SPK)
    args = parser.parse_args()

    set_seed(args.seed)

    checkpoint_root = Path(args.checkpoint_root)
    backbone_checkpoint = Path(args.backbone_checkpoint) if args.backbone_checkpoint else find_latest_backbone_checkpoint(checkpoint_root)
    log_root = Path(args.log_root)
    output_root = Path(args.output_root)
    log_dir, exp_name = get_next_exp_dir(log_root, prefix="exp")
    output_dir = output_root / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = log_dir / "identity_mlp_metrics.csv"
    writer = SummaryWriter(log_dir=str(log_dir))

    print("Preparing datasets...")
    full_dataset = Vox2Dataset(
        audio_dir=DATASET_INFO[TRAIN_DATASET_NAME]["AUDIO_DIR"],
        audio_meta_dir=DATASET_INFO[TRAIN_DATASET_NAME]["AUDIO_META_DIR"],
        musan_path=DATASET_INFO["MUSAN"]["AUDIO_DIR"],
        rir_path=DATASET_INFO["RIR"]["AUDIO_DIR"],
        suffix=DATASET_INFO[TRAIN_DATASET_NAME]["audio_suffix"],
        age_target_mode="group",
        use_acoustic_features=False,
        min_utts_per_speaker=args.train_utts_per_spk + args.val_utts_per_spk,
    )

    train_indices, val_indices, used_speakers = build_train_val_indices_by_speaker(
        full_dataset,
        train_utts_per_spk=args.train_utts_per_spk,
        val_utts_per_spk=args.val_utts_per_spk,
        seed=args.seed,
    )

    if len(used_speakers) == 0:
        raise RuntimeError("沒有任何 speaker 能滿足切分條件，請調整取樣設定或檢查資料集。")

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    num_speakers = len(full_dataset.speaker2idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Split summary | Eligible speakers: {len(used_speakers)}")
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    print(f"Speaker classes: {num_speakers}")
    print(f"Backbone checkpoint: {backbone_checkpoint}")
    print(f"Log directory: {log_dir}")
    print(f"Classifier checkpoint directory: {output_dir}")
    print(f"Running on {DEVICE}")

    speaker_extractor = SpeakerEmbeddingExtractor(model_id=MODEL_ID, device=str(DEVICE))
    speaker_extractor.eval()
    speaker_extractor.requires_grad_(False)

    backbone = LinearDecorrMLP(
        input_dim=SPEAKER_EMB_DIM,
        hidden_dims=MLP_HIDDEN_DIMS,
        output_dim=MLP_OUTPUT_DIM,
        num_speakers=num_speakers,
        num_age_groups=full_dataset.num_age_classes,
        dropout=0.1,
        age_dim=AGE_DIM,
    ).to(DEVICE)
    load_backbone_checkpoint(backbone, backbone_checkpoint)
    backbone.eval()
    backbone.requires_grad_(False)

    print("Extracting frozen full z features...")
    train_features, train_labels = extract_identity_features(backbone, speaker_extractor, train_loader)
    val_features, val_labels = extract_identity_features(backbone, speaker_extractor, val_loader)

    train_feature_loader = build_feature_loader(train_features, train_labels, args.batch_size, shuffle=True)
    val_feature_loader = build_feature_loader(val_features, val_labels, args.batch_size, shuffle=False)

    classifier = SpeakerSimpleMLP(
        input_dim=train_features.shape[1],
        hidden_dim=args.hidden_dim,
        feature_dim=args.feature_dim,
        dropout=args.dropout,
    ).to(DEVICE)

    arcface_head = ArcMarginProduct(
        in_features=args.feature_dim,
        out_features=num_speakers,
        s=ARCFACE_S,
        m=ARCFACE_M,
        easy_margin=ARCFACE_EASY_MARGIN,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        list(classifier.parameters()) + list(arcface_head.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_val_acc = -1.0
    best_epoch = -1

    with open(csv_path, mode="w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            [
                "epoch",
                "lr",
                "train_loss",
                "train_acc",
                "train_macro_f1",
                "train_top5_acc",
                "val_loss",
                "val_acc",
                "val_macro_f1",
                "val_top5_acc",
                "best_val_acc",
            ]
        )

        print("Start training identity MLP...")
        for epoch in range(args.epochs):
            current_lr = optimizer.param_groups[0]["lr"]

            train_loss, train_acc, train_macro_f1, train_top5_acc, _, _ = run_one_epoch(
                model=classifier,
                arcface_head=arcface_head,
                dataloader=train_feature_loader,
                criterion=criterion,
                optimizer=optimizer,
            )

            with torch.no_grad():
                val_loss, val_acc, val_macro_f1, val_top5_acc, _, _ = run_one_epoch(
                    model=classifier,
                    arcface_head=arcface_head,
                    dataloader=val_feature_loader,
                    criterion=criterion,
                    optimizer=None,
                )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                torch.save(
                    {
                        "classifier": classifier.state_dict(),
                        "arcface_head": arcface_head.state_dict(),
                        "backbone_checkpoint": str(backbone_checkpoint),
                        "num_speakers": num_speakers,
                        "hidden_dim": args.hidden_dim,
                        "feature_dim": args.feature_dim,
                        "dropout": args.dropout,
                        "best_epoch": best_epoch,
                        "best_val_acc": best_val_acc,
                    },
                    output_dir / "best_identity_mlp.pth",
                )

            print("\n" + "=" * 110)
            print(
                f"Epoch [{epoch + 1:3d}/{args.epochs}] | LR: {current_lr:.6f} | "
                f"Best Val Acc: {best_val_acc:.2f}%"
            )
            print(
                f"Train => Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, "
                f"Macro F1: {train_macro_f1:.4f}, Top-{TOPK}: {train_top5_acc:.2f}%"
            )
            print(
                f"Val   => Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, "
                f"Macro F1: {val_macro_f1:.4f}, Top-{TOPK}: {val_top5_acc:.2f}%"
            )
            print("=" * 110 + "\n")

            writer.add_scalar("Train/Loss", train_loss, epoch)
            writer.add_scalar("Train/Acc", train_acc, epoch)
            writer.add_scalar("Train/Macro_F1", train_macro_f1, epoch)
            writer.add_scalar(f"Train/Top{TOPK}_Acc", train_top5_acc, epoch)
            writer.add_scalar("Val/Loss", val_loss, epoch)
            writer.add_scalar("Val/Acc", val_acc, epoch)
            writer.add_scalar("Val/Macro_F1", val_macro_f1, epoch)
            writer.add_scalar(f"Val/Top{TOPK}_Acc", val_top5_acc, epoch)
            writer.add_scalar("LR", current_lr, epoch)
            writer.add_scalar("Best/Val_Acc", best_val_acc, epoch)

            csv_writer.writerow(
                [
                    epoch + 1,
                    current_lr,
                    train_loss,
                    train_acc,
                    train_macro_f1,
                    train_top5_acc,
                    val_loss,
                    val_acc,
                    val_macro_f1,
                    val_top5_acc,
                    best_val_acc,
                ]
            )
            csv_file.flush()

            scheduler.step()

        torch.save(
            {
                "classifier": classifier.state_dict(),
                        "arcface_head": arcface_head.state_dict(),
                "backbone_checkpoint": str(backbone_checkpoint),
                "num_speakers": num_speakers,
                "hidden_dim": args.hidden_dim,
                        "feature_dim": args.feature_dim,
                "dropout": args.dropout,
                "best_epoch": best_epoch,
                "best_val_acc": best_val_acc,
            },
            output_dir / "last_identity_mlp.pth",
        )

    writer.close()
    print(f"Training finished. Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}.")
    print(f"Metrics CSV: {csv_path}")
    print(f"Best checkpoint: {output_dir / 'best_identity_mlp.pth'}")
    print(f"Last checkpoint: {output_dir / 'last_identity_mlp.pth'}")


if __name__ == "__main__":
    main()