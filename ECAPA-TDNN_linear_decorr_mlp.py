import csv
import os
import warnings
from pathlib import Path

import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.vox2_loader import Vox2Dataset
from model.disentangled_model.linear_decorr_mlp import LinearDecorrLoss, LinearDecorrMLP
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
VAL_DATASET_NAME = "Vox-CA20"

EPOCHS = 40
NUM_WORKERS = 0
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
MAX_EVAL_PAIRS = 20000

SPEAKER_EMB_DIM = 192
MLP_HIDDEN_DIMS = [256]
MLP_OUTPUT_DIM = 129  # 第 0 維做年齡分類，剩下 128 維做說話者分類與驗證
MLP_DROPOUT = 0.1

LAMBDA_SPK = 1.0
LAMBDA_AGE = 1.0
LAMBDA_DECORR = 1.0

ARCFACE_S = 30.0
ARCFACE_M = 0.35
ARCFACE_EASY_MARGIN = False

EARLY_STOPPING_PATIENCE = 10


def _normalize_audio_dirs(audio_dirs):
    if isinstance(audio_dirs, (str, Path)):
        return [str(audio_dirs)]
    return [str(d) for d in audio_dirs]


def build_eval_dataset(audio_dirs, audio_meta_dir, max_pairs=20000):
    audio_dirs = _normalize_audio_dirs(audio_dirs)

    def find_audio_path(relative_path):
        for audio_dir in audio_dirs:
            audio_path = Path(audio_dir) / relative_path
            if audio_path.exists():
                return str(audio_path)
        raise FileNotFoundError(f"{relative_path} not found in audio_dirs")

    datalist = []
    with open(audio_meta_dir, "r", encoding="utf-8") as f:
        lines = f.readlines()[:max_pairs]

    for line in tqdm(lines, desc="Building eval dataset"):
        line = line.strip().split()
        if len(line) < 3:
            continue

        is_same = int(line[0])
        spk1_rel = line[1]
        spk2_rel = line[2]

        spk1_path = find_audio_path(spk1_rel)
        spk2_path = find_audio_path(spk2_rel)

        spk1_id = spk1_rel.split("/")[0]
        spk2_id = spk2_rel.split("/")[0]

        datalist.append((is_same, spk1_id, spk2_id, spk1_path, spk2_path))

    pos = sum(1 for x in datalist if x[0] == 1)
    neg = sum(1 for x in datalist if x[0] == 0)
    print(f"[Eval] Positive pairs: {pos}, Negative pairs: {neg}")
    return datalist


def eval_network(model, speaker_extractor, datalist):
    model.eval()
    speaker_extractor.eval()

    before_scores = []
    after_scores = []
    labels = []

    before_embs = []
    after_embs = []
    final_ids = []

    with torch.no_grad():
        for is_same, id1, id2, path1, path2 in tqdm(datalist, desc="Evaluating"):
            wav1, _ = torchaudio.load(path1)
            wav2, _ = torchaudio.load(path2)

            spk_emb1 = speaker_extractor(wav1)
            spk_emb2 = speaker_extractor(wav2)

            out1 = model(spk_emb1)
            out2 = model(spk_emb2)

            h1 = F.normalize(spk_emb1, p=2, dim=1)
            h2 = F.normalize(spk_emb2, p=2, dim=1)
            score_before = F.cosine_similarity(h1, h2).cpu()
            before_scores.append(score_before)

            w1 = F.normalize(out1["z_id"], p=2, dim=1)
            w2 = F.normalize(out2["z_id"], p=2, dim=1)
            score_after = F.cosine_similarity(w1, w2).cpu()
            after_scores.append(score_after)

            before_embs.append(spk_emb1.cpu())
            before_embs.append(spk_emb2.cpu())
            after_embs.append(out1["z_id"].cpu())
            after_embs.append(out2["z_id"].cpu())

            labels.append(is_same)
            final_ids.extend([id1, id2])

    final_labels = torch.tensor(labels).numpy()
    final_before_scores = torch.cat(before_scores).numpy()
    final_after_scores = torch.cat(after_scores).numpy()

    eer_before = compute_eer(final_before_scores, final_labels)
    eer_after = compute_eer(final_after_scores, final_labels)

    final_before_embs = torch.cat(before_embs, dim=0)
    final_after_embs = torch.cat(after_embs, dim=0)

    return eer_before, eer_after, final_before_embs, final_after_embs, final_ids


print("Preparing datasets...")
full_dataset = Vox2Dataset(
    audio_dir=DATASET_INFO[TRAIN_DATASET_NAME]["AUDIO_DIR"],
    audio_meta_dir=DATASET_INFO[TRAIN_DATASET_NAME]["AUDIO_META_DIR"],
    musan_path=DATASET_INFO["MUSAN"]["AUDIO_DIR"],
    rir_path=DATASET_INFO["RIR"]["AUDIO_DIR"],
    suffix=DATASET_INFO[TRAIN_DATASET_NAME]["audio_suffix"],
    age_target_mode="group",
    use_acoustic_features=False,
)

train_loader = DataLoader(
    full_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

eval_dataset = build_eval_dataset(
    audio_dirs=DATASET_INFO["VoxCeleb1"][VAL_DATASET_NAME]["AUDIO_DIR"],
    audio_meta_dir=DATASET_INFO["VoxCeleb1"][VAL_DATASET_NAME]["AUDIO_DATALIST"],
    max_pairs=MAX_EVAL_PAIRS,
)

print(f"Train size: {len(full_dataset)} | Eval pairs: {len(eval_dataset)}")
print(f"Building models on {DEVICE}...")

speaker_extractor = SpeakerEmbeddingExtractor(model_id=MODEL_ID, device=str(DEVICE))
speaker_extractor.eval()
speaker_extractor.requires_grad_(False)

model = LinearDecorrMLP(
    input_dim=SPEAKER_EMB_DIM,
    hidden_dims=MLP_HIDDEN_DIMS,
    output_dim=MLP_OUTPUT_DIM,
    num_speakers=len(full_dataset.speaker2idx),
    num_age_groups=full_dataset.num_age_classes,
    dropout=MLP_DROPOUT,
).to(DEVICE)

criterion = LinearDecorrLoss(
    latent_id_dim=MLP_OUTPUT_DIM - 1,
    num_speakers=len(full_dataset.speaker2idx),
    lambda_spk=LAMBDA_SPK,
    lambda_age=LAMBDA_AGE,
    lambda_decorr=LAMBDA_DECORR,
    arcface_s=ARCFACE_S,
    arcface_m=ARCFACE_M,
    arcface_easy_margin=ARCFACE_EASY_MARGIN,
).to(DEVICE)

optimizer = torch.optim.Adam(
    list(model.parameters()) + list(criterion.parameters()),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

log_dir = "logs/linear_decorr_mlp"
checkpoint_dir = "checkpoints/linear_decorr_mlp"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

writer = SummaryWriter(log_dir=log_dir)

csv_path = os.path.join(log_dir, "train_test_metrics.csv")
csv_file = open(csv_path, mode="w", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(
    [
        "epoch",
        "lr",
        "train_total_loss",
        "train_spk_loss",
        "train_age_loss",
        "train_decorr_loss",
        "train_spk_acc",
        "train_age_acc",
        "eer_before",
        "eer_after",
        "best_eer_after",
    ]
)

best_EER = float("inf")
best_epoch = -1
no_improve_epochs = 0

print("Start training (Linear-Decorr MLP)...")
for epoch in range(EPOCHS):
    model.train()

    train_total_loss = 0.0
    train_spk_loss = 0.0
    train_age_loss = 0.0
    train_decorr_loss = 0.0
    train_correct_spk = 0
    train_correct_age = 0
    train_total_samples = 0

    for waveform, label_spk, _, label_age in tqdm(train_loader, desc=f"Train {epoch + 1}/{EPOCHS}"):
        waveform = waveform.to(DEVICE)
        label_spk = label_spk.to(DEVICE)
        label_age = label_age.to(DEVICE).long()

        with torch.no_grad():
            speaker_emb = speaker_extractor(waveform)

        outputs = model(speaker_emb)
        loss, loss_dict = criterion(outputs=outputs, target_spk=label_spk, target_age=label_age)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_total_loss += float(loss.item())
        train_spk_loss += float(loss_dict["loss_spk"].item())
        train_age_loss += float(loss_dict["loss_age"].item())
        train_decorr_loss += float(loss_dict["loss_decorr"].item())

        pred_spk = loss_dict["arcface_logits"].argmax(dim=1)
        pred_age = outputs["logits_age"].argmax(dim=1)
        train_correct_spk += (pred_spk == label_spk).sum().item()
        train_correct_age += (pred_age == label_age).sum().item()
        train_total_samples += label_spk.size(0)

    train_total_loss /= max(1, len(train_loader))
    train_spk_loss /= max(1, len(train_loader))
    train_age_loss /= max(1, len(train_loader))
    train_decorr_loss /= max(1, len(train_loader))
    train_spk_acc = 100.0 * train_correct_spk / max(1, train_total_samples)
    train_age_acc = 100.0 * train_correct_age / max(1, train_total_samples)

    eer_before, eer_after, before_embs, after_embs, final_ids = eval_network(
        model=model,
        speaker_extractor=speaker_extractor,
        datalist=eval_dataset,
    )

    current_lr = optimizer.param_groups[0]["lr"]
    print("\n" + "=" * 130)
    print(f"Epoch [{epoch + 1:3d}/{EPOCHS}] | LR: {current_lr:.6f}")
    print(
        f"Train Loss => Total: {train_total_loss:.4f}, SPK: {train_spk_loss:.4f}, "
        f"AGE: {train_age_loss:.4f}, DECORR: {train_decorr_loss:.4f}"
    )
    print(f"Acc   => Train SPK/Age: {train_spk_acc:.2f}%/{train_age_acc:.2f}%")
    print(
        f"Eval  => EER(before disentangle): {eer_before:.4f} | "
        f"EER(after disentangle): {eer_after:.4f} | Best(after): {best_EER:.4f}"
    )
    print("=" * 130 + "\n")

    if best_EER > eer_after:
        best_EER = eer_after
        best_epoch = epoch + 1
        no_improve_epochs = 0
        torch.save(
            {"model": model.state_dict()},
            os.path.join(checkpoint_dir, "best_model.pth"),
        )
        torch.save(
            {
                "embeddings": after_embs,
                "before_embeddings": before_embs,
                "ids": final_ids,
            },
            os.path.join(checkpoint_dir, "best_disentangled_embeddings.pt"),
        )
        print(f"✓ 儲存最佳模型 EER: {best_EER * 100:.2f}%")
    else:
        no_improve_epochs += 1

    if epoch == EPOCHS - 1:
        torch.save(
            {"model": model.state_dict()},
            os.path.join(checkpoint_dir, "last_model.pth"),
        )
        torch.save(
            {
                "embeddings": after_embs,
                "before_embeddings": before_embs,
                "ids": final_ids,
            },
            os.path.join(checkpoint_dir, "last_disentangled_embeddings.pt"),
        )
        print("✓ 儲存最終模型。")

    writer.add_scalar("Train/Total_Loss", train_total_loss, epoch)
    writer.add_scalar("Train/SPK_Loss", train_spk_loss, epoch)
    writer.add_scalar("Train/AGE_Loss", train_age_loss, epoch)
    writer.add_scalar("Train/DECORR_Loss", train_decorr_loss, epoch)
    writer.add_scalar("Train/Acc_SPK", train_spk_acc, epoch)
    writer.add_scalar("Train/Acc_Age", train_age_acc, epoch)
    writer.add_scalar("Eval/EER_Before", eer_before, epoch)
    writer.add_scalar("Eval/EER_After", eer_after, epoch)
    writer.add_scalar("Eval/Best_EER_After", best_EER, epoch)
    writer.add_scalar("LR", current_lr, epoch)

    csv_writer.writerow(
        [
            epoch + 1,
            current_lr,
            train_total_loss,
            train_spk_loss,
            train_age_loss,
            train_decorr_loss,
            train_spk_acc,
            train_age_acc,
            eer_before,
            eer_after,
            best_EER,
        ]
    )
    csv_file.flush()

    scheduler.step()

    if no_improve_epochs >= EARLY_STOPPING_PATIENCE:
        print(
            f"Early stopping 觸發：連續 {EARLY_STOPPING_PATIENCE} 個 epoch 無改善。"
            f"最佳 EER(after)={best_EER:.4f}（Epoch {best_epoch}）。"
        )
        torch.save(
            {"model": model.state_dict()},
            os.path.join(checkpoint_dir, "last_model.pth"),
        )
        torch.save(
            {
                "embeddings": after_embs,
                "before_embeddings": before_embs,
                "ids": final_ids,
            },
            os.path.join(checkpoint_dir, "last_disentangled_embeddings.pt"),
        )
        print("✓ 儲存 early-stopped 最終模型。")
        break

writer.close()
csv_file.close()
print("Training finished (Linear-Decorr MLP).")
