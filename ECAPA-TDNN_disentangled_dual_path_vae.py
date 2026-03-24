import os
import csv
import warnings

import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.vox2_loader import Vox2Dataset
from model.feature_extractor.ecapa_tdnn import SpeakerEmbeddingExtractor
from model.disentangled_model.dual_path_vae import DualPathVAE, DualPathVAELoss
from params.param import DATASET_INFO, BATCH_SIZE, MODEL_ID

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
SAMPLE_RATE = 16000

EPOCHS = 50
TRAIN_TEST_SPLIT = 0.9
NUM_WORKERS = 0
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

# Dual-Path VAE 超參數
SPEAKER_EMB_DIM = 192
ACOUSTIC_DIM = 8
LATENT_AGE_DIM = 16
LATENT_ID_DIM = 64

# loss 權重
LAMBDA_RECON = 1.0
LAMBDA_KL_AGE = 1.0
LAMBDA_KL_ID = 1.0
LAMBDA_CLS_SPK = 1.0
LAMBDA_CLS_AGE = 1.0
LAMBDA_COSINE_DISENTANGLE = 0.1


# ==========================================
# 2) 準備資料集（train/test split）
# ==========================================
print("Preparing datasets...")

full_dataset = Vox2Dataset(
    audio_dir=DATASET_INFO[TRAIN_DATASET_NAME]["AUDIO_DIR"],
    audio_meta_dir=DATASET_INFO[TRAIN_DATASET_NAME]["AUDIO_META_DIR"],
    musan_path=DATASET_INFO["MUSAN"]["AUDIO_DIR"],
    rir_path=DATASET_INFO["RIR"]["AUDIO_DIR"],
    suffix=DATASET_INFO[TRAIN_DATASET_NAME]["audio_suffix"],
    age_target_mode="group",
    use_acoustic_features=True,
    acoustic_feature_type="egemaps8",
    acoustic_sample_rate=SAMPLE_RATE,
)

train_size = int(len(full_dataset) * TRAIN_TEST_SPLIT)
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(
    full_dataset,
    [train_size, test_size],
    generator=torch.Generator().manual_seed(42),
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)

print(f"Train size: {len(train_dataset)} | Test size: {len(test_dataset)}")


# ==========================================
# 3) 建立模型與優化器
# ==========================================
print(f"Building models on {DEVICE}...")

# (A) 冷凍的 Speaker Embedding 抽取器
speaker_extractor = SpeakerEmbeddingExtractor(
    model_id=MODEL_ID,
    device=str(DEVICE),
)
speaker_extractor.eval()
speaker_extractor.requires_grad_(False)

# (B) Dual-Path VAE（只吃外部 acoustic_vec）
model = DualPathVAE(
    speaker_emb_dim=SPEAKER_EMB_DIM,
    acoustic_dim=ACOUSTIC_DIM,
    latent_age_dim=LATENT_AGE_DIM,
    latent_id_dim=LATENT_ID_DIM,
    num_speakers=len(full_dataset.speaker2idx),
    num_age_groups=full_dataset.num_age_classes,
).to(DEVICE)

criterion = DualPathVAELoss(
    lambda_recon=LAMBDA_RECON,
    lambda_kl_age=LAMBDA_KL_AGE,
    lambda_kl_id=LAMBDA_KL_ID,
    lambda_cls_spk=LAMBDA_CLS_SPK,
    lambda_cls_age=LAMBDA_CLS_AGE,
    lambda_cosine_disentangle=LAMBDA_COSINE_DISENTANGLE,
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
    eta_min=1e-5,
)


# ==========================================
# 4) Logger 與 checkpoint
# ==========================================
log_dir = "logs/dual_path_vae"
checkpoint_dir = "checkpoints/dual_path_vae"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

writer = SummaryWriter(log_dir=log_dir)

csv_path = os.path.join(log_dir, "train_test_metrics.csv")
csv_file = open(csv_path, mode="w", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "epoch",
    "lr",
    "train_total_loss",
    "train_recon_loss",
    "train_kl_age",
    "train_kl_id",
    "train_cls_spk",
    "train_cls_age",
    "train_cosine_disentangle",
    "train_spk_acc",
    "train_age_acc",
    "test_total_loss",
    "test_recon_loss",
    "test_kl_age",
    "test_kl_id",
    "test_cls_spk",
    "test_cls_age",
    "test_cosine_disentangle",
    "test_spk_acc",
    "test_age_acc",
])

best_test_loss = float("inf")


# ==========================================
# 5) 訓練與測試主迴圈
# ==========================================
print("Start training...")

for epoch in range(EPOCHS):
    model.train()

    train_total_loss = 0.0
    train_recon_loss = 0.0
    train_kl_age = 0.0
    train_kl_id = 0.0
    train_cls_spk = 0.0
    train_cls_age = 0.0
    train_cosine_disentangle = 0.0
    train_correct_spk = 0
    train_correct_age = 0
    train_total_samples = 0

    for waveform, label_spk, _, label_age, acoustic_feat in tqdm(train_loader, desc=f"Train {epoch + 1}/{EPOCHS}"):
        # waveform: [B, T]，acoustic_feat: [B, 8]
        waveform = waveform.to(DEVICE)
        acoustic_feat = acoustic_feat.to(DEVICE)
        label_spk = label_spk.to(DEVICE)
        label_age = label_age.to(DEVICE).long()

        # 1) 先用冷凍 ECAPA 抽 speaker embedding
        with torch.no_grad():
            speaker_emb = speaker_extractor(waveform)

        # 2) Dual-Path VAE 前向（聲學特徵已在 loader 預先提取）
        outputs = model(
            speaker_emb=speaker_emb,
            acoustic_vec=acoustic_feat,
        )

        # 3) 損失：重建 + KL_Age + KL_ID
        loss, loss_dict = criterion(
            outputs=outputs,
            speaker_emb_target=speaker_emb,
            target_spk=label_spk,
            target_age=label_age,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_total_loss += float(loss.item())
        train_recon_loss += float(loss_dict["recon_loss"].item())
        train_kl_age += float(loss_dict["kl_age"].item())
        train_kl_id += float(loss_dict["kl_id"].item())
        train_cls_spk += float(loss_dict["cls_spk"].item())
        train_cls_age += float(loss_dict["cls_age"].item())
        train_cosine_disentangle += float(loss_dict["cosine_disentangle"].item())

        pred_spk = outputs["logits_spk"].argmax(dim=1)
        pred_age = outputs["logits_age"].argmax(dim=1)
        train_correct_spk += (pred_spk == label_spk).sum().item()
        train_correct_age += (pred_age == label_age).sum().item()
        train_total_samples += label_spk.size(0)

    # epoch 訓練平均
    train_total_loss /= len(train_loader)
    train_recon_loss /= len(train_loader)
    train_kl_age /= len(train_loader)
    train_kl_id /= len(train_loader)
    train_cls_spk /= len(train_loader)
    train_cls_age /= len(train_loader)
    train_cosine_disentangle /= len(train_loader)
    train_spk_acc = 100.0 * train_correct_spk / max(1, train_total_samples)
    train_age_acc = 100.0 * train_correct_age / max(1, train_total_samples)

    # ------------------
    # test loss
    # ------------------
    model.eval()
    test_total_loss = 0.0
    test_recon_loss = 0.0
    test_kl_age = 0.0
    test_kl_id = 0.0
    test_cls_spk = 0.0
    test_cls_age = 0.0
    test_cosine_disentangle = 0.0
    test_correct_spk = 0
    test_correct_age = 0
    test_total_samples = 0

    with torch.no_grad():
        for waveform, label_spk, _, label_age, acoustic_feat in tqdm(test_loader, desc=f"Test  {epoch + 1}/{EPOCHS}"):
            waveform = waveform.to(DEVICE)
            acoustic_feat = acoustic_feat.to(DEVICE)
            label_spk = label_spk.to(DEVICE)
            label_age = label_age.to(DEVICE).long()
            speaker_emb = speaker_extractor(waveform)

            outputs = model(speaker_emb=speaker_emb, acoustic_vec=acoustic_feat)
            test_loss, test_loss_dict = criterion(
                outputs=outputs,
                speaker_emb_target=speaker_emb,
                target_spk=label_spk,
                target_age=label_age,
            )

            test_total_loss += float(test_loss.item())
            test_recon_loss += float(test_loss_dict["recon_loss"].item())
            test_kl_age += float(test_loss_dict["kl_age"].item())
            test_kl_id += float(test_loss_dict["kl_id"].item())
            test_cls_spk += float(test_loss_dict["cls_spk"].item())
            test_cls_age += float(test_loss_dict["cls_age"].item())
            test_cosine_disentangle += float(test_loss_dict["cosine_disentangle"].item())

            pred_spk = outputs["logits_spk"].argmax(dim=1)
            pred_age = outputs["logits_age"].argmax(dim=1)
            test_correct_spk += (pred_spk == label_spk).sum().item()
            test_correct_age += (pred_age == label_age).sum().item()
            test_total_samples += label_spk.size(0)

    test_total_loss /= len(test_loader)
    test_recon_loss /= len(test_loader)
    test_kl_age /= len(test_loader)
    test_kl_id /= len(test_loader)
    test_cls_spk /= len(test_loader)
    test_cls_age /= len(test_loader)
    test_cosine_disentangle /= len(test_loader)
    test_spk_acc = 100.0 * test_correct_spk / max(1, test_total_samples)
    test_age_acc = 100.0 * test_correct_age / max(1, test_total_samples)

    # ------------------
    # checkpoint
    # ------------------
    if test_total_loss < best_test_loss:
        best_test_loss = test_total_loss
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_by_test_loss.pth"))
        print(f"✓ Save best-by-test-loss model: {best_test_loss:.6f}")

    # 每個 epoch 都存 latest
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "latest.pth"))

    # ------------------
    # log print
    # ------------------
    current_lr = optimizer.param_groups[0]["lr"]
    print("\n" + "=" * 110)
    print(f"Epoch [{epoch + 1:3d}/{EPOCHS}] | LR: {current_lr:.6f}")
    print(
        f"Train Loss => Total: {train_total_loss:.4f}, Recon: {train_recon_loss:.4f}, "
        f"KL_Age: {train_kl_age:.4f}, KL_ID: {train_kl_id:.4f}, "
        f"CLS_SPK: {train_cls_spk:.4f}, CLS_AGE: {train_cls_age:.4f}, COS: {train_cosine_disentangle:.4f}"
    )
    print(
        f"Test  Loss => Total: {test_total_loss:.4f}, Recon: {test_recon_loss:.4f}, "
        f"KL_Age: {test_kl_age:.4f}, KL_ID: {test_kl_id:.4f}, "
        f"CLS_SPK: {test_cls_spk:.4f}, CLS_AGE: {test_cls_age:.4f}, COS: {test_cosine_disentangle:.4f}"
    )
    print(
        f"Acc   => Train SPK/Age: {train_spk_acc:.2f}%/{train_age_acc:.2f}% | "
        f"Test SPK/Age: {test_spk_acc:.2f}%/{test_age_acc:.2f}%"
    )
    print("=" * 110 + "\n")

    # ------------------
    # tensorboard
    # ------------------
    writer.add_scalar("Train/Total_Loss", train_total_loss, epoch)
    writer.add_scalar("Train/Recon_Loss", train_recon_loss, epoch)
    writer.add_scalar("Train/KL_Age", train_kl_age, epoch)
    writer.add_scalar("Train/KL_ID", train_kl_id, epoch)
    writer.add_scalar("Train/CLS_SPK", train_cls_spk, epoch)
    writer.add_scalar("Train/CLS_AGE", train_cls_age, epoch)
    writer.add_scalar("Train/CosineDisentangle", train_cosine_disentangle, epoch)
    writer.add_scalar("Train/Acc_SPK", train_spk_acc, epoch)
    writer.add_scalar("Train/Acc_Age", train_age_acc, epoch)

    writer.add_scalar("Test/Total_Loss", test_total_loss, epoch)
    writer.add_scalar("Test/Recon_Loss", test_recon_loss, epoch)
    writer.add_scalar("Test/KL_Age", test_kl_age, epoch)
    writer.add_scalar("Test/KL_ID", test_kl_id, epoch)
    writer.add_scalar("Test/CLS_SPK", test_cls_spk, epoch)
    writer.add_scalar("Test/CLS_AGE", test_cls_age, epoch)
    writer.add_scalar("Test/CosineDisentangle", test_cosine_disentangle, epoch)
    writer.add_scalar("Test/Acc_SPK", test_spk_acc, epoch)
    writer.add_scalar("Test/Acc_Age", test_age_acc, epoch)
    writer.add_scalar("LR", current_lr, epoch)

    # ------------------
    # csv log
    # ------------------
    csv_writer.writerow([
        epoch + 1,
        current_lr,
        train_total_loss,
        train_recon_loss,
        train_kl_age,
        train_kl_id,
        train_cls_spk,
        train_cls_age,
        train_cosine_disentangle,
        train_spk_acc,
        train_age_acc,
        test_total_loss,
        test_recon_loss,
        test_kl_age,
        test_kl_id,
        test_cls_spk,
        test_cls_age,
        test_cosine_disentangle,
        test_spk_acc,
        test_age_acc,
    ])
    csv_file.flush()

    scheduler.step()


# ==========================================
# 6) 收尾
# ==========================================
writer.close()
csv_file.close()
print("Training finished.")
