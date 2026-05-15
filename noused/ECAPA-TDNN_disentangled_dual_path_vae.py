import os
import csv
import warnings
from pathlib import Path

import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from tool.EER import compute_eer
from data.vox2_loader import Vox2Dataset
from model.feature_extractor.ecapa_tdnn_ver2 import SpeakerEmbeddingExtractor
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
VAL_DATASET_NAME = "Vox-CA20"
SAMPLE_RATE = 16000

EPOCHS = 40
NUM_WORKERS = 0
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-5
MAX_EVAL_PAIRS = 20000

# Dual-Path VAE 超參數
SPEAKER_EMB_DIM = 192
ACOUSTIC_DIM = 8
LATENT_AGE_DIM = 24
LATENT_ID_DIM = 128

# loss 權重
LAMBDA_RECON = 0.5
LAMBDA_KL_AGE = 0.3
LAMBDA_KL_ID = 0.008
LAMBDA_CLS_SPK = 1.0
LAMBDA_CLS_AGE = 0.7
LAMBDA_ADV_AGE_ID = 0.7
ADV_GRL_LAMBDA = 1.0
KL_WARMUP_EPOCHS = 12
KL_WARMUP_START_SCALE = 0.05
EARLY_STOPPING_PATIENCE = 10


def build_opensmile_extractor():
    try:
        import opensmile
    except Exception as exc:
        raise ImportError(
            "此腳本需要 opensmile 以提取 eval acoustic feature，請先安裝: pip install opensmile"
        ) from exc

    return opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )


def extract_egemaps8_from_path(file_path: str, smile, acoustic_sample_rate: int = 16000) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(file_path)

    if sample_rate != acoustic_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, acoustic_sample_rate)

    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    wav_np = waveform.squeeze(0).detach().cpu().float().numpy().astype(np.float32, copy=False)
    feat_df = smile.process_signal(wav_np, acoustic_sample_rate)

    if feat_df is None or feat_df.empty:
        raise ValueError(f"OpenSMILE 回傳空特徵，檔案: {file_path}")

    target_map = {
        "voicedSegmentsPerSecond": ["voicedSegmentsPerSecond", "VoicedSegmentsPerSec"],
        "meanunVoicedSegmentLength": ["meanunVoicedSegmentLength", "MeanUnvoicedSegmentLength"],
        "F0semitoneFrom27.5Hz_sma3nz_amean": ["F0semitoneFrom27.5Hz_sma3nz_amean"],
        "F0semitoneFrom27.5Hz_sma3nz_stddevNorm": ["F0semitoneFrom27.5Hz_sma3nz_stddevNorm"],
        "jitterLocal_sma3nz_amean": ["jitterLocal_sma3nz_amean"],
        "shimmerLocaldB_sma3nz_amean": ["shimmerLocaldB_sma3nz_amean"],
        "HNRdB1-10kHz_sma3nz_amean": [
            "HNRdB1-10kHz_sma3nz_amean",
            "HNRdBACF_sma3nz_amean",
            "HNRdBACF_sma3_amean",
            "logHNR_sma3nz_amean",
        ],
        "alphaRatioV_sma3nz_amean": [
            "alphaRatioV_sma3nz_amean",
            "alphaRatio_sma3nz_amean",
            "alphaRatioV_sma3_amean",
            "alphaRatio_sma3_amean",
        ],
    }

    columns = list(feat_df.columns)
    lower_to_raw = {c.lower(): c for c in columns}
    resolved = {}

    for key, candidates in target_map.items():
        picked = None
        for cand in candidates:
            if cand.lower() in lower_to_raw:
                picked = lower_to_raw[cand.lower()]
                break

        if picked is None:
            for cand in candidates:
                cand_lower = cand.lower()
                contains = [c for c in columns if cand_lower in c.lower()]
                if contains:
                    picked = contains[0]
                    break

        if picked is None:
            raise KeyError(f"找不到目標欄位 '{key}'，檔案: {file_path}")

        resolved[key] = picked

    feature_order = [
        "voicedSegmentsPerSecond",
        "meanunVoicedSegmentLength",
        "F0semitoneFrom27.5Hz_sma3nz_amean",
        "F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
        "jitterLocal_sma3nz_amean",
        "shimmerLocaldB_sma3nz_amean",
        "HNRdB1-10kHz_sma3nz_amean",
        "alphaRatioV_sma3nz_amean",
    ]

    vec8 = np.array([feat_df.iloc[0][resolved[name]] for name in feature_order], dtype=np.float32)
    return torch.tensor(vec8, dtype=torch.float32)


def _normalize_audio_dirs(audio_dirs):
    if isinstance(audio_dirs, (str, Path)):
        return [str(audio_dirs)]
    return [str(d) for d in audio_dirs]

# ==========================================
# 2) 準備資料集（train + eval pair）
# ==========================================
print("Preparing datasets...")


def build_eval_dataset(audio_dirs, audio_meta_dir, smile, max_pairs=20000):
    audio_dirs = _normalize_audio_dirs(audio_dirs)

    def find_audio_path(relative_path):
        for audio_dir in audio_dirs:
            audio_path = Path(audio_dir) / relative_path
            if audio_path.exists():
                return str(audio_path)
        raise FileNotFoundError(f"{relative_path} not found in audio_dirs")

    datalist = []
    acoustic_cache = {}

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

        if spk1_path not in acoustic_cache:
            acoustic_cache[spk1_path] = extract_egemaps8_from_path(spk1_path, smile, acoustic_sample_rate=SAMPLE_RATE)
        if spk2_path not in acoustic_cache:
            acoustic_cache[spk2_path] = extract_egemaps8_from_path(spk2_path, smile, acoustic_sample_rate=SAMPLE_RATE)

        spk1_id = spk1_rel.split("/")[0]
        spk2_id = spk2_rel.split("/")[0]

        datalist.append(
            (
                is_same,
                spk1_id,
                spk2_id,
                spk1_path,
                spk2_path,
                acoustic_cache[spk1_path],
                acoustic_cache[spk2_path],
            )
        )

    pos = sum(1 for x in datalist if x[0] == 1)
    neg = sum(1 for x in datalist if x[0] == 0)
    print(f"[Eval] Positive pairs: {pos}, Negative pairs: {neg}, Unique audios: {len(acoustic_cache)}")

    return datalist


def eval_network(model, speaker_extractor, datalist, device):
    """
    Speaker verification evaluation on pairwise trials.

    Returns:
        eer_before (float)
        eer_after  (float)
        final_before_embs (Tensor)
        final_after_embs  (Tensor)
        final_ids (List[str])
    """

    model.eval()
    speaker_extractor.eval()

    before_scores = []
    after_scores = []
    labels = []

    before_embs = []
    after_embs = []
    final_ids = []

    with torch.no_grad():
        for is_same, id1, id2, path1, path2, feat1, feat2 in tqdm(datalist, desc="Evaluating"):
            emb1, sr1 = torchaudio.load(path1)
            emb2, sr2 = torchaudio.load(path2)
            
            spk_emb1 = speaker_extractor(emb1)
            spk_emb2 = speaker_extractor(emb2)

            ac1 = feat1.unsqueeze(0).to(device=device, dtype=spk_emb1.dtype)
            ac2 = feat2.unsqueeze(0).to(device=device, dtype=spk_emb2.dtype)

            out1 = model(speaker_emb=spk_emb1, acoustic_vec=ac1)
            out2 = model(speaker_emb=spk_emb2, acoustic_vec=ac2)

            h1 = F.normalize(spk_emb1, p=2, dim=1)
            h2 = F.normalize(spk_emb2, p=2, dim=1)
            score_before = F.cosine_similarity(h1, h2).cpu()
            before_scores.append(score_before)

            w1 = F.normalize(out1["mu_id"], p=2, dim=1)
            w2 = F.normalize(out2["mu_id"], p=2, dim=1)
            score_after = F.cosine_similarity(w1, w2).cpu()
            after_scores.append(score_after)

            before_embs.append(spk_emb1.cpu())
            before_embs.append(spk_emb2.cpu())
            after_embs.append(out1["mu_id"].cpu())
            after_embs.append(out2["mu_id"].cpu())

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

train_loader = DataLoader(
    full_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

smile_extractor = build_opensmile_extractor()

eval_dataset = build_eval_dataset(
    audio_dirs=DATASET_INFO["VoxCeleb1"][VAL_DATASET_NAME]["AUDIO_DIR"],
    audio_meta_dir=DATASET_INFO["VoxCeleb1"][VAL_DATASET_NAME]["AUDIO_DATALIST"],
    smile=smile_extractor,
    max_pairs=MAX_EVAL_PAIRS,
)

print(
    f"Train size: {len(full_dataset)} | "
    f"Eval pairs: {len(eval_dataset)}"
)


# ==========================================
# 3) 建立模型與優化器
# ==========================================
print(f"Building models on {DEVICE}...")

speaker_extractor = SpeakerEmbeddingExtractor(
    model_id=MODEL_ID,
    device=str(DEVICE),
)
speaker_extractor.eval()
speaker_extractor.requires_grad_(False)

model = DualPathVAE(
    speaker_emb_dim=SPEAKER_EMB_DIM,
    acoustic_dim=ACOUSTIC_DIM,
    latent_age_dim=LATENT_AGE_DIM,
    latent_id_dim=LATENT_ID_DIM,
    num_speakers=len(full_dataset.speaker2idx),
    num_age_groups=full_dataset.num_age_classes,
    adv_grl_lambda=ADV_GRL_LAMBDA,
).to(DEVICE)

criterion = DualPathVAELoss(
    lambda_recon=LAMBDA_RECON,
    lambda_kl_age=LAMBDA_KL_AGE,
    lambda_kl_id=LAMBDA_KL_ID,
    lambda_cls_spk=LAMBDA_CLS_SPK,
    lambda_cls_age=LAMBDA_CLS_AGE,
    lambda_adv_age_id=LAMBDA_ADV_AGE_ID,
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
    "train_adv_age_id",
    "train_spk_acc",
    "train_age_acc",
    "train_adv_age_acc",
    "eer_before",
    "eer_after",
    "best_eer_after",
])

best_EER = float("inf")
best_epoch = -1
no_improve_epochs = 0


# ==========================================
# 5) 訓練與測試主迴圈
# ==========================================
print("Start training...")

for epoch in range(EPOCHS):
    model.train()

    # KL warmup: 前 KL_WARMUP_EPOCHS 由 KL_WARMUP_START_SCALE 線性增加到 1.0
    if KL_WARMUP_EPOCHS <= 1:
        kl_warmup_scale = 1.0
    else:
        progress = min(1.0, epoch / float(KL_WARMUP_EPOCHS - 1))
        kl_warmup_scale = KL_WARMUP_START_SCALE + (1.0 - KL_WARMUP_START_SCALE) * progress

    criterion.lambda_kl_age = LAMBDA_KL_AGE * kl_warmup_scale
    criterion.lambda_kl_id = LAMBDA_KL_ID * kl_warmup_scale

    train_total_loss = 0.0
    train_recon_loss = 0.0
    train_kl_age = 0.0
    train_kl_id = 0.0
    train_cls_spk = 0.0
    train_cls_age = 0.0
    train_adv_age_id = 0.0
    train_correct_spk = 0
    train_correct_age = 0
    train_correct_adv_age = 0
    train_total_samples = 0

    for waveform, label_spk, _, label_age, acoustic_feat in tqdm(train_loader, desc=f"Train {epoch + 1}/{EPOCHS}"):
        waveform = waveform.to(DEVICE)
        acoustic_feat = acoustic_feat.to(DEVICE)
        label_spk = label_spk.to(DEVICE)
        label_age = label_age.to(DEVICE).long()

        with torch.no_grad():
            speaker_emb = speaker_extractor(waveform)

        outputs = model(
            speaker_emb=speaker_emb,
            acoustic_vec=acoustic_feat,
        )

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
        train_adv_age_id += float(loss_dict["adv_age_id"].item())

        pred_spk = outputs["logits_spk"].argmax(dim=1)
        pred_age = outputs["logits_age"].argmax(dim=1)
        pred_adv_age = outputs["logits_age_adv"].argmax(dim=1)
        train_correct_spk += (pred_spk == label_spk).sum().item()
        train_correct_age += (pred_age == label_age).sum().item()
        train_correct_adv_age += (pred_adv_age == label_age).sum().item()
        train_total_samples += label_spk.size(0)

    train_total_loss /= max(1, len(train_loader))
    train_recon_loss /= max(1, len(train_loader))
    train_kl_age /= max(1, len(train_loader))
    train_kl_id /= max(1, len(train_loader))
    train_cls_spk /= max(1, len(train_loader))
    train_cls_age /= max(1, len(train_loader))
    train_adv_age_id /= max(1, len(train_loader))
    train_spk_acc = 100.0 * train_correct_spk / max(1, train_total_samples)
    train_age_acc = 100.0 * train_correct_age / max(1, train_total_samples)
    train_adv_age_acc = 100.0 * train_correct_adv_age / max(1, train_total_samples)

    # ------------------
    # EER 評估（含 eval acoustic feature）
    # ------------------
    eer_before, eer_after, before_embs, after_embs, final_ids = eval_network(
        model=model,
        speaker_extractor=speaker_extractor,
        datalist=eval_dataset,
        device=DEVICE,
    )

    # ------------------
    # log print（epoch 結束後完整資訊）
    # ------------------
    current_lr = optimizer.param_groups[0]["lr"]
    print("\n" + "=" * 130)
    print(f"Epoch [{epoch + 1:3d}/{EPOCHS}] | LR: {current_lr:.6f}")
    print(
        f"KL warmup scale: {kl_warmup_scale:.4f} | "
        f"lambda_kl_age: {criterion.lambda_kl_age:.4f} | "
        f"lambda_kl_id: {criterion.lambda_kl_id:.4f}"
    )
    print(
        f"Train Loss => Total: {train_total_loss:.4f}, Recon: {train_recon_loss:.4f}, "
        f"KL_Age: {train_kl_age:.4f}, KL_ID: {train_kl_id:.4f}, "
        f"CLS_SPK: {train_cls_spk:.4f}, CLS_AGE: {train_cls_age:.4f}, ADV_AGE(z_id): {train_adv_age_id:.4f}"
    )
    print(
        f"Acc   => Train SPK/Age: {train_spk_acc:.2f}%/{train_age_acc:.2f}% | "
        f"ADV_AGE: {train_adv_age_acc:.2f}%"
    )
    print(
        f"Eval  => EER(before disentangle): {eer_before:.4f} | "
        f"EER(after disentangle): {eer_after:.4f} | Best(after): {best_EER:.4f}"
    )
    print("=" * 130 + "\n")

    # ------------------
    # checkpoint: best EER + last epoch
    # ------------------
    if best_EER > eer_after:
        best_EER = eer_after
        best_epoch = epoch + 1
        no_improve_epochs = 0
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))
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
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "last_model.pth"))
        torch.save(
            {
                "embeddings": after_embs,
                "before_embeddings": before_embs,
                "ids": final_ids,
            },
            os.path.join(checkpoint_dir, "last_disentangled_embeddings.pt"),
        )
        print("✓ 儲存最終模型。")
        
    # ------------------
    # tensorboard
    # ------------------
    writer.add_scalar("Train/Total_Loss", train_total_loss, epoch)
    writer.add_scalar("Train/Recon_Loss", train_recon_loss, epoch)
    writer.add_scalar("Train/KL_Age", train_kl_age, epoch)
    writer.add_scalar("Train/KL_ID", train_kl_id, epoch)
    writer.add_scalar("Train/CLS_SPK", train_cls_spk, epoch)
    writer.add_scalar("Train/CLS_AGE", train_cls_age, epoch)
    writer.add_scalar("Train/ADV_AGE_ID", train_adv_age_id, epoch)
    writer.add_scalar("Train/Acc_SPK", train_spk_acc, epoch)
    writer.add_scalar("Train/Acc_Age", train_age_acc, epoch)
    writer.add_scalar("Train/Acc_ADV_Age", train_adv_age_acc, epoch)

    writer.add_scalar("Eval/EER_Before", eer_before, epoch)
    writer.add_scalar("Eval/EER_After", eer_after, epoch)
    writer.add_scalar("Eval/Best_EER_After", best_EER, epoch)
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
        train_adv_age_id,
        train_spk_acc,
        train_age_acc,
        train_adv_age_acc,
        eer_before,
        eer_after,
        best_EER,
    ])
    csv_file.flush()

    scheduler.step()

    if no_improve_epochs >= EARLY_STOPPING_PATIENCE:
        print(
            f"Early stopping 觸發：連續 {EARLY_STOPPING_PATIENCE} 個 epoch 無改善。"
            f"最佳 EER(after)={best_EER:.4f}（Epoch {best_epoch}）。"
        )
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "last_model.pth"))
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


# ==========================================
# 6) 收尾
# ==========================================
writer.close()
csv_file.close()
print("Training finished.")
