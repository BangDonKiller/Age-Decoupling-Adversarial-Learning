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
from model.feature_extractor.ecapa_tdnn import SpeakerEmbeddingExtractor
from model.disentangled_model.age_codebook_vae import AgeCodebookVAE, AgeCodebookVAELossArcFace
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

# Age Codebook VAE 超參數
SPEAKER_EMB_DIM = 192
ACOUSTIC_DIM = 8
LATENT_AGE_DIM = 24
LATENT_ID_DIM = 192
NUM_AGE_TOKENS = 32
TOKEN_TEMPERATURE = 1.0
USE_GUMBEL_SOFTMAX = False
GRL_LAMBDA = 1.0

# loss 權重
LAMBDA_RECON = 0.15
LAMBDA_KL_ID = 0.002
LAMBDA_CLS_SPK = 1.2
LAMBDA_CLS_AGE = 0.7
LAMBDA_TOKEN_ALIGN = 1.5
LAMBDA_CODEBOOK_ALIGN = 1.0
LAMBDA_TOKEN_ENTROPY = 0.02
LAMBDA_SPK_ADV = 0.1
LAMBDA_COSINE_DISENTANGLE = 1000.0

# Warmup：逐步打開對齊與對抗，避免訓練初期不穩
ALIGN_WARMUP_EPOCHS = 15
ALIGN_WARMUP_START_SCALE = 0.1

EARLY_STOPPING_PATIENCE = 10

# ArcFace 超參數
ARCFACE_S = 30.0
ARCFACE_M = 0.35
ARCFACE_EASY_MARGIN = False
ARCFACE_LABEL_SMOOTHING = 0.0


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
    評估項目：
    1) before disentangle：原始 speaker embedding 的 cosine 分數
    2) after disentangle：z_id (mu_id) 的 cosine 分數
    3) 最後以 EER 量化辨識性能
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
            wav1, _ = torchaudio.load(path1)
            wav2, _ = torchaudio.load(path2)

            spk_emb1 = speaker_extractor(wav1)
            spk_emb2 = speaker_extractor(wav2)

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

print(f"Train size: {len(full_dataset)} | Eval pairs: {len(eval_dataset)}")
print(f"Building models on {DEVICE}...")

speaker_extractor = SpeakerEmbeddingExtractor(model_id=MODEL_ID, device=str(DEVICE))
speaker_extractor.eval()
speaker_extractor.requires_grad_(False)

model = AgeCodebookVAE(
    speaker_emb_dim=SPEAKER_EMB_DIM,
    acoustic_dim=ACOUSTIC_DIM,
    latent_age_dim=LATENT_AGE_DIM,
    latent_id_dim=LATENT_ID_DIM,
    num_speakers=len(full_dataset.speaker2idx),
    num_age_groups=full_dataset.num_age_classes,
    num_age_tokens=NUM_AGE_TOKENS,
    token_temperature=TOKEN_TEMPERATURE,
    use_gumbel_softmax=USE_GUMBEL_SOFTMAX,
    grl_lambda=GRL_LAMBDA,
).to(DEVICE)

criterion = AgeCodebookVAELossArcFace(
    latent_id_dim=LATENT_ID_DIM,
    num_speakers=len(full_dataset.speaker2idx),
    lambda_recon=LAMBDA_RECON,
    lambda_kl_id=LAMBDA_KL_ID,
    lambda_cls_spk=LAMBDA_CLS_SPK,
    lambda_cls_age=LAMBDA_CLS_AGE,
    lambda_token_align=LAMBDA_TOKEN_ALIGN,
    lambda_codebook_align=LAMBDA_CODEBOOK_ALIGN,
    lambda_token_entropy=LAMBDA_TOKEN_ENTROPY,
    lambda_spk_adv=LAMBDA_SPK_ADV,
    lambda_cosine_disentangle=LAMBDA_COSINE_DISENTANGLE,
    arcface_s=ARCFACE_S,
    arcface_m=ARCFACE_M,
    arcface_easy_margin=ARCFACE_EASY_MARGIN,
    speaker_label_smoothing=ARCFACE_LABEL_SMOOTHING,
).to(DEVICE)

# optimizer 要同時更新 model 與 criterion(ArcFace head) 的參數
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(criterion.parameters()),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

log_dir = "logs/age_codebook_vae"
checkpoint_dir = "checkpoints/age_codebook_vae"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

writer = SummaryWriter(log_dir=log_dir)

csv_path = os.path.join(log_dir, "train_test_metrics.csv")
csv_file = open(csv_path, mode="w", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "epoch",
    "lr",
    "align_warmup_scale",
    "train_total_loss",
    "train_recon_loss",
    "train_kl_id",
    "train_cls_spk_arcface",
    "train_cls_age",
    "train_token_align",
    "train_codebook_align",
    "train_token_sample_entropy",
    "train_token_mean_entropy",
    "train_spk_adv_loss",
    "train_cosine_disentangle",
    "train_spk_acc",
    "train_age_acc",
    "eer_before",
    "eer_after",
    "best_eer_after",
])

best_EER = float("inf")
best_epoch = -1
no_improve_epochs = 0

print("Start training (Age Codebook VAE + ArcFace)...")
for epoch in range(EPOCHS):
    model.train()
    criterion.train()

    # 對齊/對抗 warmup：避免剛開始就強壓 token 與對抗，造成梯度衝突
    if ALIGN_WARMUP_EPOCHS <= 1:
        align_scale = 1.0
    else:
        progress = min(1.0, epoch / float(ALIGN_WARMUP_EPOCHS - 1))
        align_scale = ALIGN_WARMUP_START_SCALE + (1.0 - ALIGN_WARMUP_START_SCALE) * progress

    criterion.lambda_kl_id = LAMBDA_KL_ID * align_scale
    criterion.lambda_token_align = LAMBDA_TOKEN_ALIGN * align_scale
    criterion.lambda_codebook_align = LAMBDA_CODEBOOK_ALIGN * align_scale
    criterion.lambda_spk_adv = LAMBDA_SPK_ADV * align_scale

    train_total_loss = 0.0
    train_recon_loss = 0.0
    train_kl_id = 0.0
    train_cls_spk = 0.0
    train_cls_age = 0.0
    train_token_align = 0.0
    train_codebook_align = 0.0
    train_token_sample_entropy = 0.0
    train_token_mean_entropy = 0.0
    train_spk_adv_loss = 0.0
    train_cosine_disentangle = 0.0
    train_correct_spk = 0
    train_correct_age = 0
    train_total_samples = 0

    for waveform, label_spk, _, label_age, acoustic_feat in tqdm(train_loader, desc=f"Train {epoch + 1}/{EPOCHS}"):
        waveform = waveform.to(DEVICE)
        acoustic_feat = acoustic_feat.to(DEVICE)
        label_spk = label_spk.to(DEVICE)
        label_age = label_age.to(DEVICE).long()

        with torch.no_grad():
            speaker_emb = speaker_extractor(waveform)

        outputs = model(speaker_emb=speaker_emb, acoustic_vec=acoustic_feat)

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
        train_kl_id += float(loss_dict["kl_id"].item())
        train_cls_spk += float(loss_dict["cls_spk"].item())
        train_cls_age += float(loss_dict["cls_age"].item())
        train_token_align += float(loss_dict["token_align"].item())
        train_codebook_align += float(loss_dict["codebook_align"].item())
        train_token_sample_entropy += float(loss_dict["token_sample_entropy"].item())
        train_token_mean_entropy += float(loss_dict["token_mean_entropy"].item())
        train_spk_adv_loss += float(loss_dict["spk_adv_loss"].item())
        train_cosine_disentangle += float(loss_dict["cosine_disentangle"].item())

        # speaker acc 用 ArcFace logits
        pred_spk = loss_dict["arcface_logits"].argmax(dim=1)
        pred_age = outputs["logits_age"].argmax(dim=1)
        train_correct_spk += (pred_spk == label_spk).sum().item()
        train_correct_age += (pred_age == label_age).sum().item()
        train_total_samples += label_spk.size(0)

    train_total_loss /= max(1, len(train_loader))
    train_recon_loss /= max(1, len(train_loader))
    train_kl_id /= max(1, len(train_loader))
    train_cls_spk /= max(1, len(train_loader))
    train_cls_age /= max(1, len(train_loader))
    train_token_align /= max(1, len(train_loader))
    train_codebook_align /= max(1, len(train_loader))
    train_token_sample_entropy /= max(1, len(train_loader))
    train_token_mean_entropy /= max(1, len(train_loader))
    train_spk_adv_loss /= max(1, len(train_loader))
    train_cosine_disentangle /= max(1, len(train_loader))
    train_spk_acc = 100.0 * train_correct_spk / max(1, train_total_samples)
    train_age_acc = 100.0 * train_correct_age / max(1, train_total_samples)

    eer_before, eer_after, before_embs, after_embs, final_ids = eval_network(
        model=model,
        speaker_extractor=speaker_extractor,
        datalist=eval_dataset,
        device=DEVICE,
    )

    current_lr = optimizer.param_groups[0]["lr"]
    print("\n" + "=" * 138)
    print(f"Epoch [{epoch + 1:3d}/{EPOCHS}] | LR: {current_lr:.6f}")
    print(
        f"Align warmup scale: {align_scale:.4f} | "
        f"lambda_kl_id: {criterion.lambda_kl_id:.6f} | "
        f"lambda_token_align: {criterion.lambda_token_align:.4f} | "
        f"lambda_codebook_align: {criterion.lambda_codebook_align:.4f} | "
        f"lambda_spk_adv: {criterion.lambda_spk_adv:.4f}"
    )
    print(
        f"Train Loss => Total: {train_total_loss:.4f}, Recon: {train_recon_loss:.4f}, "
        f"KL_ID: {train_kl_id:.4f}, CLS_SPK(ArcFace): {train_cls_spk:.4f}, CLS_AGE: {train_cls_age:.4f}, "
        f"TokAlign(JS): {train_token_align:.4f}, CodeAlign: {train_codebook_align:.4f}, "
        f"SpkAdv: {train_spk_adv_loss:.4f}, COS: {train_cosine_disentangle:.4f}"
    )
    print(
        f"Token Stats => SampleEntropy: {train_token_sample_entropy:.4f}, "
        f"MeanEntropy: {train_token_mean_entropy:.4f}"
    )
    print(f"Acc   => Train SPK/Age: {train_spk_acc:.2f}%/{train_age_acc:.2f}%")
    print(
        f"Eval  => EER(before disentangle): {eer_before:.4f} | "
        f"EER(after disentangle): {eer_after:.4f} | Best(after): {best_EER:.4f}"
    )
    print("=" * 138 + "\n")

    if best_EER > eer_after:
        best_EER = eer_after
        best_epoch = epoch + 1
        no_improve_epochs = 0
        torch.save(
            {
                "model": model.state_dict(),
                "criterion": criterion.state_dict(),
            },
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
            {
                "model": model.state_dict(),
                "criterion": criterion.state_dict(),
            },
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

    # TensorBoard 紀錄
    writer.add_scalar("Train/Total_Loss", train_total_loss, epoch)
    writer.add_scalar("Train/Recon_Loss", train_recon_loss, epoch)
    writer.add_scalar("Train/KL_ID", train_kl_id, epoch)
    writer.add_scalar("Train/CLS_SPK_ArcFace", train_cls_spk, epoch)
    writer.add_scalar("Train/CLS_AGE", train_cls_age, epoch)
    writer.add_scalar("Train/TokenAlign_JS", train_token_align, epoch)
    writer.add_scalar("Train/CodebookAlign", train_codebook_align, epoch)
    writer.add_scalar("Train/TokenSampleEntropy", train_token_sample_entropy, epoch)
    writer.add_scalar("Train/TokenMeanEntropy", train_token_mean_entropy, epoch)
    writer.add_scalar("Train/SpeakerAdv", train_spk_adv_loss, epoch)
    writer.add_scalar("Train/CosineDisentangle", train_cosine_disentangle, epoch)
    writer.add_scalar("Train/Acc_SPK", train_spk_acc, epoch)
    writer.add_scalar("Train/Acc_Age", train_age_acc, epoch)

    writer.add_scalar("Eval/EER_Before", eer_before, epoch)
    writer.add_scalar("Eval/EER_After", eer_after, epoch)
    writer.add_scalar("Eval/Best_EER_After", best_EER, epoch)
    writer.add_scalar("LR", current_lr, epoch)
    writer.add_scalar("Warmup/AlignScale", align_scale, epoch)

    # CSV 紀錄
    csv_writer.writerow([
        epoch + 1,
        current_lr,
        align_scale,
        train_total_loss,
        train_recon_loss,
        train_kl_id,
        train_cls_spk,
        train_cls_age,
        train_token_align,
        train_codebook_align,
        train_token_sample_entropy,
        train_token_mean_entropy,
        train_spk_adv_loss,
        train_cosine_disentangle,
        train_spk_acc,
        train_age_acc,
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
        torch.save(
            {
                "model": model.state_dict(),
                "criterion": criterion.state_dict(),
            },
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
print("Training finished (Age Codebook VAE + ArcFace).")
