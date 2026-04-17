import csv
import os
import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from data.vox2_loader import Vox2Dataset
from data.vox1_loader import PairwiseDataset
from model.disentangled_model.linear_decorr_mlp import (
    LinearDecorrLoss,
    LinearDecorrMLP,
)
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
if not torch.cuda.is_available():
    raise RuntimeError("未偵測到 CUDA 裝置，無法使用 GPU 訓練。")

DEVICE = torch.device("cuda")
torch.backends.cudnn.benchmark = True
TRAIN_DATASET_NAME = "VoxCeleb2"
VAL_DATASET_NAME = "Vox-CA20"

EPOCHS = 20
NUM_WORKERS = 0
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-5
MAX_EVAL_PAIRS = 20000

SPEAKER_EMB_DIM = 192
MLP_HIDDEN_DIMS = [256]
AGE_DIM = 1
MLP_ID_DIM = 192
MLP_OUTPUT_DIM = AGE_DIM + MLP_ID_DIM  # 前 AGE_DIM 維做年齡分類，其餘做說話者分類與驗證
MLP_DROPOUT = 0.1

LAMBDA_SPK = 1.0
LAMBDA_AGE = 1.0
LAMBDA_DECORR = 0.0
LAMBDA_DECORR_WARMUP_EPOCHS = 0
LAMBDA_DECORR_START = 0.0

ARCFACE_S = 64
ARCFACE_M = 0.2
ARCFACE_EASY_MARGIN = False

EARLY_STOPPING_ENABLED = False
EARLY_STOPPING_PATIENCE = 10
TOPK_CORR = 5
AGE_GROUP_BOUNDS = [21, 31, 41, 51, 61, 71, 81]
PLOT_EVERY_N_EPOCHS = 5
SPLIT_SEED = 42
TRAIN_UTTS_PER_SPK = 8
VAL_UTTS_PER_SPK = 2


def get_next_exp_dir(base_dir: str, prefix: str = "exp"):
    """在 base_dir 下以 expXXX 形式建立下一個實驗資料夾，並回傳 (path, name)。"""
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    max_idx = 0
    for child in base_path.iterdir():
        if not child.is_dir():
            continue
        match = pattern.match(child.name)
        if match:
            max_idx = max(max_idx, int(match.group(1)))

    next_idx = max_idx + 1
    exp_name = f"{prefix}{next_idx}"
    exp_path = base_path / exp_name
    exp_path.mkdir(parents=True, exist_ok=False)
    return str(exp_path), exp_name


def get_warmup_lambda_decorr(epoch: int) -> float:
    """線性 warmup：在前幾個 epoch 將 lambda_decorr 從起始值提升到目標值。"""
    if LAMBDA_DECORR_WARMUP_EPOCHS <= 0:
        return float(LAMBDA_DECORR)

    if epoch >= LAMBDA_DECORR_WARMUP_EPOCHS:
        return float(LAMBDA_DECORR)

    progress = float(epoch + 1) / float(LAMBDA_DECORR_WARMUP_EPOCHS)
    return float(LAMBDA_DECORR_START + progress * (LAMBDA_DECORR - LAMBDA_DECORR_START))


def print_age_neuron_correlation_stats(age_corr, tag, age_dim: int = AGE_DIM):
    """打印年齡神經元與其他神經元的相關係數統計"""
    if age_corr is None:
        return
    
    age_corr_with_others = np.abs(age_corr[:, age_dim:])
    if age_corr_with_others.size == 0:
        return
    
    print(f"\n[{tag}] |Age Neuron Correlations| with Other Neurons (age_dim={age_dim})")
    print(f"  Min |correlation|:   {age_corr_with_others.min():.6f}")
    print(f"  Max |correlation|:   {age_corr_with_others.max():.6f}")
    print(f"  Mean |correlation|:  {age_corr_with_others.mean():.6f}")
    print(f"  Std |correlation|:   {age_corr_with_others.std():.6f}")
    flat_values = np.sort(age_corr_with_others.reshape(-1))[::-1]
    print(f"  Top-{TOPK_CORR} |correlations|:  {flat_values[:TOPK_CORR].tolist()}")


def compute_age_auc(age_logits: torch.Tensor, age_targets: torch.Tensor, num_age_groups: int) -> float:
    """根據 age logits 與標籤計算 AUC；多分類時使用 macro OVR AUC。"""
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


class RunningAgeCorrelation:
    """用串流統計計算前 age_dim 個 age neuron 與其他 neuron 的 Pearson 相關係數。"""

    def __init__(self, latent_dim: int, age_dim: int = AGE_DIM):
        if age_dim < 1:
            raise ValueError("age_dim 必須 >= 1")
        if latent_dim <= age_dim:
            raise ValueError("latent_dim 必須 > age_dim")

        self.age_dim = age_dim
        other_dim = latent_dim - age_dim
        self.n = 0
        self.sum_x = np.zeros(age_dim, dtype=np.float64)
        self.sum_x2 = np.zeros(age_dim, dtype=np.float64)
        self.sum_y = np.zeros(other_dim, dtype=np.float64)
        self.sum_y2 = np.zeros(other_dim, dtype=np.float64)
        self.sum_xy = np.zeros((age_dim, other_dim), dtype=np.float64)

    def update(self, z: torch.Tensor):
        if z is None or z.numel() == 0:
            return

        z_np = z.detach().cpu().numpy().astype(np.float64, copy=False)
        x = z_np[:, : self.age_dim]
        y = z_np[:, self.age_dim :]

        self.n += z_np.shape[0]
        self.sum_x += x.sum(axis=0)
        self.sum_x2 += (x * x).sum(axis=0)
        self.sum_y += y.sum(axis=0)
        self.sum_y2 += (y * y).sum(axis=0)
        self.sum_xy += x.T @ y

    def correlations(self, eps: float = 1e-12):
        if self.n < 2:
            return None

        n = float(self.n)
        ex = self.sum_x / n
        ey = self.sum_y / n
        ex2 = self.sum_x2 / n
        ey2 = self.sum_y2 / n
        exy = self.sum_xy / n

        cov = exy - ex[:, None] * ey[None, :]
        var_x = np.maximum(ex2 - ex * ex, eps)
        var_y = np.maximum(ey2 - ey * ey, eps)
        corr_other = cov / np.sqrt(var_x[:, None] * var_y[None, :])
        corr_other = np.clip(corr_other, -1.0, 1.0)

        # 前 age_dim 維是 age neurons 與自己，對角線定義為 1.0
        return np.concatenate([np.eye(self.age_dim, dtype=np.float64), corr_other], axis=1)


def plot_age_corr_heatmap(age_corr, save_path, title, age_dim: int = AGE_DIM):
    """繪製年齡神經元與其他神經元相關係數熱力圖（age_dim x N）。"""
    if age_corr is None:
        return

    values = np.asarray(age_corr[:, age_dim:], dtype=np.float32)
    if values.size == 0:
        return

    heat = values
    plt.figure(figsize=(12, max(2.5, 0.8 * heat.shape[0] + 1.5)))
    im = plt.imshow(heat, aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
    plt.yticks(range(age_dim), [f"age neuron {i}" for i in range(age_dim)])
    plt.xlabel("Other neuron index")
    plt.title(title)
    cbar = plt.colorbar(im)
    cbar.set_label("Correlation")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def eval_network(model, speaker_extractor, eval_dataloader, num_age_groups):
    model.eval()
    speaker_extractor.eval()

    before_scores = []
    after_scores = []
    labels = []

    before_embs = []
    after_embs = []
    final_ids = []
    age_logits_all = []
    age_targets_all = []
    corr_meter = RunningAgeCorrelation(latent_dim=MLP_OUTPUT_DIM, age_dim=AGE_DIM)

    with torch.no_grad():
        for is_same, id1, id2, wav1, wav2, age1, age2 in tqdm(eval_dataloader, desc="Evaluating"):
            wav1 = wav1.to(DEVICE, non_blocking=True)
            wav2 = wav2.to(DEVICE, non_blocking=True)

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
            age_logits_all.append(out1["logits_age"].cpu())
            age_logits_all.append(out2["logits_age"].cpu())
            age_targets_all.extend(age1.cpu().tolist())
            age_targets_all.extend(age2.cpu().tolist())
            corr_meter.update(out1["z"])
            corr_meter.update(out2["z"])

            labels.extend(is_same.cpu().tolist())
            final_ids.extend(list(id1))
            final_ids.extend(list(id2))

    final_labels = np.array(labels)
    final_before_scores = torch.cat(before_scores).numpy() if before_scores else np.array([])
    final_after_scores = torch.cat(after_scores).numpy() if after_scores else np.array([])

    eer_before = compute_eer(final_before_scores, final_labels) if len(final_before_scores) > 0 else float('nan')
    eer_after = compute_eer(final_after_scores, final_labels) if len(final_after_scores) > 0 else float('nan')

    final_before_embs = torch.cat(before_embs, dim=0) if before_embs else torch.tensor([])
    final_after_embs = torch.cat(after_embs, dim=0) if after_embs else torch.tensor([])
    eval_age_corr = corr_meter.correlations()
    eval_age_auc = compute_age_auc(
        torch.cat(age_logits_all, dim=0) if age_logits_all else torch.tensor([]),
        torch.tensor(age_targets_all, dtype=torch.long),
        num_age_groups=num_age_groups,
    ) if age_logits_all and age_targets_all else float('nan')

    return eer_before, eer_after, final_before_embs, final_after_embs, final_ids, eval_age_corr, eval_age_auc


def build_train_val_indices_by_speaker(
    dataset,
    train_utts_per_spk: int = 8,
    val_utts_per_spk: int = 2,
    seed: int = 42,
):
    """以 speaker 為單位做固定數量切分。"""
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


def eval_classification_network(model, speaker_extractor, val_dataloader, criterion):
    """在同資料集切出的驗證集上，評估多分類與年齡分類表現。"""
    model.eval()
    speaker_extractor.eval()

    val_total_loss = 0.0
    val_spk_loss = 0.0
    val_age_loss = 0.0
    val_decorr_loss = 0.0
    val_correct_spk = 0
    val_correct_age = 0
    val_total_samples = 0
    val_age_logits_all = []
    val_age_targets_all = []
    corr_meter = RunningAgeCorrelation(latent_dim=MLP_OUTPUT_DIM, age_dim=AGE_DIM)

    with torch.no_grad():
        for waveform, label_spk, _, label_age in tqdm(val_dataloader, desc="Validating"):
            waveform = waveform.to(DEVICE, non_blocking=True)
            label_spk = label_spk.to(DEVICE, non_blocking=True)
            label_age = label_age.to(DEVICE, non_blocking=True).long()

            speaker_emb = speaker_extractor(waveform)
            outputs = model(speaker_emb)
            loss, loss_dict = criterion(outputs=outputs, target_spk=label_spk, target_age=label_age)

            val_total_loss += float(loss.item())
            val_spk_loss += float(loss_dict["loss_spk"].item())
            val_age_loss += float(loss_dict["loss_age"].item())
            val_decorr_loss += float(loss_dict["loss_decorr"].item())

            pred_spk = loss_dict["arcface_logits"].argmax(dim=1)
            pred_age = outputs["logits_age"].argmax(dim=1)
            val_correct_spk += (pred_spk == label_spk).sum().item()
            val_correct_age += (pred_age == label_age).sum().item()
            val_total_samples += label_spk.size(0)
            val_age_logits_all.append(outputs["logits_age"].detach().cpu())
            val_age_targets_all.append(label_age.detach().cpu())
            corr_meter.update(outputs["z"])

    num_batches = max(1, len(val_dataloader))
    val_total_loss /= num_batches
    val_spk_loss /= num_batches
    val_age_loss /= num_batches
    val_decorr_loss /= num_batches
    val_spk_acc = 100.0 * val_correct_spk / max(1, val_total_samples)
    val_age_acc = 100.0 * val_correct_age / max(1, val_total_samples)
    val_age_corr = corr_meter.correlations()
    val_age_auc = compute_age_auc(
        torch.cat(val_age_logits_all, dim=0) if val_age_logits_all else torch.tensor([]),
        torch.cat(val_age_targets_all, dim=0) if val_age_targets_all else torch.tensor([], dtype=torch.long),
        num_age_groups=model.age_head.out_features,
    )
    val_age_corr_values = val_age_corr[:, AGE_DIM:] if val_age_corr is not None else None
    val_age_corr_abs_mean = (
        float(np.mean(np.abs(val_age_corr_values))) if val_age_corr_values is not None else float("nan")
    )

    return {
        "val_total_loss": val_total_loss,
        "val_spk_loss": val_spk_loss,
        "val_age_loss": val_age_loss,
        "val_decorr_loss": val_decorr_loss,
        "val_spk_acc": val_spk_acc,
        "val_age_acc": val_age_acc,
        "val_age_auc": val_age_auc,
        "val_age_corr": val_age_corr,
        "val_age_corr_abs_mean": val_age_corr_abs_mean,
    }


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

train_indices, val_indices, used_speakers = build_train_val_indices_by_speaker(
    full_dataset,
    train_utts_per_spk=TRAIN_UTTS_PER_SPK,
    val_utts_per_spk=VAL_UTTS_PER_SPK,
    seed=SPLIT_SEED,
)

if len(used_speakers) == 0:
    raise RuntimeError(
        "沒有任何 speaker 能滿足 8 train + 2 val 的切分條件，"
        "請增加每位 speaker 可用語句或調整切分比例。"
    )

train_dataset = Subset(full_dataset, train_indices)
val_dataset = Subset(full_dataset, val_indices)

train_speaker_ids = sorted({full_dataset.datalist[idx][1] for idx in train_indices})
val_speaker_ids = sorted({full_dataset.datalist[idx][1] for idx in val_indices})

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

# 使用 PairwiseDataset 讀取測試資料集（原先的驗證流程保留不變）
test_dataset = PairwiseDataset(
    audio_dir=DATASET_INFO["VoxCeleb1"][VAL_DATASET_NAME]["AUDIO_DIR"],
    audio_meta_dir=DATASET_INFO["VoxCeleb1"][VAL_DATASET_NAME]["AUDIO_DATALIST"],
    audio_meta_csv_path=DATASET_INFO["VoxCeleb1"]["AUDIO_META_DIR"],
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

print(
    f"Split summary | Eligible speakers: {len(used_speakers)}"
)
print(
    f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)} | "
    f"Test pairs: {len(test_dataset)}"
)
print(
    f"Train speaker classes: {len(train_speaker_ids)} | "
    f"Val speaker classes: {len(val_speaker_ids)}"
)
print(
    f"Train/Val speaker class overlap: {len(set(train_speaker_ids) & set(val_speaker_ids))}"
)
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
    age_dim=AGE_DIM,
).to(DEVICE)

criterion = LinearDecorrLoss(
    latent_id_dim=MLP_OUTPUT_DIM - AGE_DIM,
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

log_root_dir = "logs/linear_decorr_mlp"
checkpoint_root_dir = "checkpoints/linear_decorr_mlp"
log_dir, exp_name = get_next_exp_dir(log_root_dir, prefix="exp")
checkpoint_dir = os.path.join(checkpoint_root_dir, exp_name)
os.makedirs(checkpoint_dir, exist_ok=True)

print(f"Experiment directory: {log_dir}")
print(f"Checkpoint directory: {checkpoint_dir}")

age_corr_plot_dir = os.path.join(log_dir, "age_correlation_plots")
train_plot_dir = os.path.join(age_corr_plot_dir, "train")
eval_plot_dir = os.path.join(age_corr_plot_dir, "eval")
os.makedirs(train_plot_dir, exist_ok=True)
os.makedirs(eval_plot_dir, exist_ok=True)

writer = SummaryWriter(log_dir=log_dir)

csv_path = os.path.join(log_dir, "train_test_metrics.csv")
csv_file = open(csv_path, mode="w", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(
    [
        "epoch",
        "lr",
        "lambda_decorr_current",
        "train_total_loss",
        "train_spk_loss",
        "train_age_loss",
        "train_decorr_loss",
        "train_spk_acc",
        "train_age_acc",
        "train_age_corr_abs_mean",
        "val_total_loss",
        "val_spk_loss",
        "val_age_loss",
        "val_decorr_loss",
        "val_spk_acc",
        "val_age_acc",
        "val_age_auc",
        "val_age_corr_abs_mean",
        "test_eer_before",
        "test_eer_after",
        "test_age_auc",
        "test_age_corr_abs_mean",
        "best_test_eer_after",
    ]
)

best_test_EER = float("inf")
best_epoch = -1
no_improve_epochs = 0

print("Start training (Linear-Decorr MLP)...")
for epoch in range(EPOCHS):
    model.train()
    current_lambda_decorr = get_warmup_lambda_decorr(epoch)
    criterion.lambda_decorr = current_lambda_decorr

    train_total_loss = 0.0
    train_spk_loss = 0.0
    train_age_loss = 0.0
    train_decorr_loss = 0.0
    train_correct_spk = 0
    train_correct_age = 0
    train_total_samples = 0
    train_corr_meter = RunningAgeCorrelation(latent_dim=MLP_OUTPUT_DIM, age_dim=AGE_DIM)

    for waveform, label_spk, _, label_age in tqdm(train_loader, desc=f"Train {epoch + 1}/{EPOCHS}"):
        waveform = waveform.to(DEVICE, non_blocking=True)
        label_spk = label_spk.to(DEVICE, non_blocking=True)
        label_age = label_age.to(DEVICE, non_blocking=True).long()

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
        
        # 用串流方式記錄年齡神經元相關係數，避免每個 batch 重複做完整相關矩陣
        train_corr_meter.update(outputs["z"])

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
    
    avg_age_corr = train_corr_meter.correlations()

    val_metrics = eval_classification_network(
        model=model,
        speaker_extractor=speaker_extractor,
        val_dataloader=val_loader,
        criterion=criterion,
    )

    test_eer_before, test_eer_after, before_embs, after_embs, final_ids, test_age_corr, test_age_auc = eval_network(
        model=model,
        speaker_extractor=speaker_extractor,
        eval_dataloader=test_dataloader,
        num_age_groups=full_dataset.num_age_classes,
    )

    train_age_corr_values = avg_age_corr[:, AGE_DIM:] if avg_age_corr is not None else None
    test_age_corr_values = test_age_corr[:, AGE_DIM:] if test_age_corr is not None else None

    train_age_corr_abs_mean = (
        float(np.mean(np.abs(train_age_corr_values))) if train_age_corr_values is not None else float("nan")
    )
    test_age_corr_abs_mean = (
        float(np.mean(np.abs(test_age_corr_values))) if test_age_corr_values is not None else float("nan")
    )

    if (epoch + 1) % PLOT_EVERY_N_EPOCHS == 0:
        plot_age_corr_heatmap(
            avg_age_corr,
            save_path=os.path.join(train_plot_dir, f"epoch_{epoch + 1:03d}_heatmap.png"),
            title=f"Train Age-Neuron Correlation Heatmap (Epoch {epoch + 1})",
            age_dim=AGE_DIM,
        )
        plot_age_corr_heatmap(
            test_age_corr,
            save_path=os.path.join(eval_plot_dir, f"epoch_{epoch + 1:03d}_heatmap.png"),
            title=f"Test Age-Neuron Correlation Heatmap (Epoch {epoch + 1})",
            age_dim=AGE_DIM,
        )

    current_lr = optimizer.param_groups[0]["lr"]
    print("\n" + "=" * 130)
    print(
        f"Epoch [{epoch + 1:3d}/{EPOCHS}] | LR: {current_lr:.6f} | "
        f"lambda_decorr: {current_lambda_decorr:.6f}"
    )
    print(
        f"Train Loss => Total: {train_total_loss:.4f}, SPK: {train_spk_loss:.4f}, "
        f"AGE: {train_age_loss:.4f}, DECORR: {train_decorr_loss:.4f}"
    )
    print(f"Acc   => Train SPK/Age: {train_spk_acc:.2f}%/{train_age_acc:.2f}%")
    print(
        f"Val   => Loss(T/SPK/AGE/DEC): {val_metrics['val_total_loss']:.4f}/"
        f"{val_metrics['val_spk_loss']:.4f}/{val_metrics['val_age_loss']:.4f}/{val_metrics['val_decorr_loss']:.4f} | "
        f"Acc SPK/Age: {val_metrics['val_spk_acc']:.2f}%/{val_metrics['val_age_acc']:.2f}% | "
        f"Age AUC: {val_metrics['val_age_auc']:.6f}"
    )
    print(
        f"Test  => EER(before disentangle): {test_eer_before:.4f} | "
        f"EER(after disentangle): {test_eer_after:.4f} | Age AUC: {test_age_auc:.6f} | "
        f"Best(after): {best_test_EER:.4f}"
    )
    print(
        f"Corr  => Train |mean|: {train_age_corr_abs_mean:.6f} | "
        f"Val |mean|: {val_metrics['val_age_corr_abs_mean']:.6f} | "
        f"Test |mean|: {test_age_corr_abs_mean:.6f}"
    )
    print_age_neuron_correlation_stats(avg_age_corr, tag="Train", age_dim=AGE_DIM)
    print_age_neuron_correlation_stats(val_metrics["val_age_corr"], tag="Val", age_dim=AGE_DIM)
    print_age_neuron_correlation_stats(test_age_corr, tag="Test", age_dim=AGE_DIM)
    print("=" * 130 + "\n")

    if best_test_EER > test_eer_after:
        best_test_EER = test_eer_after
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
        print(f"✓ 儲存最佳模型 Test EER: {best_test_EER * 100:.2f}%")
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
    writer.add_scalar("Val/Total_Loss", val_metrics["val_total_loss"], epoch)
    writer.add_scalar("Val/SPK_Loss", val_metrics["val_spk_loss"], epoch)
    writer.add_scalar("Val/AGE_Loss", val_metrics["val_age_loss"], epoch)
    writer.add_scalar("Val/DECORR_Loss", val_metrics["val_decorr_loss"], epoch)
    writer.add_scalar("Val/Acc_SPK", val_metrics["val_spk_acc"], epoch)
    writer.add_scalar("Val/Acc_Age", val_metrics["val_age_acc"], epoch)
    writer.add_scalar("Val/Age_AUC", val_metrics["val_age_auc"], epoch)
    writer.add_scalar("Test/EER_Before", test_eer_before, epoch)
    writer.add_scalar("Test/EER_After", test_eer_after, epoch)
    writer.add_scalar("Test/Age_AUC", test_age_auc, epoch)
    writer.add_scalar("Test/Best_EER_After", best_test_EER, epoch)
    writer.add_scalar("Train/Lambda_Decorr_Current", current_lambda_decorr, epoch)
    writer.add_scalar("LR", current_lr, epoch)

    csv_writer.writerow(
        [
            epoch + 1,
            current_lr,
            current_lambda_decorr,
            train_total_loss,
            train_spk_loss,
            train_age_loss,
            train_decorr_loss,
            train_spk_acc,
            train_age_acc,
            train_age_corr_abs_mean,
            val_metrics["val_total_loss"],
            val_metrics["val_spk_loss"],
            val_metrics["val_age_loss"],
            val_metrics["val_decorr_loss"],
            val_metrics["val_spk_acc"],
            val_metrics["val_age_acc"],
            val_metrics["val_age_auc"],
            val_metrics["val_age_corr_abs_mean"],
            test_eer_before,
            test_eer_after,
            test_age_auc,
            test_age_corr_abs_mean,
            best_test_EER,
        ]
    )
    csv_file.flush()

    scheduler.step()

    if EARLY_STOPPING_ENABLED and no_improve_epochs >= EARLY_STOPPING_PATIENCE:
        print(
            f"Early stopping 觸發：連續 {EARLY_STOPPING_PATIENCE} 個 epoch 無改善。"
            f"最佳 Test EER(after)={best_test_EER:.4f}（Epoch {best_epoch}）。"
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
