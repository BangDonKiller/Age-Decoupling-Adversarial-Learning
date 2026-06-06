"""
MoE 版 ECAPA-TDNN 訓練腳本。

這個腳本示範如何訓練三個 LoRA 專家組成的 MoE：
    - 專家先載入已訓練完成的 LoRA adapter
    - 可透過超參數切換 soft gate 或 top-K gate
    - 損失函數 = 分類損失 + 重要性平衡損失
    - 資料切分與參數紀錄方式參考 ECAPA-TDNN_train.py
"""

import csv
import os
import random
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from data.vox1_loader import PairwiseDataset
from data.vox2_loader import Vox2Dataset
from model.disentangled_model import LoRAMoECAPAModel
from tool.EER import compute_eer, ComputeErrorRates, ComputeMinDcf
from params.param import BATCH_SIZE, DATASET_INFO, DEVICE, LEARNING_RATE, NUM_WORKERS

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
TRAIN_DATASET_NAME = "VoxCeleb2"
TRAIN_DATASET_VARIANT = "Mini"
VAL_DATASET_NAME = "Vox-O"
EPOCHS = 5
WEIGHT_DECAY = 1e-5
SPLIT_SEED = 42
ECAPA_CHANNELS = 1024
RUN_NAME = "ecapa_tdnn_moe_lora"
EXP = 1
GATE_STRATEGY = "soft"  # 可選: "soft" 或 "topk"
MIN_LR = 1e-5
TOP_K = 1
# 執行模式："train" 或 "inference"
RUN_MODE = "train"  # or "inference"
# inference 時要載入的檢查點（可以改成實際路徑）
INFERENCE_CKPT_PATH = "checkpoints/ecapa_tdnn_moe/cls/moe_router_and_head.pth"

# 三個 LoRA 專家的 adapter 路徑，請替換成你自己的 checkpoint
EXPERT_ADAPTER_PATHS = [
    "checkpoints/ecapa_tdnn_lora/m4_r4_alpha8_LE5/lora_adapter",
    "checkpoints/ecapa_tdnn_lora/m4_r4_alpha8_Delta5/lora_adapter",
    "checkpoints/ecapa_tdnn_lora/m4_r4_alpha8_Delta20/lora_adapter",
]

PRETRAINED_PATH = "pretrained_models/pretrain.model"
MODEL_CKPT_DIR = "checkpoints/ecapa_tdnn_moe"
LOG_DIR_ROOT = "logs/ecapa_tdnn_moe"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_next_exp_dir(base_dir: str, prefix: str = "exp"):
    """在 base_dir 下建立下一個 exp 資料夾。"""
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    existing = []
    for child in base_path.iterdir():
        if child.is_dir() and child.name.startswith(prefix):
            suffix = child.name[len(prefix):]
            if suffix.isdigit():
                existing.append(int(suffix))

    next_idx = max(existing, default=0) + 1
    exp_name = f"{prefix}{next_idx}"
    exp_path = base_path / exp_name
    exp_path.mkdir(parents=True, exist_ok=False)
    return str(exp_path), exp_name


def build_zero_shot_dataloader():
    test_dataset = PairwiseDataset(
        audio_dir=DATASET_INFO["VoxCeleb1"][VAL_DATASET_NAME]["AUDIO_DIR"],
        audio_meta_dir=DATASET_INFO["VoxCeleb1"][VAL_DATASET_NAME]["AUDIO_DATALIST"],
        audio_meta_csv_path=DATASET_INFO["VoxCeleb1"]["AUDIO_META_DIR"],
    )
    return DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )


def evaluate_zero_shot(model, test_loader, device, write_pair_csv: bool = False):
    """用 cosine similarity 做 zero-shot EER 評估。

    If `write_pair_csv` is True, also write a per-pair CSV named
    `<VAL_DATASET_NAME>_<GATE_STRATEGY>.csv` under `LOG_DIR_ROOT`.
    """
    model.eval()

    before_scores = []
    labels = []

    csv_file = None
    csv_writer = None
    if write_pair_csv:
        os.makedirs(LOG_DIR_ROOT, exist_ok=True)
        pair_csv_path = os.path.join(LOG_DIR_ROOT, f"{VAL_DATASET_NAME}_{GATE_STRATEGY}.csv")
        csv_file = open(pair_csv_path, mode="w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "id1",
            "id2",
            "weights_A",
            "weights_B",
            "expert_scores",
            "moe_score",
            "label",
            "delta_age",
        ])

    with torch.no_grad():
        eval_desc = "Inference (zero-shot)"
        if write_pair_csv:
            eval_desc += " + CSV"

        for is_same, id1, id2, wav1, wav2, age1, age2 in tqdm(
            test_loader,
            desc=eval_desc,
            total=len(test_loader),
            dynamic_ncols=True,
            leave=False,
        ):
            wav1 = wav1.to(device, non_blocking=True)
            wav2 = wav2.to(device, non_blocking=True)

            # fused embeddings for final MoE score
            emb1, gate1 = model.extract_fused_embedding(wav1, aug=False)
            emb2, gate2 = model.extract_fused_embedding(wav2, aug=False)

            h1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
            h2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
            score = torch.nn.functional.cosine_similarity(h1, h2).cpu()

            before_scores.append(score)
            labels.extend(is_same.cpu().tolist())

            if write_pair_csv:
                # router weights (B=1 assumed)
                wA = gate1.detach().cpu().squeeze(0).tolist()
                wB = gate2.detach().cpu().squeeze(0).tolist()

                # expert-wise embeddings and scores
                expert_scores = []
                for expert in model.experts:
                    ea = model._encode_backbone(expert, wav1, aug=False)
                    eb = model._encode_backbone(expert, wav2, aug=False)
                    ea = torch.nn.functional.normalize(ea, p=2, dim=1)
                    eb = torch.nn.functional.normalize(eb, p=2, dim=1)
                    s = torch.nn.functional.cosine_similarity(ea, eb).cpu().item()
                    expert_scores.append(s)

                # final MoE fused score (take first value from score tensor)
                moe_score = score.cpu().item() if isinstance(score, torch.Tensor) else float(score)

                # delta age if available
                delta_age = ""
                try:
                    if age1 is not None and age2 is not None:
                        # age tensors or numbers
                        a1 = int(age1.cpu().item()) if hasattr(age1, "cpu") else int(age1)
                        a2 = int(age2.cpu().item()) if hasattr(age2, "cpu") else int(age2)
                        delta_age = abs(a1 - a2)
                except Exception:
                    delta_age = ""

                # format arrays as requested
                fmt = lambda arr: "[" + ",".join(f"{v:.6f}" for v in arr) + "]"
                weights_A_str = fmt(wA)
                weights_B_str = fmt(wB)
                expert_scores_str = fmt(expert_scores)

                csv_writer.writerow([
                    id1,
                    id2,
                    weights_A_str,
                    weights_B_str,
                    expert_scores_str,
                    f"{moe_score:.6f}",
                    int(is_same.item()) if hasattr(is_same, "item") else int(is_same),
                    delta_age,
                ])

    if csv_file is not None:
        csv_file.close()

    final_labels = np.array(labels)
    final_scores = torch.cat(before_scores).numpy() if before_scores else np.array([])
    if final_scores.size == 0:
        return float("nan"), float("nan")

    # 計算 EER, 最佳閾值與 minDCF
    eer, threshold = compute_eer(final_scores, final_labels)
    fnrs, fprs, thresholds = ComputeErrorRates(final_scores, final_labels)
    min_dcf, best_threshold = ComputeMinDcf(fnrs, fprs, thresholds, p_target=0.01, c_miss=1, c_fa=1)
    
    return eer, threshold, min_dcf


def main():
    global EXP
    set_seed(SPLIT_SEED)
    
    audio_dir = None
    audio_meta_dir = None
    suffix = None
    if TRAIN_DATASET_NAME in DATASET_INFO:
        ds_entry = DATASET_INFO[TRAIN_DATASET_NAME]
        if TRAIN_DATASET_VARIANT and isinstance(ds_entry, dict) and TRAIN_DATASET_VARIANT in ds_entry:
            var = ds_entry[TRAIN_DATASET_VARIANT]
            audio_dir = var.get("AUDIO_DIR")
            audio_meta_dir = var.get("AUDIO_META_DIR")
        else:
            audio_dir = ds_entry.get("AUDIO_DIR")
            audio_meta_dir = ds_entry.get("AUDIO_META_DIR")
        suffix = ds_entry.get("audio_suffix")

    # 建立測試集（zero-shot）總是需要
    test_loader = build_zero_shot_dataloader()

    if RUN_MODE == "train":
        print("建立訓練 Dataset...")
        full_dataset = Vox2Dataset(
            audio_dir=audio_dir,
            audio_meta_dir=audio_meta_dir,
            musan_path=DATASET_INFO["MUSAN"]["AUDIO_DIR"],
            rir_path=DATASET_INFO["RIR"]["AUDIO_DIR"],
            suffix=suffix,
            age_target_mode="group",
        )

        train_loader = DataLoader(
            full_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )

        print(f"訓練集: {len(full_dataset)} samples")
    else:
        # inference mode: no training dataset
        full_dataset = None
        train_loader = None

    print("建立模型...")
    num_speakers = len(full_dataset.speaker2idx) if full_dataset is not None else 1
    model = LoRAMoECAPAModel(
        C=ECAPA_CHANNELS,
        n_class=num_speakers,
        pretrained_path=PRETRAINED_PATH,
        expert_adapter_paths=EXPERT_ADAPTER_PATHS,
        m=0.2,
        s=64.0,
        w_importance=0.0,
        gate_strategy=GATE_STRATEGY,
        top_k=TOP_K,
    )
    model = model.to(DEVICE)
    
    print("\n========== LoRA Check ==========")

    for i, expert in enumerate(model.experts):
        print(f"\nExpert {i}")

        found = False
        for name, _ in expert.named_parameters():
            if "lora" in name.lower():
                found = True
                print(name)

        if not found:
            print("No LoRA parameters found!")

    print("================================\n")

    # Print total parameter count (all parameters, including frozen)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters (all): {total_params}")

    # Print trainable parameters for inspection
    trainable_list = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    print("Trainable parameters:")
    total_trainable = 0
    for n, p in trainable_list:
        cnt = p.numel()
        total_trainable += cnt
        print(f" - {n}: {cnt}")
    print(f"Total trainable parameters: {total_trainable}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, EPOCHS - 1),
        eta_min=MIN_LR,
    )

    # If inference mode, load checkpoint (if available) and run evaluation only
    if RUN_MODE == "inference":
        if os.path.exists(INFERENCE_CKPT_PATH):
            ck = torch.load(INFERENCE_CKPT_PATH, map_location=DEVICE)
            state = ck.get("model_state_dict", ck)
            state.pop("speaker_loss.weight", None)

            try:
                model.load_state_dict(state, strict=False)
                print(f"Loaded checkpoint from {INFERENCE_CKPT_PATH}")
            except Exception as e:
                print(f"Warning: failed to fully load checkpoint: {e}")
        else:
            print(f"Warning: checkpoint not found: {INFERENCE_CKPT_PATH}")

        # run zero-shot once and write pair csv
        test_eer, test_threshold, test_min_dcf = evaluate_zero_shot(model, test_loader, DEVICE, write_pair_csv=True)
        print(f"Zero-shot EER: {test_eer:.4f}")
        print(f"Zero-shot MinDCF: {test_min_dcf:.4f}")
        return

    exp_dir, exp_name = get_next_exp_dir(MODEL_CKPT_DIR)
    log_dir, _ = get_next_exp_dir(LOG_DIR_ROOT)
    csv_path = os.path.join(log_dir, f"{RUN_NAME}.csv")

    print(f"實驗名稱: {exp_name}")
    print(f"Device: {DEVICE}")
    print(f"Gate 策略: {GATE_STRATEGY} (top_k={TOP_K})")
    print("訓練內容: 分類損失 + 重要性平衡損失")
    print("=" * 90)

    csv_file = open(csv_path, mode="w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "epoch",
        "learning_rate",
        "train_loss",
        "cls_loss",
        "Router_loss",
        "train_acc",
        "test_eer",
        "test_min_dcf",
        "gate_entropy",
        "gate_mean_LE5",
        "gate_mean_Delta5",
        "gate_mean_Delta20",
    ])

    for epoch in range(EPOCHS):
        model.train()

        total_loss = 0.0
        total_cls_loss = 0.0
        total_imp_loss = 0.0
        total_acc = 0.0
        total_samples = 0
        num_batches = 0
        gate_sum = torch.zeros(3, device=DEVICE)
        gate_entropy_sum = 0.0

        train_pbar = tqdm(
            train_loader,
            desc=f"Train Epoch {epoch + 1}/{EPOCHS}",
            total=len(train_loader),
            dynamic_ncols=True,
            leave=False,
        )

        for batch in train_pbar:
            waveform, labels, gender, age = batch
            waveform = waveform.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True).long()

            optimizer.zero_grad()
            loss, cls_loss, imp_loss, acc, gate_weights = model(waveform, labels=labels, aug=True)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_samples += batch_size
            num_batches += 1
            total_loss += float(loss.detach().cpu())
            total_cls_loss += float(cls_loss.detach().cpu())
            total_imp_loss += float(imp_loss.detach().cpu())
            total_acc += float(acc) * batch_size
            gate_sum += gate_weights.detach().sum(dim=0)
            # accumulate entropy: per-sample entropy summed over the batch
            eps = 1e-12
            gw = gate_weights.detach().clamp(min=eps)
            batch_entropy_sum = -(gw * gw.log()).sum(dim=1).sum().item()
            gate_entropy_sum += batch_entropy_sum

            train_pbar.set_postfix(
                loss=f"{float(loss.detach().cpu()):.4f}",
                cls=f"{float(cls_loss.detach().cpu()):.4f}",
                imp=f"{float(imp_loss.detach().cpu()):.4f}",
            )

        avg_loss = total_loss / max(1, num_batches)
        avg_cls_loss = total_cls_loss / max(1, num_batches)
        avg_imp_loss = total_imp_loss / max(1, num_batches)
        avg_acc = total_acc / max(1, total_samples)
        gate_mean = (gate_sum / max(1, total_samples)).detach().cpu().tolist()
        gate_entropy = gate_entropy_sum / max(1, total_samples)

        write_pairs = (epoch == EPOCHS - 1)
        test_eer, test_threshold, test_min_dcf = evaluate_zero_shot(model, test_loader, DEVICE, write_pair_csv=write_pairs)
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_display = epoch + 1
        print(
            f"Epoch [{epoch_display}/{EPOCHS}] | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {avg_loss:.4f} (cls={avg_cls_loss:.4f}, imp={avg_imp_loss:.4f}) | "
            f"Train Acc: {avg_acc:.2f}% | "
            f"Zero-shot EER: {test_eer:.4f} | Zero-shot MinDCF: {test_min_dcf:.4f} | Gate Mean: {gate_mean} | Gate Entropy: {gate_entropy:.4f}"
        )

        csv_writer.writerow([
            epoch + 1,
            current_lr,
            avg_loss,
            avg_cls_loss,
            avg_imp_loss,
            avg_acc,
            test_eer,
            test_min_dcf,
            gate_entropy,
            gate_mean[0],
            gate_mean[1],
            gate_mean[2],
        ])
        csv_file.flush()

        # update LR scheduler (step per epoch)
        scheduler.step()

        if epoch == EPOCHS - 1:
            ckpt_path = os.path.join(exp_dir, "moe_router_and_head.pth")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "pretrained_path": PRETRAINED_PATH,
                    "expert_adapter_paths": EXPERT_ADAPTER_PATHS,
                    "gate_strategy": GATE_STRATEGY,
                    "top_k": TOP_K,
                    "n_class": num_speakers,
                    "epoch": epoch + 1,
                },
                ckpt_path,
            )
            print(f"✓ 保存 MoE 檢查點 到: {ckpt_path}")

    csv_file.close()
    print(f"訓練日誌已保存: {csv_path}")
    print("=" * 90)
    print("MoE 訓練完成")
    print("=" * 90)

if __name__ == "__main__":
    main()
