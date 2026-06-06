"""
從零開始訓練一個全新的 ECAPA-TDNN 模型。

這個腳本使用 Vox2Dataset，從預訓練處開始訓練一個全新的 ECAPA-TDNN 模型。

特點：
  - 從預訓練處開始，初始化 LoRA 模塊、分類頭權重
  - 使用 Vox2Dataset（Offline Augmentation）
  - 標準 PyTorch 訓練迴圈
    - 模型檢查點保存和 zero-shot 評估
"""
import csv
import os
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from safetensors.torch import load_file, save_file
from torch.optim.lr_scheduler import CosineAnnealingLR

from data.vox2_loader import Vox2Dataset
from data.vox1_loader import PairwiseDataset
from model.feature_extractor.ecapa_model import ECAPAModel
from params.param import BATCH_SIZE, DATASET_INFO, DEVICE, LEARNING_RATE, NUM_WORKERS
from peft import LoraConfig, get_peft_model, PeftModel
from loss.arcface import ArcMarginProduct

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
# Choose training dataset. For datasets with variants (like VoxCeleb2), set
# TRAIN_DATASET_NAME to the top-level key and TRAIN_DATASET_VARIANT to the
# specific variant name (e.g. "LE5", "Delta5", "Delta20").
TRAIN_DATASET_NAME = "VoxCeleb2"
TRAIN_DATASET_VARIANT = "Mixture"
EPOCHS = 5
WEIGHT_DECAY = 1e-5
SPLIT_SEED = 42
RANKS = [4]
ALPHA = [8]
MODULE_IDS = [4] # 1: attention, 2: layer4, 3: fc6, 4: attention+layer4+fc6
MODULES = {
    # 1: ["attention.0", "attention.4"], # 只微調 attention 模塊
    # 2: ["layer4"], # 只微調 layer4(特徵聚合) 模塊
    # 3: ["fc6"], # 只微調最後的全連接層
    4: ["attention.0", "attention.4", "layer4", "fc6"], # 微調 attention 和 layer4 模塊
}

for mid in MODULE_IDS:
    if mid not in MODULES:
        raise ValueError(f"MODULE_ID 必須是 1~4，目前有錯誤的 id: {mid}")

# zero-shot validation set name (VoxCeleb1 split key in params.param.DATASET_INFO)
VAL_DATASET_NAME = ["Vox1-H.S", "Vox1-S.S", "Vox-CA10", "Vox-CA20"]
ECAPA_CHANNELS = 1024

# 執行模式："finetune" 或 "inference"
# - finetune: 載入 pretrained 後只訓練 LoRA 模塊
# - inference: 載入模型(包括 LoRA 模塊)做推論
RUN_MODE = "inference"  # "finetune" 或 "inference"

# 檢查點配置
EXP = 1
# We'll include module & rank in experiment folders to support multiple runs

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    global EXP
    set_seed(SPLIT_SEED)

    if RUN_MODE not in {"finetune", "inference"}:
        raise ValueError(f"RUN_MODE 只能是 'finetune' 或 'inference'，目前是: {RUN_MODE}")
    
    if RUN_MODE == "finetune":
        print("建立訓練 Dataset...")
        dataset = Vox2Dataset(
            audio_dir=DATASET_INFO[TRAIN_DATASET_NAME][TRAIN_DATASET_VARIANT]["AUDIO_DIR"],
            audio_meta_dir=DATASET_INFO[TRAIN_DATASET_NAME][TRAIN_DATASET_VARIANT]["AUDIO_META_DIR"],
            musan_path=DATASET_INFO["MUSAN"]["AUDIO_DIR"],
            rir_path=DATASET_INFO["RIR"]["AUDIO_DIR"],
            suffix=DATASET_INFO[TRAIN_DATASET_NAME]["audio_suffix"],
            age_target_mode="group",
        )

        train_loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )

        print(f"訓練集: {len(dataset)} samples")
        
    # Zero-shot test datasets (VoxCeleb1 splits)
    print("建立測試 Dataset...")
    test_loaders = {}
    for val_name in VAL_DATASET_NAME:
        td = PairwiseDataset(
            audio_dir=DATASET_INFO["VoxCeleb1"][val_name]["AUDIO_DIR"],
            audio_meta_dir=DATASET_INFO["VoxCeleb1"][val_name]["AUDIO_DATALIST"],
            audio_meta_csv_path=DATASET_INFO["VoxCeleb1"]["AUDIO_META_DIR"],
        )
        test_loaders[val_name] = DataLoader(
            td,
            batch_size=1,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )
    
    # ========== 建立模型（統一處理 inference / finetune） ==========
    print("建立模型...")
    num_speakers = len(dataset.speaker2idx) if (RUN_MODE == "finetune" and dataset is not None) else 1

    if RUN_MODE == "inference":
        # 建立模型時用訓練時的類別數（會被拔掉分類頭，所以用高數字避免尺寸衝突）
        base_model = ECAPAModel(C=ECAPA_CHANNELS, n_class=5994, m=0.2, s=64.0)
        base_model = base_model.to(DEVICE)
        print("開始推論模式，載入預訓練權重...")
        base_model.load_parameters("pretrained_models/pretrain.model")
        
        # 載入 LoRA adapter（忽略 speaker_loss 不匹配）
        ADAPTER_PATH = "checkpoints/ecapa_tdnn_lora/m4_r4_alpha8_Mixture/lora_adapter"
        print(f"載入 LoRA adapter 從: {ADAPTER_PATH}")
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

        lora_param_names = [name for name, _ in model.named_parameters() if "lora_" in name]
        print(f"載入後 LoRA 參數數量: {len(lora_param_names)}")
        print(lora_param_names[:20])
        if not lora_param_names:
            raise RuntimeError("LoRA adapter 載入後沒有任何 lora_ 參數，請檢查 adapter path / config 是否正確。")

        model.eval()
        
        print("✓ 模型載入完成（分類頭將被拔掉）")
        
        # evaluate on all val splits
        for val_name, dl in test_loaders.items():
            test_eer = model.evaluate_zero_shot(dl, DEVICE)
            print(f"Zero-shot EER on VoxCeleb1 {val_name}: {test_eer:.4f}")
        return

    if RUN_MODE == "finetune":
        print("開始微調模式，從預訓練處開始訓練模型(LoRA模塊)...")
        print(f"\n開始訓練... (每個實驗總 {EPOCHS} epochs)")
        print(f"Device: {DEVICE}")
        print(f"模式: 從預訓練處開始訓練")
        print("=" * 90)

        # iterate over module, rank and alpha combinations
        for module_id in MODULE_IDS:
            for rank in RANKS:
                for alpha in ALPHA:
                    EXP_NAME = f"exp{EXP}_m{module_id}_r{rank}_alpha{alpha}_{TRAIN_DATASET_VARIANT}"
                    CHECKPOINT_DIR = f"checkpoints/ecapa_tdnn_lora/{EXP_NAME}"
                    LOG_DIR = f"logs/ecapa_tdnn_lora/{EXP_NAME}"
                    # if checkpoint dir already exists for this EXP, advance EXP until unique
                    while os.path.exists(CHECKPOINT_DIR) or os.path.exists(LOG_DIR):
                        EXP += 1
                        EXP_NAME = f"exp{EXP}_m{module_id}_r{rank}_alpha{alpha}_{TRAIN_DATASET_VARIANT}"
                        CHECKPOINT_DIR = f"checkpoints/ecapa_tdnn_lora/{EXP_NAME}"
                        LOG_DIR = f"logs/ecapa_tdnn_lora/{EXP_NAME}"

                    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
                    os.makedirs(LOG_DIR, exist_ok=True)

                    print(f"\n=> 實驗: MODULE={module_id}, RANK={rank}, ALPHA={alpha}  -> {EXP_NAME}")

                    # build model anew per experiment
                    model = ECAPAModel(C=ECAPA_CHANNELS, n_class=num_speakers, m=0.2, s=64.0)
                    model = model.to(DEVICE)
                    # 讀取預訓練權重
                    model.load_parameters("pretrained_models/pretrain.model")

                    # 如果預訓練的 speaker classifier 與當前 num_speakers 不符，重新初始化並設為可訓練
                    if hasattr(model, "speaker_loss"):
                        try:
                            cur = model.speaker_loss
                            # 如果原本是 ArcMarginProduct，保留其 s/m/easy_margin 設定
                            if isinstance(cur, ArcMarginProduct):
                                out_features = getattr(cur, "out_features", None)
                                in_features = getattr(cur, "in_features", None) or 192
                                s_val = getattr(cur, "s", 30.0)
                                m_val = getattr(cur, "m", 0.35)
                                easy = getattr(cur, "easy_margin", False)
                                if out_features is None or out_features != num_speakers:
                                    model.speaker_loss = ArcMarginProduct(in_features, num_speakers, s=s_val, m=m_val, easy_margin=easy).to(DEVICE)
                            else:
                                out_features = getattr(cur, "out_features", None)
                                in_features = getattr(cur, "in_features", None) or 192
                                if out_features is None or out_features != num_speakers:
                                    model.speaker_loss = nn.Linear(in_features, num_speakers).to(DEVICE)
                        except Exception:
                            model.speaker_loss = nn.Linear(192, num_speakers).to(DEVICE)
                    else:
                        # 如果沒有 speaker_loss 屬性，建立一個 ArcMarginProduct 作為分類頭
                        model.speaker_loss = ArcMarginProduct(192, num_speakers).to(DEVICE)

                    # 先把分類頭設為可訓練（get_peft_model 之後我們會再次確保在 wrapper 中也可訓練）
                    for p in model.speaker_loss.parameters():
                        p.requires_grad = True

                    # create LoRA config for this experiment (use current alpha)
                    config = LoraConfig(
                        r=rank,
                        lora_alpha=alpha,
                        target_modules=MODULES[module_id],
                        lora_dropout=0.05,
                        bias="none",
                    )

                    # apply LoRA
                    model = get_peft_model(model, config)
                    model.print_trainable_parameters()

                    # get_peft_model 可能把模型包成 wrapper，確保 wrapper 內的 speaker_loss 也為可訓練
                    target = None
                    if hasattr(model, "speaker_loss"):
                        target = model.speaker_loss
                    elif hasattr(model, "base_model") and hasattr(model.base_model, "speaker_loss"):
                        target = model.base_model.speaker_loss
                    elif hasattr(model, "model") and hasattr(model.model, "speaker_loss"):
                        target = model.model.speaker_loss
                    elif hasattr(model, "base_model") and hasattr(model.base_model, "model") and hasattr(model.base_model.model, "speaker_loss"):
                        target = model.base_model.model.speaker_loss

                    if target is not None:
                        for p in target.parameters():
                            p.requires_grad = True

                    # optimizer for trainable params
                    trainable_params = [p for p in model.parameters() if p.requires_grad]
                    for name, p in model.named_parameters():
                        if p.requires_grad:
                            print(name)
                    optimizer = torch.optim.Adam(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
                    model.optim = optimizer
                    # Cosine annealing scheduler (per-epoch)
                    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=0)
                    model.scheduler = scheduler

                    # csv logging: include val split Vox-O and alpha in filename
                    csv_path = os.path.join(LOG_DIR, f"ECAPA-TDNN_LoRA_Module{module_id}_Rank{rank}_alpha{alpha}_{TRAIN_DATASET_VARIANT}.csv")
                    csv_file = open(csv_path, mode="w", newline="", encoding="utf-8")
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow([
                        "epoch",
                        "learning_rate",
                        "train_loss",
                        "train_acc",
                        "test_eer_vox_o",
                    ])

                    for epoch in range(EPOCHS):
                        avg_loss, avg_acc = model.train_network(
                            epoch=epoch,
                            loader=train_loader,
                            optimizer=optimizer
                        )

                        # evaluate on each val split
                        eer_results = {}
                        for val_name, dl in test_loaders.items():
                            eer_results[val_name] = model.evaluate_zero_shot(dl, DEVICE)

                        current_lr = model.optim.param_groups[0]['lr']

                        print(
                            f"Epoch [{epoch + 1:2d}/{EPOCHS}] | "
                            f"LR: {current_lr:.2e} | "
                            f"Train Loss: {avg_loss:.4f}, Train Acc: {avg_acc:.2f}% | "
                            f"Test EERs: {eer_results} | "
                        )

                        csv_writer.writerow([
                            epoch + 1,
                            current_lr,
                            avg_loss,
                            avg_acc,
                            eer_results.get("Vox-O", ""),
                        ])
                        csv_file.flush()

                        # update LR scheduler (step per epoch)
                        if hasattr(model, "scheduler") and model.scheduler is not None:
                            model.scheduler.step()

                        # save LoRA adapter at the end of training for this experiment
                        if epoch == EPOCHS - 1:
                            adapter_dir = os.path.join(CHECKPOINT_DIR, "lora_adapter")
                            model.save_pretrained(adapter_dir)
                            
                            # 過濾掉 speaker_loss，只保存 LoRA 權重
                            adapter_file = os.path.join(adapter_dir, "adapter_model.safetensors")
                            state_dict = load_file(adapter_file)
                            state_dict_filtered = {k: v for k, v in state_dict.items() if "speaker_loss" not in k}
                            save_file(state_dict_filtered, adapter_file)
                            print(f"✓ 保存 LoRA 權重 到: {CHECKPOINT_DIR}（已過濾 speaker_loss）")

                    csv_file.close()
                    print(f"訓練日誌已保存: {csv_path}")
                    EXP += 1

        print("\n" + "=" * 90)
        print("所有實驗完成")
        print("=" * 90)

if __name__ == "__main__":
    main()
