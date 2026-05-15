"""
從零開始訓練一個全新的 ECAPA-TDNN 模型。

這個腳本使用 Vox2Dataset，從隨機初始化開始訓練一個全新的 ECAPA-TDNN 模型。
不使用任何預訓練權重。

特點：
  - 完全從零開始，隨機初始化所有權重
  - 使用你的 Vox2Dataset（包含資料強化：MUSAN 噪音、RIR 混響）
  - 標準 PyTorch 訓練迴圈
    - 模型檢查點保存和 zero-shot 評估
"""
import csv
import os
import random
import warnings
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.vox2_loader import Vox2Dataset
from data.vox1_loader import PairwiseDataset
from model.feature_extractor.ecapa_model import ECAPAModel
from params.param import BATCH_SIZE, DATASET_INFO
from peft import LoraConfig, get_peft_model, PeftModel

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
EPOCHS = 1  # 從零開始訓練，通常需要更多 epoch
LEARNING_RATE = 1e-3  # 從零開始訓練，用更高的學習率
WEIGHT_DECAY = 1e-5
SPLIT_SEED = 42

# LR warmup / annealing and margin scheduler (defaults copied from step-based script)
MARGIN_SCHED_T1 = 2000
MARGIN_SCHED_T2 = 6000
MARGIN_SCHED_M = 0.2

# zero-shot validation set name (VoxCeleb1 split key in params.param.DATASET_INFO)
VAL_DATASET_NAME = "Vox-O"

# 訓練配置
SPEAKER_EMB_DIM = 192

# ECAPA-TDNN channel width for the local implementation (conv channel C)
ECAPA_CHANNELS = 1024

# 執行模式："finetune" 或 "inference"
# - finetune: 載入(可選)pretrained 後進入訓練
# - inference: 只載入模型做推論（不訓練）
RUN_MODE = "finetune"  # 或 "inference"

# 3. 定義 LoRA 配置
config = LoraConfig(
    r=1,                       # LoRA Rank
    lora_alpha=16,             # Scaling factor
    target_modules=["attention.0", "attention.4"], # 這裡可以直接寫 Sequential 裡面的索引名，或者具體模塊名
    lora_dropout=0.05,
    bias="none",               # 是否微調 bias
    modules_to_save=[],        # 除了 LoRA，還有哪些層要解凍（通常為空）
)

# 檢查點配置
CHECKPOINT_DIR = "checkpoints/ecapa_tdnn_from_scratch"
LOG_DIR = "logs/ecapa_tdnn_from_scratch"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    set_seed(SPLIT_SEED)

    if RUN_MODE not in {"finetune", "inference"}:
        raise ValueError(f"RUN_MODE 只能是 'finetune' 或 'inference'，目前是: {RUN_MODE}")
    
    # 建立輸出目錄
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    

    dataset = None
    train_loader = None
    if RUN_MODE == "finetune":
        print("建立訓練 Dataset...")
        dataset = Vox2Dataset(
            audio_dir=DATASET_INFO[TRAIN_DATASET_NAME]["AUDIO_DIR"],
            audio_meta_dir=DATASET_INFO[TRAIN_DATASET_NAME]["AUDIO_META_DIR"],
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
        
    # Zero-shot test dataset (VoxCeleb1 split)
    print("建立測試 Dataset...")
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
    
    # ========== 建立模型（統一處理 inference / finetune） ==========
    print("建立模型...")
    num_speakers = len(dataset.speaker2idx) if dataset is not None else 1

    model = ECAPAModel(C=ECAPA_CHANNELS, n_class=num_speakers, m=0.2, s=64.0)
    model = model.to(DEVICE)
    
    if RUN_MODE == "inference":
        print("開始推論模式，載入模型權重並執行評估...")
        model.load_parameters("pretrained_models/pretrain.model")
        model = PeftModel.from_pretrained(model, "checkpoints/ecapa_tdnn_from_scratch/lora_adapter")
        model.eval()
        test_eer = model.evaluate_zero_shot(test_dataloader, DEVICE)
        print(f"Zero-shot EER on VoxCeleb1 {VAL_DATASET_NAME}: {test_eer:.4f}")
        return

    if RUN_MODE == "finetune":
        print("開始微調模式，從預訓練處開始訓練模型(LoRA模塊)...")
        # 讀取預訓練權重
        model.load_parameters("pretrained_models/pretrain.model")
        # 套用 LoRA
        model = get_peft_model(model, config)
        # 打印一下看看哪些參數在動
        model.print_trainable_parameters()
        
        # 重建 optimizer（只包含 LoRA 參數）
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        # for name, p in model.named_parameters():
        #     if p.requires_grad:
        #         print(name)
        optimizer = torch.optim.Adam(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        model.optim = optimizer  # 替換模型內的 optimizer
        
        
        # ========== 訓練迴圈 ==========
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(LOG_DIR, f"train_history_{timestamp}.csv")
        
        csv_file = open(csv_path, mode="w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "epoch",
            "learning_rate",
            "train_loss",
            "train_acc",
            "test_eer_before",
        ])
        
        print(f"\n開始訓練... (總 {EPOCHS} epochs)")
        print(f"Device: {DEVICE}")
        print(f"模式: 從預訓練處開始訓練")
        print("=" * 90)

        for epoch in range(EPOCHS):
            # 訓練
            avg_loss, lr, avg_acc = model.train_network(
                epoch=epoch,
                loader=train_loader
            )

            # zero-shot evaluation on VoxCeleb1 split (may be slow)
            test_eer_before = model.evaluate_zero_shot(test_dataloader, DEVICE)
            
            # 取得目前 learning rate
            current_lr = model.optim.param_groups[0]['lr']

            print(
                f"Epoch [{epoch + 1:2d}/{EPOCHS}] | "
                f"LR: {current_lr:.2e} | "
                f"Train Loss: {avg_loss:.4f}, Train Acc: {avg_acc:.2f}% | "
                f"Test EER(before): {test_eer_before:.4f} | "
            )

            csv_writer.writerow([
                epoch + 1,
                current_lr,
                avg_loss,
                avg_acc,
                test_eer_before,
            ])
            csv_file.flush()
            
            # 模型檢查點：保存最後模型
            if epoch == EPOCHS - 1:
                # 在訓練完成後，只保存 LoRA 權重
                # 訓練結束後（model 為 get_peft_model(...) 回傳的模型）
                model.save_pretrained("checkpoints/ecapa_tdnn_from_scratch/lora_adapter")
                print(f"✓ 保存 LoRA 權重")
        
        csv_file.close()
        
        # ========== 訓練完成摘要 ==========
        print("\n" + "=" * 90)
        print("訓練完成摘要")
        print(f"訓練日誌已保存: {csv_path}")
        print(f"模型檢查點已保存: {CHECKPOINT_DIR}")
        print("=" * 90)


if __name__ == "__main__":
    main()
