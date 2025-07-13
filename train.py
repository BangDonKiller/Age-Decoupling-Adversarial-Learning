import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
import logging
import random
from data.dataloader import VoxCeleb_dataset
from params import param
from model import ADAL_Model 

# --- 設定日誌 ---
# 設置日誌格式
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(os.path.join(param.LOG_DIR, 'training.log')),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

# --- 設定隨機種子，確保可重現性 ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False # 如果輸入大小不變，可以設為True加速

set_seed(param.RANDOM_SEED)

# --- 訓練主函數 ---
def train_model():
    device = torch.device(param.DEVICE)
    
    # --- 1. 數據加載器 ---
    train_dataset = VoxCeleb_dataset(
        data_list_file=param.DATA_LIST_FILE,
        dataset_path=param.DATA_ROOT,
        musan_path=param.MUSAN_DIR,
        rir_path=param.RIR_NOISE_DIR,
        mode='train',
        augment=True, # 訓練時啟用數據增強
    )
    print(f"Training dataset loaded with {len(train_dataset)} samples.")
    train_loader = DataLoader(
        train_dataset,
        batch_size=param.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True if param.DEVICE == 'cuda' else False,
        drop_last=True, # 確保每個批次的數據量相同
        collate_fn=train_dataset.collate_fn # 使用自定義的 collate_fn 來處理變長序列
    )
    logger.info(f"Training DataLoader created with {len(train_loader)} batches, batch size {param.BATCH_SIZE}.")

    # --- 2. 模型、優化器、損失函數 ---
    model = ADAL_Model(
        feature_dim=param.EMBEDDING_DIM,
        age_classes=param.NUM_AGE_GROUPS,
        identity_classes=param.NUM_SPEAKERS
    ).to(device)
    logger.info(f"Model initialized and moved to {device}.")

    optimizer = optim.SGD(
        model.parameters(),
        lr=param.INITIAL_LR,
        momentum=0.9,
        weight_decay=1e-4 # 常見的權重衰減，作為正則化
    )
    logger.info(f"Optimizer {param.OPTIMIZER} initialized with initial LR {param.INITIAL_LR}.")

    # --- 學習率調度器與預熱 ---
    def lr_lambda(current_epoch):
        # 線性預熱
        if current_epoch < param.LR_WARMUP_EPOCHS:
            return (current_epoch + 1) / param.LR_WARMUP_EPOCHS
        # 多步衰減
        else:
            decay_factor = 1.0
            for decay_step in param.LR_DECAY_STEPS:
                if current_epoch >= decay_step:
                    decay_factor *= param.LR_DECAY_FACTOR
            return decay_factor

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    logger.info(f"Learning rate scheduler initialized with warmup epochs {param.LR_WARMUP_EPOCHS} and decay steps {param.LR_DECAY_STEPS}.")

    # 交叉熵損失用於年齡分類
    criterion_ce = nn.CrossEntropyLoss()

    # --- 3. 訓練循環 ---
    logger.info("Starting training loop...")
    
    # 初始化一個變量來跟蹤最佳模型，通常是基於驗證集 EER
    # 這裡 train.py 不包含驗證，但實際應用會需要
    best_val_eer = float('inf') 

    for epoch in range(param.EPOCHS):
        model.train() # 設置模型為訓練模式
        total_loss = 0.0
        total_loss_id = 0.0
        total_loss_age = 0.0
        total_loss_grl = 0.0

        current_lr = optimizer.param_groups[0]['lr']
        # 檢查學習率是否達到停止條件
        if current_lr < param.MIN_LR:
            logger.info(f"Learning rate {current_lr:.6f} dropped below {param.MIN_LR}. Stopping training.")
            break
        
        logger.info(f"Epoch {epoch+1}/{param.EPOCHS}, Current LR: {current_lr:.6f}")

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch")
        for batch_idx, (mels, identity_labels, age_labels) in enumerate(pbar):
            mels = mels.to(device)
            identity_labels = identity_labels.to(device)
            age_labels = age_labels.to(device)

            optimizer.zero_grad() # 清除梯度

            # 前向傳播
            # features, z, z_age, identity_logits, age_logits_from_age, age_logits_from_id_grl
            _, _, _, identity_logits, age_logits_from_age, age_logits_from_id_grl = model(mels, identity_labels)

            # 計算損失
            # L_id: 身份分類損失 (ArcFaceLoss)
            loss_id = criterion_ce(identity_logits, identity_labels) 

            # L_age: 年齡分類損失 (監督 z_age)
            loss_age = criterion_ce(age_logits_from_age, age_labels)

            # L_grl: 對抗年齡損失 (讓 z_id 無法預測年齡)
            loss_grl = criterion_ce(age_logits_from_id_grl, age_labels)

            # 總損失 (根據論文公式5)
            loss = (param.LAMBDA_ID * loss_id +
                    param.LAMBDA_AGE * loss_age +
                    param.LAMBDA_GRL * loss_grl)

            # 反向傳播與優化
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_loss_id += loss_id.item()
            total_loss_age += loss_age.item()
            total_loss_grl += loss_grl.item()

            pbar.set_postfix({
                'Total_L': f'{loss.item():.4f}',
                'L_id': f'{loss_id.item():.4f}',
                'L_age': f'{loss_age.item():.4f}',
                'L_grl': f'{loss_grl.item():.4f}'
            })

        avg_loss = total_loss / len(train_loader)
        avg_loss_id = total_loss_id / len(train_loader)
        avg_loss_age = total_loss_age / len(train_loader)
        avg_loss_grl = total_loss_grl / len(train_loader)

        logger.info(f"Epoch {epoch+1} Summary: Avg Total Loss: {avg_loss:.4f}, Avg L_id: {avg_loss_id:.4f}, Avg L_age: {avg_loss_age:.4f}, Avg L_grl: {avg_loss_grl:.4f}")

        # 在每個 epoch 結束後更新學習率
        scheduler.step()

        # --- 4. 保存模型檢查點 ---
        # 可以在每個 epoch 結束時保存，或者基於驗證性能保存最佳模型
        checkpoint_path = os.path.join(param.CHECKPOINT_DIR, f'adal_model_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
            'current_lr': current_lr,
        }, checkpoint_path)
        logger.info(f"Model checkpoint saved to {checkpoint_path}")

        # --- (可選) 執行驗證/評估 ---
        # 實際項目中，這裡會調用一個單獨的評估函數，計算 EER/mDCF
        # 根據驗證指標來決定是否保存 'best' 模型
        # 例如：
        # val_eer, val_mDCF = evaluate(model, val_loader, device)
        # logger.info(f"Validation EER: {val_eer:.4f}%, mDCF: {val_mDCF:.4f}")
        # if val_eer < best_val_eer:
        #     best_val_eer = val_eer
        #     best_checkpoint_path = os.path.join(param.CHECKPOINT_DIR, 'best_adal_model.pth')
        #     torch.save(model.state_dict(), best_checkpoint_path)
        #     logger.info(f"New best model saved to {best_checkpoint_path} with EER {best_val_eer:.4f}%")


if __name__ == '__main__':
    train_model()
