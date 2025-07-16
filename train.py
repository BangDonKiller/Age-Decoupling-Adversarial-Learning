import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
from tool.save_system import save_system
import random
from data.dataloader import VoxCeleb_dataset
from params import param
from model import ADAL_Model 
from tool.eval_metric import compute_eer, compute_min_dcf

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
save_system = save_system()  # 初始化保存系統，確保目錄存在並創建初始文件

def prepare_dataloader():
    """
    準備數據集和數據加載器。
    這裡可以根據需要進行數據增強或其他處理。
    """
    # 數據集初始化
    train_dataset = VoxCeleb_dataset(
        data_list_file=param.DATA_LIST_FILE,
        dataset_path=param.DATA_ROOT,
        musan_path=param.MUSAN_DIR,
        rir_path=param.RIR_NOISE_DIR,
        mode='train',
        augment=False,
    )
    
    print(f"Training dataset loaded with {len(train_dataset)} samples.")
    
    # 數據加載器
    train_loader = DataLoader(
        train_dataset,
        batch_size=param.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # 根據實際情況調整
        # pin_memory=True if param.DEVICE == 'cuda' else False,
        drop_last=True,  # 確保每個批次的數據量相同
        collate_fn=train_dataset.collate_fn  # 使用自定義的 collate_fn 來處理變長序列
    )
    
    val_dataset = VoxCeleb_dataset(
        data_list_file=param.VAL_DATA_LIST_FILE,
        dataset_path=param.VAL_DATA_ROOT,
        musan_path=param.MUSAN_DIR,
        rir_path=param.RIR_NOISE_DIR,
        mode='val',
        augment=False,  # 驗證時不進行數據增強
    )
    
    print(f"Validation dataset loaded with {len(val_dataset)} samples.")
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=param.BATCH_SIZE,
        shuffle=False,  # 驗證集通常不需要打亂
        num_workers=0,  # 根據實際情況調整
        # pin_memory=True if param.DEVICE == 'cuda' else False,
        drop_last=False,  # 驗證集可以不需要確保每個批次的數據量相同
        collate_fn=val_dataset.collate_fn  # 使用自定義的 collate_fn 來處理變長序列
    )
    
    
    return train_loader, val_loader


def evaluate(model, val_loader, device):
    """
    在驗證集上評估模型性能，計算 EER 和 minDCF。
    
    :param model: 訓練好的模型。
    :param val_loader: 驗證數據加載器。
    :param device: 設備 (CPU 或 GPU)。
    :return: EER 和 minDCF。
    """
    model.eval()  # 設置模型為評估模式
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for mels, identity_labels, age_labels in tqdm(val_loader, desc="Evaluating", unit="batch"):
            mels = mels.to(device)
            identity_labels = identity_labels.to(device)
            age_labels = age_labels.to(device)

            # 前向傳播
            _, _, _, identity_logits, _, _ = model(mels, identity_labels)

            # 獲取分數 (這裡假設使用身份分類的 logits 作為分數)
            scores = torch.softmax(identity_logits, dim=1)[:, 1]  # 假設第二類是目標類別
            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(identity_labels.cpu().numpy())

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    # 計算 EER 和 minDCF
    eer, eer_threshold = compute_eer(all_scores, all_labels)
    min_dcf = compute_min_dcf(all_scores, all_labels, p_target=param.P_TARGET, c_fa=param.C_FA, c_miss=param.C_MISS)

    return eer, min_dcf

# --- 訓練主函數 ---
def train_model():
    device = torch.device(param.DEVICE)
    
    # --- 1. 數據加載器 ---
    train_loader, val_loader = prepare_dataloader()

    # --- 2. 模型、優化器、損失函數 ---
    model = ADAL_Model(
        feature_dim=param.EMBEDDING_DIM,
        age_classes=param.NUM_AGE_GROUPS,
        identity_classes=param.NUM_SPEAKERS
    ).to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=param.INITIAL_LR,
        momentum=0.9,
        weight_decay=1e-4 # 常見的權重衰減，作為正則化
    )

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

    # 交叉熵損失用於年齡分類
    criterion_ce = nn.CrossEntropyLoss()

    # --- 3. 訓練循環 ---   
    best_val_eer = float('inf') 

    for epoch in range(param.EPOCHS):
        state = "train"
        model.train() # 設置模型為訓練模式
        total_loss = 0.0
        total_loss_id = 0.0
        total_loss_age = 0.0
        total_loss_grl = 0.0

        current_lr = optimizer.param_groups[0]['lr']
        # 檢查學習率是否達到停止條件
        if current_lr < param.MIN_LR:
            break
        
        # logger.info(f"Epoch {epoch+1}/{param.EPOCHS}, Current LR: {current_lr:.6f}")

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

        # 在每個 epoch 結束後更新學習率
        scheduler.step()

        # --- 4. 保存模型檢查點 ---
        save_system.save_model(model, epoch + 1)
        
        # 寫入各種loss以及 EER 和 minDCF 到tensorboard
        save_system.write_tensorboard_log(
            param.TENSOR_BOARD_DIR,
            epoch + 1,
            current_lr,
            avg_loss_id,
            avg_loss_age,
            avg_loss_grl,
            avg_loss,
            eer=None,  # 在訓練階段不計算 EER
            min_dcf=None  # 在訓練階段不計算 minDCF
        )

        # --- 執行驗證/評估 ---
        state = "val"
        val_eer, val_mDCF = evaluate(model, val_loader, device)
        if val_eer < best_val_eer:
            best_val_eer = val_eer
            best_checkpoint_path = os.path.join(param.CHECKPOINT_DIR, 'best_adal_model.pth')
            torch.save(model.state_dict(), best_checkpoint_path)
            
        save_system.write_tensorboard_log(
            param.TENSOR_BOARD_DIR,
            state,
            epoch + 1,
            l_id=avg_loss_id,
            l_age=avg_loss_age,
            l_grl=avg_loss_grl,
            total_loss=avg_loss,
            eer=val_eer,
            min_dcf=val_mDCF
        )
            
        # 保存訓練結果到文件
        save_system.write_result_to_file(
            param.SCORE_DIR, 
            "result", 
            (epoch + 1, current_lr, avg_loss_id, avg_loss_age, avg_loss_grl, avg_loss, val_eer, val_mDCF)
        )
        
        state = "train"
        print(f"Epoch {epoch + 1}/{param.EPOCHS} completed. "
              f"Avg Loss: {avg_loss:.4f}, "
              f"Avg L_id: {avg_loss_id:.4f}, "
              f"Avg L_age: {avg_loss_age:.4f}, "
              f"Avg L_grl: {avg_loss_grl:.4f}, "
              f"Val EER: {val_eer:.4f}, "
              f"Val minDCF: {val_mDCF:.4f}")


if __name__ == '__main__':
    train_model()
