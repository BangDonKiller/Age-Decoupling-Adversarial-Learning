import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
from tool.save_system import Save_system
import random
from data.dataloader import Voxceleb2_dataset, Voxceleb1_dataset
from params import param
from model import ADAL_Model 
from tool.eval_metric import *
from torch.utils.tensorboard import SummaryWriter

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

def prepare_dataloader():
    """
    準備數據集和數據加載器。
    這裡可以根據需要進行數據增強或其他處理。
    """
    # 數據集初始化
    train_dataset = Voxceleb2_dataset(
        data_list_file=param.DATA_LIST_FILE,
        dataset_path=param.DATA_ROOT,
        musan_path=param.MUSAN_DIR,
        rir_path=param.RIR_NOISE_DIR,
        augment=False,
    )
    
    print(f"Training dataset loaded with {len(train_dataset)} samples.")
    
    # 數據加載器
    train_loader = DataLoader(
        train_dataset,
        batch_size=param.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        # pin_memory=True if param.DEVICE == 'cuda' else False,
        drop_last=True,  # 確保每個批次的數據量相同(會丟棄最後一個不完整批次)
        collate_fn=train_dataset.collate_fn  # 使用自定義的 collate_fn 來處理變長序列
    )
    
    val_dataset = Voxceleb1_dataset(
        data_list_file=param.VAL_DATA_LIST_FILE,
        dataset_path=param.VAL_DATA_ROOT,
    )
    
    print(f"Validation dataset loaded with {len(val_dataset)} samples.")
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32, # param.BATCH_SIZE
        shuffle=False,  # 驗證集通常不需要打亂
        num_workers=0,  # 根據實際情況調整
        # pin_memory=True if param.DEVICE == 'cuda' else False,
        drop_last=False,  # 驗證集可以不需要確保每個批次的數據量相同
        collate_fn=val_dataset.collate_fn  # 使用自定義的 collate_fn 來處理變長序列
    )
    
    
    return train_loader, val_loader


def init_tensorboard():
    count = 1
    while True:
        if not os.path.exists(f"tensorboard_logs/exp{count}"):
            os.makedirs(f"tensorboard_logs/exp{count}")
            break
        count += 1
    writer = SummaryWriter(comment="ADAL", log_dir=f"tensorboard_logs/exp{count}")
    return writer

def log_tensorboard(writer, step, id_loss, age_loss, age_grl_loss, acc, EER, minDCF):
    if id_loss is not None:
        writer.add_scalar("Loss/train", id_loss, step)
    if age_loss is not None:
        writer.add_scalar("Loss/train/age", age_loss, step)
    if age_grl_loss is not None:
        writer.add_scalar("Loss/train/age_grl", age_grl_loss, step)
    if acc is not None:
        writer.add_scalar("Accuracy/train", acc, step)
    if EER is not None:
        writer.add_scalar("EER/train", EER, step)
    if minDCF is not None:
        writer.add_scalar("minDCF/train", minDCF, step)

def evaluate(model, val_loader, device):
    """
    在驗證集上評估模型性能，計算 EER 和 minDCF。
    
    :param model: 訓練好的模型。
    :param val_loader: 驗證數據加載器。
    :param device: 設備 (CPU 或 GPU)。
    :return: EER 和 minDCF。
    """
    model.eval()  # 設置模型為評估模式
    all_scores = [] # 用於收集所有測試對的分數
    all_labels = [] # 用於收集所有測試對的真實標籤

    with torch.no_grad():
        for audio1, audio2, label in tqdm(val_loader, desc="Evaluating", unit="batch"):
            audio1 = audio1.to(device)
            audio2 = audio2.to(device)

            # 前向傳播
            embedding1 = model(audio1, mode="val") # 輸出形狀: (batch_size, feature_dim)
            embedding2 = model(audio2, mode="val") # 輸出形狀: (batch_size, feature_dim)
            
            # L2 正則化 (這部分是正確的)
            embedding1 = F.normalize(embedding1, p=2, dim=1)
            embedding2 = F.normalize(embedding2, p=2, dim=1)

            # --- 核心修改：計算批次中每對音頻的餘弦相似度 ---
            scores_batch = F.cosine_similarity(embedding1, embedding2, dim=1)
            
            all_scores.extend(scores_batch.cpu().numpy().tolist())
            all_labels.extend(label.cpu().numpy().tolist()) # 假設 label 也是一個 Tensor
            
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    # 計算 EER 和 minDCF (這部分不需要修改)
    EER = tuneThresholdfromScore(all_scores, all_labels, [1, 0.1])[1]
    fnrs, fprs, thresholds = ComputeErrorRates(all_scores, all_labels)
    minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)

    return EER, minDCF

# --- 訓練主函數 ---
def train_model():
    device = torch.device(param.DEVICE)
    
    # --- 1. 數據加載器 ---
    train_loader, val_loader = prepare_dataloader()
    save_system = Save_system()  # 初始化保存系統，確保目錄存在並創建初始文件
    writer = init_tensorboard()  # 初始化 TensorBoard 日誌

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
    step = 0

    for epoch in range(param.EPOCHS):
        model.train() # 設置模型為訓練模式
        total_loss = 0.0
        total_loss_id = 0.0
        total_loss_age = 0.0
        total_loss_grl = 0.0

        current_lr = optimizer.param_groups[0]['lr']
        # 檢查學習率是否達到停止條件
        if current_lr < param.MIN_LR:
            break

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch")
        for batch_idx, (mels, identity_labels, age_labels) in enumerate(pbar):
            mels = mels.to(device)
            identity_labels = identity_labels.to(device)
            age_labels = age_labels.to(device)

            optimizer.zero_grad() # 清除梯度

            # 前向傳播
            z_id, loss_id, pred_age, pred_grl_age = model(mels, mode = "train", id_label = identity_labels)
            
            # 計算損失
            # L_id: 身份分類損失
            # max_index_of_id = torch.argmax(pred_ids, dim=1)
            # loss_id = criterion_ce(pred_ids, identity_labels) 

            # L_age: 年齡分類損失 (監督 z_age)
            max_index_of_age = torch.argmax(pred_age, dim=1)
            loss_age = criterion_ce(pred_age, age_labels)
            
            max_index_of_grl_age = torch.argmax(pred_grl_age, dim=1)
            # L_grl: 對抗年齡損失 (讓 z_id 無法預測年齡)
            loss_grl = criterion_ce(pred_grl_age, age_labels)            

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
            
            # --- 4. 保存模型檢查點 ---
            if step % 640 == 0:
                # --- 執行驗證/評估 ---
                val_eer, val_mDCF = evaluate(model, val_loader, device)
                if val_eer < best_val_eer:
                    best_val_eer = val_eer
                    best_checkpoint_path = os.path.join(param.CHECKPOINT_DIR, 'best_adal_model.pth')
                    torch.save(model.state_dict(), best_checkpoint_path)
                log_tensorboard(
                    writer, 
                    step,
                    loss_id.item(),
                    loss_age.item(),
                    loss_grl.item(),
                    None,  # 這裡可以添加準確率或其他指標
                    val_eer,  # EER
                    val_mDCF   # minDCF
                )
            
            step += param.BATCH_SIZE
                
        avg_loss = total_loss / len(train_loader)
        avg_loss_id = total_loss_id / len(train_loader)
        avg_loss_age = total_loss_age / len(train_loader)
        avg_loss_grl = total_loss_grl / len(train_loader)

        # 在每個 epoch 結束後更新學習率
        scheduler.step()
            
        # 保存訓練結果到文件
        save_system.write_result_to_file(
            param.SCORE_DIR, 
            "result", 
            (epoch + 1, current_lr, avg_loss_id, avg_loss_age, avg_loss_grl, avg_loss, val_eer, val_mDCF)
        )

        print(f"Epoch {epoch + 1}/{param.EPOCHS} completed. "
              f"Avg Loss: {avg_loss:.4f}, "
              f"Avg L_id: {avg_loss_id:.4f}, "
              f"Avg L_age: {avg_loss_age:.4f}, "
              f"Avg L_grl: {avg_loss_grl:.4f}, "
              f"Val EER: {val_eer:.4f}, "
              f"Val minDCF: {val_mDCF.item():.4f}")
        
        if epoch == param.EPOCHS - 1:
            # 在最後一個 epoch 結束時保存模型
            save_system.save_model(model, epoch + 1)


if __name__ == '__main__':
    train_model()
