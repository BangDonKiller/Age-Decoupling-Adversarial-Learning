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
from model import AttributeUnlearningModel
from tool.eval_metric import *
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict

# --- 設定隨機種子，確保可重現性 ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(param.RANDOM_SEED)

# ... (data_weight, prepare_dataloader, init_tensorboard, log_tensorboard, evaluate, get_grad_norm 函數保持不變) ...
def data_weight(dataset):
    """
    計算數據集的權重。
    這裡可以根據需要進行數據增強或其他處理。
    """
    # 假設 dataset 是一個列表或其他可迭代對象
    age_counts = {}
    
    for _, _, age_label in dataset.data_list:
        if age_label not in age_counts:
            age_counts[age_label] = 1
        age_counts[age_label] += 1
    age_weights = np.array([age_counts[i] for i in range(param.NUM_AGE_GROUPS)])
    age_weights = torch.tensor(age_weights, dtype=torch.float32)

    return age_weights

def prepare_dataloader():
    """
    準備數據集和數據加載器。
    這裡可以根據需要進行數據增強或其他處理。
    """
    # 數據集初始化
    train_dataset = Voxceleb2_dataset(
        num_frames=param.NUM_FRAMES,
        data_list_file=param.DATA_LIST_FILE,
        dataset_path=param.DATA_ROOT,
        musan_path=param.MUSAN_DIR,
        rir_path=param.RIR_NOISE_DIR,
        augment=False,
    )
    
    print(f"Training dataset loaded with {len(train_dataset)} samples.")
    
    # 計算數據集權重
    age_weights = data_weight(train_dataset)
    
    # 數據加載器
    train_loader = DataLoader(
        train_dataset,
        batch_size=param.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        collate_fn=train_dataset.collate_fn,
    )
    
    val_dataset = Voxceleb1_dataset(
        data_list_file=param.VAL_DATA_LIST_FILE,
        dataset_path=param.VAL_DATA_ROOT,
        frame_num=param.NUM_FRAMES,
    )
    
    print(f"Validation dataset loaded with {len(val_dataset)} samples.")
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=param.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=val_dataset.collate_fn,
    )
    
    return train_loader, val_loader, age_weights


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


# --- 新增的輔助函數，用於計算梯度的範數 ---
def get_grad_norm(model_part):
    """
    計算一個模型部分所有參數梯度的總L2範數。
    Args:
        model_part (nn.Module): 模型的特定部分，例如 model.extractor。
    Returns:
        float: 梯度的總L2範數。如果沒有梯度，返回0。
    """
    total_norm = 0.0
    for p in model_part.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def lr_lambda(current_epoch):
    if current_epoch < param.WARM_UP_EPOCHS:
        # 线性预热
        return float(current_epoch + 1) / float(param.WARM_UP_EPOCHS)
    # 查找当前 epoch 应该对应的衰减因子
    decay_factor = 1.0
    for milestone in sorted(param.LR_DECAY_STEPS, reverse=True):
        if current_epoch >= milestone:
            decay_factor = param.LR_DECAY_FACTOR ** sorted(param.LR_DECAY_STEPS).index(milestone)
            break
    return decay_factor

def train_model():
    device = torch.device(param.DEVICE)
    train_loader, val_loader, _ = prepare_dataloader()
    save_system = Save_system()
    # writer = init_tensorboard() # 如果需要TensorBoard可以取消註釋

    model = AttributeUnlearningModel(
        num_main_classes=param.NUM_SPEAKERS,
        num_attribute_classes=param.NUM_AGE_GROUPS,
        input_channels=3,
        input_size=224
    ).to(device)

    # 【修改點】優化器定義，現在 ArcFace 的權重也需要被優化
    # optimizer_main = optim.Adam(
    #     list(model.classifier.parameters()) + list(model.extractor.parameters()),
    #     lr=param.INITIAL_LR
    # )
    # optimizer_detach = optim.Adam(
    #     list(model.extractor.parameters()) + list(model.aux_network.parameters()),
    #     lr=param.LEARNING_RATE_DETACH
    # )

    optimizer = optim.Adam(model.parameters(), lr=param.INITIAL_LR)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # scheduler_main = optim.lr_scheduler.LambdaLR(optimizer_main, lr_lambda=warmup_lambda)
    # scheduler_detach = optim.lr_scheduler.LambdaLR(optimizer_detach, lr_lambda=warmup_lambda)

    criterion_main = nn.CrossEntropyLoss()

    for epoch in range(param.EPOCHS):
        model.train()
        
        # 【新增】用於累加整個 epoch 準確率的變量
        total_correct_id = 0
        total_samples_id = 0
        index = 0
        top1 = 0
        
        # ... (其他損失和準確率的累加變量保持不變) ...
        total_loss_id = 0.0
        total_detach_loss = 0.0
        total_loss_recon = 0.0
        total_detach_loss_y = 0.0
        total_detach_loss_age = 0.0
        total_acc_age = 0.0
        total_detach_acc_ID = 0.0
        
        current_alpha = param.ALPHA
        # for e_threshold, alpha_val in sorted(param.ALPHA_SCHEDULE.items()):
        #     if epoch + 1 >= e_threshold:
        #         current_alpha = alpha_val

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch")
        for batch_idx, (mels, identity_labels, age_labels) in enumerate(pbar):
            mels = mels.to(device)
            identity_labels = identity_labels.to(device)
            age_labels = age_labels.to(device)

            # optimizer_main.zero_grad()
            # optimizer_detach.zero_grad()
            optimizer.zero_grad()

            # 【修改點】模型前向傳播，接收 loss 和 acc
            # output, h = model(mels, mode="train", id_label=identity_labels)
            loss_main, acc_main, h = model(mels, mode="train", id_label=identity_labels)
            
            # loss_main = criterion_main(output, identity_labels)

            # ... (輔助網絡的損失計算保持不變) ...
            loss_detach, loss_recon, pred_y, loss_y, pred_age, pred_detach_age_loss = model.aux_network(h, mels, identity_labels, age_labels, current_alpha, param.BETA, param.GAMMA)
            
            # --- 總損失計算和反向傳播保持不變 ---
            total_loss_for_extractor = loss_main + loss_detach
            total_loss_for_extractor.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            # optimizer_main.step()
            # optimizer_detach.step()
            optimizer.step()

            # --- 累加其他損失和準確率（用於日誌）---
            total_loss_id += loss_main.item()
            total_detach_loss += loss_detach.item()
            total_loss_recon += loss_recon.item()
            total_detach_loss_y += loss_y.item()
            total_detach_loss_age += pred_detach_age_loss.item()
            
            # max_index_of_id = torch.argmax(output, dim=1)
            # total_correct_id += (max_index_of_id == identity_labels).sum().item()
            # total_samples_id += identity_labels.size(0)
            batch_size = identity_labels.size(0)
            total_correct_id += (acc_main.item() / 100.0) * batch_size
            total_samples_id += batch_size

            max_index_of_age = torch.argmax(pred_age, dim=1)
            total_acc_age += (max_index_of_age == age_labels).sum().item()
            
            max_index_of_detach_ID = torch.argmax(pred_y, dim=1)
            total_detach_acc_ID += (max_index_of_detach_ID == identity_labels).sum().item()

            pbar.set_postfix({
                'L_id': f'{loss_main.item():.4f}',
                'Acc_id': f'{acc_main.item():.2f}%',
                'L_detach': f'{loss_detach.item():.4f}',
            })
        
        # --- 計算整個 epoch 的平均損失和準確率 ---
        avg_loss_id = total_loss_id / len(train_loader)
        
        # 【修改點】計算 epoch 的平均 ID 準確率
        avg_acc_id = (total_correct_id / total_samples_id * 100.0) if total_samples_id > 0 else 0.0

        # ... (其他平均值的計算保持不變) ...
        avg_detach_loss = total_detach_loss / len(train_loader)
        avg_loss_recon = total_loss_recon / len(train_loader)
        avg_detach_loss_y = total_detach_loss_y / len(train_loader)
        avg_detach_loss_age = total_detach_loss_age / len(train_loader)
        avg_acc_age = total_acc_age / (len(train_loader) * param.BATCH_SIZE)
        avg_detach_acc_ID = total_detach_acc_ID / (len(train_loader) * param.BATCH_SIZE)
        avg_acc_id = (total_correct_id / total_samples_id) * 100.0 if total_samples_id > 0 else 0.0

        # scheduler_main.step()
        # scheduler_detach.step()
        # scheduler.step()

        model.eval()
        val_eer, val_mDCF = evaluate(model, val_loader, device)

        # 【修改點】在保存和打印日誌時，使用新的 avg_acc_id
        # current_main_lr = optimizer_main.param_groups[0]['lr']
        # current_detach_lr = optimizer_detach.param_groups[0]['lr']

        current_main_lr = optimizer.param_groups[0]['lr']
        current_detach_lr = optimizer.param_groups[0]['lr']

        save_system.write_result_to_file(
            param.SCORE_DIR,
            "result",
            (epoch + 1, current_main_lr, current_detach_lr, current_alpha, avg_loss_id, avg_acc_id, avg_detach_loss, avg_acc_age, avg_detach_loss_age, avg_loss_recon, avg_detach_loss_y, avg_detach_acc_ID, val_eer, val_mDCF)
        )

        print(f"Epoch {epoch + 1}/{param.EPOCHS} completed. "
              f"主要任務損失: {avg_loss_id:.4f}, "
              f"主要任務準確率: {avg_acc_id:.4f}%, " # <-- 使用新的準確率
              f"輔助任務總損失: {avg_detach_loss:.4f}, "
              f"輔助任務Age損失: {avg_detach_loss_age:.4f}, "
              f"輔助任務Age準確率: {avg_acc_age:.4f}, "
              f"輔助任務重建損失: {avg_loss_recon:.4f}, "
              f"輔助任務ID損失: {avg_detach_loss_y:.4f}, "
              f"輔助任務ID準確率: {avg_detach_acc_ID:.4f}, "
              f"Val EER: {val_eer:.4f}, "
              f"Val minDCF: {val_mDCF:.4f}")

        if epoch == param.EPOCHS - 1:
            save_system.save_model(model, epoch + 1)

# train.py -> train_model()

# def train_model():
#     device = torch.device(param.DEVICE)
#     train_loader, val_loader, _ = prepare_dataloader()
#     save_system = Save_system()

#     model = AttributeUnlearningModel(
#         num_main_classes=param.NUM_SPEAKERS,
#         num_attribute_classes=param.NUM_AGE_GROUPS,
#         input_channels=3,
#         input_size=224
#     ).to(device)

#     # --- 【核心】只用一個優化器，管理所有參數 ---
#     optimizer = optim.Adam(model.parameters(), lr=param.INITIAL_LR)
    
#     criterion = nn.CrossEntropyLoss()
    
#     total_acc = []

#     for epoch in range(param.EPOCHS):
#         model.train()
#         total_loss_id = 0.0
#         total_correct_id = 0
#         total_samples = 0

#         pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch")
#         for batch_idx, (mels, identity_labels, age_labels) in enumerate(pbar):
#             mels = mels.to(device)
#             identity_labels = identity_labels.to(device)

#             optimizer.zero_grad()

#             # --- 【核心】只計算和反向傳播主要任務損失 ---
#             loss_main, acc_main, h = model(mels, mode="train", id_label=identity_labels)
#             loss_detach, loss_recon, pred_y, loss_y, pred_age, pred_detach_age_loss = model.aux_network(h, mels, identity_labels, age_labels, current_alpha, param.BETA, param.GAMMA)

#             # output, _ = model(mels, mode="train", id_label=identity_labels)

#             # loss_main = criterion(output, identity_labels)
#             loss_main.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0) # 可以稍微放寬 max_norm
#             optimizer.step()

#             # pred_ID = output.argmax(dim=1)
#             # acc_main = (pred_ID == identity_labels).float().mean() * 100.0

#             # --- 累加損失和準確率 ---
#             total_loss_id += loss_main.item()
            
#             batch_size = identity_labels.size(0)
#             total_correct_id += (acc_main.item() / 100.0) * batch_size
#             total_samples += batch_size

#             pbar.set_postfix({
#                 'L_id': f'{loss_main.item():.4f}',
#                 'Acc_id': f'{acc_main.item():.2f}%',
#             })
        
#         avg_loss_id = total_loss_id / len(train_loader)
#         avg_acc_id = (total_correct_id / total_samples) * 100.0 if total_samples > 0 else 0.0

#         # --- 驗證階段（可選，但強烈推薦）---
#         # 在這個基線階段，我們希望看到 Acc_id 上升的同時，EER 也在下降
#         model.eval()
#         val_eer, val_mDCF = evaluate(model, val_loader, device)

#         # --- 打印和保存日誌 ---
#         current_lr = optimizer.param_groups[0]['lr']
#         print(f"Epoch {epoch + 1}/{param.EPOCHS} completed. LR: {current_lr:.6f}, "
#               f"Avg ID Loss: {avg_loss_id:.4f}, "
#               f"Avg ID Acc: {avg_acc_id:.2f}%, "
#               f"Val EER: {val_eer:.4f}, "
#               f"Val minDCF: {val_mDCF:.4f}")
        
#         save_system.write_result_to_file(
#             param.SCORE_DIR,
#             "result",
#             (epoch + 1, param.INITIAL_LR, 0.0, param.ALPHA, avg_loss_id, avg_acc_id, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, val_eer, val_mDCF)
#         )



if __name__ == '__main__':
    train_model()