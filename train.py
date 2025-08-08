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
        torch.backends.cudnn.benchmark = False # 如果輸入大小不變，可以設為True加速

set_seed(param.RANDOM_SEED)

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
        # pin_memory=True if param.DEVICE == 'cuda' else False,
        drop_last=True,  # 確保每個批次的數據量相同(會丟棄最後一個不完整批次)
        collate_fn=train_dataset.collate_fn,  # 使用自定義的 collate_fn 來處理變長序列
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
        collate_fn=val_dataset.collate_fn,  # 使用自定義的 collate_fn 來處理變長序列
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

# --- 訓練主函數 ---
def train_model():
    device = torch.device(param.DEVICE)
    
    # --- 1. 數據加載器 ---
    train_loader, val_loader, age_weights = prepare_dataloader()
    save_system = Save_system()  # 初始化保存系統，確保目錄存在並創建初始文件
    writer = init_tensorboard()  # 初始化 TensorBoard 日誌

    # --- 2. 模型、優化器、損失函數 ---
    model = AttributeUnlearningModel(
        num_main_classes=param.NUM_SPEAKERS,
        num_attribute_classes=param.NUM_AGE_GROUPS,
        input_channels=3,  # 假設輸入是三通道的 (例如 RGB 圖像)
        input_size=224
    ).to(device)
    
    # 論文核心：解耦更新
    # 優化器1：用於主要任務，更新提取器和分類器
    optimizer_main = optim.Adam(
        list(model.extractor.parameters()) + list(model.classifier.parameters()),
        lr=param.INITIAL_LR
    )

    # 優化器2：用於表示分離任務，只更新提取器和輔助網絡
    # 這會對提取器產生一個對抗性的梯度，迫使它忘記屬性信息
    optimizer_detach = optim.Adam(
        list(model.extractor.parameters()) + list(model.aux_network.parameters()),
        lr=param.LEARNING_RATE_DETACH
    )

    # 交叉熵損失用於年齡分類
    ID_criterion_ce = nn.CrossEntropyLoss()     

    # --- 3. 訓練循環 ---   
    best_val_eer = float('inf') 
    step = 0

    id_count = defaultdict(list)
    age_count = defaultdict(list)

    for epoch in range(param.EPOCHS):
        model.train() # 設置模型為訓練模式
        total_loss_id = 0.0
        total_detach_loss = 0.0
        total_loss_recon = 0.0
        total_detach_loss_y = 0.0
        total_detach_loss_age = 0.0

        total_acc_id = 0.0
        total_acc_age = 0.0
        total_detach_acc_ID = 0.0

        # 計算模型預測分布
        batch_id_count = defaultdict(int)
        batch_age_count = defaultdict(int)
        
        # current_lr = optimizer.param_groups[0]['lr']
        # # 檢查學習率是否達到停止條件
        # if current_lr < param.MIN_LR:
        #     break

            
        # 初始化 batch 計數器
        for index in range(param.NUM_AGE_GROUPS):
            batch_age_count[index] = 0
        for index in range(param.NUM_SPEAKERS):
            batch_id_count[index] = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch")
        for batch_idx, (mels, identity_labels, age_labels) in enumerate(pbar):
            mels = mels.to(device)
            identity_labels = identity_labels.to(device)
            age_labels = age_labels.to(device)
            
            optimizer_main.zero_grad()


            # 順向傳播
            main_output, h = model(mels)
            
            # 計算主要任務損失
            loss_main = ID_criterion_ce(main_output, identity_labels)
            
            # 反向傳播並更新
            # `retain_graph=True` 是必須的，因為我們稍後要對同一個計算圖再次反向傳播
            loss_main.backward(retain_graph=True)
            optimizer_main.step()


            loss_detach, loss_recon, pred_y, loss_y, pred_age, pred_detach_age_loss = model.aux_network(h.detach(), mels, identity_labels, age_labels, param.ALPHA, param.BETA, param.GAMMA)

            # 反向傳播並更新
            # 這次的梯度只會影響 optimizer_detach 所管理的參數 (提取器和輔助網絡)
            loss_detach.backward()
            optimizer_detach.step()

            max_index_of_id = torch.argmax(main_output, dim=1)
            max_index_of_age = torch.argmax(pred_age, dim=1)
            max_index_of_detach_ID = torch.argmax(pred_y, dim=1)

            # 計算每個batch，模型預測的ID以及AGE數量 
            for i in range(len(max_index_of_id)):
                batch_id_count[max_index_of_id[i].item()] += 1
                batch_age_count[max_index_of_age[i].item()] += 1


            total_loss_id += loss_main.item()
            total_detach_loss += loss_detach.item()
            total_loss_recon += loss_recon.item()
            total_detach_loss_y += loss_y.item()
            total_detach_loss_age += pred_detach_age_loss.item()

            batch_acc_id = (max_index_of_id == identity_labels).sum().item() / identity_labels.size(0)
            total_acc_id += (max_index_of_id == identity_labels).sum().item()

            batch_acc_age = (max_index_of_age == age_labels).sum().item() / age_labels.size(0)
            total_acc_age += (max_index_of_age == age_labels).sum().item()

            batch_detach_acc_age = (max_index_of_detach_ID == age_labels).sum().item() / age_labels.size(0)
            total_detach_acc_ID = (max_index_of_detach_ID == identity_labels).sum().item()

            pbar.set_postfix({
                'L_id': f'{loss_main.item():.4f}',
                'acc_id': f'{batch_acc_id:.4f}',
                'L_age': f'{loss_detach.item():.4f}',
                'acc_age': f'{batch_acc_age:.4f}',
            })
            
            # --- 4. 保存模型檢查點 ---
            # if step % 640 == 0:
            #     # --- 執行驗證/評估 ---
            #     val_eer, val_mDCF = evaluate(model, val_loader, device)
            #     if val_eer < best_val_eer:
            #         best_val_eer = val_eer
            #         best_checkpoint_path = os.path.join(param.CHECKPOINT_DIR, 'best_adal_model.pth')
            #         torch.save(model.state_dict(), best_checkpoint_path)
            #     log_tensorboard(
            #         writer, 
            #         step,
            #         loss_id.item(),
            #         loss_age.item(),
            #         loss_grl.item(),
            #         None,  # 這裡可以添加準確率或其他指標
            #         val_eer,  # EER
            #         val_mDCF   # minDCF
            #     )
            
            step += param.BATCH_SIZE

        avg_loss_id = total_loss_id / len(train_loader)
        avg_detach_loss = total_detach_loss / len(train_loader)
        avg_loss_recon = total_loss_recon / len(train_loader)
        avg_detach_loss_y = total_detach_loss_y / len(train_loader)
        avg_detach_loss_age = total_detach_loss_age / len(train_loader)

        avg_acc_id = total_acc_id / (len(train_loader) * param.BATCH_SIZE)
        avg_acc_age = total_acc_age / (len(train_loader) * param.BATCH_SIZE)
        avg_detach_acc_ID = total_detach_acc_ID / (len(train_loader) * param.BATCH_SIZE)

        for key, value in batch_id_count.items():
            id_count[key].append(value)
        for key, value in batch_age_count.items():
            age_count[key].append(value)
        
        denominator = len(train_loader) * param.BATCH_SIZE
        
        with open("model_id_prediction.txt", "w") as f:
            f.write("ID, Probability\n")
            for key in sorted(id_count.keys()):  # 按照 key 由小到大排序
                value = id_count[key]
                normalized_values = [f"{v / denominator:.4f}" for v in value]
                line = f"{key}, " + "-> ".join(normalized_values) + "\n"
                f.write(line)
        with open("model_age_prediction.txt", "w") as f:
            f.write("Age Group, Probability\n")
            for key in sorted(age_count.keys()):  # 按照 key 由小到大排序
                value = age_count[key]
                normalized_values = [f"{v / denominator:.4f}" for v in value]
                line = f"{key}, " + "-> ".join(normalized_values) + "\n"
                f.write(line)

        # 在每個 epoch 結束後更新學習率
        # scheduler.step()
            
        # 保存訓練結果到文件
        # save_system.write_result_to_file(
        #     param.SCORE_DIR, 
        #     "result", 
        #     (epoch + 1, current_lr, avg_loss_id, None, None, avg_loss, val_eer, val_mDCF)
        # )
        
        
        # model.eval()  # 設置模型為評估模式
        # val_eer, val_mDCF = evaluate(model, val_loader, device)
        
        save_system.write_result_to_file(
            param.SCORE_DIR, 
            "result", 
            (epoch + 1, param.INITIAL_LR, avg_loss_id, avg_acc_id, avg_detach_loss, avg_acc_age, avg_detach_loss_age, avg_loss_recon, avg_detach_loss_y, avg_detach_acc_ID)  # 保存訓練結果
        )

        print(f"Epoch {epoch + 1}/{param.EPOCHS} completed. "
              f"主要任務損失: {avg_loss_id:.4f}, "
              f"主要任務準確率: {avg_acc_id:.4f}, "
              f"輔助任務總損失: {avg_detach_loss:.4f}, "
              f"輔助任務Age損失: {avg_detach_loss_age:.4f}, "
              f"輔助任務Age準確率: {avg_acc_age:.4f}, "
              f"輔助任務重建損失: {avg_loss_recon:.4f}, "
              f"輔助任務ID損失: {avg_detach_loss_y:.4f}, "
              f"輔助任務ID準確率: {avg_detach_acc_ID:.4f}, "
            #   f"Val EER: {val_eer:.4f}, "
            #   f"Val minDCF: {val_mDCF:.4f}"
            )
        
        if epoch == param.EPOCHS - 1:
            # 在最後一個 epoch 結束時保存模型
            save_system.save_model(model, epoch + 1)


if __name__ == '__main__':
    train_model()
