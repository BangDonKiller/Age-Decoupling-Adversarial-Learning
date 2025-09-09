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
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

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

def prepare_dataloader():
    """
    準備數據集和數據加載器。
    """
    # 預訓練的 train dataset
    train_dataset = Voxceleb2_dataset(
        num_frames=param.NUM_FRAMES,
        data_list_file=param.DATA_LIST_FILE,
        dataset_path=param.DATA_ROOT,
        musan_path=param.MUSAN_DIR,
        rir_path=param.RIR_NOISE_DIR,
        augment=param.AUGMENT,
    )
    print(f"Training dataset loaded with {len(train_dataset)} samples.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=param.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        collate_fn=train_dataset.collate_fn,
    )

    # finetune dataset (用於微調 + 驗證)
    finetune_dataset = Voxceleb1_dataset(
        dataset_path=param.VAL_DATA_ROOT,
        data_list_file=param.FINETUNE_DATA_LIST_FILE,
        frame_num=param.NUM_FRAMES,
        musan_path=param.MUSAN_DIR,
        rir_path=param.RIR_NOISE_DIR,
        augment=param.AUGMENT,
    )
    print(f"Fine-tune dataset loaded with {len(finetune_dataset)} samples.")

    # DataLoader
    finetune_loader = DataLoader(
        finetune_dataset,
        batch_size=param.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        collate_fn=finetune_dataset.collate_fn,
    )
    
    eval_dataset = Voxceleb1_dataset(
        dataset_path=param.VAL_DATA_ROOT,
        data_list_file=param.VAL_DATA_LIST_FILE,
        musan_path=param.MUSAN_DIR,
        rir_path=param.RIR_NOISE_DIR,
        frame_num=param.NUM_FRAMES,
        augment=False,  # 評估時不進行增強
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=param.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=finetune_dataset.collate_fn,
    )

    return train_loader, finetune_loader, eval_loader


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
    
    cos_scores = []
    cos_labels = []

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
            
            scores_batch = model.SNN_classifier.forward(embedding1, embedding2)

            # --- 核心修改：計算批次中每對音頻的餘弦相似度 ---
            scores_batch2 = F.cosine_similarity(embedding1, embedding2, dim=1)
            
            all_scores.extend(scores_batch.cpu().numpy().tolist())
            all_labels.extend(label.cpu().numpy().tolist()) # 假設 label 也是一個 Tensor
            
            cos_scores.extend(scores_batch2.cpu().numpy().tolist())
            cos_labels.extend(label.cpu().numpy().tolist())
            
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    cos_scores = np.array(cos_scores)
    cos_labels = np.array(cos_labels)
    
    # ==== 用餘弦相似度計算 EER 和 minDCF ====
    _, cos_EER, cos_EER_threshold, _, _ = tuneThresholdfromScore(cos_scores, cos_labels, [1, 0.1])
    fnrs, fprs, thresholds = ComputeErrorRates(cos_scores, cos_labels)
    minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)
    
    # ==== 用SNN分數計算 EER 和 minDCF ====
    _, snn_eer, SNN_EER_threshold, _, _ = tuneThresholdfromScore(all_scores, all_labels, [1, 0.1])
    fnrs, fprs, thresholds = ComputeErrorRates(all_scores, all_labels)
    minDCF_snn, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)
    
    # ==== 計算 confusion matrix ====
    threshold = cos_EER_threshold  # 可以改成 EER threshold
    preds = (cos_scores >= threshold).astype(int)

    cm = confusion_matrix(cos_labels, preds)
    cos_acc = accuracy_score(cos_labels, preds)
    cos_precision = precision_score(cos_labels, preds, zero_division=0)
    cos_recall = recall_score(cos_labels, preds, zero_division=0)

    # 畫 confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Cosine Similarity Confusion Matrix")
    plt.savefig("cos_confusion_matrix.png")
    plt.close()
    
    # ==== ROC curve ====
    fpr, tpr, _ = roc_curve(cos_labels, cos_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.savefig("cos_roc_curve.png")
    plt.close()


    # ==== SNN confusion matrix ====
    threshold = SNN_EER_threshold
    preds_snn = (all_scores >= threshold).astype(int)

    cm_snn = confusion_matrix(all_labels, preds_snn)
    acc_snn = accuracy_score(all_labels, preds_snn)
    precision_snn = precision_score(all_labels, preds_snn, zero_division=0)
    recall_snn = recall_score(all_labels, preds_snn, zero_division=0)

    # confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_snn, annot=True, fmt="d", cmap="Blues", xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("SNN Confusion Matrix")
    plt.savefig("snn_confusion_matrix.png")
    plt.close()

    # # ROC curve
    fpr_snn, tpr_snn, _ = roc_curve(all_labels, all_scores)
    roc_auc_snn = auc(fpr_snn, tpr_snn)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr_snn, tpr_snn, color="darkorange", lw=2, label=f"SNN ROC curve (AUC = {roc_auc_snn:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("SNN Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.savefig("snn_roc_curve.png")
    plt.close()

    # return cos_EER, cos_acc, cos_precision, cos_recall, cos_EER_threshold
    return cos_EER, cos_acc, cos_precision, cos_recall, cos_EER_threshold, snn_eer,acc_snn, precision_snn, recall_snn, SNN_EER_threshold

def finetune(model, train_loader, eval_loader, device, save_system):
    """
    微調模型以適應新數據集。
    Args:
        model: 要微調的模型。
        train_loader: 訓練數據加載器。
        val_loader: 驗證數據加載器。
        device: 設備 (CPU 或 GPU)。
    """
    
    best_eer = float('inf')

    optimizer = optim.Adam(model.parameters(), lr=param.FINETUNE_LR)

    for epoch in range(param.FINETUNE_EPOCHS):
        model.train()
        # model.extractor.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for audio1, audio2, label in tqdm(train_loader, desc="Finetuning", unit="batch"):
            audio1 = audio1.to(device)
            audio2 = audio2.to(device)
            label = label.to(device)

            embedding1 = model(audio1, mode="finetune") # 輸出形狀: (batch_size, feature_dim)
            embedding2 = model(audio2, mode="finetune") # 輸出形狀: (batch_size, feature_dim)

            output = model.SNN_classifier.forward(embedding1, embedding2)

            loss = F.binary_cross_entropy(output.squeeze(), label.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = (output.view(-1) > 0.5).long()
            correct = (pred == label.view(-1)).sum().item()
            total_correct += correct
            total_samples += label.size(0)


        avg_loss = total_loss / len(train_loader)
        avg_acc = total_correct / total_samples
        
        model.eval()
        cos_EER, cos_acc, cos_precision, cos_recall, cos_EER_threshold, snn_eer,acc_snn, precision_snn, recall_snn, SNN_EER_threshold = evaluate(model, eval_loader, device)

        save_system.write_result_to_file(param.FINETUNE_DIR, "finetune", (epoch + 1, avg_loss, avg_acc, cos_EER, cos_acc, cos_precision, cos_recall, cos_EER_threshold
                                                                          , snn_eer, acc_snn, precision_snn, recall_snn, SNN_EER_threshold))

        if cos_EER < best_eer:
            best_eer = cos_EER
            save_system.save_model(model, epoch + 1, mode="finetune", state="best")

        if epoch == param.FINETUNE_EPOCHS - 1:
            save_system.save_model(model, epoch + 1, mode="finetune", state="last")


    return avg_loss, avg_acc * 100.0, cos_EER, snn_eer

def train_model():
    device = torch.device(param.DEVICE)
    train_loader, finetune_loader, eval_loader = prepare_dataloader()
    save_system = Save_system()
    # writer = init_tensorboard() # 如果需要TensorBoard可以取消註釋

    model = AttributeUnlearningModel(
        num_main_classes=param.NUM_SPEAKERS,
        num_attribute_classes=param.NUM_AGE_GROUPS,
        input_channels=3,
        input_size=224
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=param.INITIAL_LR)

    criterion_main = nn.CrossEntropyLoss()

    # check if gpu is available
    print("torch GPU available:", torch.cuda.is_available())
    
    best_lost = float('inf')
    
    if param.PRETRAIN:
        for epoch in range(param.EPOCHS):
            model.train()
            
            # 【新增】用於累加整個 epoch 準確率的變量
            total_correct_id = 0
            total_samples_id = 0

            total_loss_id = 0.0
            total_detach_loss = 0.0
            total_loss_recon = 0.0
            total_detach_loss_y = 0.0
            total_detach_loss_age = 0.0
            total_acc_age = 0.0
            total_detach_acc_ID = 0.0
            
            current_alpha = param.ALPHA

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch")
            for batch_idx, (mels, identity_labels, age_labels) in enumerate(pbar):
                mels = mels.to(device)
                identity_labels = identity_labels.to(device)
                age_labels = age_labels.to(device)

                optimizer.zero_grad()

                # 【修改點】模型前向傳播，接收 loss 和 acc
                # output, h = model(mels, mode="train", id_label=identity_labels)
                loss_main, acc_main, h = model(mels, mode="train", id_label=identity_labels)
                
                # loss_main = criterion_main(output, identity_labels)

                loss_detach, loss_recon, pred_y, loss_y, pred_age, pred_detach_age_loss = model.aux_network(h, mels, identity_labels, age_labels, current_alpha, param.BETA, param.GAMMA)

                total_loss_for_extractor = loss_main + loss_detach
                total_loss_for_extractor.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
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
                    'Acc_id': f'{total_correct_id / total_samples_id * 100:.2f}%',
                    'L_detach': f'{loss_detach.item():.4f}',
                })
            
            # --- 計算整個 epoch 的平均損失和準確率 ---
            avg_loss_id = total_loss_id / len(train_loader)

            # ... (其他平均值的計算保持不變) ...
            avg_detach_loss = total_detach_loss / len(train_loader)
            avg_loss_recon = total_loss_recon / len(train_loader)
            avg_detach_loss_y = total_detach_loss_y / len(train_loader)
            avg_detach_loss_age = total_detach_loss_age / len(train_loader)
            avg_acc_age = total_acc_age / (len(train_loader) * param.BATCH_SIZE)
            avg_detach_acc_ID = total_detach_acc_ID / (len(train_loader) * param.BATCH_SIZE)
            avg_acc_id = (total_correct_id / total_samples_id) * 100.0 if total_samples_id > 0 else 0.0

            current_main_lr = optimizer.param_groups[0]['lr']
            current_detach_lr = optimizer.param_groups[0]['lr']

            # save_system.write_result_to_file(
            #     param.SCORE_DIR,
            #     "result",
            #     (epoch + 1, current_main_lr, current_detach_lr, current_alpha, avg_loss_id, avg_acc_id, avg_detach_loss, avg_acc_age, avg_detach_loss_age, avg_loss_recon, avg_detach_loss_y, avg_detach_acc_ID)
            # )
            
            print(f"Epoch {epoch + 1}/{param.EPOCHS} completed. "
                f"預訓練主要任務損失: {avg_loss_id:.4f}, "
                f"主要任務準確率: {avg_acc_id:.4f}%, "
                f"輔助任務總損失: {avg_detach_loss:.4f}, "
                f"輔助任務Age損失: {avg_detach_loss_age:.4f}, "
                f"輔助任務Age準確率: {avg_acc_age:.4f}, "
                f"輔助任務重建損失: {avg_loss_recon:.4f}, "
                f"輔助任務ID損失: {avg_detach_loss_y:.4f}, "
                f"輔助任務ID準確率: {avg_detach_acc_ID:.4f}, "
                )
            
            cos_EER, test_acc, precision, recall, EER_threshold = evaluate(model, eval_loader, device)
            print(f"Evaluation - EER: {cos_EER:.4f}, Acc: {test_acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, EER_threshold: {EER_threshold:.4f}")
            save_system.write_result_to_file(
                param.SCORE_DIR,
                "result",
                (epoch + 1, current_main_lr, avg_loss_id, avg_acc_id, cos_EER, test_acc, precision, recall, EER_threshold)
            )
            
            if best_lost > avg_loss_id:
                best_lost = avg_loss_id
                save_system.save_model(model, epoch + 1, mode="pretrain", state="best")
            
            if epoch == param.EPOCHS - 1:
                save_system.save_model(model, epoch + 1, mode="pretrain", state="last")

    else:
        model.load_state_dict(torch.load(param.PRETRAINED_WEIGHTS_PATH))
        finetune_loss, finetune_acc, cos_eer, snn_eer = finetune(model, finetune_loader, eval_loader, device, save_system)
        print("After finetune, Loss:", finetune_loss, "Accuracy:", finetune_acc, "EER:", cos_eer, "SNN EER:", snn_eer)

if __name__ == '__main__':
    train_model()