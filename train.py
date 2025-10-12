import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
from tool.save_system import Save_system
import random
from data.pretrain_loader import Train_loader
from data.finetune_loader import Finetune_loader
from params import param
from model.backbone.speechbrain_resnet import AttributeUnlearningModel
from tool.eval_metric import *
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import torchaudio

# --- 設定隨機種子，確保可重現性 ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(param.RANDOM_SEED)

def prepare_dataloader():
    """
    準備數據集和數據加載器。
    """
    # 預訓練的 train dataset
    train_dataset = Train_loader(
        num_frames=param.NUM_FRAMES,
        data_list_file=param.DATA_LIST_FILE,
        dataset_path=param.DATA_ROOT,
        musan_path=param.MUSAN_DIR,
        rir_path=param.RIR_NOISE_DIR,
        augment=param.AUGMENT,
        num_people=param.NUM_SPEAKERS
    )
    print(f"Training dataset loaded with {len(train_dataset)} samples.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=param.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    # finetune_dataset = Finetune_loader(
    #     dataset_path=param.VAL_DATA_ROOT,
    #     data_list_file=param.FINETUNE_DATA_LIST_FILE,
    #     frame_num=param.NUM_FRAMES,
    #     musan_path=param.MUSAN_DIR,
    #     rir_path=param.RIR_NOISE_DIR,
    # )
    # print(f"Fine-tune dataset loaded with {len(finetune_dataset)} samples.")

    # finetune_loader = DataLoader(
    #     finetune_dataset,
    #     batch_size=param.BATCH_SIZE,
    #     shuffle=True,
    #     num_workers=0,
    #     drop_last=False,
    # )

    return train_loader, None


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

def eval_file_processing(eval_list):
    """
    處理評估文件列表，返回音頻對和標籤。
    
    :param file_list: 包含音頻對和標籤的文件路徑。
    :return: 音頻對和標籤的列表。
    """
    files = []
    lines = open(eval_list).read().splitlines()[:500]
    for line in lines:
        files.append(line.split()[1])
        files.append(line.split()[2])
    setfiles = list(set(files))
    setfiles.sort()
    return setfiles, lines


def evaluate(model, eval_path):
    """
    在驗證集上評估模型性能，計算 EER 和 minDCF。
    
    :param model: 訓練好的模型。
    :param val_loader: 驗證數據加載器。
    :param device: 設備 (CPU 或 GPU)。
    :return: EER 和 minDCF。
    """
    model.eval()  # 設置模型為評估模式

    embeddings = {}
    setfiles, lines = eval_file_processing(param.VAL_DATA_LIST_FILE)

    for idx, file in tqdm(enumerate(setfiles), total = len(setfiles)):
        try:
            for path in eval_path:
                target_path = os.path.join(path, file)
                if os.path.exists(target_path):
                    break
        except:
            print("File not found:", file)
            continue

        audio, _  = torchaudio.load(target_path)
        # Full utterance
        data_1 = torch.FloatTensor(numpy.stack([audio],axis=0)).cuda()

        # Spliited utterance matrix
        max_audio = 200 * 160 + 240
        # 【修正 #1】: 使用 audio.shape[1] 檢查長度
        if audio.shape[1] <= max_audio:
            shortage = max_audio - audio.shape[1]
            # 【修正 #2】: 使用 torch.nn.functional.pad，而不是 numpy.pad
            # F.pad 的參數是 (pad_left, pad_right)，作用在最後一個維度上
            audio = F.pad(audio, (0, shortage), 'circular') # 'circular' 相當於 numpy 的 'wrap'
        
        feats = []
        # 【修正 #3】: 使用 torch.linspace 產生分割點
        startframe = torch.linspace(0, audio.shape[1] - max_audio, 5)
        for asf in startframe:
            start_idx = int(asf)
            feats.append(audio[:, start_idx : start_idx + max_audio])
        
        # 【修正 #4】: 使用 torch.stack，而不是 numpy.stack
        feats_tensor = torch.stack(feats, dim=0)
        # feats_tensor = feats_tensor.squeeze(1)  # 移除多餘的維度
        data_2 = feats_tensor.cuda()
        
        # 3. 提取 Speaker embeddings (保持不變)
        with torch.no_grad():
            raw_embedding_1 = model(data_1, mode="val")
            raw_embedding_2 = model(data_2, mode="val")
            cos_embedding_1 = F.normalize(raw_embedding_1, p=2, dim=1)
            cos_embedding_2 = F.normalize(raw_embedding_2, p=2, dim=1)
            
            embeddings[file] = {
                "raw": [raw_embedding_1, raw_embedding_2],
                "cos": [cos_embedding_1, cos_embedding_2]
            }
    cos_scores, snn_scores, labels  = [], [], []
    
    print("Computing scores...")
    for line in tqdm(lines):
        # 提取兩種 embedding
        embed_1_raw, embed_1_cos = embeddings[line.split()[1]]["raw"], embeddings[line.split()[1]]["cos"]
        embed_2_raw, embed_2_cos = embeddings[line.split()[2]]["raw"], embeddings[line.split()[2]]["cos"]

        # 1. ==== Cosine Similarity Score ==== (保留原有邏輯)
        embedding_11_cos, embedding_12_cos = embed_1_cos
        embedding_21_cos, embedding_22_cos = embed_2_cos
        score_1_cos = torch.mean(torch.matmul(embedding_11_cos, embedding_21_cos.T))
        score_2_cos = torch.mean(torch.matmul(embedding_12_cos, embedding_22_cos.T))
        score_cos = (score_1_cos + score_2_cos) / 2
        cos_scores.append(score_cos.detach().cpu().numpy())
        
        # 2. ==== SNN Classifier Score ==== (【新增】)
        with torch.no_grad():
            embedding_11_raw, embedding_12_raw = embed_1_raw
            embedding_21_raw, embedding_22_raw = embed_2_raw
            # 使用 SNN 分類器計算分數 (輸出通常是 0-1 之間的相似度)
            score_1_snn = model.SNN_classifier(embedding_11_raw, embedding_21_raw)
            score_2_snn = model.SNN_classifier(embedding_12_raw, embedding_22_raw)
            # SNN 分類器可能輸出 (batch, 1) 的形狀，用 squeeze() 去掉多餘維度
            score_2_snn = torch.mean(score_2_snn, dim=0, keepdim=True)
            score_snn = (score_1_snn.squeeze() + score_2_snn.squeeze()) / 2
            snn_scores.append(score_snn.detach().cpu().numpy())

        # 標籤對於兩種方法是相同的
        labels.append(int(line.split()[0]))
        
    # ======================================================================
    # ========== 1. Cosine Similarity Evaluation (保留原有邏輯) ==========
    # ======================================================================
    print("Evaluating Cosine Similarity...")
    _, cos_EER, cos_EER_threshold, _, _ = tuneThresholdfromScore(cos_scores, labels, [1, 0.1])
    fnrs, fprs, thresholds = ComputeErrorRates(cos_scores, labels)
    cos_minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)
    
    # Confusion matrix
    cos_preds = (np.array(cos_scores) >= cos_EER_threshold).astype(int)
    cos_cm = confusion_matrix(labels, cos_preds)
    cos_acc = accuracy_score(labels, cos_preds)
    cos_precision = precision_score(labels, cos_preds, zero_division=0)
    cos_recall = recall_score(labels, cos_preds, zero_division=0)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cos_cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Cosine Similarity Confusion Matrix")
    plt.savefig("cos_confusion_matrix.png")
    plt.close()
    
    # ROC curve
    cos_fpr, cos_tpr, _ = roc_curve(labels, cos_scores)
    cos_roc_auc = auc(cos_fpr, cos_tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(cos_fpr, cos_tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {cos_roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Cosine Similarity ROC")
    plt.legend(loc="lower right")
    plt.savefig("cos_roc_curve.png")
    plt.close()

    # ======================================================================
    # ================ 2. SNN Classifier Evaluation ================
    # ======================================================================
    # print("Evaluating SNN Classifier...")
    # _, snn_EER, snn_EER_threshold, _, _ = tuneThresholdfromScore(snn_scores, labels, [1, 0.1])
    # fnrs_snn, fprs_snn, thresholds_snn = ComputeErrorRates(snn_scores, labels)
    # snn_minDCF, _ = ComputeMinDcf(fnrs_snn, fprs_snn, thresholds_snn, 0.05, 1, 1)

    # # Confusion matrix
    # snn_preds = (np.array(snn_scores) >= snn_EER_threshold).astype(int)
    # snn_cm = confusion_matrix(labels, snn_preds)
    # snn_acc = accuracy_score(labels, snn_preds)
    # snn_precision = precision_score(labels, snn_preds, zero_division=0)
    # snn_recall = recall_score(labels, snn_preds, zero_division=0)

    # plt.figure(figsize=(6, 5))
    # sns.heatmap(snn_cm, annot=True, fmt="d", cmap="Greens", xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
    # plt.xlabel("Predicted")
    # plt.ylabel("True")
    # plt.title("SNN Classifier Confusion Matrix")
    # plt.savefig("snn_confusion_matrix.png")
    # plt.close()

    # # ROC curve
    # snn_fpr, snn_tpr, _ = roc_curve(labels, snn_scores)
    # snn_roc_auc = auc(snn_fpr, snn_tpr)
    # plt.figure(figsize=(6, 5))
    # plt.plot(snn_fpr, snn_tpr, color="darkgreen", lw=2, label=f"ROC curve (AUC = {snn_roc_auc:.4f})")
    # plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("SNN Classifier ROC")
    # plt.legend(loc="lower right")
    # plt.savefig("snn_roc_curve.png")
    # plt.close()

    return cos_EER, cos_acc, cos_precision, cos_recall, cos_EER_threshold
    # return cos_EER, cos_acc, cos_precision, cos_recall, cos_EER_threshold, snn_EER, snn_acc, snn_precision, snn_recall, snn_EER_threshold

def finetune(model, train_loader, eval_path, device, save_system):
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
        cos_EER, cos_acc, cos_precision, cos_recall, cos_EER_threshold = evaluate(model, eval_path, device)
        # cos_EER, cos_acc, cos_precision, cos_recall, cos_EER_threshold, snn_eer,acc_snn, precision_snn, recall_snn, SNN_EER_threshold = evaluate(model, eval_path, device)

        save_system.write_result_to_file(param.FINETUNE_DIR, "finetune", (epoch + 1, avg_loss, avg_acc, cos_EER, cos_acc, cos_precision, cos_recall, cos_EER_threshold
                                                                          , 0.0, 0.0, 0.0, 0.0, 0.0))
        # save_system.write_result_to_file(param.FINETUNE_DIR, "finetune", (epoch + 1, avg_loss, avg_acc, cos_EER, cos_acc, cos_precision, cos_recall, cos_EER_threshold
        #                                                                   , snn_eer, acc_snn, precision_snn, recall_snn, SNN_EER_threshold))

        if cos_EER < best_eer:
            best_eer = cos_EER
            save_system.save_model(model, mode="finetune", state="best")

        if epoch == param.FINETUNE_EPOCHS - 1:
            save_system.save_model(model, mode="finetune", state="last")


    return avg_loss, avg_acc * 100.0, cos_EER, 0.0 # snn_eer

def train_model():
    device = torch.device(param.DEVICE)
    train_loader, finetune_loader = prepare_dataloader()
    save_system = Save_system()
    # writer = init_tensorboard() # 如果需要TensorBoard可以取消註釋

    model = AttributeUnlearningModel(
        num_main_classes=param.NUM_SPEAKERS,
        num_attribute_classes=param.NUM_AGE_GROUPS,
        input_channels=3,
        input_size=224
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=param.INITIAL_LR)

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

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch")
            for batch_idx, (mels, identity_labels, age_labels) in enumerate(pbar):
                mels = mels.to(device)
                identity_labels = identity_labels.to(device)
                age_labels = age_labels.to(device)

                optimizer.zero_grad()

                # 【修改點】模型前向傳播，接收 loss 和 acc
                # output, h = model(mels, mode="train", id_label=identity_labels)
                loss_main, acc_main, h = model(mels, mode="train", id_label=identity_labels)
                total_loss_for_extractor = loss_main
                total_loss_for_extractor.backward()

                optimizer.step()

                # --- 累加其他損失和準確率（用於日誌）---
                total_loss_id += loss_main.item()
                
                batch_size = identity_labels.size(0)
                total_correct_id += (acc_main.item() / 100.0) * batch_size
                total_samples_id += batch_size

                pbar.set_postfix({
                    'L_id': f'{loss_main.item():.4f}',
                    'Acc_id': f'{total_correct_id / total_samples_id * 100:.2f}%',
                })
            
            # --- 計算整個 epoch 的平均損失和準確率 ---
            avg_loss_id = total_loss_id / len(train_loader)
            avg_acc_id = (total_correct_id / total_samples_id) * 100.0 if total_samples_id > 0 else 0.0

            current_main_lr = optimizer.param_groups[0]['lr']
            
            print(f"Epoch {epoch + 1}/{param.EPOCHS} completed. "
                f"預訓練主要任務損失: {avg_loss_id:.4f}, "
                f"主要任務準確率: {avg_acc_id:.4f}%, "
                )
            
            cos_EER, test_acc, precision, recall, EER_threshold = evaluate(model, param.EVAL_PATH)
            print(f"Evaluation - EER: {cos_EER:.4f}, Acc: {test_acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, EER_threshold: {EER_threshold:.4f}")
            save_system.write_result_to_file(
                param.SCORE_DIR,
                "result",
                (epoch + 1, current_main_lr, avg_loss_id, avg_acc_id, cos_EER, test_acc, precision, recall, EER_threshold)
            )
            
            if best_lost > avg_loss_id:
                best_lost = avg_loss_id
                save_system.save_model(model, mode="pretrain", state="best")
            
            if epoch == param.EPOCHS - 1:
                save_system.save_model(model, mode="pretrain", state="last")

    # 二次訓練
    elif not param.PRETRAIN and not param.EVAL:
        # model.load_state_dict(torch.load(param.PRETRAINED_WEIGHTS_PATH))
        finetune_loss, finetune_acc, cos_eer, snn_eer = finetune(model, finetune_loader, param.EVAL_PATH, device, save_system)
        print("After finetune, Loss:", finetune_loss, "Accuracy:", finetune_acc, "EER:", cos_eer, "SNN EER:", snn_eer)
        
    else:
        model.load_state_dict(torch.load(param.PRETRAINED_WEIGHTS_PATH))
        cos_EER, cos_acc, cos_precision, cos_recall, cos_EER_threshold, snn_EER, snn_acc, snn_precision, snn_recall, snn_EER_threshold = evaluate(model, param.EVAL_PATH, device)
        print(f"Evaluation - EER: {cos_EER:.4f}, Acc: {cos_acc:.4f}, Precision: {cos_precision:.4f}, Recall: {cos_recall:.4f}, EER_threshold: {cos_EER_threshold:.4f}")
        print(f"Evaluation - SNN EER: {snn_EER:.4f}, SNN Acc: {snn_acc:.4f}, SNN Precision: {snn_precision:.4f}, SNN Recall: {snn_recall:.4f}, SNN EER_threshold: {snn_EER_threshold:.4f}")

if __name__ == '__main__':
    train_model()