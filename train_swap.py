import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from pathlib import Path
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from model.disentangled_model.model_swap import JFENetworkSwap, JFELossSwap
from tool.EER import compute_eer
from data.vox2_loader import Vox2Dataset
from params.param import DATASET_INFO, BATCH_SIZE, MODEL_ID
import csv

# ==========================================
# 1. 環境設定與種子
# ==========================================
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 2. 評估函式 (沿用你的邏輯)
# ==========================================
def eval_network(model, datalist):
    model.eval()
    before_scores, after_scores, labels = [], [], []
    after_embs, before_embs, final_ids = [], [], []

    with torch.no_grad():
        for is_same, id1, id2, path1, path2, g1, g2 in tqdm(datalist, desc="Evaluating"):
            emb1, _ = torchaudio.load(path1)
            emb2, _ = torchaudio.load(path2)
            emb1, emb2 = emb1.to(device), emb2.to(device)

            # 獲取模型輸出
            out1 = model(emb1, gender=torch.tensor([g1], device=device), mode="test")
            out2 = model(emb2, gender=torch.tensor([g2], device=device), mode="test")

            # Before Disentangle
            h1 = F.normalize(out1["spkr_emb"], p=2, dim=1)
            h2 = F.normalize(out2["spkr_emb"], p=2, dim=1)
            before_scores.append(F.cosine_similarity(h1, h2).cpu())

            # After Disentangle (w_spkr)
            w1 = F.normalize(out1["w_spkr"], p=2, dim=1)
            w2 = F.normalize(out2["w_spkr"], p=2, dim=1)
            after_scores.append(F.cosine_similarity(w1, w2).cpu())

            labels.append(is_same)
            final_ids.extend([id1, id2])
            after_embs.append(out1["w_spkr"].cpu())
            before_embs.append(out1["spkr_emb"].cpu())

    final_labels = torch.tensor(labels).numpy()
    eer_before = compute_eer(torch.cat(before_scores).numpy(), final_labels)
    eer_after = compute_eer(torch.cat(after_scores).numpy(), final_labels)
    
    return eer_before, eer_after, torch.cat(before_embs), torch.cat(after_embs), final_ids

# ==========================================
# 3. 資料載入準備
# ==========================================
def build_eval_dataset(audio_dirs, audio_datalist, audio_meta_dir, max_pairs=20000):
    def find_audio_path(relative_path):
        for audio_dir in audio_dirs:
            audio_path = Path(audio_dir) / relative_path
            if audio_path.exists(): return str(audio_path)
        raise FileNotFoundError(f"{relative_path} not found")
    
    meta_data = pd.read_csv(audio_meta_dir, sep=',')
    meta_data = meta_data[['SpeakerID', 'Gender']]
    gender_map = {"f": 0, "m": 1}

    datalist = []
    with open(audio_datalist, "r") as f:
        lines = f.readlines()[:max_pairs]

    for line in lines:
        line = line.strip().split(" ")
        is_same = int(line[0]); spk1_id = line[1].split("/")[0]; spk2_id = line[2].split("/")[0]
        # 獲取性別標籤
        g1 = gender_map[meta_data[meta_data['SpeakerID'] == spk1_id]['Gender'].iloc[0].lower()]
        g2 = gender_map[meta_data[meta_data['SpeakerID'] == spk2_id]['Gender'].iloc[0].lower()]
        datalist.append((is_same, spk1_id, spk2_id, find_audio_path(line[1]), find_audio_path(line[2]), g1, g2))
            
    print(f"總共找到 {len(datalist)} 對評估語音對 (包含所有性別)")
    return datalist

train_dataset = Vox2Dataset(
    audio_dir=DATASET_INFO['VoxCeleb2']['AUDIO_DIR'],
    audio_meta_dir=DATASET_INFO['VoxCeleb2']['AUDIO_META_DIR']
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset_name = 'Vox-CA20'
eval_data = build_eval_dataset(
    audio_dirs=DATASET_INFO['VoxCeleb1'][val_dataset_name]['AUDIO_DIR'],
    audio_datalist=DATASET_INFO['VoxCeleb1'][val_dataset_name]['AUDIO_DATALIST'],
    audio_meta_dir=DATASET_INFO['VoxCeleb1']['AUDIO_META_DIR']
)

# ==========================================
# 4. 初始化模型、優化器與 Logger
# ==========================================
model = JFENetworkSwap(MODEL_ID, input_dim=192, spk_dim=256, age_dim=256).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = JFELossSwap(lambda_entropy=0.1, lambda_recon=1.0, lambda_ortho=0.0, lambda_swap=1.0, lambda_emb_recon=1.0)

log_dir = "logs/jfe_swap"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

csv_path = os.path.join(log_dir, f"{val_dataset_name}_metrics.csv")
csv_file = open(csv_path, mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "Epoch",
    "Spk_Acc",
    "Age_Acc",
    "Loss_Total",
    "Loss_Spk",
    "Loss_Age",
    "Loss_Entropy",
    "Loss_Recon",
    "Loss_Recon_Emb_Spk",
    "Loss_Recon_Emb_Age",
    "Loss_Swap",
    "Loss_Ortho",
    "EER_Before",
    "EER_After"
])

# ==========================================
# 5. 訓練迴圈 (核心：交換重構)
# ==========================================
EPOCHS = 30
best_eer = float('inf')

for epoch in range(EPOCHS):
    model.train()
    total_metrics = {"loss": 0, "spk": 0, "age": 0, "swap": 0, "ortho": 0, "recon": 0}
    correct_spk, correct_age, total_samples = 0, 0, 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for emb, label_spk, gender, label_age in pbar:
        emb, label_spk, gender, label_age = emb.to(device), label_spk.to(device), gender.to(device), label_age.to(device)
        
        # --- Step 1: 原樣本前向傳播 ---
        outputs = model(emb, gender=gender, mode="train")
        
        # --- Step 2: 特徵交換 (Latent Swapping) ---
        # 隨機打亂 Batch 順序
        idx_shuffle = torch.randperm(emb.size(0)).to(device)
        h_spk = outputs['w_spkr']
        h_age_swapped = outputs['w_age'][idx_shuffle]
        target_age_swapped = label_age[idx_shuffle]
        
        # --- Step 3: 合成新特徵並重新驗證 ---
        # 這裡是創新點：將 A 的身分 + B 的年齡丟進 Decoder 生成合成特徵
        h_spk = outputs['w_spkr']  # 原始的 spk embedding
        h_age = outputs['w_age']   # 原始的 age embedding
        
        x_swap_recon = model.decoder(torch.cat((h_spk, h_age_swapped), dim=1))
        
        # 檢查合成特徵是否真的具備 A 的身分與 B 的年齡
        h_spk_reswap, h_age_reswap = model.forward_encoder(x_swap_recon, gender)
        
        swap_results = {
            'logits_spkr': model.classifier_spkr(h_spk_reswap),
            'logits_age': model.classifier_age(h_age_reswap)
        }

        # --- Step 4: 計算 Loss 並更新 ---
        loss, loss_dict = criterion(outputs, label_spk, label_age, 
                                    swap_results=swap_results, 
                                    target_age_swapped=target_age_swapped,
                                    h_spk_reswap=h_spk_reswap,
                                    h_age_reswap=h_age_reswap,
                                    h_spk_orig=h_spk,
                                    h_age_orig=h_age)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 統計
        total_samples += emb.size(0)
        _, pred_s = torch.max(outputs['logits_spkr_main'], 1)
        _, pred_a = torch.max(outputs['logits_age_main'], 1)
        correct_spk += (pred_s == label_spk).sum().item()
        correct_age += (pred_a == label_age).sum().item()
        
        # 累積 total loss
        total_metrics["loss_total"] = total_metrics.get("loss_total", 0) + loss.item()

        # 累積各項 loss
        for k, v in loss_dict.items():
            total_metrics[k] = total_metrics.get(k, 0) + v
            
        pbar.set_postfix({"Loss": f"{loss.item():.3f}", "SpkAcc": f"{100*correct_spk/total_samples:.1f}%"})
        
    num_batches = len(train_loader)

    avg_loss_total = total_metrics["loss_total"] / num_batches
    avg_loss_spk = total_metrics["loss_spkr"] / num_batches
    avg_loss_age = total_metrics["loss_age"] / num_batches
    avg_loss_entropy = total_metrics["loss_entropy"] / num_batches
    avg_loss_recon = total_metrics["loss_recon"] / num_batches
    avg_loss_emb_spk = total_metrics.get("loss_emb_spk", 0) / num_batches
    avg_loss_emb_age = total_metrics.get("loss_emb_age", 0) / num_batches
    avg_loss_swap = total_metrics["loss_swap"] / num_batches
    avg_loss_ortho = total_metrics["loss_ortho"] / num_batches

    # ==========================================
    # 6. 驗證與存檔
    # ==========================================
    eer_before, eer_after, _, _, _ = eval_network(model, eval_data)
    
    print(f"\n[Epoch {epoch+1}] Val EER Before: {eer_before*100:.2f}% | After: {eer_after*100:.2f}%")
    
    # TensorBoard Logging
    writer.add_scalar("EER/After", eer_after, epoch)
    writer.add_scalar("Loss/Total", loss.item(), epoch)
    writer.add_scalar("Accuracy/Spk", 100*correct_spk/total_samples, epoch)
    
    csv_writer.writerow([
        epoch+1,
        f"{100*correct_spk/total_samples:.2f}",
        f"{100*correct_age/total_samples:.2f}",
        f"{avg_loss_total:.4f}",
        f"{avg_loss_spk:.4f}",
        f"{avg_loss_age:.4f}",
        f"{avg_loss_entropy:.4f}",
        f"{avg_loss_recon:.4f}",
        f"{avg_loss_emb_spk:.4f}",
        f"{avg_loss_emb_age:.4f}",
        f"{avg_loss_swap:.4f}",
        f"{avg_loss_ortho:.4f}",
        f"{eer_before*100:.2f}",
        f"{eer_after*100:.2f}"
    ])
    csv_file.flush() 
    
    if eer_after < best_eer:
        best_eer = eer_after
        torch.save(model.state_dict(), f"checkpoints/{val_dataset_name}_best_swap_model.pth")
        print(f"New Best EER: {best_eer*100:.2f}% Saved!")
    
    if epoch == EPOCHS - 1:
        torch.save(model.state_dict(), f"checkpoints/{val_dataset_name}_final_swap_model.pth")
        print("Final model saved.")

writer.close()
csv_file.close()
print("Training Finished.")