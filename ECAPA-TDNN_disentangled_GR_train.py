import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.disentangled_model.JFE import JFENetwork, JFELoss
from tool.EER import compute_eer
from data.vox2_loader_ver2 import Vox2Dataset, AgeGroupBatchSampler 
from params.param import DATASET_INFO, BATCH_SIZE, MODEL_ID
from torch.utils.tensorboard import SummaryWriter
import os
import csv
from pathlib import Path
import pandas as pd

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ==========================================
# 1. 資料準備與前處理
# ==========================================
dataset = 'VoxCeleb2'
val_dataset = 'Vox-CA20'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

g = torch.Generator()
g.manual_seed(SEED)

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
    audio_dir=DATASET_INFO[dataset]['AUDIO_DIR'],
    audio_meta_dir=DATASET_INFO[dataset]['AUDIO_META_DIR'],
)

eval_dataset = build_eval_dataset(
    audio_dirs=DATASET_INFO['VoxCeleb1'][val_dataset]['AUDIO_DIR'],
    audio_datalist=DATASET_INFO['VoxCeleb1'][val_dataset]['AUDIO_DATALIST'],
    audio_meta_dir=DATASET_INFO['VoxCeleb1']['AUDIO_META_DIR']
)

# 使用三方對齊取樣器
train_sampler = AgeGroupBatchSampler(train_dataset, batch_size=BATCH_SIZE)
train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0)

def eval_network(model, datalist):
    model.eval()
    before_scores, after_scores, labels = [], [], []
    with torch.no_grad():
        for is_same, id1, id2, path1, path2, g1, g2 in tqdm(datalist, desc="Evaluating"):
            emb1, _ = torchaudio.load(path1); emb2, _ = torchaudio.load(path2)
            emb1, emb2 = emb1.to(device), emb2.to(device)
            # 向模型餵入對應的性別 feature
            out1 = model(emb1, gender=torch.tensor([g1], device=device), mode="test")
            out2 = model(emb2, gender=torch.tensor([g2], device=device), mode="test")

            h1 = F.normalize(out1["spkr_emb"], p=2, dim=1); h2 = F.normalize(out2["spkr_emb"], p=2, dim=1)
            before_scores.append(F.cosine_similarity(h1, h2).cpu())
            w1 = F.normalize(out1["w_spkr"], p=2, dim=1); w2 = F.normalize(out2["w_spkr"], p=2, dim=1)
            after_scores.append(F.cosine_similarity(w1, w2).cpu())
            labels.append(is_same)

    final_labels = torch.tensor(labels).numpy()
    eer_before = compute_eer(torch.cat(before_scores).numpy(), final_labels)
    eer_after = compute_eer(torch.cat(after_scores).numpy(), final_labels)
    return eer_before, eer_after

# ==========================================
# 2. 模型初始化
# ==========================================
model = JFENetwork(MODEL_ID, input_dim=192, spk_dim=256, age_dim=256, num_speakers=5990, num_age_groups=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = JFELoss(lambda_entropy=0.1, lambda_mapc=0.0, lambda_recon=1.0)

LAMBDA_GR = 0.1 

# Logger 初始化
log_dir = "logs/jfe_gr"; os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)
csv_file = open(os.path.join(log_dir, f"{val_dataset}_metrics.csv"), mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["epoch", "train_loss", "train_gr_loss", "train_spk_acc", "train_age_acc", "val_eer_before", "val_eer_after"])

EPOCHS = 10
best_EER = float('inf')

# ==========================================
# 3. 訓練迴圈
# ==========================================
print(f"Start GR training on {device}...")

for epoch in range(EPOCHS):
    model.train()
    total_train_loss, train_gr_loss = 0, 0
    correct_spk, correct_age, total_samples = 0, 0, 0
    
    for combined_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        embs, spks, ages, genders = combined_batch
        embs, spks, ages, genders = embs.to(device), spks.to(device), ages.to(device), genders.to(device)
        
        # --- 數據切分 ---
        e0, s0, a0, g0 = embs[:BATCH_SIZE], spks[:BATCH_SIZE], ages[:BATCH_SIZE], genders[:BATCH_SIZE]
        e1, s1, a1, g1 = embs[BATCH_SIZE:2*BATCH_SIZE], spks[BATCH_SIZE:2*BATCH_SIZE], ages[BATCH_SIZE:2*BATCH_SIZE], genders[BATCH_SIZE:2*BATCH_SIZE]
        e2, s2, a2, g2 = embs[2*BATCH_SIZE:], spks[2*BATCH_SIZE:], ages[2*BATCH_SIZE:], genders[2*BATCH_SIZE:]

        # --- 梯度與損失獲取 ---
        def get_grads_and_loss(e, s, a, gf):
            out = model(e, gender=gf, mode="train")
            loss, _ = criterion(out, s, a)
            grads = torch.autograd.grad(loss, model.encoder.parameters(), create_graph=True, retain_graph=True)
            return grads, loss, out

        grads0, loss0, out0 = get_grads_and_loss(e0, s0, a0, g0)
        grads1, loss1, out1 = get_grads_and_loss(e1, s1, a1, g1)
        grads2, loss2, out2 = get_grads_and_loss(e2, s2, a2, g2)

        # --- 梯度正則化 (三方對齊) ---
        gr_step_loss = 0
        for g_0, g_1, g_2 in zip(grads0, grads1, grads2):
            sim01 = F.cosine_similarity(g_0.view(-1), g_1.view(-1), dim=0)
            sim21 = F.cosine_similarity(g_2.view(-1), g_1.view(-1), dim=0)
            sim02 = F.cosine_similarity(g_0.view(-1), g_2.view(-1), dim=0)
            gr_step_loss += (3.0 - (sim01 + sim21 + sim02))

        # --- 總優化 ---
        total_loss = (loss0 + loss1 + loss2) + LAMBDA_GR * gr_step_loss
        optimizer.zero_grad()
        total_loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        # --- 指標統計 ---
        total_train_loss += total_loss.item()
        train_gr_loss += gr_step_loss.item()
        
        # 合計三組的準確率
        combined_spk_logits = torch.cat([out0['logits_spkr_main'], out1['logits_spkr_main'], out2['logits_spkr_main']])
        combined_age_logits = torch.cat([out0['logits_age_main'], out1['logits_age_main'], out2['logits_age_main']])
        combined_spk_targets = torch.cat([s0, s1, s2])
        combined_age_targets = torch.cat([a0, a1, a2])
        
        _, pred_s = torch.max(combined_spk_logits, 1)
        _, pred_a = torch.max(combined_age_logits, 1)
        
        correct_spk += (pred_s == combined_spk_targets).sum().item()
        correct_age += (pred_a == combined_age_targets).sum().item()
        total_samples += (BATCH_SIZE * 3)

    # ==========================================
    # 4. 驗證與紀錄
    # ==========================================
    num_steps = len(train_loader)
    avg_loss = total_train_loss / num_steps
    avg_gr = train_gr_loss / num_steps
    acc_spk = 100 * correct_spk / total_samples
    acc_age = 100 * correct_age / total_samples
    
    eer_before, eer_after = eval_network(model, eval_dataset)

    print(f"Epoch [{epoch+1}] Spk Acc: {acc_spk:.2f}% | Age Acc: {acc_age:.2f}% | Total Loss: {avg_loss:.4f} | GR Loss: {avg_gr:.4f} | Before EER: {eer_before*100:.2f}% | After EER: {eer_after*100:.2f}%")

    if eer_after < best_EER:
        best_EER = eer_after
        torch.save(model.state_dict(), os.path.join(log_dir, f"{val_dataset}_best_disentangled_embeddings.pth"))
        print(f"儲存最佳模型 EER: {best_EER*100:.2f}%")
    if epoch == EPOCHS - 1:
        torch.save(model.state_dict(), os.path.join(log_dir, f"{val_dataset}_last_disentangled_embeddings.pth"))
        print(f"儲存最終模型 EER: {eer_after*100:.2f}%")

    # CSV 紀錄
    csv_writer.writerow([
        epoch + 1, avg_loss, avg_gr, acc_spk, acc_age, eer_before, eer_after
    ])
    csv_file.flush()
    
    # TensorBoard 紀錄
    writer.add_scalar("Loss/Train", avg_loss, epoch)
    writer.add_scalar("Accuracy/Train_Spk", acc_spk, epoch)
    writer.add_scalar("Accuracy/Train_Age", acc_age, epoch)
    writer.add_scalar("EER/Val_Before", eer_before, epoch)
    writer.add_scalar("EER/Val_After", eer_after, epoch)

writer.close()
csv_file.close()
print("GR Training Finished!")