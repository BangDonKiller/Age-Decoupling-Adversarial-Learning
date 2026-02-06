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

def build_eval_dataset(audio_dirs, audio_meta_dir, max_pairs=20000):
    def find_audio_path(relative_path):
        for audio_dir in audio_dirs:
            audio_path = Path(audio_dir) / relative_path
            if audio_path.exists():
                return str(audio_path)
        raise FileNotFoundError(f"{relative_path} not found in audio_dirs")

    datalist = []
    with open(audio_meta_dir, "r") as f:
        lines = f.readlines()[:max_pairs]

    for line in lines:
        line = line.strip().split(" ")
        is_same = int(line[0]); spk1_rel = line[1]; spk2_rel = line[2]
        spk1_path = find_audio_path(spk1_rel); spk2_path = find_audio_path(spk2_rel)
        spk1_id = spk1_rel.split("/")[0]; spk2_id = spk2_rel.split("/")[0]
        datalist.append((is_same, spk1_id, spk2_id, spk1_path, spk2_path))
    return datalist

train_dataset = Vox2Dataset(
    audio_dir=DATASET_INFO[dataset]['AUDIO_DIR'],
    audio_meta_dir=DATASET_INFO[dataset]['AUDIO_META_DIR']
)

eval_dataset = build_eval_dataset(
    audio_dirs=DATASET_INFO['VoxCeleb1'][val_dataset]['AUDIO_DIR'],
    audio_meta_dir=DATASET_INFO['VoxCeleb1'][val_dataset]['AUDIO_META_DIR']
)

# 使用自定義取樣器
train_sampler = AgeGroupBatchSampler(train_dataset, batch_size=BATCH_SIZE)
train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0)

def eval_network(model, datalist):
    model.eval()
    before_scores, after_scores, labels = [], [], []
    before_embs, after_embs, final_ids = [], [], []

    with torch.no_grad():
        for is_same, id1, id2, path1, path2 in tqdm(datalist, desc="Evaluating"):
            emb1, _ = torchaudio.load(path1); emb2, _ = torchaudio.load(path2)
            emb1, emb2 = emb1.to(device), emb2.to(device)
            out1 = model(emb1, mode="test"); out2 = model(emb2, mode="test")

            h1 = F.normalize(out1["spkr_emb"], p=2, dim=1)
            h2 = F.normalize(out2["spkr_emb"], p=2, dim=1)
            before_scores.append(F.cosine_similarity(h1, h2).cpu())

            w1 = F.normalize(out1["w_spkr"], p=2, dim=1)
            w2 = F.normalize(out2["w_spkr"], p=2, dim=1)
            after_scores.append(F.cosine_similarity(w1, w2).cpu())

            labels.append(is_same)
            after_embs.append(out1["w_spkr"].cpu()); after_embs.append(out2["w_spkr"].cpu())
            before_embs.append(out1["spkr_emb"].cpu()); before_embs.append(out2["spkr_emb"].cpu())
            final_ids.extend([id1, id2])

    final_labels = torch.tensor(labels).numpy()
    eer_before = compute_eer(torch.cat(before_scores).numpy(), final_labels)
    eer_after = compute_eer(torch.cat(after_scores).numpy(), final_labels)
    return eer_before, eer_after, torch.cat(before_embs, dim=0), torch.cat(after_embs, dim=0), final_ids

# ==========================================
# 2. 模型初始化
# ==========================================
model = JFENetwork(MODEL_ID, input_dim=192, spk_dim=256, age_dim=256, num_speakers=5990, num_age_groups=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# 這裡將 lambda_hsic 設為 0.0，因為你已經拔掉了
criterion = JFELoss(lambda_entropy=0.1, lambda_mapc=0.0, lambda_recon=1.0)

LAMBDA_GR = 0.5 

log_dir = "logs/jfe_gr"; os.makedirs(log_dir, exist_ok=True)
checkpoint_dir = "checkpoints"; os.makedirs(checkpoint_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)
csv_file = open(os.path.join(log_dir, f"{val_dataset}_metrics.csv"), mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "epoch", "train_loss", "train_loss_spkr", "train_loss_age", 
    "train_entropy_age", "train_entropy_spkr", "train_mapc", 
    "train_recon_loss", "train_gr_loss", "train_spk_acc", 
    "train_age_acc", "train_age_leak", "train_id_leak", "val_eer_before", "val_eer_after"
])

EPOCHS = 10
best_EER = float('inf')

# ==========================================
# 3. 訓練迴圈 (核心實作 GR)
# ==========================================
print(f"Start GR training on {device}...")

for epoch in range(EPOCHS):
    model.train()
    
    # 紀錄初始化
    total_train_loss = 0.0
    train_loss_spkr, train_loss_age = 0.0, 0.0
    train_entropy_age, train_entropy_spkr = 0.0, 0.0
    train_mapc, train_recon_loss = 0.0, 0.0
    train_gr_loss = 0.0
    
    correct_spk, correct_age = 0, 0
    correct_age_sub, correct_id_sub = 0, 0
    total_samples = 0
    
    for combined_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        embs, spks, ages = combined_batch
        embs, spks, ages = embs.to(device), spks.to(device), ages.to(device)
        
        emb_anc, spk_anc, age_anc = embs[:BATCH_SIZE], spks[:BATCH_SIZE], ages[:BATCH_SIZE]
        emb_oth, spk_oth, age_oth = embs[BATCH_SIZE:], spks[BATCH_SIZE:], ages[BATCH_SIZE:]

        # --- 1. 壯年組梯度 ---
        outputs_anc = model(emb_anc, mode="train")
        loss_anc, loss_dict_anc = criterion(outputs_anc, spk_anc, age_anc)
        
        optimizer.zero_grad()
        grads_anc = torch.autograd.grad(
            loss_anc, model.encoder.parameters(), create_graph=True, retain_graph=True
        )

        # --- 2. 變異組梯度 ---
        outputs_oth = model(emb_oth, mode="train")
        loss_oth, loss_dict_oth = criterion(outputs_oth, spk_oth, age_oth)
        
        grads_oth = torch.autograd.grad(
            loss_oth, model.encoder.parameters(), create_graph=True, retain_graph=True
        )

        # --- 3. 梯度正則化 (GR) ---
        gr_loss = 0
        for g0, gk in zip(grads_anc, grads_oth):
            # gr_loss -= torch.sum(g0 * gk)
            # 1. 將梯度拉平 (Flatten)
            g0_flat = g0.contiguous().view(-1)
            gk_flat = gk.contiguous().view(-1)
            
            # 2. 計算餘弦相似度 (Cosine Similarity)
            # 我們希望相似度趨近 1，所以 Loss 設為 (1 - similarity)
            # 加上 1e-8 防止除以 0
            similarity = F.cosine_similarity(g0_flat, gk_flat, dim=0, eps=1e-8)
            gr_loss += (1.0 - similarity) 

        # --- 4. 總合損失與更新 ---
        total_loss = (loss_anc + loss_oth) + LAMBDA_GR * gr_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # --- 5. 數據紀錄 ---
        total_train_loss += total_loss.item()
        train_gr_loss += gr_loss.item()
        
        # 累積兩邊的子損失 (取平均以反映 Batch 狀態)
        train_loss_spkr += (loss_dict_anc['loss_spkr'] + loss_dict_oth['loss_spkr']) / 2
        train_loss_age += (loss_dict_anc['loss_age'] + loss_dict_oth['loss_age']) / 2
        train_entropy_age += (loss_dict_anc['entropy_age'] + loss_dict_oth['entropy_age']) / 2
        train_entropy_spkr += (loss_dict_anc['entropy_spkr'] + loss_dict_oth['entropy_spkr']) / 2
        train_mapc += (loss_dict_anc['mapc'] + loss_dict_oth['mapc']) / 2
        train_recon_loss += (loss_dict_anc['loss_recon'] + loss_dict_oth['loss_recon']) / 2
        
        # 計算準確率 (以 Anchor + Other 合計)
        outputs_combined_logits_spk = torch.cat([outputs_anc['logits_spkr_main'], outputs_oth['logits_spkr_main']])
        outputs_combined_logits_age = torch.cat([outputs_anc['logits_age_main'], outputs_oth['logits_age_main']])
        targets_combined_spk = torch.cat([spk_anc, spk_oth])
        targets_combined_age = torch.cat([age_anc, age_oth])
        
        _, pred_s = torch.max(outputs_combined_logits_spk, 1)
        _, pred_a = torch.max(outputs_combined_logits_age, 1)
        correct_spk += (pred_s == targets_combined_spk).sum().item()
        correct_age += (pred_a == targets_combined_age).sum().item()
        
        # 洩漏監控
        _, pred_a_sub = torch.max(torch.cat([outputs_anc['logits_age_sub'], outputs_oth['logits_age_sub']]), 1)
        _, pred_s_sub = torch.max(torch.cat([outputs_anc['logits_spkr_sub'], outputs_oth['logits_spkr_sub']]), 1)
        correct_age_sub += (pred_a_sub == targets_combined_age).sum().item()
        correct_id_sub += (pred_s_sub == targets_combined_spk).sum().item()
        
        total_samples += (BATCH_SIZE * 2)

    # ==========================================
    # 4. 驗證與日誌
    # ==========================================
    avg_loss = total_train_loss / len(train_loader)
    avg_gr_loss = train_gr_loss / len(train_loader)
    avg_spk_loss = train_loss_spkr / len(train_loader)
    avg_age_loss = train_loss_age / len(train_loader)
    avg_ent_age = train_entropy_age / len(train_loader)
    avg_ent_spk = train_entropy_spkr / len(train_loader)
    avg_mapc = train_mapc / len(train_loader)
    avg_recon = train_recon_loss / len(train_loader)
    
    acc_spk = 100 * correct_spk / total_samples
    acc_age = 100 * correct_age / total_samples
    acc_age_leak = 100 * correct_age_sub / total_samples
    acc_id_leak = 100 * correct_id_sub / total_samples
    
    eer_before, eer_after, _, _, _ = eval_network(model, eval_dataset)

    if best_EER > eer_after:
        best_EER = eer_after
        torch.save(model.state_dict(), f'./checkpoints/{val_dataset}_best_model.pth')
        print(f"儲存最佳模型 EER: {best_EER * 100:.2f}%")

    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f} | GR: {avg_gr_loss:.4f} | Spk Acc: {acc_spk:.2f}% | Age Acc: {acc_age:.2f}% | Before EER: {eer_before * 100:.2f}% | After EER: {eer_after * 100:.2f}%")

    # TensorBoard
    writer.add_scalar("Loss/Total", avg_loss, epoch)
    writer.add_scalar("Loss/GR", avg_gr_loss, epoch)
    writer.add_scalar("Accuracy/Spk", acc_spk, epoch)
    writer.add_scalar("EER/After", eer_after, epoch)

    # CSV
    csv_writer.writerow([
        epoch + 1, avg_loss, avg_spk_loss, avg_age_loss, 
        avg_ent_age, avg_ent_spk, avg_mapc, avg_recon, 
        avg_gr_loss, acc_spk, acc_age, acc_age_leak, acc_id_leak, eer_before, eer_after
    ])
    csv_file.flush()

writer.close()
csv_file.close()
print("GR Training Finished!")