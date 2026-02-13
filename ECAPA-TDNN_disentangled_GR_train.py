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

def build_eval_dataset(audio_dirs, audio_datalist, audio_meta_dir, gender, max_pairs=20000):
    def find_audio_path(relative_path):
        for audio_dir in audio_dirs:
            audio_path = Path(audio_dir) / relative_path
            if audio_path.exists():
                return str(audio_path)
        raise FileNotFoundError(f"{relative_path} not found in audio_dirs")
    
    meta_data = pd.read_csv(audio_meta_dir, sep=',')
    # get the SpeakerID and Gender columns
    meta_data = meta_data[['SpeakerID', 'Gender']]
    gender = gender  # 只選指定性別進行評估

    datalist = []
    with open(audio_datalist, "r") as f:
        lines = f.readlines()[:max_pairs]

    for line in lines:
        line = line.strip().split(" ")
        is_same = int(line[0]); spk1_rel = line[1]; spk2_rel = line[2]
        spk1_id = spk1_rel.split("/")[0]; spk2_id = spk2_rel.split("/")[0]
        # 如果都是指定性別才加入評估清單
        spk1_gender = meta_data[meta_data['SpeakerID'] == spk1_id]['Gender'].iloc[0]
        spk2_gender = meta_data[meta_data['SpeakerID'] == spk2_id]['Gender'].iloc[0]
        if spk1_gender == gender and spk2_gender == gender:
            spk1_path = find_audio_path(spk1_rel); spk2_path = find_audio_path(spk2_rel)
            datalist.append((is_same, spk1_id, spk2_id, spk1_path, spk2_path))
            
    print(f"總共找到 {len(datalist)} 對符合性別條件的評估語音對")
    return datalist

gender = "f"  # 只訓練女性說話者

train_dataset = Vox2Dataset(
    audio_dir=DATASET_INFO[dataset]['AUDIO_DIR'],
    audio_meta_dir=DATASET_INFO[dataset]['AUDIO_META_DIR'],
    gender="f"
)

eval_dataset = build_eval_dataset(
    audio_dirs=DATASET_INFO['VoxCeleb1'][val_dataset]['AUDIO_DIR'],
    audio_datalist=DATASET_INFO['VoxCeleb1'][val_dataset]['AUDIO_DATALIST'],
    audio_meta_dir=DATASET_INFO['VoxCeleb1']['AUDIO_META_DIR'],
    gender="f"
)

# 使用自定義取樣器 (一次抽出三組索引: 0, 1, 2)
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
criterion = JFELoss(lambda_entropy=0.1, lambda_mapc=0.0, lambda_recon=1.0)

# GR 權重
LAMBDA_GR = 1.0

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
# 3. 訓練迴圈
# ==========================================
print(f"Start GR training on {device}...")

for epoch in range(EPOCHS):
    model.train()
    
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
        
        # 1. 拆分為三組數據 (0: 幼年, 1: 壯年Anchor, 2: 老年)
        e0, s0, a0 = embs[:BATCH_SIZE], spks[:BATCH_SIZE], ages[:BATCH_SIZE]
        e1, s1, a1 = embs[BATCH_SIZE:2*BATCH_SIZE], spks[BATCH_SIZE:2*BATCH_SIZE], ages[BATCH_SIZE:2*BATCH_SIZE]
        e2, s2, a2 = embs[2*BATCH_SIZE:], spks[2*BATCH_SIZE:], ages[2*BATCH_SIZE:]

        # 2. 分別計算三個梯度 (針對 encoder 參數)
        def get_grads_and_loss(e, s, a):
            out = model(e, mode="train")
            loss, l_dict = criterion(out, s, a)
            # 獲取 encoder 的梯度
            grads = torch.autograd.grad(
                loss, model.encoder.parameters(), create_graph=True, retain_graph=True
            )
            return grads, loss, l_dict, out

        g0, loss0, dict0, out0 = get_grads_and_loss(e0, s0, a0)
        g1, loss1, dict1, out1 = get_grads_and_loss(e1, s1, a1)
        g2, loss2, dict2, out2 = get_grads_and_loss(e2, s2, a2)

        # 3. 實作論文 Equation 1: 梯度正則化 (使用 Cosine Similarity 確保穩定)
        gr_step_loss = 0
        for grad0, grad1, grad2 in zip(g0, g1, g2):
            g0_f, g1_f, g2_f = grad0.view(-1), grad1.view(-1), grad2.view(-1)
            
            # 垂直對齊 (噪音 vs 乾淨)
            sim01 = F.cosine_similarity(g0_f, g1_f, dim=0, eps=1e-8)
            sim21 = F.cosine_similarity(g2_f, g1_f, dim=0, eps=1e-8)
            # 水平對齊 (噪音 vs 噪音)
            sim02 = F.cosine_similarity(g0_f, g2_f, dim=0, eps=1e-8)
            
            gr_step_loss += (1.0 - sim01) + (1.0 - sim21) + (1.0 - sim02)

        # 4. 總合損失與優化
        total_loss = (loss0 + loss1 + loss2) + LAMBDA_GR * gr_step_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0) # 梯度剪裁
        optimizer.step()
        
        # --- 5. 數據紀錄 ---
        total_train_loss += total_loss.item()
        train_gr_loss += gr_step_loss.item()
        
        # 累積三組的子損失平均
        for d in [dict0, dict1, dict2]:
            train_loss_spkr += d['loss_spkr'] / 3
            train_loss_age += d['loss_age'] / 3
            train_entropy_age += d['entropy_age'] / 3
            train_entropy_spkr += d['entropy_spkr'] / 3
            train_mapc += d['mapc'] / 3
            train_recon_loss += d['loss_recon'] / 3
        
        # 計算準確率 (三組樣本合計)
        all_logits_spk = torch.cat([out0['logits_spkr_main'], out1['logits_spkr_main'], out2['logits_spkr_main']])
        all_logits_age = torch.cat([out0['logits_age_main'], out1['logits_age_main'], out2['logits_age_main']])
        all_targets_spk = torch.cat([s0, s1, s2])
        all_targets_age = torch.cat([a0, a1, a2])
        
        _, pred_s = torch.max(all_logits_spk, 1)
        _, pred_a = torch.max(all_logits_age, 1)
        correct_spk += (pred_s == all_targets_spk).sum().item()
        correct_age += (pred_a == all_targets_age).sum().item()
        
        # 洩漏監控
        all_logits_age_sub = torch.cat([out0['logits_age_sub'], out1['logits_age_sub'], out2['logits_age_sub']])
        all_logits_spk_sub = torch.cat([out0['logits_spkr_sub'], out1['logits_spkr_sub'], out2['logits_spkr_sub']])
        _, pred_as = torch.max(all_logits_age_sub, 1)
        _, pred_ss = torch.max(all_logits_spk_sub, 1)
        correct_age_sub += (pred_as == all_targets_age).sum().item()
        correct_id_sub += (pred_ss == all_targets_spk).sum().item()
        
        total_samples += (BATCH_SIZE * 3)

    # ==========================================
    # 4. 驗證與日誌記錄
    # ==========================================
    num_steps = len(train_loader)
    avg_loss = total_train_loss / num_steps
    avg_gr = train_gr_loss / num_steps
    avg_spk_l = train_loss_spkr / num_steps
    avg_age_l = train_loss_age / num_steps
    avg_ent_a = train_entropy_age / num_steps
    avg_ent_s = train_entropy_spkr / num_steps
    avg_mapc = train_mapc / num_steps
    avg_recon = train_recon_loss / num_steps
    
    acc_spk = 100 * correct_spk / total_samples
    acc_age = 100 * correct_age / total_samples
    acc_age_leak = 100 * correct_age_sub / total_samples
    acc_id_leak = 100 * correct_id_sub / total_samples
    
    eer_before, eer_after, _, _, _ = eval_network(model, eval_dataset)

    if best_EER > eer_after:
        best_EER = eer_after
        torch.save(model.state_dict(), f'./checkpoints/{val_dataset}_best_model.pth')
        print(f"儲存最佳模型 EER: {best_EER * 100:.2f}%")

    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f} | GR: {avg_gr:.4f} | Spk Acc: {acc_spk:.2f}% | Age Acc: {acc_age:.2f}%")
    print(f"|| Before EER: {eer_before * 100:.2f}% | After EER: {eer_after * 100:.2f}%")

    # TensorBoard
    writer.add_scalar("Loss/Total", avg_loss, epoch)
    writer.add_scalar("Loss/GR", avg_gr, epoch)
    writer.add_scalar("Accuracy/Spk", acc_spk, epoch)
    writer.add_scalar("EER/After", eer_after, epoch)

    # CSV
    csv_writer.writerow([
        epoch + 1, avg_loss, avg_spk_l, avg_age_l, 
        avg_ent_a, avg_ent_s, avg_mapc, avg_recon, 
        avg_gr, acc_spk, acc_age, acc_age_leak, acc_id_leak, eer_before, eer_after
    ])
    csv_file.flush()

writer.close()
csv_file.close()
print("GR Training Finished!")