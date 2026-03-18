import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.disentangled_model.JFE import JFENetwork, JFELoss
from tool.EER import compute_eer
from data.vox2_loader import Vox2Dataset
from params.param import DATASET_INFO, BATCH_SIZE, MODEL_ID
from torch.utils.tensorboard import SummaryWriter
import os
import csv
from pathlib import Path
import pandas as pd
import random

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
        g1 = gender_map[meta_data[meta_data['SpeakerID'] == spk1_id]['Gender'].iloc[0].lower()]
        g2 = gender_map[meta_data[meta_data['SpeakerID'] == spk2_id]['Gender'].iloc[0].lower()]
        datalist.append((is_same, spk1_id, spk2_id, find_audio_path(line[1]), find_audio_path(line[2]), g1, g2))
            
    print(f"總共找到 {len(datalist)} 對評估語音對")
    return datalist

train_dataset = Vox2Dataset(
    audio_dir=DATASET_INFO[dataset]['AUDIO_DIR'],
    audio_meta_dir=DATASET_INFO[dataset]['AUDIO_META_DIR'],
    target_sample_rate=16000,
    suffix=DATASET_INFO[dataset]['audio_suffix']
)

eval_dataset = build_eval_dataset(
    audio_dirs=DATASET_INFO['VoxCeleb1'][val_dataset]['AUDIO_DIR'],
    audio_datalist=DATASET_INFO['VoxCeleb1'][val_dataset]['AUDIO_DATALIST'],
    audio_meta_dir=DATASET_INFO['VoxCeleb1']['AUDIO_META_DIR'],
    max_pairs=20000
)

def eval_network(model, datalist):
    model.eval()
    before_scores, after_scores, labels = [], [], []
    before_embs, after_embs, final_ids = [], [], []

    with torch.no_grad():
        for is_same, id1, id2, path1, path2, g1, g2 in tqdm(datalist, desc="Evaluating"):
            emb1, _ = torchaudio.load(path1); emb2, _ = torchaudio.load(path2)
            emb1, emb2 = emb1.to(device), emb2.to(device)

            out1 = model(emb1, gender=torch.tensor([g1], device=device), mode="test")
            out2 = model(emb2, gender=torch.tensor([g2], device=device), mode="test")

            h1 = F.normalize(out1["spkr_emb"], p=2, dim=1); h2 = F.normalize(out2["spkr_emb"], p=2, dim=1)
            before_scores.append(F.cosine_similarity(h1, h2).cpu())
            before_embs.append(out1["spkr_emb"].cpu()); before_embs.append(out2["spkr_emb"].cpu())

            w1 = F.normalize(out1["w_spkr"], p=2, dim=1); w2 = F.normalize(out2["w_spkr"], p=2, dim=1)
            after_scores.append(F.cosine_similarity(w1, w2).cpu())
            after_embs.append(out1["w_spkr"].cpu()); after_embs.append(out2["w_spkr"].cpu())

            labels.append(is_same); final_ids.extend([id1, id2])

    final_labels = torch.tensor(labels).numpy()
    eer_before = compute_eer(torch.cat(before_scores).numpy(), final_labels)
    eer_after = compute_eer(torch.cat(after_scores).numpy(), final_labels)
    return eer_before, eer_after, torch.cat(before_embs, dim=0), torch.cat(after_embs, dim=0), final_ids

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=g)

model = JFENetwork(MODEL_ID, input_dim=192, spk_dim=256, age_dim=256, num_speakers=5990, num_age_groups=7).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = JFELoss(lambda_entropy=0.1, lambda_mapc=0.0, lambda_recon=1.0, lambda_ortho=0.0, lambda_jacobian=1.0)

log_dir = "logs/jfe_jacobian"; checkpoint_dir = "checkpoints"
os.makedirs(log_dir, exist_ok=True); os.makedirs(checkpoint_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)
csv_file = open(os.path.join(log_dir, f"{val_dataset}_metrics.csv"), mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["epoch", "train_loss", "train_spk_acc", "train_age_acc", "train_jacobian", "val_eer_before", "val_eer_after"])

best_EER = float('inf')

print(f"Start training on {device} with Jacobian Projection...")

for epoch in range(10):
    model.train()
    total_train_loss, train_jacobian_loss = 0.0, 0.0
    correct_spk, correct_age, total_samples = 0, 0, 0
    
    for emb, label_spk, gender, label_age in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        emb, label_spk, gender, label_age = emb.to(device), label_spk.to(device), gender.to(device), label_age.to(device)
        
        # 為了計算 Jacobian，需要手動追蹤 backbone 後的特徵梯度
        with torch.no_grad():
            backbone_feat = model.backbone(emb)
        
        input_feat = torch.cat((backbone_feat, gender.unsqueeze(1).float()), dim=1)
        input_feat.requires_grad_(True) 
        
        # 手動執行網路後半部
        latent = model.encoder(input_feat)
        w_spkr, w_age = latent[:, :model.spk_dim], latent[:, model.spk_dim:]
        logits_spkr_main = model.classifier_spkr(w_spkr)
        logits_age_main = model.classifier_age(w_age)
        logits_age_sub = model.classifier_age(w_spkr)
        logits_spkr_sub = model.classifier_spkr(w_age)
        x_recon = model.decoder(torch.cat((w_spkr, w_age), dim=1))
        
        outputs = {
            "spkr_emb": backbone_feat, "w_spkr": w_spkr, "w_age": w_age,
            "logits_spkr_main": logits_spkr_main, "logits_age_main": logits_age_main,
            "logits_age_sub": logits_age_sub, "logits_spkr_sub": logits_spkr_sub, "x_recon": x_recon
        }

        loss, loss_dict = criterion(outputs, label_spk, label_age, input_feature=input_feat)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
        train_jacobian_loss += loss_dict.get('loss_jacobian', 0)
        _, pred_s = torch.max(logits_spkr_main, 1)
        _, pred_a = torch.max(logits_age_main, 1)
        correct_spk += (pred_s == label_spk).sum().item()
        correct_age += (pred_a == label_age).sum().item()
        total_samples += emb.size(0)
        
    avg_loss = total_train_loss / len(train_loader)
    avg_jac = train_jacobian_loss / len(train_loader)
    acc_spk, acc_age = 100 * correct_spk / total_samples, 100 * correct_age / total_samples
    
    eer_before, eer_after, before_embs, after_embs, final_ids = eval_network(model, eval_dataset)

    if best_EER > eer_after:
        best_EER = eer_after
        torch.save(model.state_dict(), f'./checkpoints/{val_dataset}_best_model.pth')
        print(f"最佳模型 EER: {best_EER * 100:.2f}%")

    print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Jac: {avg_jac:.4f} | Spk Acc: {acc_spk:.2f}% | EER Before: {eer_before*100:.2f}% |  EER After: {eer_after*100:.2f}%")

    csv_writer.writerow([epoch+1, avg_loss, acc_spk, acc_age, avg_jac, eer_before, eer_after])
    csv_file.flush()
    writer.add_scalar("Loss/Train", avg_loss, epoch)
    writer.add_scalar("Loss/Jacobian", avg_jac, epoch)
    writer.add_scalar("EER/Val_After", eer_after, epoch)

writer.close()
csv_file.close()
print("Training Finished!")