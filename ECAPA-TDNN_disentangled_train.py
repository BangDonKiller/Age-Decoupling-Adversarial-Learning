import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.disentangled_model.JFE import JFENetwork, JFELoss
from tool.EER import compute_eer
from data.vox2_loader import Vox2Dataset
# from data.vox1_loader import PairwiseDataset
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
        is_same = int(line[0])

        spk1_rel = line[1]
        spk2_rel = line[2]

        spk1_path = find_audio_path(spk1_rel)
        spk2_path = find_audio_path(spk2_rel)

        spk1_id = spk1_rel.split("/")[0]
        spk2_id = spk2_rel.split("/")[0]

        datalist.append((is_same, spk1_id, spk2_id, spk1_path, spk2_path))

    pos = sum(1 for x in datalist if x[0] == 1)
    neg = sum(1 for x in datalist if x[0] == 0)
    print(f"[Eval] Positive pairs: {pos}, Negative pairs: {neg}")
    
    return datalist

train_dataset = Vox2Dataset(
    audio_dir=DATASET_INFO[dataset]['AUDIO_DIR'],
    audio_meta_dir=DATASET_INFO[dataset]['AUDIO_META_DIR'],
    target_sample_rate=16000,
    suffix=DATASET_INFO[dataset]['audio_suffix']
)

eval_dataset = build_eval_dataset(
    audio_dirs=DATASET_INFO['VoxCeleb1'][val_dataset]['AUDIO_DIR'],
    audio_meta_dir=DATASET_INFO['VoxCeleb1'][val_dataset]['AUDIO_META_DIR'],
    max_pairs=20000
)

def eval_network(model, datalist):
    """
    Speaker verification evaluation on pairwise trials (e.g. Vox-O)

    Returns:
        eer_before (float)
        eer_after  (float)
        final_before_embs (Tensor)
        final_after_embs  (Tensor)
        final_ids (List[str])
    """

    model.eval()

    # =========================
    # Containers
    # =========================
    before_scores = []
    after_scores = []
    labels = []

    before_embs = []
    after_embs = []
    final_ids = []

    # =========================
    # Forward (pairwise)
    # =========================
    with torch.no_grad():
        for is_same, id1, id2, path1, path2 in tqdm(datalist, desc="Evaluating"):

            emb1, sr1 = torchaudio.load(path1)
            emb2, sr2 = torchaudio.load(path2)

            emb1 = emb1.to(device)
            emb2 = emb2.to(device)

            out1 = model(emb1, mode="test")
            out2 = model(emb2, mode="test")

            # -------- Before disentangle (h_spk) --------
            h1 = F.normalize(out1["spkr_emb"], p=2, dim=1)
            h2 = F.normalize(out2["spkr_emb"], p=2, dim=1)

            score_before = F.cosine_similarity(h1, h2).cpu()
            before_scores.append(score_before)

            before_embs.append(out1["spkr_emb"].cpu())
            before_embs.append(out2["spkr_emb"].cpu())

            # -------- After disentangle (w_spkr) --------
            w1 = F.normalize(out1["w_spkr"], p=2, dim=1)
            w2 = F.normalize(out2["w_spkr"], p=2, dim=1)

            score_after = F.cosine_similarity(w1, w2).cpu()
            after_scores.append(score_after)

            after_embs.append(out1["w_spkr"].cpu())
            after_embs.append(out2["w_spkr"].cpu())

            labels.append(is_same)
            final_ids.extend([id1, id2])

    # =========================
    # EER
    # =========================
    final_labels = torch.tensor(labels).numpy()
    final_before_scores = torch.cat(before_scores).numpy()
    final_after_scores = torch.cat(after_scores).numpy()

    eer_before = compute_eer(final_before_scores, final_labels)
    eer_after = compute_eer(final_after_scores, final_labels)

    final_before_embs = torch.cat(before_embs, dim=0)
    final_after_embs = torch.cat(after_embs, dim=0)

    return (
        eer_before,
        eer_after,
        final_before_embs,
        final_after_embs,
        final_ids
    )        

# 封裝成 DataLoader
BATCH_SIZE = BATCH_SIZE

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=g)

# ==========================================
# 2. 模型初始化
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 根據你的資料自動設定參數
model = JFENetwork(
    MODEL_ID,
    input_dim=192, 
    spk_dim=96, 
    age_dim=96, 
    num_speakers=5990,
    num_age_groups=7
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Loss Functions
age_weights = train_dataset.age_class_weights.to(device)
criterion = JFELoss(age_weights, lambda_entropy=0.1, lambda_mapc=0.5)

# 訓練參數
EPOCHS = 1
best_val_loss = float('inf')
best_score_balanced = -float('inf')
best_spk_acc = 0.0
best_leak_privacy = float('inf')
best_EER = float('inf')

# ==========================================
# TensorBoard & CSV Logger
# ==========================================
log_dir = "logs/jfe"
checkpoint_dir = "checkpoints"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

writer = SummaryWriter(log_dir=log_dir)

csv_path = os.path.join(log_dir, f"{val_dataset}_metrics.csv")
csv_file = open(csv_path, mode="w", newline="")
csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "epoch",
    "train_loss",
    "train_loss_spkr",
    "train_loss_age",
    "train_entropy_age",
    "train_entropy_spkr",
    "train_mapc",
    "train_spk_acc",
    "train_age_acc",
    "train_age_leak",
    "train_id_leak",
    "val_eer_before",
    "val_eer_after"
])

print("Logger initialized.")

# ==========================================
# 3. 訓練迴圈
# ==========================================
print(f"Start training on {device}...")

for epoch in range(EPOCHS):
    model.train()
    
    total_train_loss = 0.0
    train_loss_spkr = 0.0
    train_loss_age = 0.0
    train_entropy_age = 0.0
    train_entropy_spkr = 0.0
    train_mapc = 0.0
    correct_spk = 0
    correct_age = 0
    correct_age_sub = 0
    correct_id_sub = 0
    total_samples = 0
    
    for emb, label_spk, label_age in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        emb, label_spk, label_age = emb.to(device), label_spk.to(device), label_age.to(device)
        
        # Forward
        outputs = model(emb, mode = "train")

        # 計算 Loss
        loss, loss_dict = criterion(outputs, label_spk, label_age)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
        train_loss_spkr += loss_dict['loss_spkr']
        train_loss_age += loss_dict['loss_age']
        train_entropy_age += loss_dict['entropy_age']
        train_entropy_spkr += loss_dict['entropy_spkr']
        train_mapc += loss_dict['mapc']
        
        # 計算準確率 (監控用)
        _, pred_s_main = torch.max(outputs['logits_spkr_main'], 1) # 從 h_spk 預測說話者
        _, pred_a_main = torch.max(outputs['logits_age_main'], 1) # 從 h_age 預測年齡
        _, pred_s_sub = torch.max(outputs['logits_spkr_sub'], 1) # 從 h_age 預測說話者 (洩漏)
        _, pred_a_sub = torch.max(outputs['logits_age_sub'], 1) # 從 h_spk 預測年齡 (洩漏)
        correct_spk += (pred_s_main == label_spk).sum().item()
        correct_age += (pred_a_main == label_age).sum().item()
        correct_id_sub += (pred_s_sub == label_spk).sum().item()
        correct_age_sub += (pred_a_sub == label_age).sum().item()
        total_samples += emb.size(0)
        
    avg_loss = total_train_loss / len(train_loader)
    avg_loss_spkr = train_loss_spkr / len(train_loader)
    avg_loss_age = train_loss_age / len(train_loader)
    avg_entropy_age = train_entropy_age / len(train_loader)
    avg_entropy_spkr = train_entropy_spkr / len(train_loader)
    avg_mapc = train_mapc / len(train_loader)
    acc_spk = 100 * correct_spk / total_samples
    acc_age = 100 * correct_age / total_samples
    acc_age_leak = 100 * correct_age_sub / total_samples
    acc_id_leak = 100 * correct_id_sub / total_samples
    
    # ==========================================
    # 4. 驗證迴圈
    # ==========================================
    eer_before, eer_after, before_embs, after_embs, final_ids = eval_network(
        model,
        eval_dataset
    )

    
    if best_EER > eer_after:
        best_EER = eer_after
        torch.save(model.state_dict(), f'./checkpoints/{val_dataset}_best_model.pth')
        torch.save({
            'embeddings': after_embs,
            'before_embeddings': before_embs,
            'ids': final_ids,
        }, os.path.join(checkpoint_dir, f'{val_dataset}_best_disentangled_embeddings.pt'))
        print(f"儲存最佳模型 EER: {best_EER * 100:.2f}%")
        
    if epoch == EPOCHS - 1:
        torch.save(model.state_dict(), f'./checkpoints/{val_dataset}_last_model.pth')
        torch.save({   
            'embeddings': after_embs,
            'before_embeddings': before_embs,
            'ids': final_ids,
        }, os.path.join(checkpoint_dir, f'{val_dataset}_last_disentangled_embeddings.pt'))
        print("儲存最終模型。")
    
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Train Loss: {avg_loss:.4f} | Spk Acc: {acc_spk:.2f}% | Age Acc: {acc_age:.2f}% | Age Leak: {acc_age_leak:.2f}% | ID Leak: {acc_id_leak:.2f}% | Correlation: {avg_mapc:.4f} "
          f"|| Val EER Before: {eer_before * 100:.2f}% | After: {eer_after * 100:.2f}%")

    # ==========================================
    # TensorBoard logging
    # ==========================================
    writer.add_scalar("Loss/Train", avg_loss, epoch)
    writer.add_scalar("Loss/Train_Spk", avg_loss_spkr, epoch)
    writer.add_scalar("Loss/Train_Age", avg_loss_age, epoch)
    writer.add_scalar("Entropy/Train_Age", avg_entropy_age, epoch)
    writer.add_scalar("Entropy/Train_Spk", avg_entropy_spkr, epoch)
    writer.add_scalar("MAPC/Train", avg_mapc, epoch)
    writer.add_scalar("Accuracy/Train_Spk", acc_spk, epoch)
    writer.add_scalar("Accuracy/Train_Age", acc_age, epoch)
    writer.add_scalar("Leak/Train_Age", acc_age_leak, epoch)
    writer.add_scalar("Leak/Train_ID", acc_id_leak, epoch)
    
    writer.add_scalar("EER/Val_Before_Disentangle", eer_before, epoch)
    writer.add_scalar("EER/Val_After_Disentangle", eer_after, epoch)

    # ==========================================
    # CSV logging
    # ==========================================
    csv_writer.writerow([
        epoch + 1,
        avg_loss,
        avg_loss_spkr,
        avg_loss_age,
        avg_entropy_age,
        avg_entropy_spkr,
        avg_mapc,
        acc_spk,
        acc_age,
        acc_age_leak,
        acc_id_leak,
        eer_before,
        eer_after
    ])
    csv_file.flush() 

writer.close()
csv_file.close()

print("Training Finished!")