import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.disentangled_model.JFE import JFENetwork, JFELoss
from tool.EER import compute_eer
from data.vox2_loader import Vox2Dataset
from data.vox1_loader import PairwiseDataset
from params.param import DATASET_INFO, BATCH_SIZE
from torch.utils.tensorboard import SummaryWriter
import os
import csv

SEED = 42

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ==========================================
# 1. 資料準備與前處理
# ==========================================
dataset = 'VoxCeleb2'

train_dataset = Vox2Dataset(
    audio_dir=DATASET_INFO[dataset]['AUDIO_DIR'],
    audio_meta_dir=DATASET_INFO[dataset]['AUDIO_META_DIR'],
    target_sample_rate=16000,
    suffix=DATASET_INFO[dataset]['audio_suffix']
)

# 驗證集大小佔 10%
val_size = int(0.1 * len(train_dataset))
train_size = len(train_dataset) - val_size
g = torch.Generator()
g.manual_seed(SEED)

train_dataset, val_dataset = torch.utils.data.random_split(
    train_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

test_dataset = PairwiseDataset(
    audio_dir=DATASET_INFO['VoxCeleb1']["Train"]['AUDIO_DIR'],
    audio_meta_dir=DATASET_INFO['VoxCeleb1']["Train"]['AUDIO_META_DIR'],
)

# 封裝成 DataLoader
BATCH_SIZE = BATCH_SIZE

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=g)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, generator=g)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# 2. 模型初始化
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 根據你的資料自動設定參數
model = JFENetwork(
    input_dim=192, 
    spk_dim=96, 
    age_dim=96, 
    num_speakers=5990,
    num_age_groups=7
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Loss Functions
criterion = JFELoss(lambda_entropy=0.1, lambda_mapc=0.5)

# 訓練參數
EPOCHS = 5
best_val_loss = float('inf')
best_score_balanced = -float('inf')
best_spk_acc = 0.0
best_leak_privacy = float('inf')

# ==========================================
# TensorBoard & CSV Logger
# ==========================================
log_dir = "logs/jfe"
os.makedirs(log_dir, exist_ok=True)

writer = SummaryWriter(log_dir=log_dir)

csv_path = os.path.join(log_dir, "metrics.csv")
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
    "val_loss",
    "val_loss_spkr",
    "val_loss_age",
    "val_entropy_age",
    "val_entropy_spkr",
    "val_mapc",
    "val_spk_acc",
    "val_age_acc",
    "val_age_leak",
    "val_id_leak"
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
    model.eval()
    val_loss = 0.0
    val_loss_spkr = 0.0
    val_loss_age = 0.0
    val_entropy_age = 0.0
    val_entropy_spkr = 0.0
    val_mapc = 0.0
    val_correct_spk = 0
    val_correct_age = 0
    val_correct_age_sub = 0
    val_correct_id_sub = 0
    val_samples = 0
    
    with torch.no_grad():
        for emb, label_spk, label_age in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            val_hspk_list = []
            val_spk_label_list = []
            
            emb, label_spk, label_age = emb.to(device), label_spk.to(device), label_age.to(device)

            # 重點是看 h1 到底還殘留多少年齡資訊
            outputs = model(emb, mode = "val")

            loss, loss_dict = criterion(outputs, label_spk, label_age)

            val_loss += loss.item()
            val_loss_spkr += loss_dict['loss_spkr']
            val_loss_age += loss_dict['loss_age']
            val_entropy_age += loss_dict['entropy_age']
            val_entropy_spkr += loss_dict['entropy_spkr']
            val_mapc += loss_dict['mapc']
            _, pred_s_main = torch.max(outputs['logits_spkr_main'], 1) # 從 h_spk 預測說話者
            _, pred_a_main = torch.max(outputs['logits_age_main'], 1) # 從 h_age 預測年齡
            _, pred_s_sub = torch.max(outputs['logits_spkr_sub'], 1) # 從 h_age 預測說話者 (洩漏)
            _, pred_a_sub = torch.max(outputs['logits_age_sub'], 1) # 從 h_spk 預測年齡 (洩漏)
            val_correct_spk += (pred_s_main == label_spk).sum().item()
            val_correct_age += (pred_a_main == label_age).sum().item()
            val_correct_id_sub += (pred_s_sub == label_spk).sum().item()
            val_correct_age_sub += (pred_a_sub == label_age).sum().item()
            val_samples += label_spk.size(0)
            
    avg_val_loss = val_loss / len(val_loader)
    avg_val_loss_spkr = val_loss_spkr / len(val_loader)
    avg_val_loss_age = val_loss_age / len(val_loader)
    avg_val_entropy_age = val_entropy_age / len(val_loader)
    avg_val_entropy_spkr = val_entropy_spkr / len(val_loader)
    avg_val_mapc = val_mapc / len(val_loader)
    val_acc_spk = 100 * val_correct_spk / val_samples   
    val_acc_age = 100 * val_correct_age / val_samples
    val_acc_age_leak = 100 * val_correct_age_sub / val_samples
    val_acc_id_leak = 100 * val_correct_id_sub / val_samples
    
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Train Loss: {avg_loss:.4f} | Spk Acc: {acc_spk:.2f}% | Age Acc: {acc_age:.2f}% | Age Leak: {acc_age_leak:.2f}% | ID Leak: {acc_id_leak:.2f}% "
          f"|| Val Loss: {avg_val_loss:.4f} | Spk Acc: {val_acc_spk:.2f}% | Age Acc: {val_acc_age:.2f}% | Age Leak: {val_acc_age_leak:.2f}% | ID Leak: {val_acc_id_leak:.2f}% ")

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

    writer.add_scalar("Loss/Val", avg_val_loss, epoch)
    writer.add_scalar("Loss/Val_Spk", avg_val_loss_spkr, epoch)
    writer.add_scalar("Loss/Val_Age", avg_val_loss_age, epoch)
    writer.add_scalar("Entropy/Val_Age", avg_val_entropy_age, epoch)
    writer.add_scalar("Entropy/Val_Spk", avg_val_entropy_spkr, epoch)
    writer.add_scalar("MAPC/Val", avg_val_mapc, epoch)
    writer.add_scalar("Accuracy/Val_Spk", val_acc_spk, epoch)
    writer.add_scalar("Accuracy/Val_Age", val_acc_age, epoch)
    writer.add_scalar("Leak/Val_Age", val_acc_age_leak, epoch)
    writer.add_scalar("Leak/Val_ID", val_acc_id_leak, epoch)

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
        avg_val_loss,
        avg_val_loss_spkr,
        avg_val_loss_age,
        avg_val_entropy_age,
        avg_val_entropy_spkr,
        avg_val_mapc,
        val_acc_spk,
        val_acc_age,
        val_acc_age_leak,
        val_acc_id_leak
    ])
    csv_file.flush() 
    
    # 1. 計算綜合分數 (Balanced Score)
    # 這代表: 每犧牲 1% 的 Spk Acc，必須換來 1% 以上的 Leak 下降才划算
    # 你可以調整權重: current_score = val_acc_spk - 0.5 * val_acc_age_leak
    current_score = val_acc_spk - val_acc_age_leak
    if current_score > best_score_balanced:
        best_score_balanced = current_score
        torch.save(model.state_dict(), 'best_balanced.pth')
        print(f"  -> [Saved] Best Balanced Model (Score: {current_score:.2f} | Spk: {val_acc_spk:.2f}% | Leak: {val_acc_age_leak:.2f}%)")
    
    # 2. 條件式極致隱私 (Constrained Privacy)
    # 門檻建議: 90% (視你的容忍度而定，如果 90% 太高，可降至 85%)
    SPK_ACC_THRESHOLD = 90.0
    if val_acc_spk >= SPK_ACC_THRESHOLD:
        if val_acc_age_leak < best_leak_privacy:
            best_leak_privacy = val_acc_age_leak
            torch.save(model.state_dict(), 'best_privacy.pth')
            print(f"  -> [Saved] Best Privacy Model (Leak: {val_acc_age_leak:.2f}% | Spk: {val_acc_spk:.2f}%)")    


# 印出綜合分數最好的模型說話者準確率與洩漏率
print(f"Best Balanced Score: {best_score_balanced:.2f}")

print("Training Finished!")

# ==========================================
# 5. 提取解耦後的 Embedding (Test Set)
# ==========================================
# 載入最佳模型
model.load_state_dict(torch.load('best_balanced.pth'))
model.eval()
model.to(device)

before_scores = []       # 解耦前 cosine scores
disentangled_scores = [] # 解耦後 cosine scores
before_disentagled_list = []
after_disentagled_list = []
is_same_labels = []
final_ids = []

print("現在開始在測試集上提取解耦後的 Embedding...")

with torch.no_grad():
    for is_same, id1, id2, emb1, emb2 in tqdm(test_loader, desc="Extracting Embeddings"):
        emb1 = emb1.to(device)
        emb2 = emb2.to(device)

        # forward
        out1 = model(emb1, mode="test")
        out2 = model(emb2, mode="test")

        # =========================
        # 解耦前 (h_spk / spkr_emb)
        # =========================
        spk_emb1 = out1['spkr_emb']
        spk_emb2 = out2['spkr_emb']
        
        spk_emb1 = F.normalize(spk_emb1, p=2, dim=1)
        spk_emb2 = F.normalize(spk_emb2, p=2, dim=1)
        
        before_disentagled_list.append(out1['spkr_emb'].cpu())
        before_disentagled_list.append(out2['spkr_emb'].cpu())

        score_before = F.cosine_similarity(spk_emb1, spk_emb2, dim=1)
        before_scores.append(score_before.cpu())

        # =========================
        # 解耦後 (w_spkr)
        # =========================
        w_spk1 = out1['w_spkr']
        w_spk2 = out2['w_spkr']
        
        w_spk1 = F.normalize(w_spk1, p=2, dim=1)
        w_spk2 = F.normalize(w_spk2, p=2, dim=1)
        
        after_disentagled_list.append(out1['w_spkr'].cpu())
        after_disentagled_list.append(out2['w_spkr'].cpu())

        score_after = F.cosine_similarity(w_spk1, w_spk2, dim=1)
        disentangled_scores.append(score_after.cpu())

        is_same_labels.append(is_same.cpu())
        final_ids.extend(id1)  # 對應 out1
        final_ids.extend(id2)  # 對應 out2

# =========================
# concat & EER
# =========================
final_labels = torch.cat(is_same_labels, dim=0).numpy()
final_before_scores = torch.cat(before_scores, dim=0).numpy()
final_after_scores = torch.cat(disentangled_scores, dim=0).numpy()

eer_before = compute_eer(final_before_scores, final_labels)
eer_after = compute_eer(final_after_scores, final_labels)

print(f"Test EER (Before Disentangle, h_spk): {eer_before * 100:.2f}%")
print(f"Test EER (After  Disentangle, w_spkr): {eer_after * 100:.2f}%")

final_before_embs = torch.cat(before_disentagled_list, dim=0)
final_embs = torch.cat(after_disentagled_list, dim=0)

# 儲存結果，供後續 PCA/CCA 分析使用
torch.save({
    'embeddings': final_embs,
    'before_embeddings': final_before_embs,
    'ids': final_ids,
}, 'disentangled_test_embeddings.pt')

writer.close()
csv_file.close()

print(f"Disentangled embeddings saved with shape: {final_embs.shape}")