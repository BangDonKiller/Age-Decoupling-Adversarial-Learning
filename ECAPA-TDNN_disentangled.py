import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tqdm import tqdm

# 請確保你的模型檔案路徑正確
from model.disentangled_model.SAI_AgeDisentangler import Disentangler 

# ==========================================
# 1. 資料準備與前處理
# ==========================================
print("Loading data...")
data = torch.load('result/ECAPA-TDNN/TIMIT_ECAPA-TDNN_embeddings.pt')

# 假設你的 .pt 檔結構如下
embeddings = data['embeddings']  # Tensor [N, 192]
ages = data['ages']              # Tensor or List [N] (0~6 representing Age Groups)
speakers = data['speaker_ids']  # List of speaker labels [N]

# 確認各年齡組的比例
# unique, counts = np.unique(ages, return_counts=True)
# age_distribution = dict(zip(unique, counts))
# print("Age Distribution:", age_distribution)

# 將文字標籤 (例如 'spk001') 轉為數字 ID (0, 1, 2...)
spk_encoder = LabelEncoder()
spk_labels = spk_encoder.fit_transform(speakers)
num_speakers = len(spk_encoder.classes_)

# 確保 ages 是 Tensor 格式
if isinstance(ages, list):
    ages = torch.tensor(ages)
spk_labels = torch.tensor(spk_labels, dtype=torch.long)

print(f"Total samples: {len(embeddings)}")
print(f"Num Speakers: {num_speakers}")
print(f"Num Age Groups: {len(torch.unique(ages))}")

# 資料切割：80% 訓練, 10% 驗證, 10% 測試
# 第一次切：Train (80%) vs Temp (20%)
X_train, X_temp, y_spk_train, y_spk_temp, y_age_train, y_age_temp = train_test_split(
    embeddings, spk_labels, ages, test_size=0.2, random_state=42, stratify=ages
)

# 第二次切：Val (10%) vs Test (10%)
X_val, X_test, y_spk_val, y_spk_test, y_age_val, y_age_test = train_test_split(
    X_temp, y_spk_temp, y_age_temp, test_size=0.5, random_state=42, stratify=y_age_temp
)

print(f"Train samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")

# 封裝成 DataLoader
BATCH_SIZE = 64

train_dataset = TensorDataset(X_train, y_spk_train, y_age_train)
val_dataset = TensorDataset(X_val, y_spk_val, y_age_val)
test_dataset = TensorDataset(X_test, y_spk_test, y_age_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# 2. 模型初始化
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 根據你的資料自動設定參數
model = Disentangler(
    input_dim=192, 
    spk_dim=128, 
    age_dim=64, 
    num_spks=num_speakers,     # 自動填入說話者總數
    num_age_groups=7           # 假設 TIMIT 分成 7 組
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Loss Functions
criterion_cls = nn.CrossEntropyLoss()
criterion_recon = nn.MSELoss()

# 訓練參數
EPOCHS = 100
best_val_loss = float('inf')

# ==========================================
# 3. 訓練迴圈
# ==========================================
print(f"Start training on {device}...")

for epoch in range(EPOCHS):
    model.train()
    
    # 動態 Alpha: 隨著 Epoch 增加，讓 GRL 的影響力變大 (0 -> 1)
    # 這有助於穩定訓練：先學好特徵，再開始對抗
    p = float(epoch) / EPOCHS
    alpha = 2.0 / (1. + np.exp(-10 * p)) - 1
    
    total_train_loss = 0.0
    correct_spk = 0
    correct_age_h1 = 0 # 我們希望這個越低越好 (代表解耦成功)
    total_samples = 0
    
    for emb, label_spk, label_age in train_loader:
        emb, label_spk, label_age = emb.to(device), label_spk.to(device), label_age.to(device)
        
        # Forward
        x_recon, spk_pred, age_from_h2, age_from_h1, h_spk, h_age = model(emb, alpha=alpha)
        
        # 1. 重建 Loss (Reconstruction)
        loss_recon = criterion_recon(x_recon, emb)
        
        # 2. 說話者識別 Loss (Main Task)
        loss_spk = criterion_cls(spk_pred, label_spk)
        
        # 3. 年齡吸收 Loss (Age Absorption - h2)
        loss_age_h2 = criterion_cls(age_from_h2, label_age)
        
        # 4. 年齡對抗 Loss (Adversarial - h1)
        loss_adv_h1 = criterion_cls(age_from_h1, label_age)
        
        # 5. 正交化 Loss (Orthogonality)
        loss_ortho = model.correlation_loss(h_spk, h_age)
        
        # 總 Loss (權重可微調)
        # 建議: 重建權重稍微大一點，保證 Embedding 品質       
        if epoch < 20:
            lambda_adv = 50
        else:
            lambda_adv = 20
            
        total_loss = loss_age_h2 + loss_adv_h1 * 50 + loss_ortho * 50
        # total_loss = loss_spk + 0 * loss_recon + 0 * loss_age_h2 + loss_adv_h1 + 0 * loss_ortho
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        total_train_loss += total_loss.item()
        
        # 計算準確率 (監控用)
        _, pred_s = torch.max(spk_pred, 1)
        _, pred_a_h1 = torch.max(age_from_h1, 1)
        correct_spk += (pred_s == label_spk).sum().item()
        correct_age_h1 += (pred_a_h1 == label_age).sum().item()
        total_samples += label_spk.size(0)

    avg_loss = total_train_loss / len(train_loader)
    acc_spk = 100 * correct_spk / total_samples
    acc_age_leak = 100 * correct_age_h1 / total_samples # 洩漏率
    
    # ==========================================
    # 4. 驗證迴圈
    # ==========================================
    model.eval()
    val_loss = 0.0
    val_correct_spk = 0
    val_correct_age_h1 = 0
    val_samples = 0
    
    with torch.no_grad():
        for emb, label_spk, label_age in val_loader:
            emb, label_spk, label_age = emb.to(device), label_spk.to(device), label_age.to(device)

            # 重點是看 h1 到底還殘留多少年齡資訊
            x_recon, spk_pred, age_from_h2, age_from_h1, h_spk, h_age = model(emb, alpha=0.0)
            
            l_recon = criterion_recon(x_recon, emb)
            l_spk = criterion_cls(spk_pred, label_spk)
            l_age_h2 = criterion_cls(age_from_h2, label_age)
            # 驗證時不計算 GRL 的反向，只算 CrossEntropy 看預測準不準
            l_age_leak = criterion_cls(age_from_h1, label_age) 
            l_ortho = model.correlation_loss(h_spk, h_age)
            
            # 驗證 Loss
            batch_loss = l_age_h2 + l_age_leak * 50 + l_ortho * 50
            # batch_loss = l_spk + 2.0 * l_recon + l_age_h2 + l_age_leak + 10.0 * l_ortho
            val_loss += batch_loss.item()
            
            _, pred_s = torch.max(spk_pred, 1)
            _, pred_a_h1 = torch.max(age_from_h1, 1)
            val_correct_spk += (pred_s == label_spk).sum().item()
            val_correct_age_h1 += (pred_a_h1 == label_age).sum().item()
            val_samples += label_spk.size(0)
            
    avg_val_loss = val_loss / len(val_loader)
    val_acc_spk = 100 * val_correct_spk / val_samples
    val_acc_age_leak = 100 * val_correct_age_h1 / val_samples
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Train Loss: {avg_loss:.4f} | Spk Acc: {acc_spk:.2f}% | Age Leak: {acc_age_leak:.2f}% | Ortho Loss: {loss_ortho:.4f} |"
          f"|| Val Loss: {avg_val_loss:.4f} | Spk Acc: {val_acc_spk:.2f}% | Age Leak: {val_acc_age_leak:.2f}% | Ortho Loss: {l_ortho:.4f}")
    
    # 儲存最佳模型
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_age_disentangler.pth')
        print("  -> Model Saved!")

print("Training Finished!")

# ==========================================
# 5. 提取解耦後的 Embedding (Test Set)
# ==========================================
# 載入最佳模型
model.load_state_dict(torch.load('best_age_disentangler.pth'))
model.eval()

disentangled_embeddings = []
disentangled_ages = []

with torch.no_grad():
    for emb, _, label_age in test_loader:
        emb = emb.to(device)
        # 我們只需要 h_spk (解耦後的說話者特徵)
        _, _, _, _, h_spk, _ = model(emb, alpha=0.0)
        disentangled_embeddings.append(h_spk.cpu())
        disentangled_ages.append(label_age)

final_embs = torch.cat(disentangled_embeddings, dim=0)
final_ages = torch.cat(disentangled_ages, dim=0)

# 儲存結果，供後續 PCA/CCA 分析使用
torch.save({
    'embeddings': final_embs,
    'ages': final_ages
}, 'disentangled_test_embeddings.pt')

print(f"Disentangled embeddings saved with shape: {final_embs.shape}")