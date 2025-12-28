import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch # 你的檔案是 .pt，可能需要 torch 來載入

# --- 步驟 1: 載入你的真實資料 ---
# 你的程式碼是 np.load，但 .pt 通常是 torch 的檔案格式。
# 我提供兩種載入方式，請選擇適合你的那一種。

# 方式一: 如果 .pt 是用 np.savez 存的 (根據你的程式碼推斷)
try:
    data = np.load("./result/ECAPA-TDNN/VoxCeleb2_ECAPA-TDNN_embeddings.pt", allow_pickle=True)
    embeddings = data["embeddings"]
    genders = data["genders"]
# 方式二: 如果 .pt 是用 torch.save 存的
except (FileNotFoundError, KeyError, AttributeError):
    print("使用 NumPy 載入失敗，嘗試使用 PyTorch 載入...")
    data = torch.load("./result/ECAPA-TDNN/VoxCeleb2_ECAPA-TDNN_embeddings.pt")
    # 假設 data 是一個字典，包含 'embeddings' 和 'genders'
    embeddings = data["embeddings"].cpu().numpy() # 轉為 numpy array
    genders = data["genders"]
    speaker_ids = data["speaker_ids"]  # 如果需要的話
    
# --- 步驟 2: 資料預處理 ---
# 將嵌入向量從可能存在的 torch Tensor 轉換為 numpy array
if not isinstance(embeddings, np.ndarray):
    embeddings = embeddings.numpy()

# 將性別標籤轉換為數字 (0 for female, 1 for male)
# 假設原始標籤是 'f'/'female' 和 'm'/'male'
# 如果你的標籤本來就是 0/1，這段程式碼也不會出錯
unique_genders = np.unique(genders)
print(f"原始性別標籤: {unique_genders}")

# 建立一個映射，例如 {'female': 0, 'male': 1} 或 {'f': 0, 'm': 1}
# 這裡我們自動判斷，假設 'm' 開頭的是男性
male_label_str = [g for g in unique_genders if str(g).lower().startswith('m')][0]
female_label_str = [g for g in unique_genders if str(g).lower().startswith('f')][0]

labels = np.array([1 if g == male_label_str else 0 for g in genders])

print(f"資料準備完成，Embeddings shape: {embeddings.shape}, Labels shape: {labels.shape}")
print(f"將 '{female_label_str}' 映射到 0，'{male_label_str}' 映射到 1")
print("-" * 50)


# --- 任務二：找出那把「性別的尺」 (尋找性別方向) ---
print("執行任務二：尋找性別向量...")

# 1. 根據標籤分離男性和女性的嵌入向量
male_vectors = embeddings[labels == 1]
female_vectors = embeddings[labels == 0]

# 2. 計算平均向量
mean_male_vector = np.mean(male_vectors, axis=0)
mean_female_vector = np.mean(female_vectors, axis=0)

# 3. 相減得到「性別之箭」
gender_vector = mean_male_vector - mean_female_vector

print(f"成功計算出性別向量，維度為: {gender_vector.shape}")
print("性別向量的前5個維度值:", gender_vector[:5]) # 可以取消註解來看看
print("-" * 50)


# --- 任務三：做個「無性別」實驗 (特徵去除與驗證) ---
print("執行任務三：移除性別特徵並驗證...")

# 1. 驗證原始資料的性別可分性
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.3, random_state=42, stratify=labels)
model_original = LogisticRegression(max_iter=1000, class_weight='balanced')
model_original.fit(X_train, y_train)
accuracy_original = model_original.score(X_test, y_test)
print(f"在『原始』嵌入向量上的性別分類準確率: {accuracy_original:.4f}")

# 2. 進行「無性別化」手術
# v' = v - proj_g(v) = v - (v·g / ||g||²) * g
# 加上一個極小值 epsilon 防止除以零
gender_vector_norm_sq = np.sum(gender_vector**2) + 1e-9
debiased_embeddings = np.zeros_like(embeddings)

# 使用 NumPy 的向量化操作，比 for 迴圈快很多
projections = (embeddings @ gender_vector / gender_vector_norm_sq)[:, np.newaxis] * gender_vector
debiased_embeddings = embeddings - projections

print("已完成所有向量的去性別化處理。")

# 3. 驗證「無性別化」後的資料
X_train_db, X_test_db, y_train_db, y_test_db = train_test_split(debiased_embeddings, labels, test_size=0.3, random_state=42, stratify=labels)
model_debiased = LogisticRegression(max_iter=1000, class_weight='balanced')
model_debiased.fit(X_train_db, y_train_db)
accuracy_debiased = model_debiased.score(X_test_db, y_test_db)
print(f"在『去性別化』嵌入向量上的性別分類準確率: {accuracy_debiased:.4f}")
print("-" * 50)


# --- 任務四：當個偵探，調查「懸案」 (錯誤分析) ---
print("執行任務四：錯誤分析...")

# 使用在原始資料上訓練的模型，對完整的測試集進行預測
predictions_on_test = model_original.predict(X_test)
error_mask = predictions_on_test != y_test
error_indices_in_test = np.where(error_mask)[0]

# 找出這些錯誤樣本在原始 embeddings 陣列中的索引，方便你回溯
# 一個簡單的方法是直接對完整資料集做預測來找錯誤樣本。
full_predictions = model_original.predict(embeddings)
error_indices_in_full = np.where(full_predictions != labels)[0]


print(f"在總共 {len(embeddings)} 個樣本中，模型總共搞錯了 {len(error_indices_in_full)} 個。")
print("以下是前5個被搞錯的樣本資訊（來自完整資料集）：")

for i, idx in enumerate(error_indices_in_full[:5]):
    true_label_str = genders[idx] # 使用原始的字串標籤
    pred_label_int = full_predictions[idx]
    
    # 根據數字預測反向找到字串標籤
    pred_label_str = male_label_str if pred_label_int == 1 else female_label_str

    print(f"說話者 ID {speaker_ids[idx]} ,樣本在原始陣列中的索引 {idx}: 真實性別是 [{true_label_str}], 模型卻預測成 [{pred_label_str}]")

print("-" * 50)


# --- 任務五：單獨使用性別向量進行預測 (資訊含量分析) ---
print("執行任務五：分析性別向量本身的資訊含量...")

# 1. 計算每個嵌入向量在 gender_vector 上的投影分數 (點積)
# 這會將 192 維的向量降維成 1 維的「性別分數」
# @ 符號是 NumPy 中的矩陣乘法 (在這裡是矩陣-向量乘法)
projection_scores = embeddings @ gender_vector

# 2. 將這個 1D 的分數作為我們新的特徵
# scikit-learn 需要一個 2D 的輸入，所以我們用 reshape(-1, 1)
X_1d = projection_scores.reshape(-1, 1)

# 3. 使用這個 1D 特徵來訓練和測試一個新的邏輯回歸模型
X_train_1d, X_test_1d, y_train_1d, y_test_1d = train_test_split(X_1d, labels, test_size=0.3, random_state=42, stratify=labels)
model_1d = LogisticRegression(class_weight='balanced')
model_1d.fit(X_train_1d, y_train_1d)
accuracy_1d = model_1d.score(X_test_1d, y_test_1d)

print(f"在『原始』(192維)嵌入向量上的分類準確率: {accuracy_original:.4f}")
print(f"僅使用『性別向量投影』(1維)分數的分類準確率: {accuracy_1d:.4f}")
print("-" * 50)