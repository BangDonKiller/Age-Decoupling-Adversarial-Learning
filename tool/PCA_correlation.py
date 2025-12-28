import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

# 1. 載入資料
data = torch.load("./result/ECAPA-TDNN/GLOBE_ECAPA-TDNN_embeddings.pt", map_location='cpu')
embeddings = data["embeddings"]
genders = np.array(data["genders"])

# 2. 資料前處理
# 將 Tensor 轉為 Numpy
if torch.is_tensor(embeddings):
    embeddings = embeddings.cpu().numpy()

# 將性別標籤 (例如 'm', 'f') 轉為數值 (0, 1)
le = LabelEncoder()
y_encoded = le.fit_transform(genders)
print(f"標籤編碼對應: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# 3. 執行 PCA
# 我們觀察前 20 個主成分就足夠了
n_components = 20
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(embeddings)

# 取得解釋變異量 (Explained Variance Ratio)
explained_variance = pca.explained_variance_ratio_

# 4. 計算每個主成分與性別的相關係數 (Correlation)
correlations = []
for i in range(n_components):
    # 取出第 i 個主成分的所有樣本分數
    pc_scores = X_pca[:, i]
    # 計算該主成分與性別標籤 (0/1) 的 Pearson 相關係數
    # abs() 取絕對值，因為我們只關心相關程度，不關心正負
    corr = np.abs(np.corrcoef(pc_scores, y_encoded)[0, 1])
    correlations.append(corr)

# 5. 繪圖分析
fig, ax1 = plt.subplots(figsize=(12, 6))

# 設定 X 軸 (PC1, PC2, ...)
indices = np.arange(1, n_components + 1)
width = 0.35

# 繪製左軸：解釋變異量 (Bar Chart - Blue)
rects1 = ax1.bar(indices - width/2, explained_variance, width, label='Explained Variance', color='skyblue', alpha=0.7)
ax1.set_xlabel('Principal Component (PC) Index')
ax1.set_ylabel('Explained Variance Ratio', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.set_xticks(indices)
ax1.set_title(f'PCA Analysis: Variance vs. Gender Correlation (Dataset: GLOBE)')

# 繪製右軸：性別相關係數 (Bar Chart - Red)
ax2 = ax1.twinx()  # 共用 X 軸
rects2 = ax2.bar(indices + width/2, correlations, width, label='Correlation with Gender', color='salmon', alpha=0.9)
ax2.set_ylabel('Absolute Correlation Coefficient', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')
ax2.set_ylim(0, 1.0) # 相關係數最大為 1

# 加入圖例
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper right')

plt.tight_layout()
plt.savefig("./result/ECAPA-TDNN/GLOBE_ECAPA-TDNN_pca_variance_correlation.png", dpi=300)
plt.show()

# 6. 輸出統計數據供參考
print("-" * 30)
print(f"{'PC Index':<10} | {'Var (%)':<10} | {'Gender Corr':<15}")
print("-" * 30)
for i in range(5): # 只印出前 5 個
    print(f"PC {i+1:<7} | {explained_variance[i]*100:.2f}%     | {correlations[i]:.4f}")
print("-" * 30)