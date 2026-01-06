import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# =====================
# 載入資料
# =====================
data = torch.load("./result/ECAPA-TDNN/TIMIT_ECAPA-TDNN_embeddings.pt")

X = data["embeddings"]   # Tensor [N, D]
ages = data["ages"]      # list or tensor [N]

# =====================
# 前處理
# =====================
if torch.is_tensor(X):
    X = X.cpu().numpy()

if torch.is_tensor(ages):
    ages = ages.cpu().numpy()
else:
    ages = np.array(ages)

# =====================
# LDA 降維（7 類 → 最多 6 維，這裡取前 2）
# =====================
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, ages)

# =====================
# 繪圖（每個 age group 最多 200 點）
# =====================
plt.figure(figsize=(8, 6))

np.random.seed(42)  # 確保可重現

max_points_per_class = 200
age_groups = np.unique(ages)

for age in age_groups:
    idx = np.where(ages == age)[0]

    # 若該類樣本超過 200，隨機抽樣
    if len(idx) > max_points_per_class:
        idx = np.random.choice(idx, max_points_per_class, replace=False)

    plt.scatter(
        X_lda[idx, 0],
        X_lda[idx, 1],
        label=f"Age group {age}",
        alpha=0.7,
        s=20
    )

plt.xlabel("LDA Component 1")
plt.ylabel("LDA Component 2")
plt.title("TIMIT LDA Projection of ECAPA-TDNN Embeddings")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(
    "./result/ECAPA-TDNN/TIMIT_ECAPA-TDNN_age_lda.png",
    dpi=300
)
plt.show()
