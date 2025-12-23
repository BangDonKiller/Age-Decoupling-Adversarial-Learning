import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# =====================
# 載入資料
# =====================
data = torch.load("./result/ECAPA-TDNN/VoxCeleb2_ECAPA-TDNN_embeddings.pt")

X = data["embeddings"]          # Tensor [N, D]
y_raw = np.array(data["genders"])

# tensor -> numpy
if hasattr(X, "cpu"):
    X = X.cpu().numpy()

# label mapping
label_map = {'f': 'female', 'm': 'male'}
y = np.array([label_map[g] for g in y_raw])

color_map = {'female': '#c44e52', 'male': '#4c72b0'}

print("X shape:", X.shape)
print("y classes:", np.unique(y))

# =====================
# LDA（2 類 → 1 維）
# =====================
lda = LinearDiscriminantAnalysis(n_components=1)
X_lda = lda.fit_transform(X, y)   # shape: [N, 1]

print("X_lda shape:", X_lda.shape)

# =====================
# 視覺化（1D → 2D 用 jitter）
# =====================
plt.figure(figsize=(10, 4))

for label in np.unique(y):
    idx = y == label
    jitter = np.random.normal(0, 0.02, size=idx.sum())

    plt.scatter(
        X_lda[idx, 0],
        jitter,
        s=5,
        alpha=0.3,
        label=label,
        color=color_map[label]
    )

plt.yticks([])
plt.xlabel("LDA Component 1")
plt.title("LDA Projection of ECAPA-TDNN Embeddings (Gender)")
plt.legend()
plt.tight_layout()

plt.savefig("VoxCeleb2_lda_gender_ecapa.png", dpi=300)
plt.show()

print("LDA plot saved: VoxCeleb2_lda_gender_ecapa.png")