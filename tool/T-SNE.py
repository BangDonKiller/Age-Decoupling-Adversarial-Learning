from sklearn.manifold import TSNE
import numpy as np
import torch
import matplotlib.pyplot as plt

# 讀取資料
data = torch.load("./result/ECAPA-TDNN/VoxCeleb1_ECAPA-TDNN_embeddings.pt")

X = data["embeddings"].numpy()     # [N, D]
y = np.array(data["genders"])      # e.g. ['f', 'm']

# t-SNE
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate=200,
    random_state=42,
    init="pca"
)

X_tsne = tsne.fit_transform(X)

# ======================
# 畫圖
# ======================
plt.figure(figsize=(10, 8))

# 女生（深紅）
mask_f = y == "f"
plt.scatter(
    X_tsne[mask_f, 0],
    X_tsne[mask_f, 1],
    s=5,
    c="darkred",
    alpha=0.3,
    label="Female"
)

# 男生（深藍）
mask_m = y == "m"
plt.scatter(
    X_tsne[mask_m, 0],
    X_tsne[mask_m, 1],
    s=5,
    c="darkblue",
    alpha=0.3,
    label="Male"
)

plt.legend(markerscale=3)
plt.title("t-SNE of ECAPA-TDNN VoxCeleb1 Speaker Embeddings")
plt.xlabel("t-SNE dim 1")
plt.ylabel("t-SNE dim 2")
plt.tight_layout()
plt.savefig("VoxCeleb1_tsne_gender_ecapa.png", dpi=300)
plt.show()