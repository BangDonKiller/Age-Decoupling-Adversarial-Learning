import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


# read npy file turn to numpy
all_embeddings = np.load("ecapa_embeddings.npy")


# ========== PCA & 畫圖 ==========
X = all_embeddings.numpy()
pca = PCA(n_components=2)
X2 = pca.fit_transform(X)

plt.figure(figsize=(8, 6))
plt.scatter(X2[:, 0], X2[:, 1], s=6, alpha=0.7)
plt.title("ECAPA-TDNN PCA")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.savefig("PCA result.png")
plt.show()