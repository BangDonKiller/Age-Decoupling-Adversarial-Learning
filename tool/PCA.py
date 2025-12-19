import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

data = torch.load("result/ResNet34/ResNet34_embeddings.pt")

embeddings = data["embeddings"]      # Tensor [N, D]
genders = data["genders"]            # list[int] or list[str]

X = embeddings.cpu().numpy()         # [N, D]
y = np.array(genders)                # [N]

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)         # [N, 3]

# 3D Plotting
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

for g in np.unique(y):
    idx = y == g
    ax.scatter(
        X_pca[idx, 0],  # PC1
        X_pca[idx, 1],  # PC2
        X_pca[idx, 2],  # PC3
        label=f"gender {g}",
        alpha=0.6
    )

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.legend()
ax.set_title("ResNet34 embeddings PCA (3 components)")
plt.savefig("result/ResNet34/ResNet34_embeddings_pca3d.png")
plt.show()
