import umap
import matplotlib.pyplot as plt
import torch

# Load embeddings
data = torch.load("./result/ECAPA-TDNN/VoxCeleb2_ECAPA-TDNN_embeddings.pt")
embeddings = data["embeddings"].numpy()  # Convert to numpy array
genders = data["genders"]

print("開始 UMAP 降維…")
# UMAP 降維
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
embedding_2d = reducer.fit_transform(embeddings)
print("UMAP 降維完成！")

# 繪製結果
plt.figure(figsize=(10, 8))
for gender in set(genders):
    idxs = [i for i, g in enumerate(genders) if g == gender]
    plt.scatter(embedding_2d[idxs, 0], embedding_2d[idxs, 1], label=gender, alpha=0.7)
plt.legend()
plt.title("UMAP Projection of Speaker Embeddings by Gender")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.show()