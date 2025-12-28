import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score
import torch

data = torch.load("./result/ECAPA-TDNN/VoxCeleb2_ECAPA-TDNN_embeddings.pt")
X = data["embeddings"].numpy()     # [N, D]
y = np.array(data["genders"])      # e.g. ['f', 'm']
ids = data["speaker_ids"]


# X, y 已存在
# 1) PCA explained variance
pca = PCA(n_components=10).fit(X)
print("explained var ratio (pc1..5):", pca.explained_variance_ratio_[:5])

# 2) PCA 2D 投影（視覺化用）
X_pca2 = PCA(n_components=2).fit_transform(X)

# 3) t-SNE 固定參數
def run_tsne(X, perplexity=30):
    tsne = TSNE(n_components=2, init='pca', perplexity=perplexity, random_state=42)
    return tsne.fit_transform(X)

X_tsne = run_tsne(X, perplexity=30)

# 4) 線性分類器
clf = LogisticRegression(max_iter=2000)
print("LogReg acc:", cross_val_score(clf, X, y, cv=5).mean())

# 5) kNN local purity
knn = KNeighborsClassifier(n_neighbors=5)
print("kNN acc:", cross_val_score(knn, X, y, cv=5).mean())
# 畫出KNN分為五個群的圖形，並且只把其中的10個說話者的樣本點標示出來，其他就不用了

# 6) silhouette 在 tsne 空間
print("silhouette (tsne):", silhouette_score(X_tsne, y))
