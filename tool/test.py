import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr, pearsonr
import torch

# =========================
# 1. Load data
# =========================
data = torch.load("./result/ECAPA-TDNN/GLOBE_ECAPA-TDNN_embeddings.pt")

X = data["embeddings"].numpy()      # [N, D]
ages = np.array(data["ages"])       # 年齡組 label，例如 0~6

y = ages   # ★ 以年齡組作為預測目標（多類別）

print("X shape:", X.shape)
print("Age groups:", np.unique(y))

# =========================
# 2. PCA explained variance
# =========================
pca = PCA(n_components=10).fit(X)
print("explained var ratio (pc1..5):", pca.explained_variance_ratio_[:5])

# =========================
# 3. t-SNE（目前未使用，保留）
# =========================
def run_tsne(X, perplexity=30):
    tsne = TSNE(
        n_components=2,
        init='pca',
        perplexity=perplexity,
        metric='cosine',
        random_state=42
    )
    return tsne.fit_transform(X)

# =========================
# 4. kNN local purity
# =========================
knn = KNeighborsClassifier(n_neighbors=5)
knn_acc = cross_val_score(knn, X, y, cv=5).mean()
print("kNN (age group) acc:", knn_acc)