import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


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
# 4. Linear SVM (Age group prediction)
# =========================
svm = LinearSVC(
    C=1.0,
    class_weight='balanced',
    max_iter=3000
)

svm_acc = cross_val_score(svm, X, y, cv=5)
print("Linear SVM (age group) acc:", svm_acc)
print("Mean acc:", svm_acc.mean())

# =========================
# 5. kNN local purity
# =========================
knn = KNeighborsClassifier(n_neighbors=5)
knn_acc = cross_val_score(knn, X, y, cv=5).mean()
print("kNN (age group) acc:", knn_acc)

# =========================
# 6. Linear Regression + Correlation (Age direction)
# =========================

# train / test split（不需要 CV）
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Linear regression
reg = LinearRegression()
reg.fit(X_tr, y_tr)

# Predict age index
y_pred = reg.predict(X_te)

# Correlation analysis
pearson_corr, _ = pearsonr(y_te, y_pred)
spearman_corr, _ = spearmanr(y_te, y_pred)

print("LinearReg + Corr")
print("Pearson r:", pearson_corr)
print("Spearman ρ:", spearman_corr)

# =========================
# 7. RBF SVM (Nonlinear age structure test)
# =========================

rbf_svm = Pipeline([
    ("scaler", StandardScaler()),   # ★ RBF 必須
    ("svm", SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",              # ★ 安全預設
        class_weight="balanced"
    ))
])

rbf_acc = cross_val_score(
    rbf_svm,
    X,
    y,
    cv=5,
    n_jobs=1
)

print("RBF SVM (age group) acc:", rbf_acc)
print("Mean acc:", rbf_acc.mean())
