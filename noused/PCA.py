import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
import plotly.express as px

# =====================
# PCA utility functions
# =====================
def pca_reduce(embeddings, n_components=2):
    """
    embeddings: torch.Tensor [N, D]
    n_components: int
    return: np.ndarray [N, n_components], fitted PCA
    """
    X = embeddings.cpu().numpy()
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca, pca


def subsample_per_class(X, y, max_points_per_class=200, seed=42):
    """
    X: np.ndarray [N, d]
    y: np.ndarray [N]
    """
    np.random.seed(seed)

    X_out, y_out = [], []

    for label in np.unique(y):
        idx = np.where(y == label)[0]
        if len(idx) > max_points_per_class:
            idx = np.random.choice(idx, max_points_per_class, replace=False)
        X_out.append(X[idx])
        y_out.append(y[idx])

    return np.vstack(X_out), np.concatenate(y_out)


def plot_pca_2d(X_pca, y, title, save_path=None):
    plt.figure(figsize=(8, 6))
    plt.style.use("seaborn-v0_8-whitegrid")

    for age in np.unique(y):
        idx = (y == age)
        plt.scatter(
            X_pca[idx, 0],
            X_pca[idx, 1],
            label=f"Age group {age}",
            alpha=0.7,
            s=20
        )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    plt.show()


def plot_pca_3d_interactive(X_pca, y, save_path):
    assert X_pca.shape[1] == 3, "X_pca must be [N, 3]"

    fig = px.scatter_3d(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        z=X_pca[:, 2],
        color=y.astype(str),
        opacity=0.6
    )

    fig.update_traces(marker=dict(size=4))
    fig.update_layout(title="PCA 3D Interactive")
    fig.write_html(save_path)


# =====================
# 1. 載入資料
# =====================
data = torch.load(
    "./result/ECAPA-TDNN/VoxCeleb2_ECAPA-TDNN_embeddings.pt",
    map_location="cpu"
)

embeddings = data["embeddings"]   # Tensor [N, D]
ages = data["ages"]               # list or tensor [N]

# =====================
# 2. 前處理（統一）
# =====================
# 年齡 → numpy
if torch.is_tensor(ages):
    ages = ages.cpu().numpy()
else:
    ages = np.array(ages)

# L2 normalize（與你原 PCA 視覺化流程一致）
embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

# =====================
# 3. PCA 視覺化（2D）
# =====================
X_pca_2d, _ = pca_reduce(embeddings, n_components=2)
X_vis_2d, y_vis_2d = subsample_per_class(
    X_pca_2d,
    ages,
    max_points_per_class=200
)

plot_pca_2d(
    X_vis_2d,
    y_vis_2d,
    title="VoxCeleb2 ECAPA-TDNN Embeddings PCA (Age)",
    save_path="./result/ECAPA-TDNN/VoxCeleb2_ECAPA-TDNN_Age_PCA_2D.png"
)

print("2D PCA plot saved.")

# =====================
# 4. PCA 視覺化（3D）
# =====================
X_pca_3d, _ = pca_reduce(embeddings, n_components=3)
X_vis_3d, y_vis_3d = subsample_per_class(
    X_pca_3d,
    ages,
    max_points_per_class=200
)

plot_pca_3d_interactive(
    X_vis_3d,
    y_vis_3d,
    "./result/ECAPA-TDNN/VoxCeleb2_ECAPA-TDNN_Age_PCA_3D_interactive.html"
)

print("3D PCA interactive plot saved.")

# =====================
# 5. PCA × 年齡 Spearman 分析
# =====================
X = embeddings.cpu().numpy()

n_components = min(20, X.shape[1])
pca_full = PCA(n_components=n_components)
X_pca_full = pca_full.fit_transform(X)

explained_var = pca_full.explained_variance_ratio_

abs_corrs, raw_corrs, p_values = [], [], []

for i in range(n_components):
    rho, p = spearmanr(X_pca_full[:, i], ages)
    raw_corrs.append(rho)
    abs_corrs.append(abs(rho))
    p_values.append(p)

abs_corrs = np.array(abs_corrs)
raw_corrs = np.array(raw_corrs)
p_values = np.array(p_values)

# =====================
# 6. 列印結果（報告用）
# =====================
print("\nPC | Explained Var (%) | Spearman r | |r| | p-value")
print("-" * 55)
for i in range(n_components):
    print(
        f"PC{i+1:02d} | "
        f"{explained_var[i]*100:6.2f}% | "
        f"{raw_corrs[i]:+6.3f} | "
        f"{abs_corrs[i]:5.3f} | "
        f"{p_values[i]:.2e}"
    )

# =====================
# 7. 繪圖：|Spearman correlation|
# =====================
pc_labels = [f"PC{i+1}" for i in range(n_components)]

plt.figure(figsize=(12, 4))
plt.bar(pc_labels, abs_corrs)
plt.ylim(0, 1.0)

plt.xlabel("Principal Components")
plt.ylabel("|Spearman Correlation with Age Group (0–6)|")
plt.title("VoxCeleb2 ECAPA-TDNN Strength of Correlation between PCA Components and Age")

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("./result/ECAPA-TDNN/VoxCeleb2_ECAPA-TDNN_Age_PCA_Spearman_Correlation.png", dpi=300)
plt.show()
