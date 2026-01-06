import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import torch
import plotly.express as px

# =====================
# PCA functions
# =====================
def pca_reduce(embeddings, n_components=2):
    """
    embeddings: torch.Tensor [N, D]
    n_components: int (2 or 3)
    return: np.ndarray [N, n_components]
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

    X_out = []
    y_out = []

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
# 載入資料
# =====================
data = torch.load("./result/ECAPA-TDNN/GLOBE_ECAPA-TDNN_embeddings.pt")

embeddings = data["embeddings"]   # Tensor [N, D]
ages = data["ages"]               # list or tensor [N]

# =====================
# 前處理
# =====================
if torch.is_tensor(ages):
    ages = ages.cpu().numpy()
else:
    ages = np.array(ages)

# L2 normalize（與你之前流程一致）
embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

# =====================
# 2D PCA
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
    title="GLOBE ECAPA-TDNN Embeddings PCA",
    save_path="./result/ECAPA-TDNN/GLOBE_ECAPA-TDNN_Age_PCA_2D.png"
)

print("2D PCA plot saved.")

# =====================
# 3D PCA
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
    "./result/ECAPA-TDNN/GLOBE_ECAPA-TDNN_Age_PCA_3D_interactive.html"
)

print("3D PCA interactive plot saved.")
