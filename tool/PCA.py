import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import torch
import plotly.express as px

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


def plot_pca_2d(X_pca, y, label_map, color_map, title, save_path=None):
    """
    X_pca: np.ndarray [N, 2]
    y: np.ndarray [N]
    """
    plt.figure(figsize=(8, 6))
    plt.style.use('seaborn-v0_8-whitegrid')

    for gender_label in np.unique(y):
        idx = (y == gender_label)
        legend_name = label_map[gender_label]
        point_color = color_map[legend_name]

        plt.scatter(
            X_pca[idx, 0],
            X_pca[idx, 1],
            color=point_color,
            label=legend_name,
            alpha=0.5,
            s=15
        )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.legend()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    plt.show()


def plot_pca_3d_interactive_safe(X_pca, y, label_map, save_path):
    assert X_pca.shape[1] == 3, "X_pca must be [N,3]"

    print("X_pca shape:", X_pca.shape)
    print("y unique:", np.unique(y))

    labels = []
    for i in y:
        if i not in label_map:
            raise ValueError(f"Label {i} not in label_map")
        labels.append(label_map[i])

    fig = px.scatter_3d(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        z=X_pca[:, 2],
        color=labels,
        opacity=0.6
    )

    fig.update_traces(marker=dict(size=4))
    fig.update_layout(title="PCA 3D Interactive")

    fig.write_html(save_path)

def subsample_for_visualization(X, y, max_points=50000, seed=42):
    np.random.seed(seed)
    n = len(X)

    if n <= max_points:
        return X, y

    idx = np.random.choice(n, max_points, replace=False)
    return X[idx], y[idx]



# 載入資料
data = torch.load("./result/ECAPA-TDNN/VoxCeleb2_ECAPA-TDNN_embeddings.pt")
embeddings = data["embeddings"]
y = np.array(data["genders"])

# 把embeddings做L2正規化
embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

label_map = {'f': 'female', 'm': 'male'}
color_map = {'female': '#c44e52', 'male': '#4c72b0'}

# === 2D PCA ===
X_pca_2d, _ = pca_reduce(embeddings, n_components=2)
plot_pca_2d(
    X_pca_2d,
    y,
    label_map,
    color_map,
    title="GLOBE ECAPA-TDNN Embeddings PCA (2D)",
    save_path="./result/ECAPA-TDNN/GLOBE_ECAPA-TDNN_embeddings_pca2d.png"
)
print("2D PCA plot saved.")

# === 3D PCA ===
X_pca_3d, _ = pca_reduce(embeddings, n_components=3)
X_vis, y_vis = subsample_for_visualization(
    X_pca_3d,
    y,
    max_points=50000
)

plot_pca_3d_interactive_safe(
    X_vis,
    y_vis,
    label_map,
    "./result/ECAPA-TDNN/GLOBE_ECAPA-TDNN_embeddings_pca3d_interactive.html"
)


print("3D PCA plot saved.")