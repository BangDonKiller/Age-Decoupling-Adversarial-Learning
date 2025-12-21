import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# --- 1. 載入資料 ---
data = torch.load("result/ECAPA-TDNN/ECAPA-TDNN_embeddings.pt")
embeddings = data["embeddings"]      # Tensor [N, D]
genders = data["genders"]            # list[int] or list[str]

# --- 2. 資料準備 ---
X = embeddings.cpu().numpy()         # [N, D]
y = np.array(genders)                # [N]

# --- 3. 執行 PCA，降到 3 維 --- # <-- 變更
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)         # [N, 3]

# --- 4. 繪圖設定 (與之前相同，確保顏色一致) ---
plt.style.use('seaborn-v0_8-whitegrid')

# 你的 label_map 看起來是 'f' -> 'female', 'm' -> 'male'
label_map = {
    'f': 'female',
    'm': 'male'
}
color_map = {
    'female': '#c44e52', # 較柔和的紅色
    'male': '#4c72b0'    # 較柔和的藍色
}

# --- 5. 繪製 3D 散佈圖 --- # <-- 變更
fig = plt.figure(figsize=(10, 8))               # 建立畫布
ax = fig.add_subplot(111, projection="3d")      # 建立 3D 座標軸

for gender_label in np.unique(y):
    # 找到對應性別的資料點索引
    idx = (y == gender_label)
    
    # 從對應表中取得圖例名稱和顏色
    legend_name = label_map[gender_label]
    point_color = color_map[legend_name]
    
    ax.scatter(
        X_pca[idx, 0],  # PC1
        X_pca[idx, 1],  # PC2
        X_pca[idx, 2],  # PC3 <-- 變更：加入第三個維度
        color=point_color,
        label=legend_name,
        alpha=0.5,      # 透明度
        s=15            # 點的大小
    )

# --- 6. 美化與儲存圖表 --- # <-- 變更
ax.set_xlabel("PC1", fontsize=12)
ax.set_ylabel("PC2", fontsize=12)
ax.set_zlabel("PC3", fontsize=12) # <-- 變更：加入 Z 軸標籤
ax.set_title("ECAPA-TDNN Embeddings PCA (3D)", fontsize=14)
ax.legend()

# 如果想從特定角度儲存圖片，可以取消註解並調整下面的參數
# ax.view_init(elev=20., azim=-35) # elev 是仰角, azim 是方位角

plt.savefig("result/ECAPA-TDNN/ECAPA-TDNN_embeddings_pca3d.png", dpi=300)
plt.show()