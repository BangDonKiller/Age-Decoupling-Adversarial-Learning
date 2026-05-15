import torch
import numpy as np

def analyze_variance_ratio(file_path):
    print(f"正在分析檔案: {file_path}")
    
    # 1. 載入資料
    try:
        data = torch.load(file_path, map_location='cpu')
        embeddings = data["embeddings"]
        # 確保轉為 numpy 格式，如果是 Tensor 則轉 numpy
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.numpy()
        
        genders = np.array(data["genders"])
        
    except Exception as e:
        print(f"讀取錯誤: {e}")
        return

    # 2. 找出唯一的性別標籤 (例如 'Male', 'Female' 或 0, 1)
    unique_labels = np.unique(genders)
    if len(unique_labels) != 2:
        print(f"警告: 偵測到的性別類別數量不是 2 (偵測到: {unique_labels})，此分析主要針對二元分類。")
        # 這裡為了演示，假設只有前兩個 label
        unique_labels = unique_labels[:2]

    label_A, label_B = unique_labels[0], unique_labels[1]
    
    # 3. 分組數據
    # 篩選出 A 類 (e.g., Male) 和 B 類 (e.g., Female) 的嵌入向量
    group_A = embeddings[genders == label_A]
    group_B = embeddings[genders == label_B]
    
    print(f"  - 類別 {label_A}: {len(group_A)} 筆資料")
    print(f"  - 類別 {label_B}: {len(group_B)} 筆資料")

    # 4. 計算中心點 (Centroids / Mean Vectors)
    # axis=0 代表沿著 batch 維度取平均，得到 (192,) 的向量
    mean_A = np.mean(group_A, axis=0)
    mean_B = np.mean(group_B, axis=0)
    
    # 5. 計算 Sb (Between-class Scatter): 中心點距離的平方
    # 這是衡量兩個性別中心離多遠
    sb = np.sum((mean_A - mean_B) ** 2)
    
    # 6. 計算 Sw (Within-class Scatter): 類別內變異
    # 計算 A 類每個點到 A 類中心的距離平方的平均
    var_A = np.mean(np.sum((group_A - mean_A) ** 2, axis=1))
    # 計算 B 類每個點到 B 類中心的距離平方的平均
    var_B = np.mean(np.sum((group_B - mean_B) ** 2, axis=1))
    
    # 總體內部變異 (取平均)
    sw = (var_A + var_B) / 2
    
    # 7. 計算比率
    ratio = sb / sw
    
    print("-" * 30)
    print(f"  [結果統計]")
    print(f"  Sb (性別間隔訊號強度): {sb:.4f}")
    print(f"  Sw (內部環境雜訊強度): {sw:.4f}")
    print(f"  Ratio (Sb / Sw):      {ratio:.4f}")
    print("-" * 30)
    return sb, sw, ratio

# ==========================================
# 執行區域
# ==========================================

# 請將這裡替換成你實際的檔案路徑
file_paths = [
    "./result/ECAPA-TDNN/VoxCeleb2_ECAPA-TDNN_embeddings.pt",
    "./result/ECAPA-TDNN/VoxCeleb1_ECAPA-TDNN_embeddings.pt",
    "./result/ECAPA-TDNN/Librispeech_ECAPA-TDNN_embeddings.pt",
    "./result/ECAPA-TDNN/GLOBE_ECAPA-TDNN_embeddings.pt"
]

for path in file_paths:
    analyze_variance_ratio(path)