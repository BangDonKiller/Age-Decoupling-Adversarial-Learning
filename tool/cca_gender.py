import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler

def perform_gender_cca(csv_path, dataset_name):
    print(f"\nLoading data for {dataset_name} from {csv_path}...")
    
    # 讀取 CSV
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File {csv_path} not found. Please run the previous analysis script first.")
        return

    # ==========================================
    # 1. 資料前處理
    # ==========================================
    # 確保有 Gender 欄位
    if 'Gender' not in df.columns:
        print("Error: 'Gender' column not found in CSV.")
        return

    # 將 Gender 轉為數值 (0: Male, 1: Female)
    # 根據你的資料格式可能需要調整，這裡假設是 0/1 或是 'Male'/'Female' 字串
    if df['Gender'].dtype == 'object':
        df['Gender_Code'] = df['Gender'].apply(lambda x: 0 if str(x).lower().startswith('m') else 1)
    else:
        df['Gender_Code'] = df['Gender']

    # 取出 Embedding Columns
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    
    # Speaker-Level Averaging (非常重要，消除環境雜訊)
    # 我們對 Embedding 取平均，對 Gender 取第一個值(因為同一個人性別不會變)
    agg_dict = {col: 'mean' for col in emb_cols}
    agg_dict['Gender_Code'] = 'first'
    
    df_spk = df.groupby("Speaker").agg(agg_dict).reset_index()
    print(f"Data shape after speaker averaging: {df_spk.shape}")

    X = df_spk[emb_cols].values  # (N_speakers, 192)
    Y = df_spk[['Gender_Code']].values # (N_speakers, 1)

    # 標準化
    scaler_x = StandardScaler()
    # Y 是 0/1 類別，通常不需要標準化，但為了 CCA 算法穩定性，做一下也無妨
    # 不過為了直觀理解 R^2，我們下面會用 Linear Regression 輔助驗證
    X_sc = scaler_x.fit_transform(X)
    
    # ==========================================
    # 2. 執行 CCA (或等效分析)
    # ==========================================
    # 當 Y 只有 1 維時，CCA 最多只能找到 1 個 Component
    n_comps = 1
    cca = CCA(n_components=n_comps)
    
    # 擬合
    X_c, Y_c = cca.fit_transform(X_sc, Y)
    
    # 計算相關係數
    correlation = np.corrcoef(X_c[:, 0], Y_c[:, 0])[0, 1]
    
    print(f"\n--- Result for {dataset_name} ---")
    print(f"Canonical Correlation (Embedding vs Gender): {correlation:.4f}")
    print(f"R-squared (Variability of Gender explained by Embedding): {correlation**2:.4f}")

    # ==========================================
    # 3. 額外驗證：線性可分性 (Linear Separability)
    # ==========================================
    # 為了畫出漂亮的圖，我們看 Embedding 在第一典型變數 (CV1) 上的投影分佈
    
    plt.figure(figsize=(8, 6))
    
    # 建立一個 DataFrame 來畫圖
    plot_df = pd.DataFrame({
        'Canonical Variable 1 (Projected Embedding)': X_c[:, 0],
        'Gender': df_spk['Gender_Code'].map({0: 'Male', 1: 'Female'})
    })
    
    # 畫出直方圖 (Histogram) 或 KDE
    sns.histplot(data=plot_df, x='Canonical Variable 1 (Projected Embedding)', hue='Gender', kde=True, bins=30, palette={'Male': 'blue', 'Female': 'red'}, alpha=0.6)
    
    plt.title(f"{dataset_name}: Gender Separation in CCA Space\n(Correlation = {correlation:.3f})")
    plt.xlabel("Canonical Variate 1 (The 'Gender Axis' found by Model)")
    plt.ylabel("Count")
    
    output_file = f"{dataset_name}_Gender_CCA.png"
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")
    
    # ==========================================
    # 4. 分析 Embedding 中哪些維度跟性別最相關 (Optional)
    # ==========================================
    # 這可以告訴我們 Embedding 的哪幾個維度負責管性別
    # 計算 X (原始 Embedding) 與 CV1 (性別軸) 的相關係數
    loadings = np.corrcoef(X_sc.T, X_c.T)[:X.shape[1], X.shape[1]:]
    loadings = loadings.flatten() # 轉成 1D array
    
    # 找出絕對值最大的前 5 個維度
    top_indices = np.argsort(np.abs(loadings))[::-1][:10]
    
    print("\nTop 10 Embedding Dimensions most correlated with Gender:")
    for idx in top_indices:
        print(f"  Emb_Dim_{idx}: correlation = {loadings[idx]:.3f}")

def main():
    datasets = {
        "VoxCeleb1": "./result/ECAPA-TDNN/VoxCeleb1_raw_data.csv",
        # "Librispeech": "./result/ECAPA-TDNN/Librispeech_raw_data.csv" ,
        "VoxCeleb2": "./result/ECAPA-TDNN/VoxCeleb2_raw_data.csv",
        # "GLOBE": "./result/ECAPA-TDNN/GLOBE_raw_data.csv",
    }
    
    for name, path in datasets.items():
        if os.path.exists(path):
            perform_gender_cca(path, name)
        else:
            print(f"Skipping {name}: {path} not found.")

if __name__ == "__main__":
    main()