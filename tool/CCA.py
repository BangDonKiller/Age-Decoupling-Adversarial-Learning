import sys
import os
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import opensmile

# ==========================================
# 設定路徑與引用模組
# ==========================================
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from model.feature_extractor.ecapa_tdnn import SpeakerEmbeddingExtractor
from params.param import DATASET_INFO, MODEL_ID, DEVICE, BATCH_SIZE
# from data.librispeech_loader import InferenceDataset
# from data.vox2_loader import InferenceDataset
# from data.vox1_loader import InferenceDataset
from data.GLOBE_loader import InferenceDataset

# ==========================================
# 1. 初始化 OpenSMILE
# ==========================================
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

def extract_features_and_embeddings(dataset_name, model, device):
    """
    這部分與之前的 analysis.py 類似，負責提取資料。
    如果有存好的 CSV，可以直接讀取 CSV 跳過這一步。
    """
    if dataset_name not in DATASET_INFO:
        print(f"Dataset {dataset_name} not found.")
        return None

    AUDIO_DIR = DATASET_INFO[dataset_name]["AUDIO_DIR"]
    AUDIO_META_DIR = DATASET_INFO[dataset_name].get("AUDIO_META_DIR", None)
    
    # 初始化 Loader (根據你的 Dataset 調整參數)
    # 注意: 這裡假設你的 InferenceDataset 構造函數參數一致
    if AUDIO_META_DIR:
        dataset = InferenceDataset(AUDIO_DIR, AUDIO_META_DIR)
    else:
        dataset = InferenceDataset(AUDIO_DIR)
        
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    results = []
    print(f"Extracting features for {dataset_name}...")

    for batch in tqdm(dataloader, desc="Processing"):
        waveforms, spk_ids, genders = batch
        
        # 1. Embedding
        with torch.no_grad():
            embs = model(waveforms.to(device)).cpu().numpy()
            
        # 2. OpenSMILE
        current_bs = waveforms.size(0)
        for i in range(current_bs):
            wav_np = waveforms[i].squeeze().cpu().numpy()
            try:
                feat_df = smile.process_signal(wav_np, 16000)
                feat_dict = feat_df.iloc[0].to_dict()
                
                entry = {
                    "Speaker": spk_ids[i],
                    "Gender": genders[i] if isinstance(genders[i], str) else "Unknown"
                }
                # 存 Embedding
                for d, v in enumerate(embs[i]):
                    entry[f"emb_{d}"] = v
                # 存 Acoustic Features
                entry.update(feat_dict)
                results.append(entry)
            except:
                continue
                
    return pd.DataFrame(results)

def perform_cca_analysis(df, dataset_name):
    print(f"\nRunning CCA for {dataset_name}...")

    # 1. 資料清洗與準備
    # ----------------------------------------
    # 取出 Embedding Columns (X)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    
    # 取出 OpenSMILE Columns (Y) - 排除非數值欄位
    exclude_cols = ["Speaker", "Gender"] + emb_cols
    acoustic_cols = [c for c in df.columns if c not in exclude_cols]
    
    # 先做 Speaker-Level Averaging (消除雜訊干擾)
    # 這一步非常重要，能讓 CCA 更專注於「說話者特質」
    numeric_cols = emb_cols + acoustic_cols
    df_spk = df.groupby("Speaker")[numeric_cols].mean().reset_index()
    
    print(f"Data shape after averaging: {df_spk.shape}")
    
    X = df_spk[emb_cols].values
    Y = df_spk[acoustic_cols].values
    
    # 2. 標準化 (Standardization)
    # ----------------------------------------
    # CCA 對數據尺度很敏感，必須先做標準化
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_sc = scaler_x.fit_transform(X)
    Y_sc = scaler_y.fit_transform(Y)
    
    # 3. 執行 CCA
    # ----------------------------------------
    # n_components 決定我們要找幾對「典型變數 (Canonical Variables)」
    # 通常找前 5~10 對就足夠說明問題
    n_comps = 5
    cca = CCA(n_components=n_comps)
    
    # 進行擬合與轉換
    # X_c, Y_c 是轉換後的典型變數
    X_c, Y_c = cca.fit_transform(X_sc, Y_sc)
    
    # 4. 計算典型相關係數 (Canonical Correlations)
    # ----------------------------------------
    # sklearn 的 CCA 沒有直接屬性回傳相關係數，我們需要自己算
    corrs = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(n_comps)]
    
    print(f"\nTop {n_comps} Canonical Correlations:")
    for i, r in enumerate(corrs):
        print(f"  Component {i+1}: {r:.4f}")
        
    # 5. 分析成分構成 (Loadings Analysis)
    # ----------------------------------------
    # 我們想知道：到底哪些聲學特徵構成了第一對典型變數？
    # 計算 Y (聲學特徵) 與 Y_c (典型變數) 的相關性
    loadings = np.corrcoef(Y_sc.T, Y_c.T)[:Y.shape[1], Y.shape[1]:]
    
    # 找出與第一典型變數 (Component 1) 最相關的 Top 10 聲學特徵
    comp1_loadings = loadings[:, 0]
    # 取得索引排序
    top_indices = np.argsort(np.abs(comp1_loadings))[::-1][:15]
    top_features = [acoustic_cols[i] for i in top_indices]
    top_scores = comp1_loadings[top_indices]
    
    # 6. 繪圖
    # ----------------------------------------
    plt.figure(figsize=(12, 6))
    
    # 圖 1: 相關係數 Bar Chart
    plt.subplot(1, 2, 1)
    x_labels = [f"CV{i+1}" for i in range(n_comps)]
    # 修改點: 加上 hue=x_labels 和 legend=False
    sns.barplot(x=x_labels, y=corrs, hue=x_labels, palette="viridis", legend=False)
    plt.ylim(0, 1.1)
    plt.title(f"{dataset_name}: CCA Correlations\n(Embedding vs Acoustic)")
    plt.ylabel("Canonical Correlation")
    plt.xlabel("Canonical Variates")
    
    # 圖 2: 第一典型變數的特徵貢獻
    plt.subplot(1, 2, 2)
    sns.barplot(x=top_scores, y=top_features, hue=top_features, palette="coolwarm", legend=False)
    plt.title(f"Top Features contributing to CV1\n(What did the model learn?)")
    plt.xlabel("Correlation with CV1")
    
    plt.tight_layout()
    plt.savefig(f"{dataset_name}_CCA_Analysis.png", dpi=300)
    print(f"Saved plot to {dataset_name}_CCA_Analysis.png")
    
    # 額外驗證：如果第一相關係數 > 0.8，說明模型確實學到了聲學特徵
    if corrs[0] > 0.7:
        print("\n[Conclusion] Strong correlation found! The model DID learn acoustic features.")
    else:
        print("\n[Conclusion] Weak correlation. The model might rely on other non-standard features.")

def main():
    model = SpeakerEmbeddingExtractor(MODEL_ID, device=DEVICE)
    model.eval()

    # 在這裡切換你要分析的資料集
    # 建議跑兩次，一次 VoxCeleb1，一次 Librispeech 做對比
    datasets_to_run = ["GLOBE"] 
    
    for ds_name in datasets_to_run:
        # 1. 取得資料
        df = extract_features_and_embeddings(ds_name, model, DEVICE)
        
        if df is not None and not df.empty:
            # 存檔以備份
            df.to_csv(f"{ds_name}_raw_data.csv", index=False)
            
            # 2. 執行 CCA
            perform_cca_analysis(df, ds_name)

if __name__ == "__main__":
    main()