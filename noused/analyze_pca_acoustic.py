import sys
import os
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
import opensmile # 核心引入

# ==========================================
# 設定路徑與引用模組
# ==========================================
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# 請確保這些模組路徑正確
from model.feature_extractor.ecapa_tdnn_ver2 import SpeakerEmbeddingExtractor
from params.param import DATASET_INFO, MODEL_ID, DEVICE, BATCH_SIZE

# 根據你的資料集選擇 Loader
# from data.librispeech_loader import InferenceDataset
from data.vox2_loader import InferenceDataset
# from data.vox1_loader import InferenceDataset
# from data.GLOBE_loader import InferenceDataset

# ==========================================
# 1. OpenSMILE 初始化
# ==========================================
# eGeMAPSv02 是最常用的語音特徵集，包含 F0, Jitter, Shimmer, Formants 等 88 個特徵
# Functionals 代表我們計算一段音訊的統計值(平均、標準差)，而不是每個 frame 的值
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

def extract_opensmile_features(wav_tensor, sr=16000):
    """
    使用 OpenSMILE 提取 eGeMAPS 特徵 (88維)
    """
    # 轉成 numpy 並確保是一維 array (samples,)
    wav_np = wav_tensor.squeeze().cpu().numpy()
    
    try:
        # OpenSMILE 處理
        # process_signal 回傳一個 DataFrame，包含 88 個欄位
        df_feat = smile.process_signal(wav_np, sr)
        
        # 轉成字典 (key: 特徵名, value: 數值)
        # eGeMAPS 的特徵名稱通常很長，例如 'F0semitoneFrom27.5Hz_sma3nz_amean'
        # 我們直接使用原始名稱，稍後在圖表上再觀察
        feat_dict = df_feat.iloc[0].to_dict()
        return feat_dict
        
    except Exception as e:
        # 若音訊過短或全靜音可能導致錯誤
        return None

# ==========================================
# 2. Embedding 提取
# ==========================================
def get_model_embedding_batch(model, waveforms, device):
    with torch.no_grad():
        waveforms = waveforms.to(device)
        embeddings = model(waveforms) 
        return embeddings.cpu().numpy()

# ==========================================
# 3. 主程式
# ==========================================
def main():
    # --- 設定 ---
    DATASET = "VoxCeleb2" # 或 "VoxCeleb1"
    NUM_WORKERS = 4
    
    if DATASET not in DATASET_INFO:
        print(f"Dataset {DATASET} not found.")
        return

    AUDIO_DIR = DATASET_INFO[DATASET]["AUDIO_DIR"]
    AUDIO_META_DIR = DATASET_INFO[DATASET]["AUDIO_META_DIR"] # 如果有的話
    
    print(f"Loading {DATASET} Dataset...")

    # dataset = InferenceDataset(AUDIO_DIR)
    dataset = InferenceDataset(AUDIO_DIR, AUDIO_META_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print(f"Loading Model: {MODEL_ID} on {DEVICE}")
    model = SpeakerEmbeddingExtractor(MODEL_ID, device=DEVICE)
    model.eval()

    results = []
    
    print("Starting Feature Extraction (Embeddings + OpenSMILE)...")
    
    for batch in tqdm(dataloader, desc="Processing Batches"):
        waveforms, spk_ids, genders = batch
        
        # A. 提取 Embedding (GPU Batch)
        embs = get_model_embedding_batch(model, waveforms, DEVICE)
        
        # B. 提取 OpenSMILE 特徵 (CPU Loop)
        current_bs = waveforms.size(0)
        
        for i in range(current_bs):
            # 1. 提取 OpenSMILE
            smile_feats = extract_opensmile_features(waveforms[i], sr=16000)
            
            if smile_feats is None:
                continue # 跳過失敗樣本
            
            # 2. 處理性別
            g_raw = genders[i]
            # 簡單正規化性別字串
            if isinstance(g_raw, str):
                gender_code = 0 if g_raw.lower() in ['m', 'male'] else 1
            else:
                gender_code = 0 # Fallback
            
            # 3. 建立資料條目
            entry = {
                "Speaker": spk_ids[i],
                "Gender": gender_code,
            }
            
            # 加入 Embedding
            for dim, val in enumerate(embs[i]):
                entry[f"emb_{dim}"] = val
                
            # 加入 OpenSMILE 特徵 (88個)
            entry.update(smile_feats)
            
            results.append(entry)

    # --- 轉成 DataFrame ---
    df_raw = pd.DataFrame(results)
    print(f"Raw data shape: {df_raw.shape}")
    
    if df_raw.empty:
        print("No data extracted.")
        return

    # ==========================================
    # 4. Speaker-Level Averaging (消除環境雜訊)
    # ==========================================
    print("Performing Speaker-Level Averaging...")
    
    # 分辨哪些欄位是數值 (Embedding + OpenSMILE features)
    # 排除 Speaker, Gender
    numeric_cols = [c for c in df_raw.columns if c not in ['Speaker', 'Gender']]
    
    agg_dict = {col: 'mean' for col in numeric_cols}
    agg_dict['Gender'] = 'first' # 性別取第一個即可
    
    df_speaker = df_raw.groupby("Speaker").agg(agg_dict).reset_index()
    print(f"Final Speaker-Level data shape: {df_speaker.shape}")

    # ==========================================
    # 5. PCA 分析
    # ==========================================
    print("Running PCA on Speaker Embeddings...")
    
    emb_cols = [c for c in df_speaker.columns if c.startswith("emb_")]
    X_spk = df_speaker[emb_cols].values
    
    n_comps = min(5, X_spk.shape[0]) # 只看前 5 個主成分
    pca = PCA(n_components=n_comps)
    pca_result = pca.fit_transform(X_spk)
    
    pc_cols = []
    for k in range(n_comps):
        col_name = f"PC{k+1}"
        df_speaker[col_name] = pca_result[:, k]
        pc_cols.append(col_name)
        
    print(f"PCA Explained Variance: {pca.explained_variance_ratio_}")

    # ==========================================
    # 6. 相關性計算與篩選
    # ==========================================
    print("Calculating Correlations...")
    
    # 找出所有的 OpenSMILE 特徵欄位 (排除 emb_, PC, Speaker, Gender)
    smile_feature_cols = [c for c in df_speaker.columns 
                          if c not in emb_cols and c not in pc_cols and c not in ['Speaker', 'Gender']]
    
    # 加入性別一起比較
    target_cols = smile_feature_cols + ['Gender']
    
    # 計算相關矩陣 (只算 PC 與 特徵 的部分)
    # rows: PC1~PC5, cols: 88 features + Gender
    corr_matrix = df_speaker[pc_cols + target_cols].corr().loc[pc_cols, target_cols]
    
    # --- 篩選最相關的特徵畫圖 (因為 88 個太多了) ---
    # 計算每個特徵與任一 PC 的最大相關係數絕對值
    max_corr_per_feature = corr_matrix.abs().max(axis=0)
    
    # 取出相關性最高的 Top 20 特徵
    top_n = 20
    top_features = max_corr_per_feature.sort_values(ascending=False).head(top_n).index.tolist()
    
    # 確保 Gender 在裡面，方便觀察
    if 'Gender' not in top_features:
        top_features.insert(0, 'Gender')
    
    plot_data = corr_matrix[top_features]

    # ==========================================
    # 7. 繪圖
    # ==========================================
    plt.figure(figsize=(14, 8))
    sns.heatmap(plot_data.T, annot=True, cmap="coolwarm", center=0, vmin=-1, vmax=1, fmt=".2f")
    # 注意: 把特徵放在 Y 軸 (plot_data.T)，PC 放在 X 軸，這樣比較好閱讀長檔名
    
    plt.title(f"{DATASET} (Speaker-Avg): Top {top_n} Features correlated with PCA")
    plt.xlabel("Principal Components")
    plt.ylabel("Acoustic Features (OpenSMILE eGeMAPSv02)")
    
    output_filename = f"{DATASET}_OpenSMILE_Analysis.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_filename}")
    
    # ==========================================
    # 8. 文字報告
    # ==========================================
    print("\n=== Analysis Report: Top Determinants for each PC ===")
    for pc in pc_cols:
        # 找出該 PC 最相關的三個特徵
        top3 = corr_matrix.loc[pc].abs().sort_values(ascending=False).head(3)
        print(f"\n{pc} is most correlated with:")
        for feat, val in top3.items():
            original_sign = corr_matrix.loc[pc, feat] # 取回正負號
            print(f"  - {feat}: {original_sign:.3f}")

if __name__ == "__main__":
    main()