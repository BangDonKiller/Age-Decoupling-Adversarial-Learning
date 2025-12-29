import sys
from pathlib import Path

# 設定路徑 (保留你的原始設定)
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import torch
import torchaudio.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

# 引用你的模組
from model.feature_extractor.speechbrain_model import SpeakerEmbeddingExtractor
from params.param import DATASET_INFO, MODEL_ID, DEVICE, BATCH_SIZE
# from data.vox2_loader import InferenceDataset
from data.librispeech_loader import InferenceDataset

# ==========================================
# 1. 定義聲學特徵提取函數 (Kaldi Pitch Batch)
# ==========================================
def extract_acoustic_features_batch(waveforms, sr=16000):
    """
    waveforms: (Batch, Samples)
    return: f0_means, f0_stds
    """
    waveforms = waveforms.cpu()

    batch_size = waveforms.size(0)
    means, stds = [], []

    for i in range(batch_size):
        wav = waveforms[i]

        # 回傳每一 frame 的 f0（Hz），無聲為 0
        f0 = F.detect_pitch_frequency(
            wav,
            sample_rate=sr,
            frame_time=0.01,
            win_length=25
        )

        valid_f0 = f0[f0 > 0]

        if valid_f0.numel() > 0:
            means.append(valid_f0.mean().item())
            stds.append(valid_f0.std().item())
        else:
            means.append(0.0)
            stds.append(0.0)

    return means, stds

# ==========================================
# 2. 模型 Embedding 提取
# ==========================================
def get_model_embedding_batch(model, waveforms, device):
    """
    輸入: (Batch, Samples)
    輸出: numpy array (Batch, 192)
    """
    with torch.no_grad():
        waveforms = waveforms.to(device)
        embeddings = model(waveforms) 
        return embeddings.cpu().numpy()

# ==========================================
# 3. 主程式
# ==========================================
def main():
    DATASET = "LibriSpeech"
    
    if DATASET not in DATASET_INFO:
        print(f"Dataset {DATASET} not found in DATASET_INFO.")
        return

    AUDIO_DIR = DATASET_INFO[DATASET]["AUDIO_DIR"]
    AUDIO_META_DIR = DATASET_INFO[DATASET]["AUDIO_META_DIR"]
    
    # 初始化 Dataset
    print(f"Loading {DATASET} Dataset...")
    dataset = InferenceDataset(AUDIO_DIR, AUDIO_META_DIR)
    
    # num_workers=4 可以加速讀檔，如果報錯可以改回 0
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    num_samples = len(dataset)
    print(f"Analyzing {num_samples} samples with Batch Size {BATCH_SIZE}...")

    # --- 載入模型 ---
    print(f"Loading Model: {MODEL_ID} on {DEVICE}")
    device = DEVICE
    model = SpeakerEmbeddingExtractor(MODEL_ID, device=device)
    model.eval()
    # ----------------

    data_records = []
    embeddings_list = []

    # 迴圈讀取資料 (Batch Processing)
    for batch in tqdm(dataloader, desc="Extracting Features"):
        # 解包 Batch
        waveforms, spk_ids, genders = batch
        
        # 1. 批量提取聲學特徵 (極速版)
        f0_means, f0_stds = extract_acoustic_features_batch(waveforms, sr=16000)
        
        # 2. 批量提取 Embedding
        embs = get_model_embedding_batch(model, waveforms, device)
        
        # 3. 將 Batch 資料拆開存入 List
        current_batch_size = waveforms.size(0)
        
        for i in range(current_batch_size):
            g_val = genders[i] 
            # 處理性別標籤
            gender_code = 0 if g_val == 'M' else 1
            
            data_records.append({
                "Speaker": spk_ids[i],
                "Gender": gender_code,
                "F0_Mean": f0_means[i],
                "F0_Std": f0_stds[i]
            })
            
            embeddings_list.append(embs[i])

    # 轉成 DataFrame
    df_features = pd.DataFrame(data_records)
    X_emb = np.array(embeddings_list)

    print(f"Features extracted. Shape: {X_emb.shape}")
    
    # 確保資料量足夠做 10 個主成分
    n_comps = min(10, X_emb.shape[0], X_emb.shape[1])
    if n_comps < 2:
        print("Not enough data for PCA.")
        return

    print("Running PCA...")
    # 4. 執行 PCA
    pca = PCA(n_components=n_comps) 
    pca_result = pca.fit_transform(X_emb)
    
    # 把 PC 分數合併進 DataFrame
    for k in range(n_comps):
        df_features[f"PC{k+1}"] = pca_result[:, k]

    # ==========================================
    # 4. 計算相關性矩陣並畫圖
    # ==========================================
    print("Calculating Correlation...")
    
    target_cols = ["F0_Mean", "F0_Std", "Gender"]
    pc_cols = [f"PC{k+1}" for k in range(n_comps)]
    
    # 計算相關係數 (Pearson)
    corr_matrix = df_features[pc_cols + target_cols].corr()
    
    # 取出我們感興趣的區塊
    plot_data = corr_matrix.loc[pc_cols, target_cols]

    # 畫熱力圖
    plt.figure(figsize=(10, 8))
    sns.heatmap(plot_data, annot=True, cmap="coolwarm", center=0, vmin=-1, vmax=1, fmt=".2f")
    plt.title(f"Correlation: {DATASET} PCA Components vs Acoustic Features")
    plt.xlabel("Physical Features")
    plt.ylabel("Principal Components (Embedding)")
    plt.tight_layout()
    
    save_path = f"{DATASET}_ECAPA-TDNN_pc_f0_correlation.png"
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")
    # plt.show() # 如果是在 Server 上跑可以註解掉這行

    # 輸出文字報告
    print("\nTop Correlations:")
    for pc in pc_cols:
        best_feat = plot_data.loc[pc].abs().idxmax()
        score = plot_data.loc[pc, best_feat]
        print(f"{pc} is most correlated with {best_feat}: {score:.3f}")

if __name__ == "__main__":
    main()