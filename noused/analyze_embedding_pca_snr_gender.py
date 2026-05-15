import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from tqdm import tqdm
# from data.vox2_loader import InferenceDataset
# from data.vox1_loader import InferenceDataset
from data.GLOBE_loader import InferenceDataset
# from data.librispeech_loader import InferenceDataset
from model.feature_extractor.ecapa_tdnn_ver2 import SpeakerEmbeddingExtractor
from params.param import DEVICE, BATCH_SIZE, DATASET_INFO, MODEL_ID


# 設定你的路徑
DATASET = "GLOBE"  # "VoxCeleb1", "VoxCeleb2", "LibriSpeech", "GLOBE"
AUDIO_DIR = DATASET_INFO[DATASET]["AUDIO_DIR"]
# META_DIR = DATASET_INFO[DATASET]["AUDIO_META_DIR"]

def calculate_approx_snr(waveform, frame_length=0.020, hop_length=0.010, sample_rate=16000):
    """
    基於能量分佈估算 SNR (Signal-to-Noise Ratio)。
    原理：假設音檔中能量最高的前 5% 是 Speech，能量最低的前 10% 是 Noise (背景音)。
    """
    # 轉為 numpy
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.squeeze().numpy()
    
    # 簡單的防呆，避免全靜音導致 log error
    if np.sum(waveform**2) < 1e-9:
        return 0.0

    # 1. 切分 Frame (Framing)
    frame_size = int(frame_length * sample_rate)
    hop_size = int(hop_length * sample_rate)
    
    # 簡單計算每個 Frame 的能量 (Energy)
    # 為了效能，這裡用簡單的 sliding window
    num_frames = 1 + (len(waveform) - frame_size) // hop_size
    energies = []
    
    for i in range(num_frames):
        start = i * hop_size
        end = start + frame_size
        frame = waveform[start:end]
        energy = np.sum(frame ** 2) / frame_size
        energies.append(energy)
    
    energies = np.array(energies)
    
    # 避免 log(0)
    energies = np.maximum(energies, 1e-10)

    # 2. 定義 Signal 和 Noise 的能量閾值
    # 假設：能量最大的前 10% 是說話聲 (Signal)
    # 假設：能量最小的前 10% 是背景雜訊 (Noise)
    # 注意：你的 dataset 有做 zero-padding，這會導致很多 0 能量。
    # 我們需要排除掉 padding 的全 0 部分，只看有聲音的部分。
    valid_energies = energies[energies > 1e-9]
    
    if len(valid_energies) == 0:
        return 0.0

    signal_power = np.percentile(valid_energies, 90) # P90 作為訊號強度代理
    noise_power = np.percentile(valid_energies, 10)  # P10 作為噪聲強度代理
    
    # 3. 計算 SNR (dB)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def run_analysis(model, device="cuda"):
    """
    主分析流程
    Args:
        model: 已經 load 好的 ECAPA-TDNN 模型
    """
    # 1. 初始化 Dataset 與 DataLoader
    # dataset = InferenceDataset(AUDIO_DIR, META_DIR)
    dataset = InferenceDataset(AUDIO_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    embeddings_list = []
    snr_list = []
    gender_list = [] # 0: Male, 1: Female

    print("Extracting features and calculating SNR...")
    model.eval()
    model.to(device)

    with torch.no_grad():
        for waveforms, speaker_ids, genders in tqdm(dataloader):
            waveforms = waveforms.to(device)
            
            # --- A. 取得 Embeddings ---
            # 假設你的模型 forward 輸出就是 embedding
            # 如果你的模型需要特定的 input shape，請在這裡調整
            emb_batch = model(waveforms) 
            embeddings_list.append(emb_batch.cpu().numpy())

            # --- B. 計算 SNR ---
            # SNR 計算在 CPU 上做即可
            waveforms_np = waveforms.cpu()
            for i in range(waveforms_np.shape[0]):
                snr = calculate_approx_snr(waveforms_np[i])
                snr_list.append(snr)
                
                # 簡單將性別轉為數字
                g_code = 0 if genders[i] == 'male' else 1
                gender_list.append(g_code)

    # 整理數據
    X = np.concatenate(embeddings_list, axis=0) # shape: (N, 192)
    snr_values = np.array(snr_list)
    gender_values = np.array(gender_list)

    print(f"Data collected. Shape: {X.shape}")

    # --- C. 執行 PCA ---
    print("Running PCA...")
    pca = PCA(n_components=10) # 取前 10 個主成分
    X_pca = pca.fit_transform(X)

    # --- D. 計算相關係數 (Correlation) ---
    # 目標：計算 PC1, PC2... 與 SNR 的相關性
    correlations = []
    p_values = []
    
    # 同時計算與 Gender 的相關性做對比
    gender_corrs = []

    for i in range(10): # 對前 10 個 PC
        pc_scores = X_pca[:, i]
        
        # 計算與 SNR 的相關係數
        corr, p_val = pearsonr(pc_scores, snr_values)
        correlations.append(abs(corr)) # 取絕對值方便比較強度
        
        # 計算與 Gender 的相關係數 (Point-biserial 其實就是 Pearson)
        g_corr, _ = pearsonr(pc_scores, gender_values)
        gender_corrs.append(abs(g_corr))

        print(f"PC{i+1}: Corr with SNR = {corr:.4f}, Corr with Gender = {g_corr:.4f}")

    # --- E. 畫圖驗證 ---
    plot_results(correlations, gender_corrs, pca.explained_variance_ratio_)

def plot_results(snr_corrs, gender_corrs, explained_var):
    plt.figure(figsize=(12, 6))

    # 圖 1: PC 與 SNR/Gender 的相關性比較
    indices = np.arange(1, 11)
    width = 0.35

    plt.bar(indices - width/2, snr_corrs, width, label='Correlation with SNR (Noise)', color='orange')
    plt.bar(indices + width/2, gender_corrs, width, label='Correlation with Gender', color='skyblue')
    
    plt.xlabel('Principal Component (PC)')
    plt.ylabel('Absolute Pearson Correlation')
    plt.title(f'What does each PC represent? ({DATASET})')
    plt.xticks(indices)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)    
    plt.tight_layout()
    plt.savefig(
        f"{DATASET}_ECAPA-TDNN_pc_snr_gender_correlation.png",
        dpi=300,
    )
    plt.show()

if __name__ == "__main__":
    model = SpeakerEmbeddingExtractor(
        model_id=MODEL_ID,
        device=DEVICE
    )
    
    run_analysis(model, device=DEVICE)