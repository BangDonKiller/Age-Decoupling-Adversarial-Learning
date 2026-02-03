from model.feature_extractor.speechbrain_model import SpeakerEmbeddingExtractor
from data.vox2_loader_ver2 import Vox2Dataset
from torch.utils.data import DataLoader
from params.param import BATCH_SIZE, MODEL_ID, DATASET_INFO, DEVICE
import torch
from tqdm import tqdm

# --- 1. 定義計算 Fisher Score 的獨立函數 (完全不影響原始流程) ---
def run_fisher_analysis(embeddings, spk_ids, age_labels):
    """
    統計學分析：計算各年齡組的身分可分度 (Fisher Score)
    """
    unique_age_groups = sorted(list(set(age_labels)))
    results = {}

    print("\n" + "="*50)
    print(" [統計分析] 各年齡組身分穩定度 (Fisher Score) 報告")
    print("="*50)

    for age in unique_age_groups:
        # 篩選出屬於該年齡組的索引
        idx = [i for i, a in enumerate(age_labels) if a == age]
        if len(idx) < 10: continue # 樣本太少不計入

        # 該年齡組的數據
        sub_embeds = embeddings[idx]
        sub_spks = [spk_ids[i] for i in idx]
        unique_spks = list(set(sub_spks))
        
        if len(unique_spks) < 2: continue

        # 計算組內全局中心點
        global_mean = sub_embeds.mean(dim=0)
        s_between = 0
        s_within = 0

        for spk in unique_spks:
            # 找到該說話人的所有樣本索引
            spk_idx = [i for i, s in enumerate(sub_spks) if s == spk]
            spk_embeds = sub_embeds[spk_idx]
            
            n_i = spk_embeds.size(0)
            spk_mean = spk_embeds.mean(dim=0)
            
            # 類間方差 (身分間的區分度)
            s_between += n_i * torch.norm(spk_mean - global_mean).pow(2).item()
            
            # 類內方差 (同一個人的不穩定度)
            if n_i > 1:
                s_within += torch.norm(spk_embeds - spk_mean, dim=1).pow(2).sum().item()

        # Fisher Score = 區分度 / 不穩定度
        score = s_between / (s_within + 1e-8)
        results[age] = score
        print(f"年齡組 {age:2} | Fisher Score: {score:8.4f} | 樣本數: {len(idx):5} | 說話人數: {len(unique_spks):4}")

    print("="*50)
    if results:
        best_age = max(results, key=results.get)
        print(f" 統計結論：年齡組 {best_age} 具有最高的分離度與穩定性。")
        print(f" 建議：以此組作為 Gradient Regularization 的壯年期錨點 (Anchor)。")
    print("="*50 + "\n")


if __name__ == "__main__":  
    # --- 原本的功能：初始化 ---
    extractor = SpeakerEmbeddingExtractor(model_id=MODEL_ID, device=DEVICE)

    dataset = Vox2Dataset(
        audio_dir=DATASET_INFO["VoxCeleb2"]["AUDIO_DIR"],
        audio_meta_dir=DATASET_INFO["VoxCeleb2"]["AUDIO_META_DIR"],
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_embeddings = []
    all_spk_ids = []    # 新增以進行統計
    all_age_labels = [] # 新增以進行統計

    # --- 原本的功能：提取嵌入 ---
    # 這裡只做數據收集，不做任何計算
    for batch in tqdm(dataloader, desc="Extracting Embeddings"):
        waveforms, spk_id, age_labels = batch  
        
        with torch.no_grad():
            embeddings = extractor(waveforms) 
            
        all_embeddings.append(embeddings.cpu())
        all_spk_ids.extend(spk_id.tolist())      # 儲存 ID 用於統計
        all_age_labels.extend(age_labels.tolist()) # 儲存標籤用於統計

    # 將所有嵌入合併
    all_embeddings = torch.cat(all_embeddings, dim=0)

    print(f"提取完成，共有 {all_embeddings.size(0)} 個嵌入向量。")

    # --- 執行 Fisher 分析 (僅在最後執行一次) ---
    run_fisher_analysis(all_embeddings, all_spk_ids, all_age_labels)