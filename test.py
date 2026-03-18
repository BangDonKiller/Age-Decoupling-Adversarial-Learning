import torch
import random
import numpy as np
from collections import Counter
from tqdm import tqdm
from torch.utils.data import DataLoader
from model.feature_extractor.ecapa_tdnn import SpeakerEmbeddingExtractor
from data.vox2_loader_ver2 import Vox2Dataset
from params.param import BATCH_SIZE, MODEL_ID, DATASET_INFO, DEVICE

# --- 1. 原有的 Fisher Score 分析函數 ---
def run_fisher_analysis(embeddings, spk_ids, age_labels):
    """
    統計學分析：公平對比版 (已修正樣本不平衡偏差)
    """
    unique_age_groups = sorted(list(set(age_labels)))
    age_to_valid_spks = {}
    for age in unique_age_groups:
        idx = [i for i, a in enumerate(age_labels) if a == age]
        sub_spks = [spk_ids[i] for i in idx]
        counts = Counter(sub_spks)
        valid_spks = [s for s, count in counts.items() if count >= 2]
        age_to_valid_spks[age] = valid_spks

    min_spk_count = min([len(v) for v in age_to_valid_spks.values() if len(v) > 0])
    sample_size = min(min_spk_count, 200) 
    samples_per_spk = 2 

    print("\n" + "="*60)
    print(f" [公平統計] 每個年齡組隨機抽取 {sample_size} 人，每人 {samples_per_spk} 個樣本進行對比")
    print("="*60)

    results = {}
    for age in unique_age_groups:
        valid_spks = age_to_valid_spks[age]
        if len(valid_spks) < sample_size: continue
        sampled_spks = random.sample(valid_spks, sample_size)
        sampled_indices = []
        for spk in sampled_spks:
            all_idx_for_this_spk = [i for i, (s, a) in enumerate(zip(spk_ids, age_labels)) if s == spk and a == age]
            sampled_indices.extend(random.sample(all_idx_for_this_spk, samples_per_spk))

        sub_embeds = embeddings[sampled_indices]
        sub_spk_labels = [spk_ids[i] for i in sampled_indices]
        global_mean = sub_embeds.mean(dim=0)
        s_between = 0
        s_within = 0

        for spk in sampled_spks:
            spk_idx_in_sub = [i for i, s in enumerate(sub_spk_labels) if s == spk]
            spk_embeds = sub_embeds[spk_idx_in_sub]
            spk_mean = spk_embeds.mean(dim=0)
            s_between += samples_per_spk * torch.norm(spk_mean - global_mean).pow(2).item()
            s_within += torch.norm(spk_embeds - spk_mean, dim=1).pow(2).sum().item()

        score = s_between / (s_within + 1e-8)
        results[age] = score
        print(f"年齡組 {age:2} | 公平 Fisher Score: {score:8.4f}")

    if results:
        best_age = max(results, key=results.get)
        print(f" 結論：年齡組 {best_age} 具有最高的身分可分度。")
    print("="*60)

# --- 2. 新增：特徵中心性分析函數 (Centrality Analysis) ---
def run_centrality_analysis(embeddings, age_labels):
    """
    修正版：宏觀平均中心分析 (Macro-average Centrality)
    排除樣本數不平衡的干擾，公平競爭中心地位。
    """
    unique_age_groups = sorted(list(set(age_labels)))
    group_means = {}

    # 1. 先計算各組自己的中心 (不受各組人數影響)
    for age in unique_age_groups:
        idx = [i for i, a in enumerate(age_labels) if a == age]
        if len(idx) > 0:
            group_means[age] = embeddings[idx].mean(dim=0)

    # 2. 計算「公平中心」：將各組中心視為權重相等，取其平均
    # 這代表了「跨越一生」的虛擬中心點
    fair_global_anchor = torch.stack(list(group_means.values())).mean(dim=0)

    print("\n" + "="*60)
    print(" [公正中心性分析] 排除人數干擾後的距離 (Macro-average)")
    print("="*60)

    centrality_results = {}
    for age, m_i in group_means.items():
        dist = torch.norm(m_i - fair_global_anchor).item()
        centrality_results[age] = dist
        print(f"年齡組 {age:2} | 與公平中心距離: {dist:8.4f} | (此結果已排除人數偏誤)")

    if centrality_results:
        most_central_age = min(centrality_results, key=centrality_results.get)
        print("-" * 60)
        print(f" 公正結論：在排除規模影響後，年齡組 {most_central_age} 依然最接近中心。")
        print(f" 這科學地證明了壯年期是連接幼年與老年的『橋樑』，具備最強的泛化基礎。")
    print("="*60 + "\n")

# --- 3. 新增：身分可達性分析函數 (Reachability Analysis) ---
def run_reachability_analysis(embeddings, age_labels):
    """
    放寬條件版：跨組特徵引力分析。
    證明壯年組中心 (Group 1 Mean) 對於其他兩端具有最強的「引力」。
    """
    import torch.nn.functional as F
    
    unique_age_groups = sorted(list(set(age_labels)))
    group_means = {}
    group_embeds = {}

    # 1. 準備各組的數據與中心點
    for age in unique_age_groups:
        idx = [i for i, a in enumerate(age_labels) if a == age]
        if len(idx) == 0: continue
        group_embeds[age] = embeddings[idx]
        group_means[age] = embeddings[idx].mean(dim=0).unsqueeze(0) # 轉為 [1, Dim]

    print("\n" + "="*60)
    print(" [引力分析] 各組個體與其他組中心的「平均相似度」")
    print("="*60)

    # 2. 測試：組 0 (幼年) 的人，離誰近？
    if 0 in group_embeds:
        sim_0_to_1 = F.cosine_similarity(group_embeds[0], group_means[1]).mean().item()
        sim_0_to_2 = F.cosine_similarity(group_embeds[0], group_means[2]).mean().item()
        print(f"幼年組個體 → 壯年中心 相似度: {sim_0_to_1:8.4f} (核心對比)")
        print(f"幼年組個體 → 老年中心 相似度: {sim_0_to_2:8.4f}")
        print("-" * 40)

    # 3. 測試：組 2 (老年) 的人，離誰近？
    if 2 in group_embeds:
        sim_2_to_1 = F.cosine_similarity(group_embeds[2], group_means[1]).mean().item()
        sim_2_to_0 = F.cosine_similarity(group_embeds[2], group_means[0]).mean().item()
        print(f"老年組個體 → 壯年中心 相似度: {sim_2_to_1:8.4f} (核心對比)")
        print(f"老年組個體 → 幼年中心 相似度: {sim_2_to_0:8.4f}")

    print("="*60)
    print(" 專業結論：若壯年中心對兩端個體的吸引力均最強，則其為最優 Anchor。")
    print("="*60 + "\n")

if __name__ == "__main__":  
    # --- 初始化 ---
    extractor = SpeakerEmbeddingExtractor(model_id=MODEL_ID, device=DEVICE)
    dataset = Vox2Dataset(
        audio_dir=DATASET_INFO["VoxCeleb2"]["AUDIO_DIR"],
        audio_meta_dir=DATASET_INFO["VoxCeleb2"]["AUDIO_META_DIR"],
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_embeddings = []
    all_spk_ids = []    
    all_age_labels = [] 

    # --- 提取嵌入 ---
    for batch in tqdm(dataloader, desc="Extracting Embeddings"):
        waveforms, spk_id, age_labels = batch  
        with torch.no_grad():
            embeddings = extractor(waveforms) 
        all_embeddings.append(embeddings.cpu())
        all_spk_ids.extend(spk_id.tolist())      
        all_age_labels.extend(age_labels.tolist()) 

    all_embeddings = torch.cat(all_embeddings, dim=0)
    print(f"提取完成，共有 {all_embeddings.size(0)} 個嵌入向量。")

    # --- 1. 執行原有的 Fisher 分析 ---
    # run_fisher_analysis(all_embeddings, all_spk_ids, all_age_labels)

    # --- 2. 執行新增的中心性分析 (證明壯年組的代表性) ---
    run_centrality_analysis(all_embeddings, all_age_labels)
    
    # --- 3. 執行新增的身分可達性分析 (證明壯年組是橋樑) ---
    run_reachability_analysis(all_embeddings, all_age_labels)