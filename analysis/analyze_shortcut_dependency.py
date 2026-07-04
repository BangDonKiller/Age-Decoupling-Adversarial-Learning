import os
from pathlib import Path
import warnings
import argparse

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm

from params.param import DATASET_INFO, MODEL_ID, DEVICE
from model.feature_extractor.ecapa_tdnn_ver2 import SpeakerEmbeddingExtractor
from model.disentangled_model.linear_decorr_mlp import LinearDecorrMLP

warnings.filterwarnings(
    "ignore",
    message=r".*torchaudio\.load_with_torchcodec.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*StreamingMediaDecoder has been deprecated.*",
    category=UserWarning,
)

mpl.rcParams["font.sans-serif"] = [
    "Microsoft JhengHei",
    "Microsoft YaHei",
    "Noto Sans CJK TC",
    "Noto Sans CJK SC",
    "SimHei",
    "DejaVu Sans",
]
mpl.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")

# ==========================================
# 基本設定
# ==========================================
DEVICE = torch.device(DEVICE if isinstance(DEVICE, str) else str(DEVICE))
DATASET_NAME_DEFAULT = "Vox-O"
EASY_NEG_MAX_SAMPLES = 10000
HARD_NEG_MAX_SAMPLES = 10000

# 模型架構設定
SPEAKER_EMB_DIM = 192
AGE_DIM = 1
MLP_HIDDEN_DIMS = [256]
MLP_OUTPUT_DIM = 193
NUM_SPEAKERS = 5128
NUM_AGE_GROUPS = 7


def find_checkpoint_in_exp(exp_path: Path):
    """在 exp 資料夾中尋找 epoch_020_model.pth，若無則回傳 best_model.pth 或 last_model.pth"""
    if not exp_path.exists():
        return None
    
    epoch_020 = exp_path / "epoch_020_model.pth"
    if epoch_020.exists():
        return epoch_020
    
    for name in ("best_model.pth", "last_model.pth"):
        p = exp_path / name
        if p.exists():
            return p
    
    return None


def load_checkpoint_to_model(model: torch.nn.Module, ckpt_path: Path, device: torch.device):
    """從 checkpoint 載入模型權重"""
    checkpoint = torch.load(ckpt_path, map_location=device)
    state = checkpoint.get("model", checkpoint)
    cleaned = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(cleaned, strict=False)
    print(f"✓ Checkpoint 已載入: {ckpt_path}")


def load_raw_age_lookup(meta_path: str):
    """讀取 VoxCeleb1 metadata，建立 (speaker_id, utterance) -> 原始年齡 的查詢表。"""
    lookup = {}
    df = pd.read_csv(meta_path, sep=",", header=0, usecols=[0, 1, 2, 3], dtype={"age": str})

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading meta", leave=False):
        speaker = row["speaker_id"] if "speaker_id" in df.columns else row.iloc[0]
        utt = row["utterance"] if "utterance" in df.columns else row.iloc[1]
        age_str = row["age"] if "age" in df.columns else row.iloc[2]

        try:
            lookup[(str(speaker).strip(), str(utt).strip())] = float(age_str)
        except (TypeError, ValueError):
            continue

    return lookup


def load_pairs_from_vox_o_txt(datalist_path: str, audio_dirs, raw_age_lookup, rng, easy_max_samples=20000, hard_max_samples=20000, max_lines=None):
    """從 Vox-O datalist 直接讀取配對並解析為已解析的 audio pairs。
    datalist 格式預期與常見的 pairwise 試驗檔相同：
        is_same spk1_rel_path spk2_rel_path
    回傳項目格式：(wav1_path, age1, wav2_path, age2, label, subset)
    """
    resolved = []
    lines = []
    with open(datalist_path, "r", encoding="utf-8") as f:
        for i, l in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            l = l.strip()
            if not l:
                continue
            lines.append(l)

    easy_count = 0
    hard_count = 0

    for line in tqdm(lines, desc="Reading Vox-O datalist", leave=False):
        try:
            parts = line.split()
            if len(parts) < 3:
                # 嘗試以逗號分割
                parts = [p.strip() for p in line.replace(',', ' ').split()]
                if len(parts) < 3:
                    continue

            label = int(parts[0])
            spk1_rel = parts[1]
            spk2_rel = parts[2]

            # 找到實際檔案
            wav1 = find_audio_path(spk1_rel, audio_dirs)
            wav2 = find_audio_path(spk2_rel, audio_dirs)

            # 取得 speaker id 與 utterance id
            spk1_id = spk1_rel.split('/')[0]
            spk2_id = spk2_rel.split('/')[0]
            spk1_utt = spk1_rel.split('/')[1] if '/' in spk1_rel else None
            spk2_utt = spk2_rel.split('/')[1] if '/' in spk2_rel else None

            age1 = None
            age2 = None
            if raw_age_lookup:
                if spk1_utt:
                    age1 = raw_age_lookup.get((spk1_id, spk1_utt), None)
                if spk2_utt:
                    age2 = raw_age_lookup.get((spk2_id, spk2_utt), None)

            if age1 is None or age2 is None:
                # 跳過缺少年齡標籤
                continue

            age_gap = abs(age1 - age2)
            if label == 1:
                subset = "positive"
                resolved.append((wav1, age1, wav2, age2, 1, subset))
            else:
                if age_gap > 15:
                    subset = "easy_negative"
                    if easy_count >= easy_max_samples:
                        continue
                    resolved.append((wav1, age1, wav2, age2, 0, subset))
                    easy_count += 1
                elif age_gap <= 2:
                    subset = "hard_negative"
                    if hard_count >= hard_max_samples:
                        continue
                    resolved.append((wav1, age1, wav2, age2, 0, subset))
                    hard_count += 1
                else:
                    # skip negatives that are neither easy nor hard
                    continue

            # 如果兩個子集都達到上限，提前停止
            if easy_count >= easy_max_samples and hard_count >= hard_max_samples:
                break
        except Exception:
            continue

    return resolved


def find_audio_path(relative_path, audio_dirs):
    for audio_dir in audio_dirs:
        audio_path = Path(audio_dir) / relative_path
        if audio_path.exists():
            return str(audio_path)
    raise FileNotFoundError(f"Audio file {relative_path} not found in any of the provided directories.")


def load_audio_tensor(wav_path: str):
    signal, sr = torchaudio.load(wav_path)
    if sr != 16000:
        resampler = T.Resample(orig_freq=sr, new_freq=16000)
        signal = resampler(signal)
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)
    return signal.squeeze(0)


def generate_pairs_from_meta(raw_age_lookup, rng, easy_max_samples=20000, hard_max_samples=20000):
    """
        從 meta 信息生成「負對」配對列表（先在 meta 階段切分）。
        負對：不同說話者，依 age_gap 切分為
            - Easy Negatives: age_gap > 15
            - Hard Negatives: age_gap <= 2
        返回配對列表，每項為
            (speaker1, session1, age1, speaker2, session2, age2, label, subset)
    """
        # 組織 meta 信息：speaker_id -> sessions
    speaker_sessions_meta = {}
    speaker_ages = {}
    
    for (speaker_id, utterance), age in raw_age_lookup.items():
        if speaker_id not in speaker_ages:
            speaker_ages[speaker_id] = age
        
        # meta 只有會話資訊，直接視為 session id
        session_id = utterance if "/" not in utterance else utterance.split("/")[0]
        
        if speaker_id not in speaker_sessions_meta:
            speaker_sessions_meta[speaker_id] = {}
        if session_id not in speaker_sessions_meta[speaker_id]:
            speaker_sessions_meta[speaker_id][session_id] = True
    
    pairs = []
    easy_count = 0
    hard_count = 0
    speaker_list = list(speaker_sessions_meta.keys())

    # 生成負對（不同說話者）
    for idx1 in tqdm(range(len(speaker_list)), desc="Building meta negative pairs", leave=False):
        if easy_count >= easy_max_samples and hard_count >= hard_max_samples:
            break

        for idx2 in range(idx1 + 1, len(speaker_list)):
            if easy_count >= easy_max_samples and hard_count >= hard_max_samples:
                break

            speaker1, speaker2 = speaker_list[idx1], speaker_list[idx2]
            age1, age2 = speaker_ages[speaker1], speaker_ages[speaker2]
            
            sessions1 = list(speaker_sessions_meta[speaker1].keys())
            sessions2 = list(speaker_sessions_meta[speaker2].keys())
            
            if not sessions1 or not sessions2:
                continue
            
            age_gap = abs(age1 - age2)
            if age_gap > 15:
                subset = "easy_negative"
                if easy_count >= easy_max_samples:
                    continue
            elif age_gap <= 2:
                subset = "hard_negative"
                if hard_count >= hard_max_samples:
                    continue
            else:
                continue

            # 會話先固定（語句晚點在 audio dir 隨機挑）
            session1 = rng.choice(sessions1)
            session2 = rng.choice(sessions2)

            pairs.append((speaker1, session1, age1, speaker2, session2, age2, 0, subset))

            if subset == "easy_negative":
                easy_count += 1
            else:
                hard_count += 1
    
    return pairs


def find_session_wavs(speaker_id, session_id, audio_dirs):
    """
    從音頻目錄中查找某 speaker/session 下所有可用語句 wav。
    """
    wavs = []
    for audio_dir in audio_dirs:
        audio_path = Path(audio_dir)
        session_path = audio_path / speaker_id / session_id
        if session_path.exists() and session_path.is_dir():
            wavs.extend([str(p) for p in session_path.glob("*.wav")])
    return wavs


def resolve_pairs_to_audio(pairs, audio_dirs, rng):
    """
    將 meta 配對映射為實際 audio 配對。
    meta 只有會話，這裡在每個會話中隨機挑一段語句。
    """
    resolved = []
    session_wavs_cache = {}

    for speaker1, session1, age1, speaker2, session2, age2, label, subset in tqdm(
        pairs,
        total=len(pairs),
        desc="Resolving pairs to audio",
        leave=False,
    ):
        key1 = (speaker1, session1)
        key2 = (speaker2, session2)

        if key1 not in session_wavs_cache:
            session_wavs_cache[key1] = find_session_wavs(speaker1, session1, audio_dirs)
        if key2 not in session_wavs_cache:
            session_wavs_cache[key2] = find_session_wavs(speaker2, session2, audio_dirs)

        wavs1 = session_wavs_cache[key1]
        wavs2 = session_wavs_cache[key2]
        if len(wavs1) == 0 or len(wavs2) == 0:
            continue

        wav1_path = rng.choice(wavs1)
        wav2_path = rng.choice(wavs2)
        resolved.append((wav1_path, age1, wav2_path, age2, label, subset))

    return resolved


def compute_pair_scores_with_age_gap(resolved_pairs, speaker_extractor, model, device, progress_desc="Scoring pairs"):
    """
    從配對列表計算相似度分數與年齡差。
    每對格式：(wav1_path, age1, wav2_path, age2, label, subset)
    返回 DataFrame。
    """
    records = []
    
    with torch.no_grad():
        for wav1_path, age1, wav2_path, age2, label, subset in tqdm(
            resolved_pairs,
            total=len(resolved_pairs),
            desc=progress_desc,
            leave=False,
        ):
            try:
                wav1 = load_audio_tensor(wav1_path).to(device)
                wav2 = load_audio_tensor(wav2_path).to(device)
                
                emb1 = speaker_extractor(wav1.unsqueeze(0))
                emb2 = speaker_extractor(wav2.unsqueeze(0))
                
                out1 = model(emb1)["z"]
                out2 = model(emb2)["z"]
                
                v1 = out1.squeeze().cpu().numpy()
                v2 = out2.squeeze().cpu().numpy()
                
                num = np.dot(v1, v2)
                den = np.linalg.norm(v1) * np.linalg.norm(v2)
                cosine = float(num / (den + 1e-12))
                
                age_gap = abs(age1 - age2)
                records.append({
                    "label": label,
                    "score": cosine,
                    "age_gap": age_gap,
                    "age1": age1,
                    "age2": age2,
                    "subset": subset,
                })
            except Exception:
                continue
    
    return pd.DataFrame(records)


def create_subsets(df):
    """
    根據 label 和 age_gap 切分數據。
    - Positives: label == 1
    - Easy Negatives: label == 0 且 age_gap > 15
    - Hard Negatives: label == 0 且 age_gap <= 2
    """
    positives = df[df["label"] == 1].copy()
    easy_negatives = df[(df["label"] == 0) & (df["age_gap"] > 15)].copy()
    hard_negatives = df[(df["label"] == 0) & (df["age_gap"] <= 2)].copy()
    
    return {
        "Positives": positives,
        "Easy Negatives": easy_negatives,
        "Hard Negatives": hard_negatives,
    }


def print_statistics(subsets):
    """列印每個子集的統計資訊。"""
    print("\n" + "=" * 80)
    print("數據統計")
    print("=" * 80)
    for subset_name, subset_df in subsets.items():
        if len(subset_df) > 0:
            scores = subset_df["score"].values
            age_gaps = subset_df["age_gap"].values
            print(f"\n{subset_name}:")
            print(f"  樣本數: {len(subset_df)}")
            print(f"  平均分: {scores.mean():.4f}")
            print(f"  標準差: {scores.std():.4f}")
            print(f"  最小值: {scores.min():.4f}")
            print(f"  最大值: {scores.max():.4f}")
            print(f"  平均年齡差: {age_gaps.mean():.2f}")


def plot_analysis(baseline_df, decoupled_df, output_prefix):
    """
    繪製三張子圖：
    1. Score Distribution (KDE)
    2. FAR vs Threshold
    3. Shortcut Dependency (Bar Chart)
    """
    # ==========================================
    # 切分子集
    # ==========================================
    baseline_subsets = create_subsets(baseline_df)
    decoupled_subsets = create_subsets(decoupled_df)
    
    print("\n--- Baseline Model Statistics ---")
    print_statistics(baseline_subsets)
    print("\n--- Decoupled Model Statistics ---")
    print_statistics(decoupled_subsets)
    
    # ==========================================
    # 建立 1x3 的子圖
    # ==========================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # ==========================================
    # 圖 1: Score Distribution (KDE Density Plot)
    # ==========================================
    ax1 = axes[0]
    colors_map = {
        "Easy Negatives": "#FF6B6B",
        "Hard Negatives": "#FF9999",
    }
    
    # Baseline - Easy Negatives
    if len(baseline_subsets["Easy Negatives"]) > 0:
        baseline_subsets["Easy Negatives"]["score"].plot.kde(
            ax=ax1, linewidth=2, label="Baseline - Easy Negatives", color="#FF6B6B"
        )
    
    # Baseline - Hard Negatives
    if len(baseline_subsets["Hard Negatives"]) > 0:
        baseline_subsets["Hard Negatives"]["score"].plot.kde(
            ax=ax1, linewidth=2, label="Baseline - Hard Negatives", color="#FF9999"
        )
    
    # Decoupled - Easy Negatives
    if len(decoupled_subsets["Easy Negatives"]) > 0:
        decoupled_subsets["Easy Negatives"]["score"].plot.kde(
            ax=ax1, linewidth=2, label="Decoupled - Easy Negatives", color="#4ECDC4"
        )
    
    # Decoupled - Hard Negatives
    if len(decoupled_subsets["Hard Negatives"]) > 0:
        decoupled_subsets["Hard Negatives"]["score"].plot.kde(
            ax=ax1, linewidth=2, label="Decoupled - Hard Negatives", color="#8DD9D2"
        )
    
    ax1.set_xlabel("Cosine Similarity Score", fontsize=11)
    ax1.set_ylabel("Density", fontsize=11)
    ax1.set_title("Score Distribution", fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, loc="best")
    ax1.grid(True, alpha=0.3)
    
    # ==========================================
    # 圖 2: FAR vs Threshold
    # ==========================================
    ax2 = axes[1]
    thresholds = np.linspace(-0.5, 0.5, 100)
    
    # Baseline - Easy Negatives
    if len(baseline_subsets["Easy Negatives"]) > 0:
        easy_neg = baseline_subsets["Easy Negatives"]["score"].values
        far_easy = [(easy_neg > t).sum() / len(easy_neg) for t in thresholds]
        ax2.plot(thresholds, far_easy, linewidth=2, label="Baseline - Easy Negatives", color="#FF6B6B")
    
    # Baseline - Hard Negatives
    if len(baseline_subsets["Hard Negatives"]) > 0:
        hard_neg = baseline_subsets["Hard Negatives"]["score"].values
        far_hard = [(hard_neg > t).sum() / len(hard_neg) for t in thresholds]
        ax2.plot(thresholds, far_hard, linewidth=2, label="Baseline - Hard Negatives", color="#FF9999")
    
    # Decoupled - Easy Negatives
    if len(decoupled_subsets["Easy Negatives"]) > 0:
        easy_neg_dec = decoupled_subsets["Easy Negatives"]["score"].values
        far_easy_dec = [(easy_neg_dec > t).sum() / len(easy_neg_dec) for t in thresholds]
        ax2.plot(thresholds, far_easy_dec, linewidth=2, label="Decoupled - Easy Negatives", color="#4ECDC4")
    
    # Decoupled - Hard Negatives
    if len(decoupled_subsets["Hard Negatives"]) > 0:
        hard_neg_dec = decoupled_subsets["Hard Negatives"]["score"].values
        far_hard_dec = [(hard_neg_dec > t).sum() / len(hard_neg_dec) for t in thresholds]
        ax2.plot(thresholds, far_hard_dec, linewidth=2, label="Decoupled - Hard Negatives", color="#8DD9D2")
    
    ax2.set_xlabel("Threshold", fontsize=11)
    ax2.set_ylabel("False Accept Rate (FAR)", fontsize=11)
    ax2.set_title("FAR vs Threshold", fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, loc="best")
    ax2.grid(True, alpha=0.3)
    
    # ==========================================
    # 圖 3: Shortcut Dependency (Bar Chart)
    # ==========================================
    ax3 = axes[2]
    
    # 計算 Score Delta
    delta_baseline = np.nan
    delta_decoupled = np.nan
    
    if len(baseline_subsets["Hard Negatives"]) > 0 and len(baseline_subsets["Easy Negatives"]) > 0:
        hard_mean = baseline_subsets["Hard Negatives"]["score"].mean()
        easy_mean = baseline_subsets["Easy Negatives"]["score"].mean()
        delta_baseline = hard_mean - easy_mean
    
    if len(decoupled_subsets["Hard Negatives"]) > 0 and len(decoupled_subsets["Easy Negatives"]) > 0:
        hard_mean_dec = decoupled_subsets["Hard Negatives"]["score"].mean()
        easy_mean_dec = decoupled_subsets["Easy Negatives"]["score"].mean()
        delta_decoupled = hard_mean_dec - easy_mean_dec
    
    models = ["Baseline", "Decoupled"]
    deltas = [delta_baseline, delta_decoupled]
    colors_bar = ["#FF6B6B", "#4ECDC4"]
    
    bars = ax3.bar(models, deltas, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # 在柱子上標註數值
    for bar, delta in zip(bars, deltas):
        height = bar.get_height()
        if not np.isnan(height):
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax3.set_ylabel("Score Delta (Hard - Easy)", fontsize=11)
    ax3.set_title("Shortcut Dependency\n(Higher = More Age-Gap Dependent)", fontsize=12, fontweight='bold')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_prefix, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n✓ 分析圖表已儲存為 {output_prefix}")


if __name__ == "__main__":
    # ==========================================
    # 命令列參數解析
    # ==========================================
    parser = argparse.ArgumentParser(
        description="Shortcut Dependency Analysis: Age-Gap Impact on Cosine Similarity."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DATASET_NAME_DEFAULT,
        help=f"Dataset name (default: {DATASET_NAME_DEFAULT}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for pair/session/utterance sampling.",
    )
    parser.add_argument(
        "--vox-o-datalist",
        type=str,
        default=None,
        help="Path to Vox-O datalist (overrides DATASET_INFO AUDIO_DATALIST).",
    )
    parser.add_argument(
        "--easy-max-samples",
        type=int,
        default=EASY_NEG_MAX_SAMPLES,
        help=f"Max number of easy negatives (default: {EASY_NEG_MAX_SAMPLES}).",
    )
    parser.add_argument(
        "--hard-max-samples",
        type=int,
        default=HARD_NEG_MAX_SAMPLES,
        help=f"Max number of hard negatives (default: {HARD_NEG_MAX_SAMPLES}).",
    )
    args = parser.parse_args()
    DATASET_NAME = args.dataset
    rng = np.random.default_rng(args.seed)

    # 動態生成輸出檔名
    ds_suffix = DATASET_NAME.lower().replace("-", "_")
    OUTPUT_PLOT = f"shortcut_dependency_{ds_suffix}.png"

    print("=" * 100)
    print(f"Shortcut Dependency Analysis ({DATASET_NAME} Dataset)")
    print("=" * 100)
    
    # ==========================================
    # 初始化資料集
    # ==========================================
    ds_info = DATASET_INFO.get("VoxCeleb1", {}).get(DATASET_NAME)
    if ds_info is None:
        raise RuntimeError(f"找不到 DATASET_INFO 中的 {DATASET_NAME}")

    audio_dirs = ds_info["AUDIO_DIR"]
    meta_csv = DATASET_INFO["VoxCeleb1"]["AUDIO_META_DIR"]
    vox_o_datalist = args.vox_o_datalist or ds_info.get("AUDIO_DATALIST")

    print(f"\n載入原始年齡查詢表...")
    raw_age_lookup = load_raw_age_lookup(meta_csv)

    # 如果使用 Vox-O，直接從 datalist 讀取配對並解析（省去 meta 階段配對）
    if DATASET_NAME.lower().startswith("vox-o") and vox_o_datalist is not None:
        print(f"從 Vox-O datalist 讀取配對: {vox_o_datalist} ...")
        resolved_pairs = load_pairs_from_vox_o_txt(
            vox_o_datalist,
            audio_dirs,
            raw_age_lookup,
            rng,
            easy_max_samples=args.easy_max_samples,
            hard_max_samples=args.hard_max_samples,
        )
        print(f"✓ 直接從 datalist解析並套用上限後，得到 {len(resolved_pairs)} 對可用音檔配對")

        # 不輸出 CSV，只顯示簡單統計
        easy_count = sum(1 for r in resolved_pairs if r[5] == "easy_negative")
        hard_count = sum(1 for r in resolved_pairs if r[5] == "hard_negative")
        pos_count = sum(1 for r in resolved_pairs if r[5] == "positive")
        print(f"  Easy Negatives（resolved）: {easy_count} (上限 {args.easy_max_samples})")
        print(f"  Hard Negatives（resolved）: {hard_count} (上限 {args.hard_max_samples})")
        print(f"  Positives（resolved）: {pos_count}")
    else:
        print(f"從 meta 生成並切分負對（easy/hard negatives）...")
        print(f"  Easy Negatives 上限: {args.easy_max_samples}")
        print(f"  Hard Negatives 上限: {args.hard_max_samples}")
        pairs = generate_pairs_from_meta(
            raw_age_lookup,
            rng=rng,
            easy_max_samples=args.easy_max_samples,
            hard_max_samples=args.hard_max_samples,
        )
        print(f"✓ 生成 {len(pairs)} 對 meta 配對")

        easy_meta = sum(1 for p in pairs if p[6] == 0 and p[7] == "easy_negative")
        hard_meta = sum(1 for p in pairs if p[6] == 0 and p[7] == "hard_negative")
        print(f"  Easy Negatives（meta）: {easy_meta}")
        print(f"  Hard Negatives（meta）: {hard_meta}")

        print(f"從 audio dir 解析配對並在會話內隨機選語句...")
        resolved_pairs = resolve_pairs_to_audio(pairs, audio_dirs, rng=rng)
        print(f"✓ 可用音檔配對數 = {len(resolved_pairs)}")

        easy_resolved = sum(1 for p in resolved_pairs if p[4] == 0 and p[5] == "easy_negative")
        hard_resolved = sum(1 for p in resolved_pairs if p[4] == 0 and p[5] == "hard_negative")
        print(f"  Easy Negatives（resolved）: {easy_resolved}")
        print(f"  Hard Negatives（resolved）: {hard_resolved}")

    # ==========================================
    # 初始化模型
    # ==========================================
    print(f"初始化說話者特徵抽取器 ({MODEL_ID})...")
    speaker_extractor = SpeakerEmbeddingExtractor(MODEL_ID, device=str(DEVICE))
    speaker_extractor.eval()

    CHECKPOINT_DIR = Path("checkpoints/linear_decorr_mlp")
    BASELINE_EXP = "exp44"
    DECOUPLED_EXP = "exp35"

    # ==========================================
    # 載入 Baseline 模型
    # ==========================================
    print(f"\n載入 Baseline 模型 ({BASELINE_EXP})...")
    baseline_exp_path = CHECKPOINT_DIR / BASELINE_EXP
    baseline_ckpt = find_checkpoint_in_exp(baseline_exp_path)
    if baseline_ckpt is None:
        raise RuntimeError(f"找不到 {BASELINE_EXP} 的 checkpoint")

    baseline_model = LinearDecorrMLP(
        input_dim=SPEAKER_EMB_DIM,
        hidden_dims=MLP_HIDDEN_DIMS,
        output_dim=MLP_OUTPUT_DIM,
        num_speakers=NUM_SPEAKERS,
        num_age_groups=NUM_AGE_GROUPS,
        age_dim=AGE_DIM,
    )
    baseline_model.to(DEVICE)
    baseline_model.eval()
    load_checkpoint_to_model(baseline_model, baseline_ckpt, DEVICE)

    print("計算 Baseline 模型的分數...")
    baseline_df = compute_pair_scores_with_age_gap(
        resolved_pairs,
        speaker_extractor,
        baseline_model,
        DEVICE,
        progress_desc="Scoring Baseline",
    )
    print(f"✓ Baseline 完成：樣本數 = {len(baseline_df)}")

    # ==========================================
    # 載入 Decoupled 模型
    # ==========================================
    print(f"\n載入 Decoupled 模型 ({DECOUPLED_EXP})...")
    decoupled_exp_path = CHECKPOINT_DIR / DECOUPLED_EXP
    decoupled_ckpt = find_checkpoint_in_exp(decoupled_exp_path)
    if decoupled_ckpt is None:
        raise RuntimeError(f"找不到 {DECOUPLED_EXP} 的 checkpoint")

    decoupled_model = LinearDecorrMLP(
        input_dim=SPEAKER_EMB_DIM,
        hidden_dims=MLP_HIDDEN_DIMS,
        output_dim=MLP_OUTPUT_DIM,
        num_speakers=NUM_SPEAKERS,
        num_age_groups=NUM_AGE_GROUPS,
        age_dim=AGE_DIM,
    )
    decoupled_model.to(DEVICE)
    decoupled_model.eval()
    load_checkpoint_to_model(decoupled_model, decoupled_ckpt, DEVICE)

    print("計算 Decoupled 模型的分數...")
    decoupled_df = compute_pair_scores_with_age_gap(
        resolved_pairs,
        speaker_extractor,
        decoupled_model,
        DEVICE,
        progress_desc="Scoring Decoupled",
    )
    print(f"✓ Decoupled 完成：樣本數 = {len(decoupled_df)}")

    # ==========================================
    # 繪製分析圖表
    # ==========================================
    print("\n繪製分析圖表...")
    plot_analysis(baseline_df, decoupled_df, OUTPUT_PLOT)

    print("\n" + "=" * 100)
    print("分析完成！")
    print("=" * 100)
