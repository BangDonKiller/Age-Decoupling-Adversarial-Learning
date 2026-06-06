"""
三個 LoRA 專家的 ECAPA-TDNN 推論腳本。

這個腳本只做 inference：
    - 載入預訓練 ECAPA-TDNN backbone
    - 分別載入三個 LoRA expert adapter
    - 對每一筆 VoxCeleb1 pair 計算三個 expert 各自的 cosine score
    - 針對每一筆 pair，判定哪個 expert 的分數最接近標準答案，當作該局勝者
    - 同時記錄 pair 的年齡差
    - 將逐筆結果與整體 win rate 輸出成 CSV

預設使用 VoxCeleb1 的 Vox-O split；如果你要換成 Vox-E / Vox-H，
只要修改 `VAL_DATASET_NAME` 即可。
"""

import csv
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.vox1_loader import PairwiseDataset
from model.feature_extractor.ecapa_model import ECAPAModel
from params.param import DATASET_INFO, DEVICE, NUM_WORKERS
from tool.EER import compute_eer

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


# ==========================================
# 1) 基本設定
# ==========================================
ECAPA_CHANNELS = 1024
PRETRAINED_PATH = "pretrained_models/pretrain.model"

# 三個專家的 adapter 路徑。請依你的實際 checkpoint 調整。
EXPERT_ADAPTER_PATHS = [
    "checkpoints/ecapa_tdnn_lora/m4_r4_alpha8_LE5/lora_adapter",
    "checkpoints/ecapa_tdnn_lora/m4_r4_alpha8_Delta5/lora_adapter",
    "checkpoints/ecapa_tdnn_lora/m4_r4_alpha8_Delta20/lora_adapter",
]

# VoxCeleb1 pair split，預設 Vox-O。
VAL_DATASET_NAME = "Vox-CA20"

# cosine similarity 的「標準答案」定義。
# 正樣本越接近 1 越好；負樣本預設以 -1 作為標準答案。
NEGATIVE_TARGET_SCORE = -1.0

# 輸出位置
OUTPUT_DIR = Path("logs") / "ecapa_tdnn_lora" / "three_expert_inference"
OUTPUT_CSV = OUTPUT_DIR / f"{VAL_DATASET_NAME}_three_expert_scores.csv"
EER_SUMMARY_CSV = OUTPUT_DIR / f"{VAL_DATASET_NAME}_three_expert_eer_summary.csv"
THRESHOLD_BOXPLOT_PNG = OUTPUT_DIR / f"{VAL_DATASET_NAME}_three_expert_threshold_boxplot.png"


def build_test_loader() -> DataLoader:
    """建立 zero-shot pairwise dataloader。"""
    test_dataset = PairwiseDataset(
        audio_dir=DATASET_INFO["VoxCeleb1"][VAL_DATASET_NAME]["AUDIO_DIR"],
        audio_meta_dir=DATASET_INFO["VoxCeleb1"][VAL_DATASET_NAME]["AUDIO_DATALIST"],
        audio_meta_csv_path=DATASET_INFO["VoxCeleb1"]["AUDIO_META_DIR"],
    )
    return DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )


def load_expert_model(adapter_path: str) -> PeftModel:
    """載入一個已訓練完成的 LoRA expert。"""
    base_model = ECAPAModel(C=ECAPA_CHANNELS, n_class=5994, m=0.2, s=64.0)
    base_model = base_model.to(DEVICE)
    base_model.load_parameters(PRETRAINED_PATH)

    expert_model = PeftModel.from_pretrained(base_model, adapter_path)
    expert_model.eval()
    return expert_model


def get_speaker_encoder(model: torch.nn.Module) -> torch.nn.Module:
    """兼容 PeftModel wrapper 的 speaker_encoder 存取。"""
    if hasattr(model, "speaker_encoder"):
        return model.speaker_encoder
    if hasattr(model, "base_model") and hasattr(model.base_model, "speaker_encoder"):
        return model.base_model.speaker_encoder
    if hasattr(model, "model") and hasattr(model.model, "speaker_encoder"):
        return model.model.speaker_encoder
    if (
        hasattr(model, "base_model")
        and hasattr(model.base_model, "model")
        and hasattr(model.base_model.model, "speaker_encoder")
    ):
        return model.base_model.model.speaker_encoder
    raise AttributeError("找不到 speaker_encoder，請確認模型包裝結構是否正確。")


def extract_embedding(model: torch.nn.Module, waveform: torch.Tensor) -> torch.Tensor:
    """從單一 expert 提取 embedding。"""
    speaker_encoder = get_speaker_encoder(model)
    with torch.no_grad():
        return speaker_encoder.forward(waveform, aug=False)


def cosine_score(emb1: torch.Tensor, emb2: torch.Tensor) -> float:
    """計算兩個 embedding 的 cosine similarity。"""
    h1 = F.normalize(emb1, p=2, dim=1)
    h2 = F.normalize(emb2, p=2, dim=1)
    score = F.cosine_similarity(h1, h2).detach().cpu().item()
    return float(score)


def target_score(is_same_value: int) -> float:
    """依 pair 標籤決定標準答案分數。"""
    return 1.0 if is_same_value == 1 else NEGATIVE_TARGET_SCORE


def score_distance(score: float, target: float) -> float:
    """分數與標準答案的距離。越小代表越接近標準答案。"""
    return abs(score - target)


def main():
    if len(EXPERT_ADAPTER_PATHS) != 3:
        raise ValueError("目前腳本固定載入三個 expert，EXPERT_ADAPTER_PATHS 必須剛好有 3 個路徑。")

    print("建立資料集...")
    test_loader = build_test_loader()

    print("載入三個 expert...")
    experts = []
    for idx, adapter_path in enumerate(EXPERT_ADAPTER_PATHS, start=1):
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(f"找不到第 {idx} 個 expert adapter 路徑: {adapter_path}")
        print(f"  - Expert {idx}: {adapter_path}")
        experts.append(load_expert_model(adapter_path))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"開始推論，結果將寫入: {OUTPUT_CSV}")
    with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "pair_index",
                "is_same",
                "speaker_1",
                "speaker_2",
                "age_1_year",
                "age_2_year",
                "age_gap_year",
                "target_score",
                "expert_1_cosine",
                "expert_2_cosine",
                "expert_3_cosine",
                "expert_1_abs_error",
                "expert_2_abs_error",
                "expert_3_abs_error",
                "winner",
                "winner_abs_error",
                "cosine_mean",
            ]
        )

        win_counts = [0, 0, 0]
        tie_count = 0
        total_pairs = 0
        skipped_pairs = 0
        # 收集每個 expert 在正對/負對的 scores，用於後續統計
        pos_scores = [[], [], []]
        neg_scores = [[], [], []]
        # EER 計算需要每個 expert 的全資料分數與標籤
        all_scores = [[], [], []]
        all_labels = []
        # 收集每筆 pair 的勝出幅度，方便最後列出表現最明顯的前 10 筆例子
        top_examples = []

        for pair_index, batch in enumerate(tqdm(test_loader, desc="Pairs", unit="pair"), start=1):
            is_same, speaker_1, speaker_2, wav1, wav2, age1, age2 = batch

            wav1 = wav1.to(DEVICE, non_blocking=True)
            wav2 = wav2.to(DEVICE, non_blocking=True)

            age_1_year = int(age1.item())
            age_2_year = int(age2.item())
            if age_1_year == -1 or age_2_year == -1:
                skipped_pairs += 1
                continue

            age_gap_year = abs(age_1_year - age_2_year)
            is_same_value = int(is_same.item())
            target = target_score(is_same_value)
            speaker_1_value = speaker_1[0] if isinstance(speaker_1, (list, tuple)) else str(speaker_1)
            speaker_2_value = speaker_2[0] if isinstance(speaker_2, (list, tuple)) else str(speaker_2)

            scores = []
            for expert in experts:
                emb1 = extract_embedding(expert, wav1)
                emb2 = extract_embedding(expert, wav2)
                scores.append(cosine_score(emb1, emb2))

            # 記錄正/負對的 scores
            if is_same_value == 1:
                for i, s in enumerate(scores):
                    pos_scores[i].append(s)
            else:
                for i, s in enumerate(scores):
                    neg_scores[i].append(s)

            all_labels.append(is_same_value)
            for i, s in enumerate(scores):
                all_scores[i].append(s)

            distances = [score_distance(score, target) for score in scores]
            score_mean = sum(scores) / len(scores)
            best_distance = min(distances)
            winner_indices = [idx for idx, dist in enumerate(distances) if abs(dist - best_distance) < 1e-12]
            sorted_distances = sorted(distances)
            margin_to_second_best = (
                sorted_distances[1] - sorted_distances[0]
                if len(sorted_distances) > 1
                else float("nan")
            )

            # 決定 winner（原有 single-winner 機制）
            if len(winner_indices) == 1:
                winner_idx = winner_indices[0]
                win_counts[winner_idx] += 1
                winner_label = f"expert_{winner_idx + 1}"
                top_examples.append(
                    {
                        "pair_index": pair_index,
                        "is_same": is_same_value,
                        "speaker_1": speaker_1_value,
                        "speaker_2": speaker_2_value,
                        "age_1_year": age_1_year,
                        "age_2_year": age_2_year,
                        "age_gap_year": age_gap_year,
                        "winner": winner_label,
                        "best_distance": best_distance,
                        "margin_to_second_best": margin_to_second_best,
                        "scores": scores,
                        "distances": distances,
                    }
                )
            else:
                tie_count += 1
                winner_label = "tie"

            # 不再逐筆印出結果，改為最後統計輸出以減少終端機雜訊

            writer.writerow(
                [
                    pair_index,
                    is_same_value,
                    speaker_1_value,
                    speaker_2_value,
                    age_1_year,
                    age_2_year,
                    age_gap_year,
                    target,
                    scores[0],
                    scores[1],
                    scores[2],
                    distances[0],
                    distances[1],
                    distances[2],
                    winner_label,
                    best_distance,
                    score_mean,
                ]
            )
            csv_file.flush()
            total_pairs += 1

    if total_pairs == 0:
        print("沒有可用的 pair，無法計算 win rate。")
        return

    win_rates = [count / total_pairs for count in win_counts]
    print("\n=== Final Win Rate ===")
    print(f"Expert 1 win count: {win_counts[0]} / {total_pairs} ({win_rates[0]:.2%})")
    print(f"Expert 2 win count: {win_counts[1]} / {total_pairs} ({win_rates[1]:.2%})")
    print(f"Expert 3 win count: {win_counts[2]} / {total_pairs} ({win_rates[2]:.2%})")
    print(f"Tie count: {tie_count} / {total_pairs} ({tie_count / total_pairs:.2%})")
    print(f"Skipped pairs with missing age: {skipped_pairs}")

    print("\n=== Per-Expert EER & Tuned Threshold ===")
    eer_results = []
    for i in range(3):
        labels_arr = np.asarray(all_labels, dtype=np.int32)
        scores_arr = np.asarray(all_scores[i], dtype=np.float32)
        pos_count = int(np.sum(labels_arr == 1))
        neg_count = int(np.sum(labels_arr == 0))

        if len(labels_arr) == 0 or pos_count == 0 or neg_count == 0:
            eer = float("nan")
            threshold = float("nan")
            print(
                f"Expert {i+1}: 樣本不足，無法計算 EER (pos={pos_count}, neg={neg_count})"
            )
        else:
            eer, threshold = compute_eer(scores_arr, labels_arr)
            print(
                f"Expert {i+1}: EER={eer:.4%}, threshold={threshold:.6f}, "
                f"pos={pos_count}, neg={neg_count}"
            )

        eer_results.append(
            {
                "expert": f"expert_{i+1}",
                "eer": eer,
                "threshold": threshold,
                "pos_count": pos_count,
                "neg_count": neg_count,
            }
        )

    # 計算並輸出每個 expert 對於正對 / 負對的平均數與標準差（以 population std）
    def mean_std(lst):
        if len(lst) == 0:
            return float("nan"), float("nan")
        m = sum(lst) / len(lst)
        var = sum((x - m) ** 2 for x in lst) / len(lst)
        return m, var ** 0.5

    print("\n=== Expert Positive/Negative Score Statistics ===")
    for i in range(3):
        p_m, p_s = mean_std(pos_scores[i])
        n_m, n_s = mean_std(neg_scores[i])
        print(
            f"Expert {i+1} - Positive: mean={p_m:.6f}, std={p_s:.6f}, n={len(pos_scores[i])}"
        )
        print(
            f"Expert {i+1} - Negative: mean={n_m:.6f}, std={n_s:.6f}, n={len(neg_scores[i])}"
        )

    # 將統計附加到 CSV 檔案
    try:
        with open(OUTPUT_CSV, mode="a", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([])
            writer.writerow(["summary_type", "expert", "mean", "std", "count"])
            for i in range(3):
                p_m, p_s = mean_std(pos_scores[i])
                n_m, n_s = mean_std(neg_scores[i])
                writer.writerow(["positive", f"expert_{i+1}", p_m, p_s, len(pos_scores[i])])
                writer.writerow(["negative", f"expert_{i+1}", n_m, n_s, len(neg_scores[i])])

            writer.writerow([])
            writer.writerow(["summary_type", "expert", "eer", "eer_threshold", "pos_count", "neg_count"])
            for item in eer_results:
                writer.writerow(
                    [
                        "eer",
                        item["expert"],
                        item["eer"],
                        item["threshold"],
                        item["pos_count"],
                        item["neg_count"],
                    ]
                )
    except Exception as e:
        print(f"無法將統計寫入 CSV: {e}")

    # 另存一份純 EER 摘要 CSV，方便後續分析或畫圖
    try:
        with open(EER_SUMMARY_CSV, mode="w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["expert", "eer", "threshold", "pos_count", "neg_count"])
            for item in eer_results:
                writer.writerow(
                    [
                        item["expert"],
                        item["eer"],
                        item["threshold"],
                        item["pos_count"],
                        item["neg_count"],
                    ]
                )
        print(f"EER 摘要已保存到: {EER_SUMMARY_CSV}")
    except Exception as e:
        print(f"無法寫入 EER 摘要 CSV: {e}")

    # 畫每個 expert 的 score 盒狀圖，並在盒狀圖上標出各自 EER threshold
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.boxplot(
            all_scores,
            labels=["Expert 1", "Expert 2", "Expert 3"],
            showfliers=True,
        )

        for idx, item in enumerate(eer_results, start=1):
            thr = item["threshold"]
            if np.isfinite(thr):
                ax.scatter(idx, thr, color="red", s=70, zorder=3)
                ax.text(
                    idx + 0.03,
                    thr,
                    f"thr={thr:.4f}",
                    color="red",
                    fontsize=9,
                    va="center",
                )

        ax.set_title(f"{VAL_DATASET_NAME} Expert Score Boxplot with EER Threshold")
        ax.set_xlabel("Expert")
        ax.set_ylabel("Cosine Score")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        fig.tight_layout()
        fig.savefig(THRESHOLD_BOXPLOT_PNG, dpi=200)
        plt.close(fig)
        print(f"Threshold 盒狀圖已保存到: {THRESHOLD_BOXPLOT_PNG}")
    except Exception as e:
        print(f"無法繪製 threshold 盒狀圖: {e}")

    # 列出前 10 筆「勝出最明顯」且贏家為 expert_3 的例子，並包含年齡差距
    expert3_examples = [it for it in top_examples if it.get("winner") == "expert_3"]
    top_examples_sorted = sorted(
        expert3_examples,
        key=lambda item: item["margin_to_second_best"],
        reverse=True,
    )[:10]

    print("\n=== Top 10 Most Confident Winning Examples (expert_3 only) ===")
    if len(top_examples_sorted) == 0:
        print("沒有可用的勝出例子可供列印。")
    else:
        for rank, item in enumerate(top_examples_sorted, start=1):
            print(
                f"#{rank} pair_index={item['pair_index']}, winner={item['winner']}, "
                f"is_same={item['is_same']}, age_gap_year={item['age_gap_year']}, "
                f"age_1={item['age_1_year']}, age_2={item['age_2_year']}, "
                f"speaker_1={item['speaker_1']}, speaker_2={item['speaker_2']}, "
                f"best_distance={item['best_distance']:.6f}, "
                f"margin_to_second_best={item['margin_to_second_best']:.6f}, "
                f"scores={[round(s, 6) for s in item['scores']]}, "
                f"distances={[round(d, 6) for d in item['distances']]}"
            )

    # 另外也把前 10 筆摘要附加到 CSV，方便離線檢視
    try:
        with open(OUTPUT_CSV, mode="a", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([])
            writer.writerow([
                "top_example_rank",
                "pair_index",
                "winner",
                "is_same",
                "age_gap_year",
                "age_1_year",
                "age_2_year",
                "speaker_1",
                "speaker_2",
                "best_distance",
                "margin_to_second_best",
                "expert_1_cosine",
                "expert_2_cosine",
                "expert_3_cosine",
            ])
            for rank, item in enumerate(top_examples_sorted, start=1):
                writer.writerow([
                    rank,
                    item["pair_index"],
                    item["winner"],
                    item["is_same"],
                    item["age_gap_year"],
                    item["age_1_year"],
                    item["age_2_year"],
                    item["speaker_1"],
                    item["speaker_2"],
                    item["best_distance"],
                    item["margin_to_second_best"],
                    item["scores"][0],
                    item["scores"][1],
                    item["scores"][2],
                ])
    except Exception as e:
        print(f"無法將前 10 筆例子寫入 CSV: {e}")

    print(f"完成，CSV 已保存到: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()