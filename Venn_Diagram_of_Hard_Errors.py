"""
Venn Diagram of Hard Errors

在 Vox-O 測試集上，比較小專家、中專家、大專家的錯誤 Pair 集合：
  - 錯誤 Pair = False Accept (score >= threshold, label=0)
               + False Reject (score <  threshold, label=1)
  - Threshold 由每個專家各自的 EER 決定

專家 checkpoint 路徑：
  - 小專家: checkpoints/siamese_full_finetune/full_ft_lr1e4_small_seed42/siamese_best.pt
  - 中專家: checkpoints/siamese_full_finetune/full_ft_lr1e4_medium_seed42/siamese_best.pt
  - 大專家: checkpoints/siamese_full_finetune/full_ft_lr1e4_large_seed42/siamese_best.pt

輸出：
  - 印出各專家 EER、錯誤數量
  - 印出兩兩 / 三方 IoU（Intersection / Union）
  - 儲存文氏圖到 logs/venn_diagram/
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.vox1_loader import PairwiseDataset
from model.disentangled_model.siamese_network import SiameseNetwork
from params.param import DATASET_INFO, DEVICE
from tool.EER import compute_eer

warnings.filterwarnings("ignore")

# ==========================================
# 設定
# ==========================================
CHECKPOINT_ROOT = Path("checkpoints/siamese_full_finetune")

EXPERTS = {
    "small":  CHECKPOINT_ROOT / "full_ft_lr1e4_small_seed42"  / "siamese_best.pt",
    "medium": CHECKPOINT_ROOT / "full_ft_lr1e4_medium_seed42" / "siamese_best.pt",
    "large":  CHECKPOINT_ROOT / "full_ft_lr1e4_large_seed42"  / "siamese_best.pt",
}

DATASET_NAME    = "VoxCeleb1"
DATASET_VARIANT = "Vox-O"

OUTPUT_DIR = Path("logs/venn_diagram")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 工具函式
# ==========================================
def build_model() -> SiameseNetwork:
    model = SiameseNetwork().to(DEVICE)
    return model


def build_test_loader() -> DataLoader:
    test_audio_dir = DATASET_INFO[DATASET_NAME][DATASET_VARIANT]["AUDIO_DIR"]
    test_pair_meta = DATASET_INFO[DATASET_NAME][DATASET_VARIANT]["AUDIO_DATALIST"]
    test_meta_csv  = DATASET_INFO[DATASET_NAME]["AUDIO_META_DIR"]

    dataset = PairwiseDataset(
        audio_dir=test_audio_dir,
        audio_meta_dir=test_pair_meta,
        audio_meta_csv_path=test_meta_csv,
    )
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)


def get_error_pairs(model: SiameseNetwork, loader: DataLoader):
    """
    回傳此模型在 Vox-O 上判斷錯誤的 Pair index 集合。

    Returns:
        errors   (set[int]): 錯誤 pair 的索引集合
        eer      (float)
        threshold(float)
        labels_np(np.ndarray)
    """
    model.eval()
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for pair_label, _, _, wav1, wav2, _, _ in tqdm(
            loader, desc="  Inference", leave=False, dynamic_ncols=True
        ):
            pair_label = pair_label.to(DEVICE, non_blocking=True)
            wav1 = wav1.to(DEVICE, non_blocking=True)
            wav2 = wav2.to(DEVICE, non_blocking=True)

            _, _, cosine_score = model(wav1, wav2, spec_aug=False)

            all_scores.append(cosine_score.detach().cpu().item())
            all_labels.append(int(pair_label.detach().cpu().item()))

    scores_np = np.array(all_scores)
    labels_np = np.array(all_labels)

    eer, threshold = compute_eer(scores_np, labels_np)

    predictions = (scores_np >= threshold).astype(int)
    errors = set(int(i) for i in np.where(predictions != labels_np)[0])

    fa_count = int(np.sum((predictions == 1) & (labels_np == 0)))
    fr_count = int(np.sum((predictions == 0) & (labels_np == 1)))
    print(f"  EER: {eer:.4f} | Threshold: {threshold:.4f}")
    print(f"  總錯誤數: {len(errors)}  (FA={fa_count}, FR={fr_count}) / {len(scores_np)} pairs")

    return errors, eer, threshold, labels_np


def summarize_exclusive_errors(
    focus_name: str,
    other_name: str,
    focus_errors: set,
    other_errors: set,
    labels_np: np.ndarray,
) -> None:
    exclusive_errors = focus_errors - other_errors
    fa_count = sum(1 for idx in exclusive_errors if labels_np[idx] == 0)
    fr_count = sum(1 for idx in exclusive_errors if labels_np[idx] == 1)

    print(f"\n{focus_name}專家錯/{other_name}專家對 的 {len(exclusive_errors)} 個樣本中：")
    print(f"  - FA (把異人誤認成同人): {fa_count}")
    print(f"  - FR (把同人誤認成異人): {fr_count}")


def pairwise_iou(set_a: set, set_b: set) -> float:
    union = len(set_a | set_b)
    return len(set_a & set_b) / union if union > 0 else 0.0


def draw_venn3(error_sets: dict, output_path: Path) -> None:
    """使用 matplotlib-venn 畫三圓文氏圖"""
    try:
        from matplotlib_venn import venn3
    except ImportError:
        print("[警告] 找不到 matplotlib-venn，請執行: pip install matplotlib-venn")
        return

    labels = list(error_sets.keys())
    A, B, C = [error_sets[k] for k in labels]

    fig, ax = plt.subplots(figsize=(9, 7))
    venn3([A, B, C], set_labels=labels, ax=ax)
    ax.set_title(
        "Venn Diagram of Hard Error Pairs\n(Small / Medium / Large Expert on Vox-O)",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()
    print(f"\n文氏圖已儲存: {output_path}")


def draw_venn2(error_sets: dict, label_a: str, label_b: str, output_path: Path) -> None:
    """使用 matplotlib-venn 畫雙圓文氏圖（僅兩個可用 expert 時）"""
    try:
        from matplotlib_venn import venn2
    except ImportError:
        print("[警告] 找不到 matplotlib-venn，請執行: pip install matplotlib-venn")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    venn2([error_sets[label_a], error_sets[label_b]], set_labels=[label_a, label_b], ax=ax)
    ax.set_title(
        f"Venn Diagram of Hard Error Pairs\n({label_a} vs {label_b} Expert on Vox-O)",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()
    print(f"\n文氏圖已儲存: {output_path}")


# ==========================================
# 主程式
# ==========================================
def main():
    print("載入測試資料集 (Vox-O)...")
    test_loader = build_test_loader()

    model = build_model()

    error_sets: dict[str, set] = {}
    eer_info:   dict[str, dict] = {}
    labels_by_expert: dict[str, np.ndarray] = {}

    for expert_name, ckpt_path in EXPERTS.items():
        print(f"\n{'=' * 64}")
        print(f"專家: {expert_name}  |  checkpoint: {ckpt_path}")

        if not ckpt_path.exists():
            print(f"  [警告] checkpoint 不存在，略過: {ckpt_path}")
            continue

        state_dict = torch.load(str(ckpt_path), map_location="cpu")
        model.load_state_dict(state_dict)
        model = model.to(DEVICE)

        errors, eer, threshold, labels_np = get_error_pairs(model, test_loader)
        error_sets[expert_name] = errors
        eer_info[expert_name]   = {"eer": eer, "threshold": threshold}
        labels_by_expert[expert_name] = labels_np

    available = list(error_sets.keys())
    print(f"\n可用專家: {available}")

    if len(available) < 2:
        print("[錯誤] 可用專家數量不足 2，無法計算 IoU 或繪製文氏圖。")
        return

    # ---- 計算兩兩 IoU ----
    print("\n" + "=" * 64)
    print("交集比例 (IoU = Intersection / Union)")
    print("=" * 64)
    for i in range(len(available)):
        for j in range(i + 1, len(available)):
            a, b = available[i], available[j]
            inter = len(error_sets[a] & error_sets[b])
            union = len(error_sets[a] | error_sets[b])
            iou   = inter / union if union > 0 else 0.0
            print(f"  {a:8s} ∩ {b:8s} = {inter:5d}  |  ∪ = {union:5d}  |  IoU = {iou:.4f}")

    # ---- 三方 IoU（若三個都可用）----
    if len(available) == 3:
        a, b, c = available
        inter3 = len(error_sets[a] & error_sets[b] & error_sets[c])
        union3 = len(error_sets[a] | error_sets[b] | error_sets[c])
        iou3   = inter3 / union3 if union3 > 0 else 0.0
        print(f"  {a:8s} ∩ {b:8s} ∩ {c:8s} = {inter3:5d}  |  ∪ = {union3:5d}  |  三方 IoU = {iou3:.4f}")

    # ---- 印出 EER 摘要 ----
    print("\n" + "=" * 64)
    print("EER 摘要")
    print("=" * 64)
    for name, info in eer_info.items():
        print(f"  {name:8s}: EER={info['eer']:.4f}  |  Threshold={info['threshold']:.4f}")

    # ---- 中小專家互補錯誤分析 ----
    if "small" in error_sets and "medium" in error_sets:
        print("\n" + "=" * 64)
        print("中小專家互補錯誤分析")
        print("=" * 64)

        labels_np = labels_by_expert["small"]
        summarize_exclusive_errors(
            focus_name="中",
            other_name="小",
            focus_errors=error_sets["medium"],
            other_errors=error_sets["small"],
            labels_np=labels_np,
        )
        summarize_exclusive_errors(
            focus_name="小",
            other_name="中",
            focus_errors=error_sets["small"],
            other_errors=error_sets["medium"],
            labels_np=labels_np,
        )

    # ---- 繪製文氏圖 ----
    if len(available) >= 3:
        draw_venn3(error_sets, OUTPUT_DIR / "venn_diagram_3experts.png")
    else:
        draw_venn2(error_sets, available[0], available[1], OUTPUT_DIR / "venn_diagram_2experts.png")


if __name__ == "__main__":
    main()
