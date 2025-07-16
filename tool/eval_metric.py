# eval_metric.py
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

def compute_eer(scores, labels):
    """
    計算等錯誤率 (EER)。
    Args:
        scores (np.ndarray): 模型的預測分數，可以是相似度值。
        labels (np.ndarray): 真實標籤，1 表示目標 (target)，0 表示非目標 (non-target)。
    Returns:
        float: 等錯誤率 (EER)。
    """
    # 計算 ROC 曲線
    # fpr (False Positive Rate): 誤報率，即將非目標錯誤判斷為目標的比例。
    # tpr (True Positive Rate): 真陽性率，即將目標正確判斷為目標的比例。
    # thresholds: 對應的閾值。
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)

    # 尋找 EER 點，即 FPR 和 (1 - TPR) 最接近的點
    # 1 - TPR = FNR (False Negative Rate): 漏報率，即將目標錯誤判斷為非目標的比例。
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    
    # 找到對應 EER 的閾值
    eer_threshold = interp1d(fpr, thresholds)(eer)
    
    return eer, eer_threshold

def compute_min_dcf(scores, labels, p_target=0.01, c_fa=1, c_miss=1):
    """
    計算最小正規化檢測成本函數 (minDCF)。
    Args:
        scores (np.ndarray): 模型的預測分數。
        labels (np.ndarray): 真實標籤，1 表示目標 (target)，0 表示非目標 (non-target)。
        p_target (float): 目標分佈的先驗概率 (Prior probability of target speaker)。論文中為 1e-2 (0.01)。
        c_fa (int): 誤報成本 (Cost of false alarm)。論文中為 1。
        c_miss (int): 漏報成本 (Cost of missed detection)。論文中為 1。
    Returns:
        float: 最小正規化檢測成本函數 (minDCF)。
    """
    # 排序分數並找到對應的閾值
    # 這裡可以採用與 EER 類似的方式，遍歷所有可能的閾值
    # 或者直接使用 ROC 曲線的 FPR, TPR 來計算每個閾值下的 DCF

    # 計算 ROC 曲線
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)

    # 確保閾值是降序排列的，這與通常的 ROC 曲線輸出一致
    # 閾值通常是從大到小排列，如果不是，需要反轉
    if thresholds[-1] < thresholds[0]: # 檢查是否是降序
        thresholds = thresholds[::-1]
        fpr = fpr[::-1]
        tpr = tpr[::-1]

    min_dcf = float('inf')

    # 計算非目標和目標的數量
    n_nontargets = np.sum(labels == 0)
    n_targets = np.sum(labels == 1)

    # 遍歷所有可能的閾值 (或 ROC 曲線上的點)
    for i in range(len(thresholds)):
        fa_rate = fpr[i]
        miss_rate = 1 - tpr[i] # miss_rate = 1 - recall

        # 計算當前閾值下的檢測成本函數 (DCF)
        dcf = c_miss * miss_rate * p_target + c_fa * fa_rate * (1 - p_target)

        # 正規化 DCF
        # 這是檢測成本函數的標準正規化因子
        normalized_dcf = dcf / (c_miss * p_target + c_fa * (1 - p_target))

        if normalized_dcf < min_dcf:
            min_dcf = normalized_dcf

    return min_dcf

# 示例用法 (在實際 eval.py 中會用模型生成的 scores 和 labels)
# if __name__ == '__main__':
#     # 假設有一些分數和真實標籤
#     # scores 越大，表示越可能是目標
#     example_scores = np.array([0.1, 0.4, 0.35, 0.8, 0.6, 0.9, 0.2, 0.7, 0.5, 0.15])
#     # labels: 1代表目標 (same speaker), 0代表非目標 (different speaker)
#     example_labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 0])

#     eer_val, eer_threshold = compute_eer(example_scores, example_labels)
#     min_dcf_val = compute_min_dcf(example_scores, example_labels, p_target=0.01, c_fa=1, c_miss=1)

#     print(f"Example EER: {eer_val*100:.2f}% (Threshold: {eer_threshold:.4f})")
#     print(f"Example minDCF: {min_dcf_val:.4f}")

#     # 另一個測試 EER 的簡單例子
#     scores_simple = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
#     labels_simple = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0]) # 5 targets, 5 non-targets

#     eer_val_simple, _ = compute_eer(scores_simple, labels_simple)
#     print(f"\nSimple EER Test: {eer_val_simple*100:.2f}%") # 預期在 0.5 (50%) 附近