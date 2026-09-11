import numpy as np
from sklearn.metrics import roc_curve
from operator import itemgetter

def compute_eer(scores, labels):
    """
    scores: np.ndarray [N]  cosine similarity
    labels: np.ndarray [N]  1 = same speaker, 0 = different
    """

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    tuned_threshold = thresholds[eer_idx]
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2

    return eer, tuned_threshold

def ComputeErrorRates(scores, labels):
    """
    根據預測分數 (scores) 和實際標籤 (labels)，
    計算 False Negative Rates (FNRs)、False Positive Rates (FPRs)，
    以及相對應的判斷閾值 (thresholds)。
    
    Args:
        scores (list of float): 模型輸出的分數。
        labels (list of int): 實際標籤，1 表示正樣本，0 表示負樣本。

    Returns:
        fnrs (list of float): 各閾值下的 False Negative Rate。
        fprs (list of float): 各閾值下的 False Positive Rate。
        thresholds (list of float): 與錯誤率對應的閾值。
    """

    # 先將所有分數與索引配對並依分數排序，取出排序後的索引與對應閾值。
    # thresholds 是排序後的分數列表，sorted_indexes 是排序後的索引順序。
    sorted_indexes, thresholds = zip(*sorted(
        [(index, threshold) for index, threshold in enumerate(scores)],
        key=itemgetter(1)))  # 根據分數 (threshold) 升序排列

    labels = [labels[i] for i in sorted_indexes]

    fnrs = []  # False Negative Rates
    fprs = []  # False Positive Rates

    """
    遍歷每個排序後的樣本，逐步累積 FN 和 FP。
    若當前樣本是正樣本 (label=1)，表示若閾值設在這邊會錯過一個正樣本 (FN)。
    若是負樣本 (label=0)，表示會誤判為正樣本 (FP)。
    """
    for i in range(0, len(labels)):
        if i == 0:
            fnrs.append(labels[i])
            fprs.append(1 - labels[i])
        else:
            fnrs.append(fnrs[i-1] + labels[i])
            fprs.append(fprs[i-1] + 1 - labels[i])

    """
    正樣本數 = 用於計算 FN 的總數
    負樣本數 = 用於計算 FP 的總數
    """
    fnrs_norm = sum(labels)
    fprs_norm = len(labels) - fnrs_norm

    """
    將 FN 累積數正規化為比率，得到每個閾值下的 FNR
    """
    fnrs = [x / float(fnrs_norm) for x in fnrs]

    """
    同理，先計算 True Positive Rate (TPR) = TP / 所有負樣本數
    再用 1 - TPR 得到 FPR
    """
    fprs = [1 - x / float(fprs_norm) for x in fprs]

    return fnrs, fprs, thresholds

def ComputeMinDcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa):
    """
    計算最小化的偵測成本函數（minDCF），並回傳對應的閾值。

    參數：
        - fnrs: 各閾值下的 False Negative Rates (FNR) 列表。
        - fprs: 各閾值下的 False Positive Rates (FPR) 列表。
        - thresholds: 對應於 fnrs 與 fprs 的決策閾值列表。
        - p_target: 目標事件（positive class）的先驗機率 P_target。
        - c_miss: 錯過（miss）一個正樣本的成本，對應於 C_miss。
        - c_fa: 把負樣本誤判為正樣本（false alarm）的成本，對應於 C_fa。

    回傳值：
        - min_dcf: 正規化後的最小偵測成本（minDCF）。
        - min_c_det_threshold: 使得偵測成本最小化的最佳閾值。
    """

    # 初始化：設定目前最小成本為無限大，並預設最佳閾值為列表中的第一個
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]

    # 遍歷所有閾值位置，計算對應的 detection cost C_det
    for i in range(len(fnrs)):
        # 加權求和：C_det = C_miss * FNR * P_target + C_fa * FPR * (1 - P_target)
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)

        # 若當前成本更低，則更新最小成本及對應閾值
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]

    # 計算預設成本 C_def，用以正規化
    # C_def = min(C_miss * P_target, C_fa * (1 - P_target))
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))

    # 正規化後的最小偵測成本
    min_dcf = min_c_det / c_def

    return min_dcf, min_c_det_threshold
