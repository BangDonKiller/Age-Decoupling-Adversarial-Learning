import numpy as np
from sklearn.metrics import roc_curve

def compute_eer(scores, labels):
    """
    scores: np.ndarray [N]  cosine similarity
    labels: np.ndarray [N]  1 = same speaker, 0 = different
    """

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2

    return eer
