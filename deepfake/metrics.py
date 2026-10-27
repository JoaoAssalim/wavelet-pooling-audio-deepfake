import numpy as np
from sklearn.metrics import roc_curve


def llr_score(logits):
    return logits[:, 1] - logits[:, 0]


def compute_eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2)
