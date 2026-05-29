from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, log_loss

from audit.utils import entropy_binary


def expected_calibration_error_binary(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi if i < n_bins - 1 else y_prob <= hi)
        if not np.any(mask):
            continue
        acc = np.mean(y_true[mask] == (y_prob[mask] >= 0.5))
        conf = np.mean(y_prob[mask])
        ece += (np.sum(mask) / n) * abs(acc - conf)
    return float(ece)


def summarize_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    prevalence = float(np.mean(y_true))
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = float("nan")
    try:
        nll = float(log_loss(y_true, np.vstack([1 - y_prob, y_prob]).T, labels=[0, 1]))
    except Exception:
        nll = float("nan")
    h_s = entropy_binary(prevalence)
    usable_info = float(h_s - nll / np.log(2)) if np.isfinite(nll) else float("nan")
    return {
        "prevalence": prevalence,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": auc,
        "nll": nll,
        "ece": expected_calibration_error_binary(y_true, y_prob, n_bins=n_bins),
        "entropy_bits": h_s,
        "usable_info_bits": usable_info,
    }
