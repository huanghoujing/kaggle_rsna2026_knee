"""Metrics: macro AUC over the 12 findings (the competition metric)."""
import numpy as np
from sklearn.metrics import roc_auc_score

from knee.constants import LABELS


def auc_table(y_true, y_pred, threshold=0.5):
    """Per-label ROC AUC and their mean. Soft targets are binarized at `threshold`. Labels with one class are NaN."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    out = {}
    for j, name in enumerate(LABELS):
        t, p = y_true[:, j], y_pred[:, j]
        ok = ~np.isnan(t) & ~np.isnan(p)
        tb = t[ok] > threshold
        out[name] = roc_auc_score(tb, p[ok]) if 0 < tb.sum() < len(tb) else np.nan
    out['macro'] = float(np.nanmean([out[n] for n in LABELS]))
    return out
