"""Threshold and slice diagnostics for labeled probability predictions."""

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, confusion_matrix, log_loss, roc_auc_score


def classification_summary(labels, probabilities):
    y = np.asarray(labels)
    p = np.asarray(probabilities, dtype=float)
    if (y.ndim != 1 or p.shape != y.shape or len(y) == 0
            or not np.isin(y, [0, 1]).all() or not np.isfinite(p).all()
            or not ((p >= 0) & (p <= 1)).all()):
        raise ValueError("Expected nonempty binary labels and aligned finite probabilities in [0, 1]")
    tn, fp, fn, tp = confusion_matrix(y, p >= 0.5, labels=[0, 1]).ravel()
    return {
        "rows": len(y), "human": int((y == 0).sum()), "ai": int((y == 1).sum()),
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])), "threshold": 0.5,
        "true_negative": int(tn), "false_positive": int(fp),
        "false_negative": int(fn), "true_positive": int(tp),
        "false_positive_rate": float(fp / (tn + fp)) if tn + fp else None,
        "false_negative_rate": float(fn / (tp + fn)) if tp + fn else None,
    }


def length_slices(predictions):
    bins = pd.cut(predictions.word_count, [-1, 99, 199, 299, np.inf],
                  labels=["<100", "100-199", "200-299", ">=300"])
    rows = []
    for label in bins.cat.categories:
        frame = predictions[bins == label]
        row = {"word_count_bin": label, "rows": len(frame), "human": 0, "ai": 0, "auroc": None}
        if len(frame):
            row.update(classification_summary(frame.label, frame.probability))
        rows.append(row)
    return pd.DataFrame(rows)


def select_errors(predictions, per_class=5):
    """Choose the most confidently wrong cases, not a representative error sample."""
    fp = predictions[(predictions.label == 0) & (predictions.probability >= 0.5)]
    fn = predictions[(predictions.label == 1) & (predictions.probability < 0.5)]
    return pd.concat([
        fp.sort_values("probability", ascending=False).head(per_class).assign(error_type="FP"),
        fn.sort_values("probability").head(per_class).assign(error_type="FN"),
    ], ignore_index=True)
