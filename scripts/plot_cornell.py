"""Create descriptive ROC and calibration plots from validation predictions."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, roc_curve

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--baseline", type=Path, required=True)
parser.add_argument("--selected", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
if args.out.exists():
    raise FileExistsError(args.out)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
reference = None
for name, path, color in [("Baseline", args.baseline, "#596579"),
                          ("CV-selected", args.selected, "#167d9a")]:
    frame = pd.read_csv(path, dtype={"id": str}).sort_values("id").reset_index(drop=True)
    truth = frame[["id", "label"]]
    if reference is not None:
        pd.testing.assert_frame_equal(reference, truth)
    reference = truth
    fpr, tpr, _ = roc_curve(frame.label, frame.probability)
    axes[0].plot(fpr, tpr, color=color, label=f"{name}: AUC {roc_auc_score(frame.label, frame.probability):.4f}")
    observed, predicted = calibration_curve(frame.label, frame.probability, n_bins=10, strategy="quantile")
    axes[1].plot(predicted, observed, "o-", color=color, label=name, markersize=4)
for ax in axes:
    ax.plot([0, 1], [0, 1], "--", color="#bbbbbb", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=.15)
    ax.legend(loc="lower right", frameon=False)
axes[0].set(title="ROC on shared validation", xlabel="False positive rate", ylabel="True positive rate")
axes[1].set(title="Calibration: 10 equal-count bins", xlabel="Mean predicted P(AI)", ylabel="Observed AI fraction")
fig.suptitle("Cornell Model 2 reproduction — one 2,000-row development split", fontsize=12)
args.out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.out, dpi=180)
plt.close(fig)
