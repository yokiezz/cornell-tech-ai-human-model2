"""Reproducible baseline evaluation and final full-data submission."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
import warnings

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score

from model2.data import make_submission, partition, read_data, split_manifest
from model2.model import build_calibrated_model, build_pipeline


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_output(*args):
    result = subprocess.run(["git", *args], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["split", "evaluate", "submit"])
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.json"))
    parser.add_argument("--split", type=Path, default=Path("artifacts/shared-split/split.csv"))
    parser.add_argument("--out", type=Path, required=True, help="New output directory; never overwritten")
    args = parser.parse_args(argv)
    train_path = args.data_dir / "train.csv"
    train = read_data(train_path, labeled=True, expected_rows=10000)
    if train.label.value_counts().to_dict() != {0: 5000, 1: 5000}:
        raise ValueError("Expected 5,000 examples per class in full train.csv")
    config = json.loads(args.config.read_text())
    inputs = {"train.csv": digest(train_path), "config": digest(args.config)}
    args.out.mkdir(parents=True, exist_ok=False)
    start = perf_counter()
    metrics = {}
    # Do not silently accept an unconverged experiment.
    warnings.filterwarnings("error", category=ConvergenceWarning)
    if args.command == "split":
        manifest = split_manifest(train)
        manifest.to_csv(args.out / "split.csv", index=False)
        metrics["duplicate_text_rows"] = int(train.text.duplicated().sum())
        fit, val = partition(train, manifest)
        metrics["exact_text_overlap_between_partitions"] = len(set(fit.text) & set(val.text))
    elif args.command == "evaluate":
        inputs["split"] = digest(args.split)
        manifest = pd.read_csv(args.split, dtype=str, keep_default_na=False)
        fit, val = partition(train, manifest)
        if len(fit) != 8000 or len(val) != 2000:
            raise ValueError("Team contract requires 8,000 training and 2,000 validation rows")
        if set(fit.text) & set(val.text):
            raise ValueError("Exact duplicate texts cross the split; resolve with the team before evaluation")
        raw = build_pipeline(config).fit(fit.text, fit.label)
        calibrated = build_calibrated_model(config).fit(fit.text, fit.label)
        for name, frame in [("train", fit), ("validation", val)]:
            margin = raw.decision_function(frame.text)
            p = calibrated.predict_proba(frame.text)[:, 1]
            metrics[f"{name}_margin_auroc"] = roc_auc_score(frame.label, margin)
            metrics[f"{name}_probability_auroc"] = roc_auc_score(frame.label, p)
            pd.DataFrame({"id": frame.id, "label": frame.label, "margin": margin,
                          "probability": p, "word_count": frame.text.str.split().str.len()}
                         ).to_csv(args.out / f"{name}_predictions.csv", index=False)
        metrics.update(train_rows=len(fit), validation_rows=len(val))
    else:
        test_path = args.data_dir / "test.csv"
        sample_path = args.data_dir / "sample_submission.csv"
        test = read_data(test_path, expected_rows=3000)
        sample = pd.read_csv(sample_path, dtype={"id": str}, keep_default_na=False)
        if set(train.id) & set(test.id):
            raise ValueError("Training and test IDs overlap")
        inputs.update({"test.csv": digest(test_path), "sample_submission.csv": digest(sample_path)})
        model = build_calibrated_model(config).fit(train.text, train.label)
        result = make_submission(test, model.predict_proba(test.text)[:, 1], sample)
        result.to_csv(args.out / "submission.csv", index=False)
        metrics.update(train_rows=len(train), test_rows=len(test))
    record = {
        "command": args.command, "argv": sys.argv[1:] if argv is None else argv,
        "config": config, "input_sha256": inputs, "metrics": metrics,
        "elapsed_seconds": perf_counter() - start, "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name)
                     for name in ["numpy", "pandas", "scipy", "scikit-learn"]},
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status": git_output("status", "--porcelain"),
    }
    (args.out / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
