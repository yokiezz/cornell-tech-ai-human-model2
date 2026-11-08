"""Audit official files and run training-only staged model selection."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold, cross_val_score

from model2.analysis import classification_summary, length_slices, select_errors
from model2.cli import digest, git_output
from model2.data import read_data, partition, make_submission
from model2.model import build_pipeline
from model2.rehearsal import normalize_for_audit, nearest_training_similarity


def audit(data_dir, split_path, out):
    frames = {}
    report = {"files": {}, "overlaps": {}}
    for name, size, labeled in [("train", 10000, True), ("test", 3000, False),
                                ("unlabeled", 3000, False), ("train_starter_2000", 2000, True)]:
        path = data_dir / f"{name}.csv"
        frame = read_data(path, labeled=labeled, expected_rows=size)
        frame["normalized"] = frame.text.map(normalize_for_audit)
        frames[name] = frame
        row = {"rows": len(frame), "sha256": digest(path),
               "normalized_duplicate_extra_rows": int(frame.normalized.duplicated().sum()),
               "word_count_quantiles": frame.text.str.split().str.len().quantile([0, .25, .5, .75, 1]).to_dict()}
        if labeled:
            row["class_counts"] = frame.label.value_counts().sort_index().to_dict()
            row["conflicting_normalized_groups"] = int((frame.groupby("normalized").label.nunique() > 1).sum())
        report["files"][name] = row
    for left, right in [("train", "test"), ("train", "unlabeled"), ("test", "unlabeled")]:
        report["overlaps"][f"{left}:{right}"] = {
            "ids": len(set(frames[left].id) & set(frames[right].id)),
            "normalized_texts": len(set(frames[left].normalized) & set(frames[right].normalized))}
    train = frames["train"]
    starter = frames["train_starter_2000"].set_index("id")
    full = train.set_index("id")
    report["starter_exact_subset"] = set(starter.index) <= set(full.index) and starter.equals(full.loc[starter.index])
    sample = pd.read_csv(data_dir / "sample_submission.csv", dtype={"id": str}, keep_default_na=False)
    make_submission(frames["test"], np.full(3000, .5), sample)
    report["sample_submission_matches_test_ids"] = True
    manifest = pd.read_csv(split_path, dtype=str, keep_default_na=False)
    fit, val = partition(train, manifest)
    report["split_sha256"] = digest(split_path)
    report["split_counts"] = {name: frame.label.value_counts().sort_index().to_dict()
                              for name, frame in [("train", fit), ("validation", val)]}
    report["normalized_overlap_across_split"] = len(set(fit.normalized) & set(val.normalized))
    similarity = nearest_training_similarity(fit, val)
    similarity.to_csv(out / "validation_overlap.csv", index=False)
    report["approximate_overlap"] = {"method": "hashed word trigram cosine; diagnostic only",
                                     "rows_ge_0_9": int((similarity.word_trigram_cosine >= .9).sum()),
                                     "maximum": float(similarity.word_trigram_cosine.max())}
    report["starter_validation_rows"] = len(set(starter.index) & set(val.id))
    (out / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def tune(data_dir, split_path, base_path, plan_path, out):
    train = read_data(data_dir / "train.csv", labeled=True, expected_rows=10000)
    fit, val = partition(train, pd.read_csv(split_path, dtype=str, keep_default_na=False))
    if len(fit) != 8000 or len(val) != 2000:
        raise ValueError("Cornell experiments require the agreed 8,000/2,000 split")
    if set(fit.text.map(normalize_for_audit)) & set(val.text.map(normalize_for_audit)):
        raise ValueError("Normalized text overlap across the split requires review")
    base, plan = json.loads(base_path.read_text()), json.loads(plan_path.read_text())
    folds = list(StratifiedKFold(plan["cv_folds"], shuffle=True, random_state=plan["seed"]).split(fit.text, fit.label))
    trials = []
    def run(config):
        start = perf_counter()
        scores = cross_val_score(build_pipeline(config), fit.text, fit.label, cv=folds,
                                 scoring="roc_auc", n_jobs=1, error_score="raise")
        row = {"config": config, "fold_auroc": scores.tolist(), "mean_auroc": float(scores.mean()),
               "fold_std": float(scores.std()), "elapsed_seconds": perf_counter() - start}
        trials.append(row)
        (out / "trials.json").write_text(json.dumps(trials, indent=2) + "\n")
        print(json.dumps(row), flush=True)
    for analyzer in plan["analyzers"]:
        for ngram in plan["ngram_ranges"]:
            run(base | {"analyzer": analyzer, "ngram_range": ngram, "C": plan["stage_one_C"]})
    representation = max(trials, key=lambda row: row["mean_auroc"])["config"]
    for c in plan["stage_two_C"]:
        run(representation | {"C": c})
    best = max(trials, key=lambda row: row["mean_auroc"])
    (out / "selected_config.json").write_text(json.dumps(best["config"], indent=2) + "\n")
    record = {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "plan": plan,
              "train_sha256": digest(data_dir / "train.csv"), "split_sha256": digest(split_path),
              "source_sha256": {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
              "git_commit": git_output("rev-parse", "HEAD"), "git_status": git_output("status", "--porcelain"),
              "trials": trials, "selected_config": best["config"], "selected_mean_auroc": best["mean_auroc"],
              "validation_used_for_selection": False,
              "limitation": "One 3-fold partition; staged search, not exhaustive; CV fold dispersion is not a confidence interval."}
    (out / "selection.json").write_text(json.dumps(record, indent=2) + "\n")


def diagnose(data_dir, predictions_path, out):
    predictions = pd.read_csv(predictions_path, dtype={"id": str})
    raw = read_data(data_dir / "train.csv", labeled=True, expected_rows=10000)
    truth = raw.set_index("id").label
    if predictions.id.duplicated().any() or not predictions.id.isin(truth.index).all():
        raise ValueError("Predictions must have unique known IDs")
    if not np.array_equal(predictions.label, predictions.id.map(truth)):
        raise ValueError("Prediction labels disagree with original training data")
    predictions["word_count"] = predictions.id.map(raw.set_index("id").text.str.split().str.len())
    summary = classification_summary(predictions.label, predictions.probability)
    length_slices(predictions).to_csv(out / "length_slices.csv", index=False)
    errors = select_errors(predictions).merge(raw[["id", "text"]], on="id", validate="one_to_one")
    errors.to_csv(out / "error_cases_local_only.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["audit", "tune", "diagnose"])
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--split", type=Path, default=Path("artifacts/shared-split/split.csv"))
    parser.add_argument("--baseline", type=Path, default=Path("configs/baseline.json"))
    parser.add_argument("--plan", type=Path, default=Path("configs/cornell_experiment.json"))
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    warnings.filterwarnings("error", category=ConvergenceWarning)
    if args.command == "audit":
        print(json.dumps(audit(args.data_dir, args.split, args.out), indent=2))
    elif args.command == "tune":
        tune(args.data_dir, args.split, args.baseline, args.plan, args.out)
    elif args.predictions is None:
        parser.error("diagnose requires --predictions")
    else:
        diagnose(args.data_dir, args.predictions, args.out)


if __name__ == "__main__":
    main()
