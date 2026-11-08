"""Run a small, explicitly separate DAIGT rehearsal; never writes Cornell inputs."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
from time import perf_counter
import unicodedata
import warnings
import zipfile

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import roc_auc_score

from model2.analysis import classification_summary, length_slices, select_errors
from model2.cli import digest, git_output
from model2.data import partition, split_manifest, make_submission
from model2.model import build_calibrated_model, build_pipeline


def normalize_for_audit(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def prepare_sample(frame, size, seed):
    required = {"text", "label", "prompt_name", "source"}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError("DAIGT columns missing or dataset empty")
    if not frame.label.isin([0, 1]).all() or frame.label.nunique() != 2:
        raise ValueError("DAIGT requires both binary labels")
    if size < 20 or size % 10:
        raise ValueError("Choose a sample size >=20 divisible by 10")
    frame = frame.copy()
    # Position within the versioned archive is the stable source identifier.
    frame["id"] = [f"daigt-v2-row-{i}" for i in range(len(frame))]
    invalid = frame.text.isna() | frame.text.fillna("").str.strip().eq("")
    valid = frame[~invalid].copy()
    valid["normalized"] = valid.text.map(normalize_for_audit)
    groups = valid.groupby("normalized").label.nunique()
    conflicts = set(groups[groups > 1].index)
    audit = {
        "source_rows": len(frame), "source_class_counts": frame.label.value_counts().sort_index().to_dict(),
        "blank_rows_excluded": int(invalid.sum()),
        "normalized_duplicate_extra_rows": int(valid.normalized.duplicated().sum()),
        "conflicting_normalized_groups": len(conflicts),
        "conflicting_rows_excluded": int(valid.normalized.isin(conflicts).sum()),
    }
    # Exclude all contradictory groups; retain the first original row otherwise.
    valid = valid[~valid.normalized.isin(conflicts)].drop_duplicates("normalized", keep="first")
    if (valid.label.value_counts() < size // 2).any():
        raise ValueError("Not enough unique texts per class for requested balanced sample")
    sample = pd.concat([valid[valid.label == label].sample(n=size // 2, random_state=seed)
                        for label in [0, 1]]).sort_index().reset_index(drop=True)
    sample["text_sha256"] = sample.text.map(lambda text: hashlib.sha256(text.encode()).hexdigest())
    audit["eligible_unique_rows"] = len(valid)
    audit["sample_rows"] = len(sample)
    audit["sample_class_counts"] = sample.label.value_counts().sort_index().to_dict()
    audit["sample_word_count_quantiles"] = sample.text.str.split().str.len().quantile([0, .25, .5, .75, 1]).to_dict()
    return sample, audit


def nearest_training_similarity(fit, val):
    """Approximate overlap diagnostic, not proof of absence of near duplicates."""
    hasher = HashingVectorizer(n_features=2**18, analyzer="word", ngram_range=(3, 3),
                              alternate_sign=False, norm="l2", dtype=np.float32)
    xtrain = hasher.transform(fit.text)
    xval = hasher.transform(val.text)
    similarities, ids = [], []
    for start in range(0, len(val), 128):
        scores = (xval[start:start + 128] @ xtrain.T).toarray()
        positions = scores.argmax(axis=1)
        similarities.extend(scores[np.arange(len(positions)), positions].tolist())
        ids.extend(fit.id.iloc[positions].tolist())
    return pd.DataFrame({"id": val.id.to_numpy(), "nearest_train_id": ids,
                         "word_trigram_cosine": similarities})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/rehearsal.json"))
    parser.add_argument("--baseline", type=Path, default=Path("configs/baseline.json"))
    args = parser.parse_args(argv)
    plan = json.loads(args.config.read_text())
    base = json.loads(args.baseline.read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    warnings.filterwarnings("error", category=ConvergenceWarning)
    start = perf_counter()
    print("Reading and auditing the versioned external dataset...", flush=True)
    with zipfile.ZipFile(args.archive) as archive:
        with archive.open(plan["csv_member"]) as stream:
            frame = pd.read_csv(stream, keep_default_na=False)
    sample, audit = prepare_sample(frame, plan["sample_size"], plan["seed"])
    manifest = split_manifest(sample)
    fit, val = partition(sample, manifest)
    assert not set(fit.normalized) & set(val.normalized)
    manifest.to_csv(args.out / "split.csv", index=False)
    sample.drop(columns="normalized").to_csv(args.out / "sample.csv", index=False)
    sample[["id", "label", "text_sha256", "source", "prompt_name"]].to_csv(args.out / "sample_manifest.csv", index=False)
    for name, subset in [("train", fit), ("validation", val)]:
        audit[f"{name}_class_counts"] = subset.label.value_counts().sort_index().to_dict()
        counts = subset.groupby(["source", "label"]).size().reset_index(name="rows")
        counts.to_csv(args.out / f"{name}_source_counts.csv", index=False)
    audit["normalized_text_overlap"] = 0
    (args.out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print("Audited balanced sample:", plan["sample_size"], "rows", flush=True)
    similarity = nearest_training_similarity(fit, val)
    similarity.to_csv(args.out / "overlap_diagnostic.csv", index=False)
    results = []
    for variant in plan["variants"]:
        config = base | {key: value for key, value in variant.items() if key != "name"}
        name = variant["name"]
        print("Fitting", name, flush=True)
        model = build_pipeline(config)
        t0 = perf_counter()
        model.fit(fit.text, fit.label)
        fit_seconds = perf_counter() - t0
        t0 = perf_counter()
        train_margin = model.decision_function(fit.text)
        val_margin = model.decision_function(val.text)
        row = {"variant": name, "config": config, "fit_seconds": fit_seconds,
               "score_train_and_validation_seconds": perf_counter() - t0,
               "train_margin_auroc": roc_auc_score(fit.label, train_margin),
               "validation_margin_auroc": roc_auc_score(val.label, val_margin),
               "vocabulary_size": len(model.named_steps["tfidf"].vocabulary_)}
        results.append(row)
        pd.DataFrame({"id": val.id, "label": val.label, "margin": val_margin}).to_csv(
            args.out / f"{name}_validation_margins.csv", index=False)
        print(json.dumps({k: v for k, v in row.items() if k != "config"}), flush=True)
    # Calibrate the predeclared baseline, not whichever variant scored highest.
    chosen = next(row for row in results if row["variant"] == plan["calibrate"])
    print("Calibrating the predeclared baseline...", flush=True)
    t0 = perf_counter()
    calibrated = build_calibrated_model(chosen["config"]).fit(fit.text, fit.label)
    calibration_seconds = perf_counter() - t0
    predictions = val[["id", "label", "source", "prompt_name"]].copy()
    predictions["probability"] = calibrated.predict_proba(val.text)[:, 1]
    predictions["word_count"] = val.text.str.split().str.len()
    predictions = predictions.merge(similarity, on="id", validate="one_to_one")
    predictions.to_csv(args.out / "validation_predictions.csv", index=False)
    slices = length_slices(predictions)
    slices.to_csv(args.out / "length_slices.csv", index=False)
    source_metrics = [{"source": source, **classification_summary(group.label, group.probability)}
                      for source, group in predictions.groupby("source")]
    pd.DataFrame(source_metrics).to_csv(args.out / "source_slices.csv", index=False)
    errors = select_errors(predictions).merge(val[["id", "text"]], on="id", validate="one_to_one")
    errors.to_csv(args.out / "error_cases_local_only.csv", index=False)
    # Exercise the same submission contract on held-out rehearsal rows only.
    # This is not a Kaggle submission or a replacement for Cornell's test.csv.
    stub = pd.DataFrame({"id": predictions.id, "label": 0.5})
    make_submission(predictions, predictions.probability, stub, expected_rows=len(val)).to_csv(
        args.out / "REHEARSAL_ONLY_probability_export.csv", index=False)
    lower_overlap = predictions[predictions.word_trigram_cosine < .9]
    output = {
        "purpose": plan["purpose"], "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan": plan, "input_sha256": {"archive": digest(args.archive), "plan": digest(args.config),
                                         "baseline": digest(args.baseline), "split": digest(args.out / "split.csv")},
        "source_sha256": {name: digest(Path(__file__).with_name(name))
                          for name in ["rehearsal.py", "analysis.py", "model.py", "data.py", "cli.py"]},
        "split_policy": {"method": "stratified train_test_split in sampled source-row order",
                         "validation_fraction": 0.2, "seed": 42},
        "audit": audit, "raw_variants": results,
        "calibrated_baseline": {"variant": plan["calibrate"], "fit_seconds": calibration_seconds,
                                **classification_summary(predictions.label, predictions.probability)},
        "overlap_diagnostic": {"method": "hashed word trigram cosine, 2^18 features; approximate",
                               "validation_rows_ge_0_9": int((predictions.word_trigram_cosine >= .9).sum()),
                               "max_similarity": float(predictions.word_trigram_cosine.max()),
                               "sensitivity_excluding_ge_0_9": classification_summary(lower_overlap.label, lower_overlap.probability)
                               if len(lower_overlap) else None},
        "elapsed_seconds": perf_counter() - start, "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn"]},
        "git_commit": git_output("rev-parse", "HEAD"), "git_status": git_output("status", "--porcelain"),
        "limitations": ["External essay dataset, not Cornell data", "One random split and one seed",
                        "Three predeclared variants viewed on the same development validation set",
                        "Prompt/source families can occur in both partitions; no unseen-domain claim",
                        "Approximate similarity audit can miss near duplicates; label provenance not independently verified",
                        "Baseline calibration was predeclared; no final Cornell hyperparameter selection"],
    }
    (args.out / "results.json").write_text(json.dumps(output, indent=2) + "\n")
    print("Rehearsal completed:", json.dumps(output["calibrated_baseline"]), flush=True)


if __name__ == "__main__":
    main()
