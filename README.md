# Cornell Tech CS5785 — Model 2 reproduction

Character TF-IDF with a sigmoid-calibrated Linear SVM. Human = 0, AI = 1; the primary metric is AUROC. This is a locally adapted snapshot of JD-TB1's implementation, pinned to `1b1d379964c8b2ceb75e4eef531bbf90623ad176`. See [source attribution and AI disclosure](PROVENANCE.md).

## Install

Python 3.11 or newer is required. The recorded environment uses Python 3.11 on macOS ARM64. From this directory:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt -e '.[dev,plots,kaggle]'
.venv/bin/python -m pip check
.venv/bin/python -m pytest -q
```

`requirements-lock.txt` records exact versions used here; `pyproject.toml` declares supported ranges. Kaggle 2.2.5 imports `six` but did not install it transitively in this environment, so it is explicitly included in the Kaggle extra. Do not install into global Python.

## Acquire verified data

The local reproduction reused an already downloaded official competition ZIP. All five CSV hashes and the ZIP hash match inherited evidence. No API download is claimed. Existing data are never overwritten.

```sh
.venv/bin/python scripts/acquire_data.py \
  --archive /absolute/path/to/competition.zip \
  --download-dir data/raw/archives/reuse-002 \
  --record records/reproductions/run-002/acquisition.json
```

For an API acquisition on a fresh checkout, authenticate locally through the official client, then omit `--archive`:

```sh
export KAGGLE_CONFIG_DIR="$PWD/.kaggle"
.venv/bin/kaggle auth login
.venv/bin/kaggle competitions files cornell-tech-cs-5785-combined-2026-ai-vs-human-text
.venv/bin/python scripts/acquire_data.py \
  --download-dir data/raw/archives/download-001 \
  --record records/reproductions/download-001/acquisition.json
```

Keep credentials in the ignored `.kaggle/` directory; never paste them into chat or commit them. API authentication has not been completed on this machine. Client help and the missing-authentication response were checked. The download workflow is implemented, but authenticated end-to-end API acquisition remains unverified. Downloading does not authorize submissions.

For a refresh, use both a new `--download-dir` and a new `--out`; investigate any CSV hash differences before running models or reusing the canonical split. Preserve original inputs and archives.

## Reproduce

The canonical manifest is `records/cornell/shared_split/split.csv`: 8,000 training rows and 2,000 validation rows, balanced within each partition. Join string IDs; do not regenerate membership. The starter is a subset, with 386 validation IDs, and must not be used directly as a smaller training set.

```sh
.venv/bin/python scripts/reproduce.py --run-id run-002
```

Use a new run ID for every run. This performs audit, training-only exploration, unchanged baseline and selected fits, diagnostics, full-data local prediction generation, overlap sensitivity, and plots. It does not retune or upload anything. A failed step stops the sequence; consult its execution record and rerun the failed command with new output/record directories. Completed output directories are never overwritten.

Each command has UTC start/completion timestamps, return status, input/configuration/source hashes, package versions, and Git state in `records/reproductions/<run-id>/<step>/execution.json`. Raw prediction files and example texts are in ignored `artifacts/<run-id>/`; only aggregate records are curated into Git. `scripts/run_recorded.py` can wrap individual commands using `--record NEW_DIRECTORY -- COMMAND ...`.

## Method and interpretation

Baseline: character 3–5 grams; selected configuration: character 2–4 grams. Both use C=1, min_df=2, max_features=100000, lowercase and sublinear term frequency, and seed 42. Preprocessing lives inside each fold. Three-fold sigmoid calibration uses training rows only; the final calibrated pipeline fits all supplied training rows. Validation evaluation fits 8,000 rows; the final local export fits all 10,000 labeled rows. Test and unlabeled text are never included in supervised fitting.

The eight-configuration training-only selection is inherited from the reference project; it was not repeated. Reproducing a previously used validation set is not an independent generalization test. The small selected-versus-baseline gain is not established as reliable. Twelve validation passages have high hashed word-trigram similarity to training; sensitivity excludes those validation IDs from scoring without changing or refitting the models. Source/generator-held-out performance remains unmeasured.

See `records/reproductions/run-001/` for new evidence and `records/cornell/` for attributed historical evidence. New run results and interpretation are summarized in `RESULTS.md` when verified. The submission CSV contains exactly `id,label`, where label means P(AI), unlike the ground-truth label in validation predictions.

## Scope

This repository contributes Model 2. Complete course submission additionally needs Models 1/3, official report requirements/template, team selection, and separately authorized publication/submission. These reproduction scripts do not create repositories, push code, open PRs, or upload Kaggle predictions. GitHub publication is handled separately with user authorization.
