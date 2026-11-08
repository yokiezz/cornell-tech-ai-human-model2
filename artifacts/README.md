# Generated outputs

This directory holds local run outputs and is ignored by Git except for this README.
Use a new subdirectory for every run; existing outputs are never overwritten.

- `shared-split/`: local copy of the canonical ID split, when using CLI defaults.
- `cornell/`: audits, search results, predictions, diagnostics, and exploration outputs.
- `rehearsal/`: historical DAIGT outputs; removed during cleanup.

A directory alone does not establish completion. Evaluation, submission, and
exploration write `manifest.json`; staged tuning writes `selection.json` after
all trials finish. Audit and diagnostic commands write their respective JSON
reports. Failed commands can leave partial outputs.

The canonical split is tracked under `records/cornell/shared_split/`. Curated
metrics and provenance belong in `records/`; raw examples, bulk predictions,
submission CSVs, and model binaries remain local.
