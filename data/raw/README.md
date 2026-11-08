# Competition inputs

Place the original competition CSVs in this directory. All inputs, including
archives in `archives/`, are ignored by Git.

| File | Rows | Purpose |
|---|---:|---|
| `train.csv` | 10,000 | Labeled examples: 5,000 human and 5,000 AI |
| `train_starter_2000.csv` | 2,000 | Optional balanced subset of `train.csv` |
| `unlabeled.csv` | 3,000 | Unlabeled examples for separate experiments |
| `test.csv` | 3,000 | Prediction inputs with hidden labels |
| `sample_submission.csv` | 3,000 | Submission schema and test IDs |

Preserve the original file contents. The training/submission CLI validates the
full competition row counts; unit tests use synthetic fixtures. The audit command
also requires the starter file to check subset membership.

The starter is not additional data. It contains 386 IDs assigned to validation by
the [canonical split](../../records/cornell/shared_split/README.md), so smaller
training subsets must be drawn from the 8,000 training IDs instead.

The archived input ZIP passed its CRC check, and extracted CSV bytes match the
[provenance hashes](../../records/cornell/data_manifest.json). See the
[experiment record](../../records/cornell/README.md) for the complete audit.
