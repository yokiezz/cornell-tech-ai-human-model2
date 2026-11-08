# Cornell split v1

Canonical training/validation membership for Cornell model development. Use the
supplied `split.csv` for all model comparisons; a seed alone does not preserve
membership if input row order or splitting software changes.

- Original train.csv order; stratified 80/20 split, random_state=42.
- 8,000 training rows (4,000 per class), 2,000 validation rows (1,000 per class).
- Join split.csv by string id; columns are id,split. No text is included.
- Original train.csv SHA-256: 325e345faf88fb1c4203278098c282e40319a87b4bf17e094d2f235f51d0c07e
- split.csv SHA-256: 15340e28ee050a0023b017abeddd2d3b72b7fc05612b9d299b59d88d31a1cb2b
- No exact normalized duplicates across the split. An approximate hashed word-trigram check flags 12 high-similarity validation cases; results must disclose this and include a sensitivity check.
- The supplied starter contains 386 validation IDs. Draw smaller training subsets from the training IDs instead.
- Fit TF-IDF, feature selection, calibration and tuning only within the training partition.
- Export validation id,label,probability (label is ground truth, probability is P(AI)); align models by ID.
- Alternative group-aware robustness splits require separate versioned manifests and consistent membership across compared models.
