# Validation diagnostics

These are descriptive analyses of the same reused development split, not new generalization tests. At threshold 0.5, baseline has 70 false positives and 102 false negatives; selected has 68 and 103. Calibrated validation AUROC is 0.969466 and 0.970440 respectively.

For selected, AUROC by word-count bin is 0.92767 (<100; 59 rows, only 6 human), 0.95995 (100–199; 887 rows), 0.97901 (200–299; 883 rows), and 0.98832 (>=300; 171 rows). The smallest slice is sparse and strongly imbalanced; these figures do not establish a causal effect of length and should not be used for new tuning on this validation set.

Codex inspected excerpts from the five most confident false positives and false negatives for each model. The selected model's false-positive examples include conversational posts and reference-style technical prose. False negatives include recipe-like text, a technical abstract, a conversational passage, and biographical prose. These are informal descriptions of selected excerpts, not verified source labels or frequencies across all errors. They illustrate that topic or form alone is not a reliable authorship rule. The selection deliberately focuses on confident mistakes and is not representative. No ground-truth labels were changed.

The selected model's in-sample training AUROC is 0.999996, versus 0.970440 on validation. This gap warrants caution about generalization. Source/generator labels are unavailable, so source-held-out performance is not measured. Validation reuse, near-duplicate passages, and possible test distribution shift remain limitations.

Raw texts and per-row predictions remain in ignored artifacts. Aggregate length-slice tables and confusion/calibration metrics are curated separately. No Kaggle score is available.
