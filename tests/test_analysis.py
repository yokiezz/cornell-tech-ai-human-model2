import numpy as np
import pandas as pd
import pytest

from model2.analysis import classification_summary, length_slices, select_errors
from model2.rehearsal import prepare_sample


def test_known_confusion_and_single_class_slices():
    summary = classification_summary([0, 0, 1, 1], [.1, .8, .2, .9])
    assert summary["auroc"] == .75
    assert [summary[key] for key in ["true_negative", "false_positive", "false_negative", "true_positive"]] == [1, 1, 1, 1]
    frame = pd.DataFrame({"label": [0, 0, 1, 1], "probability": [.1, .8, .2, .9],
                          "word_count": [99, 100, 200, 300]})
    slices = length_slices(frame)
    assert slices.rows.tolist() == [1, 1, 1, 1]
    assert slices.auroc.isna().all()
    assert classification_summary([1, 1], [.2, .9])["false_positive_rate"] is None


def test_error_direction_and_empty_bins():
    frame = pd.DataFrame({"id": ["h", "a", "ok"], "label": [0, 1, 1],
                          "probability": [.9, .1, .8], "word_count": [200, 200, 200]})
    assert select_errors(frame)[["id", "error_type"]].to_dict("list") == {
        "id": ["h", "a"], "error_type": ["FP", "FN"]}
    assert length_slices(frame).rows.tolist() == [0, 0, 3, 0]
    with pytest.raises(ValueError, match="finite probabilities"):
        classification_summary([0, 1], [np.nan, .8])


def test_sampling_excludes_conflicts_and_normalized_duplicates():
    frame = pd.DataFrame({"text": [f"distinct text {i}" for i in range(60)],
                          "label": [i % 2 for i in range(60)],
                          "source": ["fixture"] * 60, "prompt_name": ["fixture"] * 60})
    additions = pd.DataFrame({"text": ["DISTINCT   TEXT 0", "distinct text 1"],
                              "label": [0, 0], "source": ["fixture"] * 2,
                              "prompt_name": ["fixture"] * 2})
    sample, audit = prepare_sample(pd.concat([frame, additions], ignore_index=True), 20, 42)
    assert audit["normalized_duplicate_extra_rows"] == 2
    assert audit["conflicting_normalized_groups"] == 1
    assert audit["conflicting_rows_excluded"] == 2
    assert audit["eligible_unique_rows"] == 59
    assert sample.label.value_counts().to_dict() == {0: 10, 1: 10}
    assert not sample.normalized.duplicated().any()
    assert "distinct text 1" not in sample.normalized.tolist()
    again, _ = prepare_sample(pd.concat([frame, additions], ignore_index=True), 20, 42)
    pd.testing.assert_frame_equal(sample, again)
