import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from model2.cornell import tune, diagnose
from model2.data import split_manifest


def fixture_files(tmp_path):
    frame = pd.DataFrame({"id": [str(i) for i in range(10000)],
                          "text": [f"unique text {i}" for i in range(10000)],
                          "label": [i % 2 for i in range(10000)]})
    frame.to_csv(tmp_path / "train.csv", index=False)
    manifest = split_manifest(frame)
    manifest.to_csv(tmp_path / "split.csv", index=False)
    return frame, manifest


def test_staged_search_never_passes_outer_validation_to_cv(tmp_path, monkeypatch):
    frame, manifest = fixture_files(tmp_path)
    held_out = set(frame.loc[manifest.split == "validation", "text"])
    calls = []
    def fake_cv(estimator, texts, labels, *, cv, scoring, n_jobs, error_score):
        assert len(texts) == 8000
        assert not set(texts) & held_out
        assert list(estimator.named_steps) == ["tfidf", "svm"]
        assert scoring == "roc_auc" and error_score == "raise"
        for fit, val in cv:
            assert not set(fit) & set(val)
            assert set(fit) | set(val) == set(range(8000))
        calls.append(estimator.named_steps["svm"].C)
        return np.array([.9, .91, .92]) if calls[-1] == 10 else np.array([.7, .71, .72])
    monkeypatch.setattr("model2.cornell.cross_val_score", fake_cv)
    root = Path(__file__).parents[1]
    out = tmp_path / "out"
    out.mkdir()
    tune(tmp_path, tmp_path / "split.csv", root / "configs/baseline.json",
         root / "configs/cornell_experiment.json", out)
    result = json.loads((out / "selection.json").read_text())
    assert calls == [1.0] * 6 + [.1, 10.0]
    assert result["selected_config"]["C"] == 10
    assert result["validation_used_for_selection"] is False


def test_diagnostics_reject_incorrect_true_labels(tmp_path):
    frame, _ = fixture_files(tmp_path)
    wrong = frame.head(2)[["id", "label"]].copy()
    wrong["label"] = 1 - wrong.label
    wrong["probability"] = .5
    wrong.to_csv(tmp_path / "predictions.csv", index=False)
    with pytest.raises(ValueError, match="labels disagree"):
        diagnose(tmp_path, tmp_path / "predictions.csv", tmp_path)
