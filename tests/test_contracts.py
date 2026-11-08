import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from model2.data import make_submission, partition, read_data, split_manifest
from model2.model import build_calibrated_model, build_pipeline


def toy_data():
    return pd.DataFrame({"id": [f"{i:04}" for i in range(40)],
                         "text": [f"{'casual hello friend' if i % 2 == 0 else 'formal analysis conclusion'} {i}"
                                  for i in range(40)], "label": [i % 2 for i in range(40)]})


def test_split_is_disjoint_stratified_and_joins_by_id():
    frame = toy_data()
    manifest = split_manifest(frame)
    pd.testing.assert_frame_equal(manifest, split_manifest(frame))
    train, val = partition(frame.sample(frac=1, random_state=7), manifest)
    assert len(train) == 32 and len(val) == 8
    assert not set(train.id) & set(val.id)
    assert val.label.value_counts().to_dict() == {0: 4, 1: 4}


def test_input_preserves_ids_and_literal_na(tmp_path):
    path = tmp_path / "input.csv"
    path.write_text('id,text,label\n001,NA,0\n002,hello,1\n')
    frame = read_data(path, labeled=True)
    assert frame.id.tolist() == ["001", "002"] and frame.text.iloc[0] == "NA"
    path.write_text('id,text,label\n001,hello,0\n001,world,1\n')
    with pytest.raises(ValueError, match="unique"):
        read_data(path, labeled=True)


@pytest.mark.parametrize("p", [[0.2, np.nan], [-0.1, 0.5], [0.2, 1.1], [0.2]])
def test_invalid_submission_rejected(p):
    test = pd.DataFrame({"id": ["002", "001"]})
    sample = pd.DataFrame({"id": ["001", "002"], "label": [0, 0]})
    with pytest.raises(ValueError):
        make_submission(test, p, sample, expected_rows=2)


def test_submission_follows_test_order_and_rejects_wrong_ids():
    test = pd.DataFrame({"id": ["002", "001"]})
    sample = pd.DataFrame({"id": ["001", "002"], "label": [0, 0]})
    result = make_submission(test, [0.9, 0.1], sample, expected_rows=2)
    assert result.to_dict("list") == {"id": ["002", "001"], "label": [0.9, 0.1]}
    sample.loc[0, "id"] = "003"
    with pytest.raises(ValueError, match="Sample"):
        make_submission(test, [0.9, 0.1], sample, expected_rows=2)


def test_calibration_and_vocabulary_boundary():
    config = json.loads((Path(__file__).parents[1] / "configs/baseline.json").read_text())
    frame = toy_data()
    train, val = partition(frame, split_manifest(frame))
    raw = build_pipeline(config).fit(train.text, train.label)
    vocab_before = raw.named_steps["tfidf"].vocabulary_.copy()
    raw.decision_function(["ΩΩΩ held out only"])
    assert raw.named_steps["tfidf"].vocabulary_ == vocab_before
    assert "ΩΩΩ" not in vocab_before
    model = build_calibrated_model(config)
    assert list(model.estimator.named_steps) == ["tfidf", "svm"]
    model.fit(train.text, train.label)
    assert model.classes_.tolist() == [0, 1]
    p = model.predict_proba(val.text)[:, 1]
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    assert roc_auc_score(val.label, p) > 0.9  # Separable synthetic fixture only.
