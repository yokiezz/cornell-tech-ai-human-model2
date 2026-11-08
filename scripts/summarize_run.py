"""Verify ID alignment, sensitivity, and full-data predictions; curate aggregates."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from model2.analysis import classification_summary
from model2.cli import digest
from model2.data import make_submission, read_data

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run-id', required=True)
args = parser.parse_args()
base = Path('artifacts') / args.run_id
record = Path('records/reproductions') / args.run_id
split = pd.read_csv('records/cornell/shared_split/split.csv', dtype=str)
validation_ids = set(split.loc[split.split == 'validation', 'id'])
training_ids = set(split.loc[split.split == 'train', 'id'])
overlap = pd.read_csv(base/'audit/validation_overlap.csv', dtype={'id': str})
assert overlap.id.is_unique and set(overlap.id) == validation_ids
flagged = set(overlap.loc[overlap.word_trigram_cosine >= .9, 'id'])
result = {'method': 'hashed word trigram cosine to training; exclude validation IDs with similarity >= 0.9; no refit',
          'threshold': .9, 'flagged_count': len(flagged), 'flagged_ids': sorted(flagged), 'models': {}}
for name in ['baseline', 'selected']:
    frame = pd.read_csv(base/name/'validation_predictions.csv', dtype={'id': str})
    assert frame.id.is_unique and set(frame.id) == validation_ids
    keep = frame.loc[~frame.id.isin(flagged)]
    result['models'][name] = {'all_validation': classification_summary(frame.label, frame.probability),
                             'excluding_high_similarity': classification_summary(keep.label, keep.probability)}
    for filename in ['metrics.json', 'length_slices.csv']:
        shutil.copy2(base/(name+'-diagnostics')/filename, record/(name+'-diagnostics')/filename)
    shutil.copy2(base/name/'manifest.json', record/name/'manifest.json')
examples = pd.read_csv(base/'exploration/examples_local_only.csv', dtype={'id': str})
assert len(examples) == 6 and examples.id.is_unique and set(examples.id) <= training_ids
assert examples.label.value_counts().to_dict() == {0: 3, 1: 3}
result['exploration'] = {'example_ids': examples.id.tolist(), 'all_six_training_only': True,
                         'user_review': 'pending; no user interpretation received'}
submission = pd.read_csv(base/'full-data/submission.csv', dtype={'id': str})
test = read_data('data/raw/test.csv', expected_rows=3000)
sample = pd.read_csv('data/raw/sample_submission.csv', dtype={'id': str})
assert list(submission.columns) == ['id', 'label']
assert submission.id.tolist() == test.id.tolist()
checked = make_submission(test, submission.label, sample)
assert len(checked) == 3000
result['submission'] = {'rows': len(submission), 'unique_ids': submission.id.nunique(),
                        'matches_test_order': True, 'finite_in_unit_interval': True,
                        'min_probability': float(submission.label.min()), 'max_probability': float(submission.label.max()),
                        'sha256': digest(base/'full-data/submission.csv'), 'uploaded': False}
for name, filenames in [('audit', ['audit.json']), ('exploration', ['summary.csv', 'overview.png', 'manifest.json']),
                        ('full-data', ['manifest.json'])]:
    for filename in filenames:
        shutil.copy2(base/name/filename, record/name/filename)
(record/'sensitivity_and_validation.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
