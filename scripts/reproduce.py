"""Run the prescribed fixed-configuration reproduction without tuning or uploads."""
import argparse
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run-id', required=True)
args = parser.parse_args()
if Path(args.run_id).name != args.run_id or args.run_id in {'.', '..'}:
    parser.error('run-id must be a single directory name')
base = Path('artifacts') / args.run_id
record = Path('records/reproductions') / args.run_id
split = ['--split', 'records/cornell/shared_split/split.csv']
steps = [
 ('audit', ['model2.cornell', 'audit', *split]),
 ('exploration', ['model2.explore', *split]),
 ('baseline', ['model2.cli', 'evaluate', *split, '--config', 'configs/baseline.json']),
 ('selected', ['model2.cli', 'evaluate', *split, '--config', 'configs/cornell_selected.json']),
 ('baseline-diagnostics', ['model2.cornell', 'diagnose', '--predictions', str(base/'baseline/validation_predictions.csv')]),
 ('selected-diagnostics', ['model2.cornell', 'diagnose', '--predictions', str(base/'selected/validation_predictions.csv')]),
 ('full-data', ['model2.cli', 'submit', '--config', 'configs/cornell_selected.json']),
]
for name, command in steps:
    print('Starting', name, flush=True)
    subprocess.run([sys.executable, 'scripts/run_recorded.py', '--record', str(record/name),
                    '--', sys.executable, '-m', *command, '--out', str(base/name)], check=True)
subprocess.run([sys.executable, 'scripts/run_recorded.py', '--record', str(record/'verification'),
                '--', sys.executable, 'scripts/summarize_run.py', '--run-id', args.run_id], check=True)
subprocess.run([sys.executable, 'scripts/run_recorded.py', '--record', str(record/'plots'),
                '--', sys.executable, 'scripts/plot_cornell.py',
                '--baseline', str(base/'baseline/validation_predictions.csv'),
                '--selected', str(base/'selected/validation_predictions.csv'),
                '--out', str(record/'diagnostics.png')], check=True)
