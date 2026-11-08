"""Run a command with UTC timestamps and immutable input/source provenance."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == '--':
        command = command[1:]
    if not command:
        parser.error('A command is required after --')
    args.record.mkdir(parents=True, exist_ok=False)
    paths = [*Path('src').rglob('*.py'), *Path('scripts').glob('*.py'),
             *Path('configs').glob('*.json'), *Path('data/raw').glob('*.csv'),
             Path('records/cornell/shared_split/split.csv'), Path('pyproject.toml')]
    def git(*parts):
        proc = subprocess.run(['git', *parts], capture_output=True, text=True)
        return proc.stdout.strip() if proc.returncode == 0 else None
    record = {'started_at_utc': now(), 'command': command,
              'python': platform.python_version(), 'platform': platform.platform(),
              'reference_revision': '1b1d379964c8b2ceb75e4eef531bbf90623ad176',
              'git_commit': git('rev-parse', 'HEAD'), 'git_status': git('status', '--porcelain'),
              'sha256': {str(p): sha(p) for p in sorted(paths) if p.is_file()},
              'packages': {p: importlib.metadata.version(p) for p in
                           ['numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib', 'kaggle']},
              'status': 'running'}
    destination = args.record / 'execution.json'
    destination.write_text(json.dumps(record, indent=2) + '\n')
    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path('src').resolve())
    env['MPLCONFIGDIR'] = str(Path('.mplconfig').resolve())
    try:
        with (args.record / 'console.log').open('x') as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        record['returncode'] = result.returncode
        record['status'] = 'completed' if result.returncode == 0 else 'failed'
    except BaseException as error:
        record['status'] = 'failed'
        record['error'] = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        record['completed_at_utc'] = now()
        destination.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'record': str(args.record), 'status': record['status']}), flush=True)
    sys.exit(result.returncode)


if __name__ == '__main__':
    main()
