"""Acquire official competition CSVs from Kaggle or a verified existing ZIP."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import zipfile

COMPETITION = 'cornell-tech-cs-5785-combined-2026-ai-vs-human-text'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def extract_verified(archive, destination, manifest):
    """Validate all members and bytes before extracting any CSV."""
    payloads = {}
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            path = PurePosixPath(member.filename)
            if path.is_absolute() or '..' in path.parts or '\\' in member.filename:
                raise ValueError('Unsafe archive path')
            if member.is_dir():
                continue
            if path.name in manifest['files']:
                if path.name in payloads:
                    raise ValueError('Ambiguous duplicate CSV member')
                data = zipped.read(member)
                if sha(data) != manifest['files'][path.name]['sha256']:
                    raise ValueError(f'CSV differs from reference: {path.name}')
                payloads[path.name] = data
    if set(payloads) != set(manifest['files']):
        raise ValueError('Archive does not contain all required CSVs')
    if any((destination / name).exists() for name in payloads):
        raise FileExistsError('Refusing to overwrite input CSVs; use a new directory')
    destination.mkdir(parents=True, exist_ok=True)
    for name, data in payloads.items():
        with (destination / name).open('xb') as file:
            file.write(data)
    return {name: sha(data) for name, data in payloads.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='Reuse an existing local ZIP; no API call')
    parser.add_argument('--out', type=Path, default=Path('data/raw'))
    parser.add_argument('--download-dir', type=Path, required=True, help='New archive directory')
    parser.add_argument('--record', type=Path, required=True)
    args = parser.parse_args()
    started = datetime.now(timezone.utc).isoformat()
    args.download_dir.mkdir(parents=True, exist_ok=False)
    mode = 'local_archive_reuse' if args.archive else 'official_kaggle_api'
    if args.archive:
        archive = args.download_dir / args.archive.name
        shutil.copy2(args.archive, archive)
    else:
        executable = str(Path(sys.executable).parent / 'kaggle')
        env = os.environ.copy()
        env.setdefault('KAGGLE_CONFIG_DIR', str(Path('.kaggle').resolve()))
        subprocess.run([executable, 'competitions', 'download', COMPETITION,
                        '-p', str(args.download_dir)], check=True, env=env)
        archives = list(args.download_dir.glob('*.zip'))
        if len(archives) != 1:
            raise ValueError('Expected exactly one downloaded ZIP')
        archive = archives[0]
    manifest = json.loads(Path('records/cornell/data_manifest.json').read_text())
    hashes = extract_verified(archive, args.out, manifest)
    record = {'mode': mode, 'competition': COMPETITION,
              'started_at_utc': started,
              'completed_at_utc': datetime.now(timezone.utc).isoformat(),
              'kaggle_version': importlib.metadata.version('kaggle'),
              'archive_sha256': sha(archive.read_bytes()), 'csv_sha256': hashes,
              'all_csv_hashes_match_reference': True,
              'original_download_time': None,
              'note': 'Timestamps describe this acquisition/reuse, not an inferred historical download.'}
    args.record.parent.mkdir(parents=True, exist_ok=True)
    with args.record.open('x') as file:
        file.write(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
