import hashlib
import importlib.util
from pathlib import Path
import zipfile
import pytest

spec = importlib.util.spec_from_file_location('acquire_data', Path(__file__).parents[1] / 'scripts/acquire_data.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_archive_hash_validation_and_no_overwrite(tmp_path):
    data = b'id,text,label\n1,hello,0\n'
    manifest = {'files': {'train.csv': {'sha256': hashlib.sha256(data).hexdigest()}}}
    archive = tmp_path / 'input.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('nested/train.csv', data)
    out = tmp_path / 'raw'
    module.extract_verified(archive, out, manifest)
    assert (out / 'train.csv').read_bytes() == data
    with pytest.raises(FileExistsError):
        module.extract_verified(archive, out, manifest)
    manifest['files']['train.csv']['sha256'] = 'wrong'
    with pytest.raises(ValueError, match='differs'):
        module.extract_verified(archive, tmp_path / 'rejected', manifest)
    assert not (tmp_path / 'rejected').exists()


def test_archive_traversal_rejected_before_writing(tmp_path):
    archive = tmp_path / 'unsafe.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('../train.csv', 'bad')
    with pytest.raises(ValueError, match='Unsafe'):
        module.extract_verified(archive, tmp_path / 'raw', {'files': {}})
    assert not (tmp_path / 'train.csv').exists()
