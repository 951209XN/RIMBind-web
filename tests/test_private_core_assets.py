import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('private_core_builder', ROOT/'tools/prepare_private_core_assets.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class PrivateAssets(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        (root/'configs').mkdir()
        checkpoints, database = root/'core', root/'db'
        checkpoints.mkdir()
        database.mkdir()
        content = b'not-a-pickle; hashing-only fixture'
        (checkpoints/'sample.pt').write_bytes(content)
        row = dict(archive_path='sample.pt', bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
        (root/'configs/assets_manifest.json').write_text(json.dumps(dict(sscp_checkpoints=[row], training_memory=[])))
        files = []
        for task in ('dna', 'rna'):
            for suffix in ('', '.dbtype', '.index', '_h', '_h.dbtype', '_h.index',
                           '_ss', '_ss.dbtype', '_ss.index', '_ca', '_ca.dbtype', '_ca.index'):
                path = database/(task+suffix)
                path.write_bytes(b'database-hash-fixture')
                files.append(dict(path='reference/memory/'+path.name, sha256=builder.sha(path)))
        lock = root/'lock.json'
        lock.write_text(json.dumps(dict(paths=dict(memory_db='reference/memory'), files=files)))
        return root, checkpoints, database, lock

    def test_locked_file_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root, checkpoints, database, lock = self.fixture(directory)
            with patch.object(builder, 'ROOT', root):
                self.assertEqual(len(builder.inventory(checkpoints, database, lock)), 25)

    def test_changed_database_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, checkpoints, database, lock = self.fixture(directory)
            (database/'dna').write_bytes(b'changed')
            with patch.object(builder, 'ROOT', root), self.assertRaises(RuntimeError):
                builder.inventory(checkpoints, database, lock)

    def test_unexpected_prediction_file_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, checkpoints, database, lock = self.fixture(directory)
            (database/'test_labels.pt').write_bytes(b'not allowed')
            with patch.object(builder, 'ROOT', root), self.assertRaises(RuntimeError):
                builder.inventory(checkpoints, database, lock)


if __name__ == '__main__':
    unittest.main()
