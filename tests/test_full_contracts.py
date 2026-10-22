import ast
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys
import zipfile

import numpy as np

from backend.inputs import parse_pdb
from backend.results import write_results

ROOT = Path(__file__).resolve().parents[1]


class FullContracts(unittest.TestCase):
    def test_frozen_model_bytes_unchanged(self):
        digest = hashlib.sha256((ROOT/'vendor/frozen_model_single.py').read_bytes()).hexdigest()
        self.assertEqual(digest, 'cb7e9b754ef15555d211168d809f078fb424d3dfbff7718833e010dfa5a2819c')

    def test_exports_retain_mapping_and_probability_scale(self):
        protein = parse_pdb((ROOT/'examples/6jhe_A/example.pdb').read_text(), 'A')
        n = len(protein.sequence)
        result = dict(sscp=np.full(n, .2), rimbind=np.full(n, .345),
                      provenance=dict(task='rna', rule='NO_TEMPLATE_COUNT'))
        with tempfile.TemporaryDirectory() as directory:
            table, files = write_results(Path(directory)/'result', protein, result, dict(memory_hits=0))
            with Path(files[0]).open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), n)
            self.assertEqual([r['amino_acid'] for r in rows], list(protein.sequence))
            self.assertEqual(rows[0]['author_residue_number'], str(protein.residue_keys[0]['author_residue_number']))
            self.assertAlmostEqual(float(rows[0]['rimbind_probability']), .345)
            first = Path(files[1]).read_text().splitlines()[0]
            self.assertEqual(float(first[60:66]), .34)  # Python's exact float formatting
            provenance = json.loads(Path(files[2]).read_text())
            self.assertFalse(provenance['metrics_computed'])
            self.assertFalse(provenance['query_labels_read'])
            self.assertEqual(provenance['rule'], 'NO_TEMPLATE_COUNT')

    def test_invalid_output_never_exported_as_success(self):
        protein = parse_pdb((ROOT/'examples/6jhe_A/example.pdb').read_text(), 'A')
        n = len(protein.sequence)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                write_results(Path(directory)/'result', protein,
                              dict(sscp=np.ones(n), rimbind=np.full(n, np.nan)), {})

    def test_new_full_app_gate_is_closed_by_default(self):
        source = (ROOT/'deploy/hf-full/app.py').read_text()
        self.assertIn("os.environ.get('RIMBIND_ENABLE_PREDICTION') != '1'", source)
        self.assertIn('validate_certificate(CERTIFICATE, SERVICE.assets.lock_sha256)', source)
        self.assertIn('current_runtime=runtime_fingerprint()', source)
        self.assertIn('interactive=ENABLED', source)
        self.assertIn('@spaces.GPU(duration=120)', source)
        self.assertIn('api_name=False', source)
        self.assertIn('query', (ROOT/'backend/service.py').read_text())

    def test_no_automatic_download_in_encoder_adapter(self):
        source = (ROOT/'backend/encoders.py').read_text()
        tree = ast.parse(source)
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        names = [getattr(node.func, 'id', '') for node in calls]
        self.assertNotIn('snapshot_download', names)
        self.assertIn('local_files_only=True', source)

    def test_full_sources_parse_without_execution(self):
        for path in [* (ROOT/'backend').glob('*.py'), ROOT/'deploy/hf-full/app.py',
                     ROOT/'tools/acceptance_raw.py', ROOT/'tools/acceptance_prepared.py']:
            ast.parse(path.read_text(), filename=str(path))

    def test_flat_full_package_excludes_weights_and_private_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/'package'
            subprocess.run([sys.executable, str(ROOT/'tools/prepare_full_space.py'), '--output', str(out)],
                           check=True, capture_output=True, text=True)
            self.assertEqual({p.name for p in out.iterdir()}, {'app.py', 'runtime.zip', 'README.md', 'requirements.txt', 'LICENSE'})
            with zipfile.ZipFile(out/'runtime.zip') as archive:
                self.assertFalse(any(name.endswith(('.pt', '.pth', '.bin')) for name in archive.namelist()))
                self.assertIn('backend/service.py', archive.namelist())
                self.assertIn('vendor/__init__.py', archive.namelist())
            digest = hashlib.sha256((out/'runtime.zip').read_bytes()).hexdigest()
            self.assertIn(digest, (out/'app.py').read_text())


if __name__ == '__main__':
    unittest.main()
