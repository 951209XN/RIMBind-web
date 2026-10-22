"""Static deployment contracts; these do not test hosted inference."""
import ast
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SPACE = Path(__file__).resolve().parents[1] / 'deploy/hf-space'

class SpacePackage(unittest.TestCase):
    def setUp(self):
        self.metadata = dict(line.split(':', 1) for line in (SPACE / 'README.md').read_text().split('---')[1].strip().splitlines())
        self.metadata = {key: value.strip() for key, value in self.metadata.items()}

    def test_python_version_is_quoted_string(self):
        self.assertEqual(self.metadata['python_version'], "'3.12'")

    def test_sdk_requirement_matches(self):
        self.assertIn('gradio==' + self.metadata['sdk_version'], (SPACE / 'requirements.txt').read_text().splitlines())

    def test_card_colors_are_supported(self):
        allowed = {'red','yellow','green','blue','indigo','purple','pink','gray'}
        for field in ('colorFrom', 'colorTo'):
            self.assertIn(self.metadata[field], allowed)

    def test_import_does_not_launch_server(self):
        source = (SPACE / 'app.py').read_text()
        tree = ast.parse(source)
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'launch']
        guards = [node for node in tree.body if isinstance(node, ast.If) and ast.unparse(node.test) == "__name__ == '__main__'"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(guards), 1)
        self.assertIn(calls[0], list(ast.walk(guards[0])))

    def test_flat_bundle_exact_files_and_unchanged_parser(self):
        repo = SPACE.parents[1]
        with tempfile.TemporaryDirectory(prefix='rimbind-flat-test-') as temporary:
            target = Path(temporary) / 'package'
            subprocess.run([sys.executable, str(repo / 'tools/prepare_space.py'), '--flat', '--output', str(target)], check=True, capture_output=True)
            self.assertEqual({p.name for p in target.iterdir()}, {'app.py','inputs.py','README.md','requirements.txt','LICENSE'})
            self.assertTrue(all(p.is_file() for p in target.iterdir()))
            self.assertIn('from inputs import parse_pdb', (target / 'app.py').read_text())
            digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            self.assertEqual(digest(target / 'inputs.py'), digest(repo / 'backend/inputs.py'))

    def test_original_bundle_still_has_backend_directory(self):
        repo = SPACE.parents[1]
        with tempfile.TemporaryDirectory(prefix='rimbind-nested-test-') as temporary:
            target = Path(temporary) / 'package'
            subprocess.run([sys.executable, str(repo / 'tools/prepare_space.py'), '--output', str(target)], check=True, capture_output=True)
            self.assertTrue((target / 'backend/inputs.py').is_file())
            self.assertIn('from backend.inputs import parse_pdb', (target / 'app.py').read_text())

if __name__ == '__main__':
    unittest.main()
