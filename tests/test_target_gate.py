import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

from backend.target_gate import ROOT, PROTOCOL, sha, source_digest, validate_certificate


class TargetGate(unittest.TestCase):
    def certificate(self):
        protocol = json.loads(PROTOCOL.read_text())
        rows = {}
        for task, case in protocol['cases'].items():
            rows[task] = dict(candidate_id=case['id'], memory_exact_match=True,
                             same_runtime_equivalence_passed=True, same_runtime_tolerance=1e-5,
                             same_runtime_max_probability_abs_error=0., same_runtime_max_sscp_abs_error=0.,
                             historical_tolerance=1e-5, historical_equivalence_passed=False,
                             historical_max_probability_abs_error=.002,
                             reference_sources_verified=True, runtime=dict(gpu='fixed GPU', torch='fixed runtime'))
        return dict(schema='rimbind_target_numerical_certificate_v1',
                    protocol_sha256=sha(PROTOCOL), source_sha256=source_digest(),
                    asset_lock_sha256='fixed-lock', cases=rows)

    def test_owner_approval_does_not_remove_historical_failures(self):
        cert = self.certificate()
        self.assertTrue(validate_certificate(cert, 'fixed-lock', current_runtime=cert['cases']['dna']['runtime']))
        self.assertFalse(cert['cases']['dna']['historical_equivalence_passed'])
        self.assertFalse(json.loads(PROTOCOL.read_text())['public_release_authorized_by_this_file'])

    def test_each_task_required(self):
        cert = self.certificate()
        del cert['cases']['rna']
        with self.assertRaises(RuntimeError):
            validate_certificate(cert, 'fixed-lock')

    def test_changed_source_asset_protocol_and_gpu_rejected(self):
        for key in ('source_sha256', 'asset_lock_sha256', 'protocol_sha256'):
            cert = self.certificate()
            cert[key] = 'changed'
            with self.assertRaises(RuntimeError):
                validate_certificate(cert, 'fixed-lock')
        with self.assertRaises(RuntimeError):
            validate_certificate(self.certificate(), 'fixed-lock', current_runtime=dict(gpu='new GPU'))

    def test_both_tasks_must_use_same_runtime(self):
        cert = self.certificate()
        cert['cases']['rna']['runtime']['gpu'] = 'another GPU'
        with self.assertRaises(RuntimeError):
            validate_certificate(cert, 'fixed-lock')

    def test_failures_and_incomplete_memory_do_not_open_gate(self):
        for key, value in [('same_runtime_max_probability_abs_error', .00001),
                           ('same_runtime_max_sscp_abs_error', .001), ('memory_exact_match', False),
                           ('reference_sources_verified', False), ('historical_max_probability_abs_error', float('inf')),
                           ('same_runtime_tolerance', .01), ('historical_equivalence_passed', True)]:
            cert = self.certificate()
            cert['cases']['dna'][key] = value
            with self.assertRaises(RuntimeError, msg=key):
                validate_certificate(cert, 'fixed-lock')

    def test_original_reference_files_hash_pinned(self):
        for name, expected in json.loads(PROTOCOL.read_text())['reference_sources'].items():
            self.assertEqual(sha(ROOT/name), expected)

    def test_package_and_repository_fingerprint_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/'package'
            subprocess.run([sys.executable, str(ROOT/'tools/prepare_full_space.py'), '--output', str(out)],
                           check=True, capture_output=True)
            target = Path(directory)/'extracted'
            with zipfile.ZipFile(out/'runtime.zip') as archive:
                archive.extractall(target)
                self.assertIn('vendor/frozen_c0_prediction.py', archive.namelist())
                self.assertIn('configs/target_acceptance_protocol.json', archive.namelist())
                self.assertFalse(any('cases/' in name or name.endswith('.csv') for name in archive.namelist()))
            command = [sys.executable, '-c', 'from backend.target_gate import source_digest; print(source_digest())']
            result = subprocess.run(command, cwd=target, capture_output=True, text=True, check=True)
            self.assertEqual(result.stdout.strip(), source_digest())


if __name__ == '__main__':
    unittest.main()
