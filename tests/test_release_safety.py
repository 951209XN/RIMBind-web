"""CPU-only tests of unchanged serving code; no GPU, assets or network.

Extract the real class/UI handler to avoid importing large encoder runtimes in CI.
Dependency boundaries are mocked explicitly; these are not hosted browser tests.
"""
import ast
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import uuid

import numpy as np
from backend.inputs import parse_pdb
from backend.results import write_results
from backend.worker_transport import call_worker, WorkerExecutionError

ROOT = Path(__file__).resolve().parents[1]


def extract(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


class ReleaseSafety(unittest.TestCase):
    def setUp(self):
        self.pdb = (ROOT/'examples/6jhe_A/example.pdb').read_text()
        self.protein = parse_pdb(self.pdb, 'A')
        self.n = len(self.protein.sequence)

    def test_reject_oversize_and_bad_chain_before_processing(self):
        for chain in ('', 'AB', ';'):
            with self.subTest(chain=chain), self.assertRaises(ValueError):
                parse_pdb(self.pdb, chain)
        with self.assertRaisesRegex(ValueError, '2 MiB'):
            parse_pdb(' ' * 2097153, 'A')

    def test_missing_ca_duplicate_atom_and_nan_rejected(self):
        atom = next(l for l in self.pdb.splitlines() if l.startswith('ATOM  '))
        no_ca = '\n'.join(l for l in self.pdb.splitlines() if not (l.startswith('ATOM  ') and l[12:16].strip() == 'CA'))
        ca = next(l for l in self.pdb.splitlines() if l.startswith('ATOM  ') and l[12:16].strip() == 'CA')
        with self.assertRaisesRegex(ValueError, 'C-alpha'):
            parse_pdb(no_ca, 'A')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            parse_pdb(atom+'\n'+self.pdb, 'A')
        with self.assertRaisesRegex(ValueError, 'coordinates'):
            parse_pdb(self.pdb.replace(ca, ca[:30]+'     nan'+ca[38:], 1), 'A')

    def test_first_model_is_used_and_nonstandard_not_silently_converted(self):
        atoms = '\n'.join(l for l in self.pdb.splitlines() if l.startswith('ATOM  '))+'\n'
        result = parse_pdb('MODEL        1\n'+atoms+'ENDMDL\nMODEL        2\n'+atoms+'ENDMDL\n', 'A')
        self.assertEqual(result.sequence, self.protein.sequence)
        first = next(l for l in self.pdb.splitlines() if l.startswith('ATOM  '))
        mse = 'HETATM'+first[6:17]+'MSE'+first[20:]
        with self.assertRaisesRegex(ValueError, 'MSE'):
            parse_pdb(mse+'\n', 'A')

    def test_insertion_code_and_author_number_preserved(self):
        atoms = [l for l in self.pdb.splitlines() if l.startswith('ATOM  ') and int(l[22:26]) == self.protein.residue_keys[0]['author_residue_number']]
        modified = '\n'.join(l[:21]+'B'+f'{-2:4d}'+'C'+l[27:] for l in atoms)+'\n'
        result = parse_pdb(modified, 'B')
        self.assertEqual(result.residue_keys[0]['author_residue_number'], -2)
        self.assertEqual(result.residue_keys[0]['insertion_code'], 'C')
        self.assertEqual(result.chain, 'B')

    def service_class(self, query, retrieve):
        ns = dict(Path=Path, re=re, shutil=shutil, tempfile=tempfile, time=time, uuid=uuid,
                  parse_pdb=parse_pdb, prepare_query=query, retrieve=retrieve, write_results=write_results)
        return extract(ROOT/'backend/service.py', 'Service', ns)

    def test_bad_task_chain_sequence_and_accession_never_reach_gpu(self):
        query, retrieve, gpu = Mock(), Mock(), Mock()
        Service = self.service_class(query, retrieve)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/'input.pdb'
            path.write_text(self.pdb)
            service = Service(SimpleNamespace(), root/'results')
            cases = [('A', '', 'bad', ''), ('Z', '', 'dna', ''),
                     ('A', 'AAAA', 'rna', ''), ('A', '', 'dna', 'invalid;id')]
            for chain, sequence, task, accession in cases:
                with self.subTest(task=task, chain=chain, accession=accession), self.assertRaises(ValueError):
                    service.predict(path, chain, sequence, task, accession, gpu)
            query.assert_not_called()
            retrieve.assert_not_called()
            gpu.assert_not_called()
            self.assertEqual(list(service.results_root.iterdir()), [])

    def test_failed_model_request_cleans_work_and_exports_nothing(self):
        work = []
        def query(pdb, candidate, folder, foldseek):
            work.append(folder.parent)
            folder.mkdir()
            (folder/'owned-test-file').write_text('test')
            return 'query-db', 'A'*self.n
        retrieve = Mock(return_value={'hits': 0})
        Service = self.service_class(query, retrieve)
        assets = SimpleNamespace(paths={'foldseek':Path('foldseek'), 'memory_db':Path('memory'),
                                        'checkpoint_root':Path('weights')}, labels_sha=lambda task:'sha')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            path=root/'input.pdb'; path.write_text(self.pdb)
            service=Service(assets,root/'results')
            gpu=Mock(side_effect=TimeoutError('simulated allocation failure'))
            with self.assertRaises(TimeoutError):
                service.predict(path,'A','','dna','6JHE',gpu)
            gpu.assert_called_once()
            self.assertEqual(list(service.results_root.iterdir()),[])
            self.assertTrue(all(not p.exists() for p in work))

    def test_cleanup_removes_only_expired_owned_uuid_directories(self):
        Service=self.service_class(Mock(),Mock())
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            service=Service(SimpleNamespace(),root/'results')
            expired=service.results_root/('a'*32); expired.mkdir()
            (expired/'test.txt').write_text('temporary test data')
            os.utime(expired,(time.time()-3700,)*2)
            recent=service.results_root/('b'*32); recent.mkdir()
            other=service.results_root/'keep-other'; other.mkdir()
            os.utime(other,(time.time()-3700,)*2)
            outside=root/'outside'; outside.mkdir()
            (outside/'keep.txt').write_text('temporary test data')
            link=service.results_root/('c'*32); link.symlink_to(outside,target_is_directory=True)
            service.cleanup()
            self.assertFalse(expired.exists())
            for p in (recent,other,outside/'keep.txt',link):
                self.assertTrue(p.exists())

    def test_ui_consent_gate_and_sanitized_failure(self):
        class UIError(Exception):
            pass
        service=Mock()
        ns=dict(enabled_now=lambda:True,SERVICE=service,gr=SimpleNamespace(Error=UIError),
                current_gpu_prediction=Mock())
        predict=extract(ROOT/'deploy/hf-full/app.py','predict',ns)
        with self.assertRaisesRegex(UIError,'permission|Confirm'):
            predict('input.pdb','A','','dna','',False)
        service.predict.assert_not_called()
        service.predict.side_effect=RuntimeError('private internal diagnostic must not be public')
        with self.assertRaises(UIError) as caught:
            predict('input.pdb','A','','dna','',True)
        self.assertNotIn('private internal',str(caught.exception))
        self.assertIn('No partial prediction',str(caught.exception))

    def test_worker_timeout_kills_only_owned_mock_process_and_cleans_temp(self):
        prepared=dict(protein=self.protein,three_di='A'*self.n,task='dna',memory=dict(
            memory_score=np.full(self.n,np.nan),template_count=np.zeros(self.n),best_tm=np.zeros(self.n)))
        assets=SimpleNamespace(config_path=Path('config.json'),lock_sha256='lock')
        process=Mock(pid=424242)
        process.wait.side_effect=[subprocess.TimeoutExpired('mock',110),0]
        seen=[]
        def popen(command,**kwargs):
            seen.append(Path(command[-1]))
            self.assertTrue((seen[-1]/'request.json').exists())
            self.assertTrue(kwargs['start_new_session'])
            return process
        with patch('backend.worker_transport.subprocess.Popen',side_effect=popen), patch('backend.worker_transport.os.killpg') as kill:
            with self.assertRaisesRegex(WorkerExecutionError,'timed out'):
                call_worker('python',assets,'predict',prepared,certificate={})
            kill.assert_called_once_with(424242,signal.SIGKILL)
        self.assertTrue(all(not p.exists() for p in seen))

    def test_queue_and_cache_policy_is_explicit_not_unlimited(self):
        source=(ROOT/'deploy/hf-full/app.py').read_text()
        self.assertIn('max_size=8',source)
        self.assertIn("concurrency_id='frozen_model'",source)
        self.assertIn('delete_cache=(300, 3600)',source)
        self.assertIn("show_error=False",source)
        self.assertIn('time.sleep(300)',source)


if __name__ == '__main__':
    unittest.main()
