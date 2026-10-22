import ast
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from backend.worker_environment import child_environment, MODEL_ENV_CONFIG, ensure_model_python
from backend.worker_transport import json_memory

ROOT = Path(__file__).resolve().parents[1]


class WorkerIsolation(unittest.TestCase):
    def test_ui_and_model_dependencies_are_separated(self):
        ui = (ROOT/'deploy/hf-full/requirements.txt').read_text().splitlines()
        model = json.loads(MODEL_ENV_CONFIG.read_text())
        self.assertIn('gradio==6.30.0', ui)
        self.assertFalse(any(row.startswith(('transformers', 'esm', 'huggingface-hub==0.')) for row in ui))
        self.assertIn('transformers==4.43.3', model['requirements'])
        self.assertIn('esm==3.1.1', model['requirements'])
        self.assertIn('huggingface-hub==0.36.2', model['requirements'])
        self.assertFalse(any(row.startswith('gradio') for row in model['requirements']))

    def test_child_cannot_receive_secrets_or_parent_pythonpath(self):
        env = dict(HF_TOKEN='private', HUGGING_FACE_HUB_TOKEN='private', RIMBIND_VALIDATION_KEY='private',
                   PYTHONPATH='parent-deps', PYTHONHOME='parent-home', PIP_INDEX_URL='private-index',
                   CUDA_VISIBLE_DEVICES='allocated-device', PATH='/usr/bin')
        with patch.dict(os.environ, env, clear=True):
            result = child_environment()
        for key in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'RIMBIND_VALIDATION_KEY', 'PYTHONPATH', 'PYTHONHOME', 'PIP_INDEX_URL'):
            self.assertNotIn(key, result)
        self.assertEqual(result['CUDA_VISIBLE_DEVICES'], 'allocated-device')
        self.assertEqual(result['HF_HUB_OFFLINE'], '1')

    def test_no_implicit_model_environment_install(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(ensure_model_python())

    def test_memory_transport_preserves_nan_as_json_null(self):
        memory = dict(memory_score=np.array([np.nan, .25], dtype=np.float32),
                      template_count=np.array([0, 2], dtype=np.int32),
                      best_tm=np.array([0., .5], dtype=np.float32))
        result = json_memory(memory)
        self.assertEqual(result['memory_score'], [None, .25])
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)

    def test_internal_transport_never_accepts_user_pickle_or_shell(self):
        text = (ROOT/'backend/worker_transport.py').read_text()
        self.assertIn('allow_pickle=False', text)
        self.assertIn('timeout=110', text)
        self.assertIn('start_new_session=True', text)
        self.assertNotIn('shell=True', text)
        self.assertNotIn('torch.load', text)
        worker = (ROOT/'backend/model_worker.py').read_text()
        self.assertIn('libc.prctl(1, signal.SIGKILL', worker)
        self.assertIn('os.getppid() != parent_pid', worker)

    def test_venv_never_inherits_ui_deps_or_skips_resolver(self):
        text = (ROOT/'backend/worker_environment.py').read_text()
        self.assertIn('system_site_packages=False', text)
        self.assertIn("'-m', 'pip', 'check'", text)
        tree = ast.parse(text)
        literals = [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)]
        self.assertNotIn('--no-deps', literals)

    def test_latest_certificate_crosses_zerogpu_fork_explicitly(self):
        text = (ROOT/'deploy/hf-full/app.py').read_text()
        self.assertIn('def allocated_prediction(protein, three_di, memory, task, certificate):', text)
        self.assertIn('allocated_prediction(protein, three_di, memory, task, CERTIFICATE)', text)
        self.assertIn('memory=memory, task=task), certificate)', text)


if __name__ == '__main__':
    unittest.main()
