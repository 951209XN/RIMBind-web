"""Internal JSON/NPZ transport to the frozen model interpreter.

Called only inside an allocated GPU callback. No user pickle, shell command,
credential forwarding or laboratory-server request. All outputs return on CPU.
"""
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile

import numpy as np

from backend.worker_environment import child_environment

ROOT = Path(__file__).resolve().parents[1]


class WorkerExecutionError(RuntimeError):
    pass


def json_memory(memory):
    result = {}
    for key in ('memory_score', 'template_count', 'best_tm'):
        result[key] = [None if not np.isfinite(x) else float(x) for x in np.asarray(memory[key])]
    return result


def call_worker(python, assets, operation, prepared, certificate=None):
    if operation not in {'predict', 'validate'}:
        raise ValueError('Unsupported internal worker operation')
    request = dict(operation=operation, asset_config=str(assets.config_path),
                   asset_lock_sha256=assets.lock_sha256, parent_pid=os.getpid(),
                   protein=asdict(prepared['protein']), three_di=prepared['three_di'],
                   memory=json_memory(prepared['memory']), task=prepared['task'])
    if operation == 'validate':
        request.update(candidate_id=prepared['candidate_id'], historical=prepared['historical'])
    else:
        request['certificate'] = certificate
    with tempfile.TemporaryDirectory(prefix='rimbind_model_call_') as directory:
        directory = Path(directory)
        path = directory/'request.json'
        path.write_text(json.dumps(request, allow_nan=False)+'\n')
        command = [str(python), '-m', 'backend.model_worker', str(directory)]
        with (directory/'worker.log').open('w') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=child_environment(),
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=110)
            except subprocess.TimeoutExpired:
                # Only our exact subprocess group, never the ZeroGPU scheduler
                # worker. Linux parent-death protection also covers preemption.
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
                raise WorkerExecutionError('Isolated model call timed out') from None
        result_path = directory/'result.json'
        if code or not result_path.is_file():
            error = {}
            if result_path.is_file():
                error = json.loads(result_path.read_text())
            raise WorkerExecutionError('Isolated model failed: '+str(error.get('error_type', 'ProcessError')))
        result = json.loads(result_path.read_text())
        if operation == 'validate':
            return result
        with np.load(directory/'probabilities.npz', allow_pickle=False) as data:
            arrays = {name: data[name].copy() for name in ('sscp', 'rimbind')}
        n = len(prepared['protein'].sequence)
        if any(x.shape != (n,) or not np.isfinite(x).all() or not ((x >= 0) & (x <= 1)).all()
               for x in arrays.values()):
            raise WorkerExecutionError('Invalid isolated model output')
        return dict(**arrays, provenance=result['provenance'])
