"""Trusted internal model subprocess entry; not an HTTP/upload endpoint."""
import ctypes
import json
import os
from pathlib import Path
import signal
import sys


def bind_to_parent(parent_pid):
    if sys.platform == 'linux':
        # Do not keep a CUDA child alive after ZeroGPU kills/preempts its parent.
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
            raise RuntimeError('Cannot install model subprocess parent-death guard')
    if os.getppid() != parent_pid:
        raise RuntimeError('Allocated parent ended before model startup')


def main():
    directory = Path(sys.argv[1]).resolve()
    request = json.loads((directory/'request.json').read_text())
    bind_to_parent(request['parent_pid'])
    result_path = directory/'result.json'
    try:
        import numpy as np
        from backend.assets import Assets
        from backend.inputs import ProteinInput
        from backend.target_gate import validate_certificate, runtime_fingerprint
        assets = Assets(request['asset_config'])
        if assets.lock_sha256 != request['asset_lock_sha256']:
            raise RuntimeError('Asset lock changed between interface and model')
        protein = ProteinInput(**request['protein'])
        memory = {key: np.asarray([np.nan if x is None else x for x in request['memory'][key]],
                                 dtype=np.int32 if key == 'template_count' else np.float32)
                  for key in ('memory_score', 'template_count', 'best_tm')}
        if request['operation'] == 'validate':
            from backend.target_validation import run_gpu_case
            prepared = dict(protein=protein, three_di=request['three_di'], memory=memory,
                            task=request['task'], candidate_id=request['candidate_id'],
                            historical=request['historical'])
            result = run_gpu_case(assets, prepared)
        elif request['operation'] == 'predict':
            from backend.service import predict_gpu
            validate_certificate(request['certificate'], assets.lock_sha256,
                                 current_runtime=runtime_fingerprint())
            result = predict_gpu(assets, protein, request['three_di'], memory, request['task'])
            np.savez(directory/'probabilities.npz', sscp=result.pop('sscp'), rimbind=result.pop('rimbind'))
            result['provenance']['execution_mode'] = 'isolated_frozen_model_process'
        else:
            raise RuntimeError('Unknown internal model operation')
        result_path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    except Exception as exc:
        result_path.write_text(json.dumps(dict(error_type=type(exc).__name__))+'\n')
        # Credentials are stripped before starting this process. Do not return
        # exception strings/private paths through the public Gradio callback.
        raise


if __name__ == '__main__':
    main()
