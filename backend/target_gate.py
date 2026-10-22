"""Fail-closed deployment gate; numerical QA is not public-service acceptance.

Only the administrator's local certificate is accepted, never a user upload.
Hardware checks are called exclusively inside an allocated GPU function.
"""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'configs/target_acceptance_protocol.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_digest():
    paths = sorted([* (ROOT/'backend').glob('*.py'), * (ROOT/'vendor').glob('*.py'),
                    * (ROOT/'configs').glob('*.json')])
    app = ROOT/'app_runtime.py'
    if not app.exists():
        app = ROOT/'deploy/hf-full/app.py'
    paths.append(app)
    rows = [('app_runtime.py' if p == app else str(p.relative_to(ROOT)), sha(p)) for p in paths]
    return hashlib.sha256(json.dumps(rows, separators=(',', ':')).encode()).hexdigest()


def runtime_fingerprint():
    import importlib.metadata
    import platform
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('A real allocated CUDA device is required')
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    return dict(python=platform.python_version(), torch=torch.__version__,
                transformers=importlib.metadata.version('transformers'),
                esm=importlib.metadata.version('esm'), numpy=importlib.metadata.version('numpy'),
                cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                gpu=properties.name, capability=list(torch.cuda.get_device_capability(device)),
                gpu_memory_bytes=properties.total_memory,
                matmul_precision=torch.get_float32_matmul_precision(),
                allow_tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                allow_tf32_cudnn=torch.backends.cudnn.allow_tf32,
                allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled())


def validate_certificate(certificate, asset_lock_sha256, *, current_runtime=None):
    protocol = json.loads(PROTOCOL.read_text())
    if (certificate.get('schema') != 'rimbind_target_numerical_certificate_v1' or
            protocol.get('owner_approved_same_hardware_comparator') is not True or
            certificate.get('protocol_sha256') != sha(PROTOCOL) or
            certificate.get('source_sha256') != source_digest() or
            certificate.get('asset_lock_sha256') != asset_lock_sha256):
        raise RuntimeError('Target certificate does not match this source, protocol or assets')
    cases = certificate.get('cases', {})
    if set(cases) != set(protocol['required_tasks']):
        raise RuntimeError('Both DNA and RNA acceptance results are required')
    runtimes = []
    for task, expected in protocol['cases'].items():
        row = cases[task]
        error = row.get('same_runtime_max_probability_abs_error')
        base_error = row.get('same_runtime_max_sscp_abs_error')
        if (row.get('candidate_id') != expected['id'] or row.get('memory_exact_match') is not True or
                row.get('same_runtime_equivalence_passed') is not True or
                row.get('same_runtime_tolerance') != protocol['tolerance'] or
                not isinstance(error, (float, int)) or not math.isfinite(error) or not 0 <= error < protocol['tolerance'] or
                not isinstance(base_error, (float, int)) or not math.isfinite(base_error) or not 0 <= base_error < protocol['tolerance'] or
                row.get('historical_tolerance') != protocol['historical_tolerance'] or
                not isinstance(row.get('historical_equivalence_passed'), bool) or
                not isinstance(row.get('historical_max_probability_abs_error'), (float, int)) or
                row.get('reference_sources_verified') is not True or not row.get('runtime')):
            raise RuntimeError('Incomplete or failed target numerical acceptance')
        historical_error = row['historical_max_probability_abs_error']
        if not math.isfinite(historical_error) or not historical_error >= 0 or row['historical_equivalence_passed'] != (historical_error < protocol['historical_tolerance']):
            raise RuntimeError('Historical drift must retain its actual pass/fail status')
        runtimes.append(row['runtime'])
    if runtimes[0] != runtimes[1]:
        raise RuntimeError('DNA and RNA acceptance ran on different target runtimes')
    if current_runtime is not None and runtimes[0] != current_runtime:
        raise RuntimeError('Assigned GPU/runtime changed; target self-test must be repeated')
    return True
