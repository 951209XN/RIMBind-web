"""Prepare a clean model venv without downgrading the public Gradio interface.

Installs from official PyPI only when the owner enables installation. Never
uses --no-deps, modifies the UI venv or passes HF secrets to pip/model children.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]
MODEL_ENV_CONFIG = ROOT/'configs/model_environment.json'


class ModelEnvironmentError(RuntimeError):
    pass


def child_environment():
    environment = {k: v for k, v in os.environ.items()
                   if k not in {'HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'RIMBIND_VALIDATION_KEY', 'PYTHONPATH', 'PYTHONHOME'}
                   and not k.startswith('PIP_')}
    environment['TOKENIZERS_PARALLELISM'] = 'false'
    environment['HF_HUB_OFFLINE'] = '1'
    environment['HF_HUB_DISABLE_TELEMETRY'] = '1'
    environment['TRANSFORMERS_OFFLINE'] = '1'
    return environment


def inspect_model_python(python):
    program = ('import importlib.metadata as m,json,sys; '
               'names=json.loads(sys.argv[1]); '
               'print(json.dumps({name:m.version(name) for name in names}))')
    expected = json.loads(MODEL_ENV_CONFIG.read_text())['required_versions']
    result = subprocess.run([str(python), '-c', program, json.dumps(list(expected))],
                            env=child_environment(), capture_output=True, text=True, timeout=30, check=True)
    actual = json.loads(result.stdout)
    if any(actual.get(name, '').split('+')[0] != version for name, version in expected.items()):
        raise ModelEnvironmentError('Frozen model environment version mismatch')
    subprocess.run([str(python), '-m', 'pip', 'check'], env=child_environment(),
                   check=True, capture_output=True, text=True, timeout=60)
    # Import the actual model factories, not merely distribution metadata. No
    # CUDA initialization or model weight loading is performed by this check.
    subprocess.run([str(python), '-c',
                    'from transformers import EsmModel,EsmTokenizer; from esm.models.esm3 import ESM3'],
                   env=child_environment(), check=True, capture_output=True, text=True, timeout=90)
    return actual


def ensure_model_python():
    config = json.loads(MODEL_ENV_CONFIG.read_text())
    configured = os.environ.get('RIMBIND_MODEL_PYTHON')
    if configured:
        python = Path(configured).resolve()
        inspect_model_python(python)
        return python
    if os.environ.get('RIMBIND_INSTALL_ASSETS') != '1':
        return None
    if os.environ.get('RIMBIND_THIRD_PARTY_TERMS_CONFIRMED') != '1':
        raise ModelEnvironmentError('Owner third-party acknowledgement is required')
    digest = hashlib.sha256(MODEL_ENV_CONFIG.read_bytes()).hexdigest()
    parent = Path('/tmp/rimbind_isolated_model_envs')
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory = parent/digest[:20]
    if directory.exists() and not (directory/'.rimbind_owned').exists():
        raise ModelEnvironmentError('Refuse to alter a non-installer environment')
    directory.mkdir(exist_ok=True, mode=0o700)
    (directory/'.rimbind_owned').touch(mode=0o600)
    python = directory/'bin/python'
    ready = directory/'READY.json'
    if not ready.exists():
        print('Preparing isolated frozen model environment; no model training or GPU allocation.', flush=True)
        venv.EnvBuilder(with_pip=True, system_site_packages=False).create(directory)
        requirements = directory/'requirements.txt'
        requirements.write_text('\n'.join(config['requirements'])+'\n')
        # Inherit no private Hub credentials, pip credentials or extra index.
        # Explicit PyPI wheels avoid arbitrary source-build commands.
        command = [str(python), '-m', 'pip', '--disable-pip-version-check', 'install',
                   '--index-url', 'https://pypi.org/simple', '--only-binary=:all:',
                   '--timeout', '60', '--retries', '2', '-r', str(requirements)]
        try:
            subprocess.run(command, env=child_environment(), check=True, timeout=1200)
            versions = inspect_model_python(python)
        except (subprocess.SubprocessError, OSError, ValueError) as exc:
            raise ModelEnvironmentError('Frozen model environment preparation failed: '+type(exc).__name__) from None
        ready.write_text(json.dumps(dict(config_sha256=digest, versions=versions), indent=2)+'\n')
    else:
        if json.loads(ready.read_text()).get('config_sha256') != digest:
            raise ModelEnvironmentError('Isolated model environment lock changed')
        inspect_model_python(python)
    return python
