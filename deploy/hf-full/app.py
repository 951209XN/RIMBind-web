"""Full frozen inference candidate; release gate stays closed until target QA."""
from pathlib import Path
import json
import os
import secrets
import tempfile
import threading
import time
import uuid

import spaces  # Register ZeroGPU before torch is imported by backend modules.
import gradio as gr

from backend.assets import Assets
from backend.service import Service
from backend.target_gate import validate_certificate
from backend.target_validation import prepare_case, new_certificate
from backend.worker_environment import ensure_model_python
from backend.worker_transport import call_worker

RESULTS = Path(os.environ.get('RIMBIND_RESULTS_DIR', '/tmp/rimbind_results')).resolve()
RESULTS.mkdir(parents=True, exist_ok=True)
CONFIG = Path(os.environ.get('RIMBIND_ASSET_CONFIG', 'private_assets/serving_assets.json'))
SERVICE, STATUS = None, 'Frozen offline assets have not been installed or verified.'
MODEL_PYTHON = None
try:
    MODEL_PYTHON = ensure_model_python()
except Exception as exc:
    STATUS = 'Isolated frozen model environment is not ready ('+type(exc).__name__+'). No prediction is enabled.'
INSTALLED_CASES = None
if MODEL_PYTHON is not None and os.environ.get('RIMBIND_INSTALL_ASSETS') == '1':
    try:
        from backend.hosted_install import install_authorized_assets
        CONFIG, INSTALLED_CASES = install_authorized_assets()
    except Exception as exc:
        STATUS = 'Authorized asset installation did not complete ('+type(exc).__name__+'). Check owner setup; no prediction is enabled.'
try:
    SERVICE = Service(Assets(CONFIG), RESULTS)
    if MODEL_PYTHON is not None:
        STATUS = 'Frozen asset checksums and isolated model environment verified. Target-Space acceptance is still required.'
except (OSError, ValueError, KeyError, RuntimeError):
    # Do not leak administrator paths or checkpoint metadata into a public error.
    pass

CERTIFICATE_PATH = Path(os.environ.get('RIMBIND_CERTIFICATE', '/tmp/rimbind_target_certificate.json'))
CASES_ROOT = Path(os.environ.get('RIMBIND_VALIDATION_CASES', str(INSTALLED_CASES or 'private_assets/validation_cases')))
VALIDATION_KEY = os.environ.get('RIMBIND_VALIDATION_KEY', '')
CERTIFICATE = {}
if SERVICE is not None:
    CERTIFICATE = new_certificate(SERVICE.assets.lock_sha256)
    try:
        loaded = json.loads(CERTIFICATE_PATH.read_text())
        validate_certificate(loaded, SERVICE.assets.lock_sha256)
        CERTIFICATE = loaded
    except (OSError, ValueError, KeyError, TypeError, RuntimeError):
        pass


def enabled_now():
    if SERVICE is None or MODEL_PYTHON is None or os.environ.get('RIMBIND_ENABLE_PREDICTION') != '1':
        return False
    try:
        validate_certificate(CERTIFICATE, SERVICE.assets.lock_sha256)
        return True
    except (ValueError, KeyError, TypeError, RuntimeError):
        return False


# An environment flag alone can no longer open the predictor.
ENABLED = enabled_now()


@spaces.GPU(duration=120)
def allocated_prediction(protein, three_di, memory, task, certificate):
    if SERVICE is None:
        raise gr.Error('Frozen assets are unavailable.')
    if os.environ.get('RIMBIND_ENABLE_PREDICTION') != '1':
        raise gr.Error('Both target numerical acceptance cases must pass first.')
    validate_certificate(certificate, SERVICE.assets.lock_sha256)
    return call_worker(MODEL_PYTHON, SERVICE.assets, 'predict',
                       dict(protein=protein, three_di=three_di, memory=memory, task=task), certificate)


def current_gpu_prediction(protein, three_di, memory, task):
    # ZeroGPU can reuse a fork created before the latest owner validation. Pass
    # the main-process certificate explicitly, not via stale worker globals.
    return allocated_prediction(protein, three_di, memory, task, CERTIFICATE)


@spaces.GPU(duration=120)
def allocated_validation(prepared):
    if SERVICE is None:
        raise gr.Error('Frozen assets are unavailable.')
    return call_worker(MODEL_PYTHON, SERVICE.assets, 'validate', prepared)


def validate_target(key, task):
    if not VALIDATION_KEY or not secrets.compare_digest(str(key).encode(), VALIDATION_KEY.encode()):
        raise gr.Error('Owner validation key is required. Never send it in chat.')
    if SERVICE is None or MODEL_PYTHON is None or task not in {'dna', 'rna'}:
        raise gr.Error('Verified assets and a valid task are required.')
    try:
        with tempfile.TemporaryDirectory(prefix='target_case_') as directory:
            prepared = prepare_case(SERVICE.assets, CASES_ROOT, task, directory)
            report = allocated_validation(prepared)
    except Exception as exc:
        # Retain a failed attempt, invalidate the task gate, and avoid exposing
        # administrator paths or credentials from third-party exception strings.
        report = dict(task=task, same_runtime_equivalence_passed=False,
                      error_type=type(exc).__name__, release_authorized=False)
    CERTIFICATE['cases'][task] = report
    CERTIFICATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = CERTIFICATE_PATH.with_suffix('.tmp')
    temporary.write_text(json.dumps(CERTIFICATE, indent=2)+'\n')
    os.chmod(temporary, 0o600)
    os.replace(temporary, CERTIFICATE_PATH)
    directory = RESULTS/uuid.uuid4().hex
    directory.mkdir()
    output = directory/'target_numerical_certificate.json'
    output.write_text(json.dumps(CERTIFICATE, indent=2)+'\n')
    ready = enabled_now()
    return (CERTIFICATE, str(output), gr.Button(interactive=ready,
            value='Predict binding residues' if ready else 'Prediction awaiting deployment acceptance'))


def predict(path, chain, sequence, task, accession, consent):
    if not enabled_now() or SERVICE is None:
        raise gr.Error('Live inference is not enabled for this deployment.')
    if not consent:
        raise gr.Error('Confirm the upload/privacy notice before submitting.')
    if not path:
        raise gr.Error('Choose a protein PDB file.')
    try:
        return SERVICE.predict(path, chain, sequence, task, accession, current_gpu_prediction)
    except (ValueError, UnicodeError) as exc:
        raise gr.Error(str(exc)) from None
    except Exception:
        raise gr.Error('Prediction failed or the GPU request expired. No partial prediction is returned; retry later or use a smaller input.') from None


with gr.Blocks(title='RIMBind · frozen inference candidate', delete_cache=(300, 3600), analytics_enabled=False) as demo:
    gr.Markdown('# RIMBind\nDNA / RNA binding-residue prediction from a protein structure.\n'
                '**Deployment candidate — not yet a formally accepted public service.** '
                'This build contains the real model path, not a simulated predictor. '
                'The public prediction gate remains disabled until the owner completes target-Space acceptance.')
    gr.Markdown(STATUS)
    with gr.Row():
        with gr.Column():
            pdb = gr.File(label='Protein structure · PDB · ≤2 MiB', file_types=['.pdb'], type='filepath')
            chain = gr.Textbox(label='Single protein chain', value='A', max_length=1)
            task = gr.Radio(choices=[('DNA', 'dna'), ('RNA', 'rna')], value='dna', label='Prediction task')
            sequence = gr.Textbox(label='Optional matching observed sequence / FASTA')
            accession = gr.Textbox(label='Optional original PDB accession',
                                   info='Only if known. Enables exclusion of templates from the same PDB accession. Filenames are not used as identity.')
            consent = gr.Checkbox(label='I have permission to upload this structure and have read the privacy notice.')
            button = gr.Button('Predict binding residues' if ENABLED else 'Prediction awaiting deployment acceptance', interactive=ENABLED, variant='primary')
        with gr.Column():
            table = gr.Dataframe(headers=['Position', 'AA', 'Chain', 'Author residue', 'Insertion code',
                                         'SSCP probability', 'RIMBind probability', 'Binding at 0.5'],
                                 interactive=False, label='Per-residue probabilities — not accuracy scores')
            files = gr.File(label='Download CSV, probability-colored PDB and provenance JSON', file_count='multiple')
    button.click(predict, [pdb, chain, sequence, task, accession, consent], [table, files],
                 api_name=False, concurrency_limit=1, concurrency_id='frozen_model')
    with gr.Accordion('Owner-only deployment validation — not a model accuracy test', open=False,
                      visible=bool(VALIDATION_KEY)):
        gr.Markdown('Run both fixed cases on this assigned GPU. The unchanged historical feature/C0 scripts '
                    'are rerun in a fresh temporary workspace. Same-hardware tolerance remains 0.00001; '
                    'historical cross-hardware drift is retained separately. This uses shared GPU quota. '
                    'A numerical pass is not certification of public availability or unlimited free inference.')
        validation_key = gr.Textbox(label='Owner validation key (set as a Space Secret; never send in chat)', type='password')
        validation_task = gr.Radio(['dna', 'rna'], value='dna', label='Fixed acceptance case')
        validation_button = gr.Button('Run fixed target comparison', interactive=SERVICE is not None and MODEL_PYTHON is not None)
        certificate_output = gr.JSON(label='Numerical certificate and historical drift')
        certificate_download = gr.File(label='Download numerical acceptance record')
        validation_button.click(validate_target, [validation_key, validation_task],
                                [certificate_output, certificate_download, button], api_name=False,
                                concurrency_limit=1, concurrency_id='frozen_model')
    gr.Markdown('### Interpretation\nDNA uses frozen FULL fusion; RNA uses frozen CR (`NO_TEMPLATE_COUNT`). '
                'Five frozen fold probabilities are averaged. No retraining or threshold fitting occurs. '
                'The 0.5 decision threshold is an explicit display rule, not a calibrated confidence guarantee. '
                'AP, AUROC, F1 and MCC cannot be reported for an unlabeled query. '
                'CSV retains residue numbering; the PDB B-field contains probability 0–1, rounded to two decimals.\n\n'
                '### Limits and privacy\nUp to 600 observed standard amino acids in one chain; every residue needs Cα coordinates. '
                'Missing residues, nonstandard residues and mismatching sequences are not silently remapped. '
                'Uploads are processed on Hugging Face infrastructure, not on a laboratory server. '
                'Do not submit confidential, personal or unpublished sensitive data. '
                'Working files are removed after computation; application results and Gradio caches are eligible for removal '
                'after one hour (checked every five minutes while the app runs). Download files promptly. '
                'This does not control hosting-platform logs or backups. Free GPU queues, quota limits and interruptions apply.')


def cleanup_loop():
    while True:
        time.sleep(300)
        if SERVICE is not None:
            SERVICE.cleanup()


def launch():
    threading.Thread(target=cleanup_loop, daemon=True).start()
    demo.queue(max_size=8, default_concurrency_limit=1).launch(
        server_name='0.0.0.0', show_error=False, max_file_size='2mb', allowed_paths=[str(RESULTS.resolve())])


if __name__ == '__main__':
    launch()
