"""Full frozen inference candidate; release gate stays closed until target QA."""
from pathlib import Path
import os
import threading
import time

import spaces  # Register ZeroGPU before torch is imported by backend modules.
import gradio as gr

from backend.assets import Assets
from backend.service import Service, predict_gpu

RESULTS = Path(os.environ.get('RIMBIND_RESULTS_DIR', '/tmp/rimbind_results')).resolve()
RESULTS.mkdir(parents=True, exist_ok=True)
CONFIG = Path(os.environ.get('RIMBIND_ASSET_CONFIG', 'private_assets/serving_assets.json'))
SERVICE, STATUS = None, 'Frozen offline assets have not been installed or verified.'
try:
    SERVICE = Service(Assets(CONFIG), RESULTS)
    STATUS = 'Frozen asset checksums verified. Target-Space model acceptance is still required.'
except (OSError, ValueError, KeyError, RuntimeError):
    # Do not leak administrator paths or checkpoint metadata into a public error.
    pass

# This flag is an operator-controlled test gate, not evidence of acceptance.
ENABLED = SERVICE is not None and os.environ.get('RIMBIND_ENABLE_PREDICTION') == '1'


@spaces.GPU(duration=120)
def allocated_prediction(protein, three_di, memory, task):
    if SERVICE is None:
        raise gr.Error('Frozen assets are unavailable.')
    return predict_gpu(SERVICE.assets, protein, three_di, memory, task)


def predict(path, chain, sequence, task, accession, consent):
    if not ENABLED or SERVICE is None:
        raise gr.Error('Live inference is not enabled for this deployment.')
    if not consent:
        raise gr.Error('Confirm the upload/privacy notice before submitting.')
    if not path:
        raise gr.Error('Choose a protein PDB file.')
    try:
        return SERVICE.predict(path, chain, sequence, task, accession, allocated_prediction)
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
                 api_name=False, concurrency_limit=1, concurrency_id='full_prediction')
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
