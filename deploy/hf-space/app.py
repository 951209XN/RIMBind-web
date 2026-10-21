"""Safe external-host preflight. No fabricated probabilities or paid compute."""
from pathlib import Path
import gradio as gr
from backend.inputs import parse_pdb

def inspect_structure(path,chain,sequence):
    if not path:raise gr.Error('Choose a protein PDB file')
    p=Path(path)
    if p.stat().st_size>2097152:raise gr.Error('PDB exceeds 2 MiB')
    try:
        item=parse_pdb(p.read_text(),chain,sequence or None)
    except (ValueError,UnicodeError) as exc:raise gr.Error(str(exc))
    return dict(status='input_inspection_only',model_prediction_run=False,chain=item.chain,
        observed_residues=len(item.sequence),sequence=item.sequence,input_sha256=item.input_sha256,
        original_residue_mapping=item.residue_keys,
        release_gates=['target GPU numerical acceptance','frozen memory structures','full preprocessing chain','privacy and external browser acceptance'])

with gr.Blocks(title='RIMBind Deployment Preflight',delete_cache=(3600,86400)) as demo:
    gr.Markdown('# RIMBind deployment preflight\n**Not an online predictor.** This diagnostic app checks structure formatting only. It does not run a GPU model, generate probabilities, train a model or contact a laboratory server. Uploads are processed by Hugging Face; use a public example rather than confidential unpublished data during testing.')
    file=gr.File(label='Protein structure (PDB, max 2 MiB)',file_types=['.pdb'],type='filepath')
    chain=gr.Textbox(value='A',label='Single chain identifier')
    sequence=gr.Textbox(label='Optional matching observed sequence')
    button=gr.Button('Inspect input — not prediction')
    result=gr.JSON(label='Input contract report')
    button.click(inspect_structure,[file,chain,sequence],result,api_name=False)
    gr.Button('Full RIMBind prediction unavailable',interactive=False)
    gr.Markdown('Task rules to preserve: DNA FULL and RNA CR (NO_TEMPLATE_COUNT). No performance metrics are computed for unlabeled queries.')
demo.queue(max_size=8,default_concurrency_limit=1).launch(server_name='0.0.0.0',show_error=False,max_file_size='2mb')
