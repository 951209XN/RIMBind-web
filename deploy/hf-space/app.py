"""Safe external-host preflight. No fabricated probabilities or paid compute."""
from pathlib import Path
import spaces
import torch
import gradio as gr
from backend.inputs import parse_pdb

@spaces.GPU(duration=20)
def gpu_diagnostic():
    """Real CUDA allocation/arithmetic check; explicitly not model inference."""
    if not torch.cuda.is_available():
        raise gr.Error('No CUDA GPU was allocated. This is not a successful GPU test.')
    with torch.inference_mode():
        values=torch.arange(1,257,device='cuda',dtype=torch.float32)
        observed=float(torch.dot(values,values).item())
        expected=256*257*513//6
        torch.cuda.synchronize()
        if observed!=expected:
            raise gr.Error('CUDA arithmetic check failed. Model inference remains disabled.')
        device_name=torch.cuda.get_device_name(0)
    return dict(status='gpu_hardware_diagnostic_passed',device='cuda',device_name=device_name,
        torch_version=torch.__version__,sum_of_squares=observed,expected=expected,
        model_prediction_run=False,full_model_validated=False,
        note='GPU allocation and a small CUDA calculation only; not RIMBind numerical acceptance.')

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

with gr.Blocks(title='RIMBind Deployment Preflight',delete_cache=(3600,86400),analytics_enabled=False) as demo:
    gr.Markdown('# RIMBind deployment preflight\n**Not an online predictor.** This diagnostic app checks structure formatting only. It does not run a GPU model, generate probabilities, train a model or contact a laboratory server. Uploads are processed by Hugging Face; use a public example rather than confidential unpublished data during testing.')
    file=gr.File(label='Protein structure (PDB, max 2 MiB)',file_types=['.pdb'],type='filepath')
    chain=gr.Textbox(value='A',label='Single chain identifier')
    sequence=gr.Textbox(label='Optional matching observed sequence')
    button=gr.Button('Inspect input — not prediction')
    result=gr.JSON(label='Input contract report')
    button.click(inspect_structure,[file,chain,sequence],result,api_name=False)
    gr.Markdown('## ZeroGPU environment check\nOptional: the button below requests a short shared-GPU allocation and performs a small CUDA calculation. It uses platform GPU quota and may queue. It does not load model weights, generate residue probabilities or prove RIMBind correctness. Input inspection above stays on CPU.')
    gpu_button=gr.Button('Check GPU availability — uses shared GPU quota')
    gpu_result=gr.JSON(label='GPU hardware report — not model inference')
    gpu_button.click(gpu_diagnostic,inputs=[],outputs=gpu_result,api_name=False,concurrency_limit=1)
    gr.Button('Full RIMBind prediction unavailable',interactive=False)
    gr.Markdown('Task rules to preserve: DNA FULL and RNA CR (NO_TEMPLATE_COUNT). No performance metrics are computed for unlabeled queries.')
if __name__=='__main__':
    demo.queue(max_size=8,default_concurrency_limit=1).launch(server_name='0.0.0.0',show_error=False,max_file_size='2mb')
