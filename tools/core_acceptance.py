"""Local prepared-feature fixture acceptance, not an end-to-end web test."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from backend.fusion import fuse
from backend.model_loader import load_model_c,source_sha256

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    a=argparse.ArgumentParser();a.add_argument('--assets',type=Path,required=True);a.add_argument('--fixture',type=Path,required=True);a.add_argument('--output',type=Path,required=True);args=a.parse_args()
    manifest=json.loads((ROOT/'configs/assets_manifest.json').read_text())
    for row in manifest['sscp_checkpoints']+manifest['training_memory']:
        assert sha(args.assets/row['archive_path'])==row['sha256']
    fixture=torch.load(args.fixture,map_location='cpu',weights_only=False)
    assert fixture['contains_labels'] is False and fixture['domain']=='dna'
    torch.set_num_threads(2);torch.manual_seed(0);prob=[]
    with torch.inference_mode():
        for fold in range(5):
            model=load_model_c().eval()
            state=torch.load(args.assets/f'checkpoints/dna/fold{fold}/best.pt',map_location='cpu',weights_only=False)
            assert state['variant']=='C';model.load_state_dict(state['state_dict'],strict=True)
            z=model([fixture['saprot_embedding'].float()],[fixture['raw_ca_angstrom'].float()],[fixture['esm3_embedding'].float()])
            prob.append(torch.sigmoid(z).float())
    base=torch.stack(prob).mean(0).numpy().astype(np.float64)
    final=fuse(base,fixture['memory_score'].numpy(),fixture['template_count'].numpy(),fixture['best_tm'].numpy(),'dna')
    with (ROOT/'examples/6jhe_A/expected_output.csv').open() as f:reference=list(csv.DictReader(f))
    error=max(abs(float(p)-float(r['rimbind_probability'])) for p,r in zip(final,reference))
    assert len(reference)==len(final)==53
    passed=error<1e-5
    audit=dict(prepared_feature_core_test_passed=passed,end_to_end_pdb_test_passed=False,ZeroGPU_test_passed=False,
        torch_version=torch.__version__,device='cpu',residues=53,max_probability_abs_error=error,tolerance=1e-5,
        strict_checkpoint_state_loading=True,all_ten_checkpoints_and_two_memories_sha256_verified=True,
        source_sha256=source_sha256(),fixture_sha256=sha(args.fixture),source_edit='execution-AST removal of unused E/F Mamba import; model C architecture and source bytes unchanged',
        limitation='Does not test SaProt/ESM3 preprocessing, Foldseek search, new raw structures, GPU or web requests')
    args.output.write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))
    if not passed:sys.exit(2)
if __name__=='__main__':main()
