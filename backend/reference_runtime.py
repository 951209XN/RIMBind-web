"""Execute unchanged historical feature/C0 CLIs on a fresh one-case workspace.

Not a second copy of the web forward/fusion implementation. Compatibility is
explicit: local ESM3 snapshot pinning and removal of an unused E/F import from
the execution AST. No historical probabilities are used to compute a result.
"""
import contextlib
import gc
import io
import json
from pathlib import Path
import sys
import types

import numpy as np
import torch

from backend.encoders import pin_esm3_snapshot
from backend.model_loader import load_model_class, SOURCE
from backend.predictor import file_sha256
from backend.target_gate import PROTOCOL, ROOT
from vendor import frozen_embeddings, frozen_c0_prediction, frozen_memory_pipeline


def verify_reference_sources():
    protocol = json.loads(PROTOCOL.read_text())
    for path, digest in protocol['reference_sources'].items():
        if file_sha256(ROOT/path) != digest:
            raise RuntimeError('Frozen original reference source changed')


def invoke_original(function, argv):
    previous = sys.argv
    try:
        sys.argv = ['frozen_reference', *argv]
        # Historical scripts print trusted local paths. Do not put those in
        # public logs or query output; the numerical certificate stores hashes.
        with contextlib.redirect_stdout(io.StringIO()):
            function()
    finally:
        sys.argv = previous
        gc.collect()
        torch.cuda.empty_cache()


def original_reference(assets, protein, three_di, memory, task, pid, work):
    verify_reference_sources()
    work = Path(work)
    for folder in ('data/manifests', 'data/model_inputs', 'audit', 'configs', f'data/embeddings/3di/{task}'):
        (work/folder).mkdir(parents=True, exist_ok=True)
    pdb = work/'query.pdb'
    pdb.write_text(protein.normalized_pdb)
    item = dict(observed_sequence=protein.sequence, observed_residues=len(protein.sequence),
                raw_ca_angstrom=torch.tensor(protein.ca, dtype=torch.float32), source_pdb=str(pdb),
                residue_keys=[('A', r['residue_name'], i, '') for i, r in enumerate(protein.residue_keys, 1)],
                contains_labels=False, nucleic_acid_coordinates_read=False)
    input_path = work/f'data/model_inputs/{pid}.pt'
    torch.save(item, input_path)
    row = dict(candidate_id=pid, domain=task, technical_supported=True,
               input_path=str(input_path), input_sha256=file_sha256(input_path),
               observed_residues=len(protein.sequence))
    (work/'data/manifests/LABEL_FREE_MODEL_INPUTS.jsonl').write_text(json.dumps(row)+'\n')
    (work/'configs/SMOKE_IDS.json').write_text(json.dumps({task: [pid]}))
    (work/f'data/embeddings/3di/{task}/{pid}.3di').write_text(three_di+'\n')
    frozen = assets.manifest
    checkpoints = [dict(domain=r['domain'], fold=r['fold'], sha256=r['sha256'],
                        path=str(assets.paths['checkpoint_root']/r['archive_path']))
                   for r in frozen['sscp_checkpoints']]
    setting = dict(files=dict(c0_model_code=dict(path=str(SOURCE), sha256=file_sha256(SOURCE))),
                   c0_checkpoints=checkpoints)
    (work/'audit/MODEL_ASSETS_FROZEN.json').write_text(json.dumps(setting))
    device = 'cuda'
    for kind in ('saprot', 'esm3'):
        args = ['--root', str(work), '--kind', kind, '--device', device, '--scope', 'smoke', '--domain', task]
        if kind == 'saprot':
            args += ['--saprot-model', str(assets.paths['saprot'])]
        else:
            pin_esm3_snapshot(assets.paths['esm3_snapshot'])
        invoke_original(frozen_embeddings.main, args)
    # The historical C0 CLI imports model_single. Supply the same frozen class,
    # preserving its constructor/forward and original per-fold loading order.
    previous = sys.modules.get('model_single')
    previous_paths = list(sys.path)
    module = types.ModuleType('model_single')
    module.SingleNABindModel = load_model_class()
    try:
        sys.modules['model_single'] = module
        invoke_original(frozen_c0_prediction.main,
                        ['--root', str(work), '--domain', task, '--device', device, '--scope', 'smoke'])
    finally:
        sys.path[:] = previous_paths
        if previous is None:
            sys.modules.pop('model_single', None)
        else:
            sys.modules['model_single'] = previous
    payload = torch.load(work/f'predictions/c0/smoke/{task}.pt', map_location='cpu', weights_only=False)
    if payload['rows'][0]['candidate_id'] != pid or len(payload['rows']) != 1:
        raise RuntimeError('Original C0 reference returned unexpected cases')
    base = payload['rows'][0]['probability'].numpy()
    mode, alpha, tau = ('FULL', .2, 1.) if task == 'dna' else ('NO_TEMPLATE_COUNT', .8, 4.)
    memory_score = np.nan_to_num(np.asarray(memory['memory_score'], dtype=np.float64), nan=0.)
    # Preserve the historical main() call site's float64 conversions as well
    # as the unchanged evidence()/sigmoid() functions.
    evidence = frozen_memory_pipeline.evidence(memory_score, np.asarray(memory['template_count'], dtype=np.float64),
                np.asarray(memory['best_tm'], dtype=np.float64), .1, tau, 2., mode)
    clipped = np.clip(base.astype(np.float64), 1e-6, 1-1e-6)
    final = frozen_memory_pipeline.sigmoid(np.log(clipped/(1-clipped)) + alpha*evidence).numpy()
    return dict(sscp=base, rimbind=final)
