"""Raw-PDB scientific acceptance on administrator-owned offline assets.

Uses existing label-free case probabilities; never scores binding labels.
This is a server-side validation utility, not a public upload endpoint.
"""
import argparse
import csv
import gc
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.inputs import parse_pdb
from backend.predictor import FrozenPredictor, file_sha256
from backend.encoders import load_saprot, load_esm3, saprot_embedding, esm3_embedding
from backend.memory import prepare_query, retrieve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original-assets', type=Path, required=True)
    parser.add_argument('--checkpoint-root', type=Path, required=True)
    parser.add_argument('--memory-db', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--pdb', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--out', type=Path, required=True)
    a = parser.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    audit = dict(raw_pdb_end_to_end_passed=False, hosted_space_model_passed=False,
                 query_labels_read=False, training_performed=False, tolerance=1e-5, steps={})
    def save():
        audit['elapsed_seconds'] = time.monotonic() - start
        (a.out/'REPORT.json').write_text(json.dumps(audit, indent=2)+'\n')
    def step(name, **info):
        audit['steps'][name] = info
        save()
        print(json.dumps(dict(step=name, **info)), flush=True)
    try:
        settings = json.loads(a.original_assets.read_text())
        for row in settings['saprot']['files']:
            if file_sha256(row['path']) != row['sha256']:
                raise RuntimeError('SaProt asset changed')
        esm = settings['esm3']
        if file_sha256(esm['weight_path']) != esm['weight_sha256']:
            raise RuntimeError('ESM3 asset changed')
        item = torch.load(a.fixture, map_location='cpu', weights_only=False)
        if item.get('contains_labels') is not False:
            raise RuntimeError('Fixture is not label-free')
        task, pid, n = item['domain'], item['candidate_id'], len(item['sequence'])
        audit.update(candidate_id=pid, task=task, residues=n,
                     torch_version=torch.__version__, device=a.device,
                     transformers_version=importlib.metadata.version('transformers'),
                     esm_version=importlib.metadata.version('esm'), cuda_version=torch.version.cuda,
                     gpu_name=torch.cuda.get_device_name(torch.device(a.device)),
                     fixture_sha256=file_sha256(a.fixture), pdb_sha256=file_sha256(a.pdb),
                     reference_sha256=file_sha256(a.reference))
        parsed = parse_pdb(a.pdb.read_text(), 'A', item['sequence'])
        ca = np.asarray(parsed.ca, np.float32)
        if not np.array_equal(ca, item['raw_ca_angstrom'].numpy()):
            raise RuntimeError('Input parser changed raw C-alpha coordinates')
        step('input', sequence_and_ca_match=True)
        foldseek = settings['files']['foldseek_binary']['path']
        db, three_di = prepare_query(parsed.normalized_pdb, pid, a.out/'foldseek', foldseek)
        labels = settings['files'][f'training_memory_{task}']
        memory = retrieve(db, pid, n, task, foldseek, a.memory_db/task, labels['path'], labels['sha256'])
        memory_match = all(np.array_equal(memory[k], item[k].numpy(), equal_nan=True)
                           for k in ('memory_score', 'template_count', 'best_tm'))
        step('memory', exact_match=memory_match, hits=memory['hits'],
             alignment_sha256=memory['alignment_sha256'])
        device = torch.device(a.device)
        tokenizer, model = load_saprot(Path(settings['saprot']['files'][0]['path']).parent, device)
        saprot = saprot_embedding(tokenizer, model, parsed.sequence, three_di, device)
        saprot_error = float((saprot.float() - item['saprot_embedding'].float()).abs().max())
        step('saprot', max_embedding_abs_error=saprot_error,
             exact_half_tensor_match=torch.equal(saprot, item['saprot_embedding']))
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        model = load_esm3(esm['snapshot'], device)
        normalized = a.out/'normalized.pdb'
        normalized.write_text(parsed.normalized_pdb)
        esm3, provenance = esm3_embedding(model, normalized, parsed.sequence, device)
        esm_error = float((esm3.float() - item['esm3_embedding'].float()).abs().max())
        step('esm3', max_embedding_abs_error=esm_error,
             exact_half_tensor_match=torch.equal(esm3, item['esm3_embedding']), **provenance)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        predictor = FrozenPredictor(a.checkpoint_root, task, device)
        result = predictor.predict(saprot, ca, esm3, memory['memory_score'], memory['template_count'], memory['best_tm'])
        with a.reference.open() as f:
            rows = list(csv.DictReader(f))
        if len(rows) != n or ''.join(r['amino_acid'] for r in rows) != parsed.sequence:
            raise RuntimeError('Reference sequence/order mismatch')
        error = float(np.abs(result['rimbind'].astype(np.float64) -
                             np.array([float(r['rimbind_probability']) for r in rows])).max())
        audit.update(max_probability_abs_error=error,
                     raw_pdb_end_to_end_passed=bool(error < 1e-5 and memory_match),
                     provenance=result['provenance'])
        with (a.out/'probabilities.csv').open('x', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['residue_index', 'amino_acid', 'sscp_probability', 'rimbind_probability'])
            for i, (aa, base, final) in enumerate(zip(parsed.sequence, result['sscp'], result['rimbind']), 1):
                writer.writerow([i, aa, format(float(base), '.9g'), format(float(final), '.9g')])
        step('prediction', max_probability_abs_error=error, fixed_tolerance=1e-5)
        return 0 if audit['raw_pdb_end_to_end_passed'] else 2
    except Exception as exc:
        audit['exception'] = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        save()


if __name__ == '__main__':
    sys.exit(main())
