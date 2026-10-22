"""Two fixed deployment checks, not model scoring or case selection."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile

import numpy as np

from backend.inputs import parse_pdb
from backend.memory import prepare_query, retrieve
from backend.numerical_audit import compare_probabilities
from backend.predictor import file_sha256
from backend.target_gate import PROTOCOL, sha, source_digest, runtime_fingerprint


def prepare_case(assets, cases_root, task, work):
    protocol = json.loads(PROTOCOL.read_text())
    if task not in protocol['cases']:
        raise ValueError('Select DNA or RNA')
    case = protocol['cases'][task]
    pid = case['id']
    root, work = Path(cases_root), Path(work)
    for suffix, key in (('pdb', 'pdb_sha256'), ('csv', 'historical_csv_sha256'), ('json', 'memory_json_sha256')):
        if file_sha256(root/f'{pid}.{suffix}') != case[key]:
            raise RuntimeError('Fixed acceptance input/reference checksum mismatch')
    expected = json.loads((root/f'{pid}.json').read_text())
    if expected.get('query_labels_included') is not False or expected['fixture_sha256'] != case['fixture_sha256']:
        raise RuntimeError('Unexpected fixed-case metadata')
    protein = parse_pdb((root/f'{pid}.pdb').read_text(), 'A', expected['sequence'])
    if len(protein.sequence) != case['residues'] or not np.array_equal(
            np.asarray(protein.ca, np.float32), np.asarray(expected['raw_ca_angstrom'], np.float32)):
        raise RuntimeError('Fixed-case sequence/coordinate mapping changed')
    db, three_di = prepare_query(protein.normalized_pdb, pid, work/'foldseek', assets.paths['foldseek'])
    memory = retrieve(db, pid, len(protein.sequence), task, assets.paths['foldseek'],
                      assets.paths['memory_db']/task,
                      assets.paths['checkpoint_root']/f'training_memory/{task}.pt', assets.labels_sha(task))
    if len(three_di) != len(protein.sequence):
        raise RuntimeError('Fixed-case 3Di mapping changed')
    for key in ('memory_score', 'template_count', 'best_tm'):
        values = np.asarray([np.nan if x is None else x for x in expected[key]], dtype=memory[key].dtype)
        if not np.array_equal(memory[key], values, equal_nan=True):
            raise RuntimeError('Frozen retrieval/projection differs from the original fixed-case memory')
    with (root/f'{pid}.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    if (len(rows) != len(protein.sequence) or ''.join(r['amino_acid'] for r in rows) != protein.sequence or
            [int(r['residue_index']) for r in rows] != list(range(1, len(rows)+1))):
        raise RuntimeError('Historical reference residue order changed')
    return dict(task=task, candidate_id=pid, protein=protein, three_di=three_di, memory=memory,
                historical=[float(r['rimbind_probability']) for r in rows])


def run_gpu_case(assets, prepared):
    # Import CUDA components only inside the caller's allocated GPU boundary.
    from backend.service import predict_gpu
    from backend.reference_runtime import original_reference
    fingerprint = runtime_fingerprint()
    protein, memory, task = prepared['protein'], prepared['memory'], prepared['task']
    candidate = predict_gpu(assets, protein, prepared['three_di'], memory, task)
    candidate_runtime = runtime_fingerprint()
    with tempfile.TemporaryDirectory(prefix='frozen_reference_') as directory:
        reference = original_reference(assets, protein, prepared['three_di'], memory, task,
                                       prepared['candidate_id'], directory)
    reference_runtime = runtime_fingerprint()
    if fingerprint != candidate_runtime or fingerprint != reference_runtime:
        raise RuntimeError('Runtime changed during the fixed-case comparison')
    comparison = compare_probabilities(candidate['rimbind'], reference['rimbind'], prepared['historical'],
                                      candidate_runtime=candidate_runtime, reference_runtime=reference_runtime)
    comparison.pop('policy_change_approved')  # The signed-off protocol, not this helper, records approval.
    base_error = float(np.abs(candidate['sscp'].astype(np.float64)-reference['sscp']).max())
    comparison['same_runtime_equivalence_passed'] &= base_error < 1e-5
    comparison.update(candidate_id=prepared['candidate_id'], task=task, residues=len(protein.sequence),
                      same_runtime_max_sscp_abs_error=base_error, memory_exact_match=True,
                      reference_sources_verified=True, runtime=fingerprint,
                      owner_approved_same_hardware_comparator=True,
                      query_labels_read=False, performance_metrics_computed=False,
                      observed_utc=datetime.now(timezone.utc).isoformat())
    # Approval belongs to the protocol, not the diagnostic helper's verdict.
    # release_authorized remains False until operational acceptance is complete.
    return comparison


def new_certificate(asset_lock_sha256):
    return dict(schema='rimbind_target_numerical_certificate_v1',
                protocol_sha256=sha(PROTOCOL), source_sha256=source_digest(),
                asset_lock_sha256=asset_lock_sha256, cases={},
                public_service_acceptance_completed=False,
                note='Numerical gate only. Does not certify uptime, quota, privacy or complete public-service operation.')
