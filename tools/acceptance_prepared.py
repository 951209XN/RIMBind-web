"""Compare a trusted label-free fixture to an existing probability reference.

Never computes AP/AUROC or adjusts the fixed 1e-5 numerical tolerance.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.predictor import FrozenPredictor, file_sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    item = torch.load(args.fixture, map_location='cpu', weights_only=False)
    if item.get('contains_labels') is not False:
        raise RuntimeError('Fixture must be explicitly label-free')
    torch.set_num_threads(2)
    torch.manual_seed(0)
    predictor = FrozenPredictor(args.assets, item['domain'], args.device)
    result = predictor.predict(item['saprot_embedding'], item['raw_ca_angstrom'],
                               item['esm3_embedding'], item['memory_score'].numpy(),
                               item['template_count'].numpy(), item['best_tm'].numpy())
    with args.reference.open() as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(item['sequence']) or any(row['amino_acid'] != aa for row, aa in zip(rows, item['sequence'])):
        raise RuntimeError('Reference sequence/order mismatch')
    expected = np.array([float(r['rimbind_probability']) for r in rows])
    errors = np.abs(result['rimbind'].astype(np.float64) - expected)
    report = dict(prepared_feature_core_test_passed=bool(errors.max() < 1e-5),
                  raw_pdb_end_to_end_passed=False, hosted_space_model_passed=False,
                  tolerance=1e-5, max_probability_abs_error=float(errors.max()),
                  residues=len(rows), candidate_id=item['candidate_id'],
                  fixture_sha256=file_sha256(args.fixture), reference_sha256=file_sha256(args.reference),
                  provenance=result['provenance'],
                  predictions=result['rimbind'].tolist(), sscp=result['sscp'].tolist())
    if torch.device(args.device).type == 'cuda':
        report['gpu_name'] = torch.cuda.get_device_name(torch.device(args.device))
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in {'predictions', 'sscp'}}, indent=2))
    return 0 if report['prepared_feature_core_test_passed'] else 2


if __name__ == '__main__':
    sys.exit(main())
