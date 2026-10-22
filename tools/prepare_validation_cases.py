"""Export only the two pre-existing, checksum-pinned label-free QA fixtures.

Administrator-only; never deserialize an anonymous website upload. The output
is PRIVATE and is not included in the public source repository or source ZIP.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from backend.predictor import file_sha256
    from backend.target_gate import PROTOCOL
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(PROTOCOL.read_text())
    for task, row in protocol['cases'].items():
        pid = row['id']
        path = args.fixtures/f'{pid}.pt'
        if file_sha256(path) != row['fixture_sha256']:
            raise RuntimeError('Not the original label-free fixture')
        item = torch.load(path, map_location='cpu', weights_only=False)
        if item.get('contains_labels') is not False or item['candidate_id'] != pid or item['domain'] != task:
            raise RuntimeError('Fixture boundary failure')
        data = dict(candidate_id=pid, task=task, sequence=item['sequence'],
                    raw_ca_angstrom=item['raw_ca_angstrom'].tolist(),
                    memory_score=[float(v) if float(v) == float(v) else None for v in item['memory_score']],
                    template_count=item['template_count'].tolist(), best_tm=item['best_tm'].tolist(),
                    fixture_sha256=row['fixture_sha256'], query_labels_included=False)
        memory_path = args.out/f'{pid}.json'
        memory_path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
        for suffix, key in (('pdb', 'pdb_sha256'), ('csv', 'historical_csv_sha256')):
            source = args.fixtures/f'{pid}.{suffix}'
            if file_sha256(source) != row[key]:
                raise RuntimeError('Original case input/reference changed')
            shutil.copyfile(source, args.out/source.name)
        print(json.dumps(dict(task=task, file=memory_path.name, sha256=file_sha256(memory_path))))


if __name__ == '__main__':
    main()
