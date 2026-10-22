"""Run the approved same-hardware comparison; no training or query scoring."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from backend.assets import Assets
    from backend.target_validation import prepare_case, run_gpu_case, new_certificate
    from backend.target_gate import validate_certificate
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    assets = Assets(args.assets)
    certificate = new_certificate(assets.lock_sha256)
    for task in ('dna', 'rna'):
        with tempfile.TemporaryDirectory(prefix='target_case_') as directory:
            prepared = prepare_case(assets, args.cases, task, directory)
            report = run_gpu_case(assets, prepared)
        certificate['cases'][task] = report
        (args.out/f'{task}.json').write_text(json.dumps(report, indent=2)+'\n')
        (args.out/'certificate.json').write_text(json.dumps(certificate, indent=2)+'\n')
        print(json.dumps(report), flush=True)
    validate_certificate(certificate, assets.lock_sha256)
    print('Same-target numerical gate passed; hosted public-service acceptance is a separate step.', flush=True)


if __name__ == '__main__':
    main()
