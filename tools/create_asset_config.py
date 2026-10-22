"""Lock an administrator's existing offline assets; never uploads or downloads."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.predictor import file_sha256
from backend.assets import Assets


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--original-assets', type=Path, required=True)
    p.add_argument('--checkpoint-root', type=Path, required=True)
    p.add_argument('--memory-db', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise RuntimeError('Refuse to overwrite an existing asset lock')
    original = json.loads(a.original_assets.read_text())
    for row in original['saprot']['files']:
        if file_sha256(row['path']) != row['sha256']:
            raise RuntimeError('SaProt files differ from the original frozen manifest')
    if file_sha256(original['esm3']['weight_path']) != original['esm3']['weight_sha256']:
        raise RuntimeError('ESM3 weights differ from the original frozen manifest')
    paths = dict(checkpoint_root=a.checkpoint_root.resolve(), memory_db=a.memory_db.resolve(),
                 foldseek=Path(original['files']['foldseek_binary']['path']).resolve(),
                 saprot=Path(original['saprot']['files'][0]['path']).parent.resolve(),
                 esm3_snapshot=Path(original['esm3']['snapshot']).resolve())
    # Keep manifest paths rooted in the snapshot, not HF blob symlink locations.
    logical_files = [paths['foldseek']]
    for key in ('checkpoint_root', 'saprot', 'esm3_snapshot'):
        logical_files += [p for p in paths[key].rglob('*') if p.is_file() and not p.name.startswith('.')]
    logical_files += [p for p in paths['memory_db'].iterdir() if p.is_file()]
    parent = a.out.resolve().parent
    config = dict(schema='rimbind_serving_local_assets_v1',
                  paths={k: os.path.relpath(v, parent) for k, v in paths.items()},
                  files=[dict(path=os.path.relpath(p, parent), bytes=p.stat().st_size, sha256=file_sha256(p))
                         for p in sorted(set(logical_files))],
                  query_labels_included=False,
                  note='Administrator-only asset config; never serve this file as a public query result.')
    a.out.write_text(json.dumps(config, indent=2)+'\n')
    verified = Assets(a.out)
    print(json.dumps(dict(verified=True, files=len(config['files']), lock_sha256=verified.lock_sha256)))


if __name__ == '__main__':
    main()
