"""Build a flat source-only candidate; excludes all weights and study data."""
import argparse
import hashlib
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = '''"""Verified local source archive bootstrap; never extracts query archives."""
from pathlib import Path, PurePosixPath
import hashlib
import sys
import tempfile
import zipfile

source = Path(__file__).with_name('runtime.zip')
if hashlib.sha256(source.read_bytes()).hexdigest() != '__RUNTIME_SHA256__':
    raise RuntimeError('Source archive checksum mismatch')
with zipfile.ZipFile(source) as archive:
    if sum(row.file_size for row in archive.infolist()) > 5_000_000:
        raise RuntimeError('Unexpected source archive size')
    for row in archive.infolist():
        path = PurePosixPath(row.filename)
        if path.is_absolute() or '..' in path.parts or (row.external_attr >> 16) & 0o170000 == 0o120000:
            raise RuntimeError('Unsafe local source archive')
    target = tempfile.mkdtemp(prefix='rimbind_source_')
    archive.extractall(target)
sys.path.insert(0, target)
from app_runtime import demo, launch
if __name__ == '__main__':
    launch()
'''


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    archive = a.output/'runtime.zip'
    sources = list((ROOT/'backend').glob('*.py'))
    sources += [ROOT/'vendor'/name for name in ('frozen_model_single.py', 'frozen_embeddings.py', 'frozen_memory_pipeline.py')]
    sources += [ROOT/'configs/assets_manifest.json']
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(sources):
            z.write(path, str(path.relative_to(ROOT)))
        z.writestr('vendor/__init__.py', '')
        z.write(ROOT/'deploy/hf-full/app.py', 'app_runtime.py')
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (a.output/'app.py').write_text(BOOTSTRAP.replace('__RUNTIME_SHA256__', digest))
    for name in ('README.md', 'requirements.txt'):
        shutil.copyfile(ROOT/'deploy/hf-full'/name, a.output/name)
    shutil.copyfile(ROOT/'LICENSE', a.output/'LICENSE')
    print(f'Source-only candidate created: {a.output}; runtime SHA256 {digest}')


if __name__ == '__main__':
    main()
