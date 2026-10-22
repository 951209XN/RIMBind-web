"""Prepare an administrator's PRIVATE core bundle; no upload, no encoders.

This does not grant public redistribution rights to weights or training memory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def inventory(checkpoints, database, database_lock):
    manifest = json.loads((ROOT/'configs/assets_manifest.json').read_text())
    lock = json.loads(Path(database_lock).read_text())
    directory = Path(lock['paths']['memory_db'])
    database_hashes = {Path(row['path']).name: row['sha256'] for row in lock['files']
                       if Path(row['path']).parent == directory}
    if not database_hashes:
        raise RuntimeError('Authoritative database lock is empty')
    files = []
    for row in manifest['sscp_checkpoints'] + manifest['training_memory']:
        path = Path(checkpoints)/row['archive_path']
        if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise RuntimeError('Core checkpoint/memory differs from frozen manifest')
        files.append((path, row['archive_path'], row['sha256']))
    for path in sorted(Path(database).iterdir()):
        if not re.fullmatch(r'(dna|rna)(?:_(?:ca|h|ss))?(?:\.(?:dbtype|index|lookup|source))?', path.name):
            raise RuntimeError('Unexpected database file; refuse accidental data inclusion')
        if path.is_symlink() or not path.is_file():
            raise RuntimeError('Only regular frozen database files are supported')
        digest = sha(path)
        if database_hashes.get(path.name) != digest:
            raise RuntimeError('Database file differs from the authoritative asset lock')
        files.append((path, 'memory_db/'+path.name, digest))
    for task in ('dna', 'rna'):
        for suffix in ('', '.dbtype', '.index', '_h', '_h.dbtype', '_h.index',
                       '_ss', '_ss.dbtype', '_ss.index', '_ca', '_ca.dbtype', '_ca.index'):
            if not any(name == 'memory_db/'+task+suffix for _, name, _ in files):
                raise RuntimeError('Incomplete frozen memory database')
    return files


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoints', type=Path, required=True)
    p.add_argument('--database', type=Path, required=True)
    p.add_argument('--database-lock', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise RuntimeError('Refuse to overwrite an existing private bundle')
    files = inventory(a.checkpoints, a.database, a.database_lock)
    note = dict(schema='rimbind_private_core_bundle_v1', visibility='PRIVATE',
                permitted_use='Owner-authorized non-commercial scientific hosting; not public weight redistribution',
                query_test_labels_included=False, training_memory_labels_included=True,
                foundation_encoders_included=False, foldseek_binary_included=False,
                public_release_ready=False,
                database_lock_sha256=sha(a.database_lock),
                files=[dict(path=name, bytes=path.stat().st_size, sha256=digest)
                       for path, name, digest in files])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(a.output, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path, name, _ in files:
            archive.write(path, 'rimbind_core/'+name)
        archive.writestr('rimbind_core/PRIVATE_BUNDLE_MANIFEST.json', json.dumps(note, indent=2)+'\n')
    with zipfile.ZipFile(a.output) as archive:
        for _, name, digest in files:
            if hashlib.sha256(archive.read('rimbind_core/'+name)).hexdigest() != digest:
                raise RuntimeError('Archive verification failed')
    metadata = dict(files=len(files), zip_bytes=a.output.stat().st_size,
                    zip_sha256=sha(a.output), public_upload_authorized=False,
                    note='Private asset preparation only; does not install or release a WebServer')
    a.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
