"""Owner-opted-in serving bootstrap: read private assets and pinned upstream files.

No upload or account mutation. The operator supplies HF_TOKEN as a Space Secret;
tokens never enter manifests, query results, code or public diagnostic messages.
This module is not called by default and is never used to automate account setup.
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
INSTALL_MANIFEST = ROOT/'configs/hosted_assets_manifest.json'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def extract_locked_zip(source, expected_sha, target, expected_files, max_bytes):
    if sha(source) != expected_sha:
        raise RuntimeError('Private archive checksum mismatch')
    with zipfile.ZipFile(source) as archive:
        rows = archive.infolist()
        if (len({r.filename for r in rows}) != len(rows) or
                {r.filename for r in rows} != set(expected_files) or
                sum(r.file_size for r in rows) > max_bytes):
            raise RuntimeError('Unexpected private archive inventory or size')
        for row in rows:
            path = PurePosixPath(row.filename)
            if (path.is_absolute() or '..' in path.parts or '\\' in row.filename or row.is_dir() or
                    (row.external_attr >> 16) & 0o170000 == 0o120000):
                raise RuntimeError('Unsafe private archive member')
            with archive.open(row) as handle:
                digest = hashlib.file_digest(handle, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(handle.read()).hexdigest()
            if digest != expected_files[row.filename]:
                raise RuntimeError('Private archive file checksum mismatch')
        for row in rows:
            destination = Path(target)/row.filename
            for parent in (destination, *destination.parents):
                if parent == Path(target).parent:
                    break
                if parent.is_symlink():
                    raise RuntimeError('Private installation paths must not be symlinks')
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with archive.open(row) as source_handle, destination.open('wb') as output:
                shutil.copyfileobj(source_handle, output)
            os.chmod(destination, 0o600)


def install_authorized_assets():
    if (os.environ.get('RIMBIND_INSTALL_ASSETS') != '1' or
            os.environ.get('RIMBIND_THIRD_PARTY_TERMS_CONFIRMED') != '1'):
        raise RuntimeError('Explicit owner installation and third-party terms acknowledgement are required')
    token = os.environ.get('HF_TOKEN')
    if not token:
        raise RuntimeError('Private asset read Secret is missing')
    from huggingface_hub import HfApi, hf_hub_download
    manifest = json.loads(INSTALL_MANIFEST.read_text())
    if manifest.get('schema') != 'rimbind_authorized_hosted_install_v1':
        raise RuntimeError('Unsupported hosted installation manifest')
    repo = manifest['private_repo']
    if HfApi().model_info(repo, token=token).private is not True:
        raise RuntimeError('RIMBind model assets must remain private')
    root = Path(os.environ.get('RIMBIND_INSTALLED_ASSETS', '/tmp/rimbind_frozen_assets')).resolve()
    if root.exists() and any(root.iterdir()) and not (root/'.rimbind_installer_owned').exists():
        raise RuntimeError('Refuse to install into an existing non-installer directory')
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    (root/'.rimbind_installer_owned').touch(mode=0o600)
    core = manifest['core_bundle']
    core_zip = Path(hf_hub_download(repo, core['filename'], token=token))
    if sha(core_zip) != core['sha256']:
        raise RuntimeError('Core archive differs from the frozen bundle')
    with zipfile.ZipFile(core_zip) as archive:
        name = 'rimbind_core/PRIVATE_BUNDLE_MANIFEST.json'
        metadata = archive.read(name)
        frozen = json.loads(metadata)
    if frozen.get('visibility') != 'PRIVATE' or frozen.get('query_test_labels_included') is not False:
        raise RuntimeError('Core bundle metadata boundary failure')
    expected = {'rimbind_core/'+row['path']: row['sha256'] for row in frozen['files']}
    expected[name] = hashlib.sha256(metadata).hexdigest()
    extract_locked_zip(core_zip, core['sha256'], root, expected, 500_000_000)
    support = manifest['support_bundle']
    support_zip = Path(hf_hub_download(repo, support['filename'], token=token))
    extract_locked_zip(support_zip, support['sha256'], root,
                       {row['path']: row['sha256'] for row in support['files']}, 100_000_000)
    os.chmod(root/'bin/foldseek', 0o700)
    foundation_paths = {}
    foundation_files = []
    for model in manifest['foundation_models']:
        directory = root/model['key']
        for row in model['files']:
            # No main-branch or latest-weight fallback; renamed Hub repositories
            # may redirect normally, but exact revision AND bytes are mandatory.
            source = Path(hf_hub_download(model['repo_id'], row['path'], revision=model['revision'], token=token))
            if sha(source) != row['sha256']:
                raise RuntimeError('Upstream encoder file differs from the frozen asset')
            destination = directory/row['path']
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            if destination.exists() or destination.is_symlink():
                if sha(destination) != row['sha256']:
                    raise RuntimeError('Installed foundation file changed')
            else:
                destination.symlink_to(source.resolve())
            foundation_files.append(dict(path=os.path.relpath(destination, root), sha256=row['sha256']))
        foundation_paths[model['key']] = os.path.relpath(directory, root)
    config = dict(schema='rimbind_serving_local_assets_v1',
                  paths=dict(checkpoint_root='rimbind_core', memory_db='rimbind_core/memory_db', foldseek='bin/foldseek', **foundation_paths),
                  files=[dict(path=name, sha256=digest) for name, digest in expected.items()]+
                        [dict(path=row['path'], sha256=row['sha256']) for row in support['files']]+foundation_files,
                  query_labels_included=False)
    config_path = root/'serving_assets.json'
    config_path.write_text(json.dumps(config, indent=2)+'\n')
    os.chmod(config_path, 0o600)
    # The normal Assets constructor performs an independent strict verification
    # before any trusted checkpoint or training-memory pickle is deserialized.
    return config_path, root/'validation_cases'
