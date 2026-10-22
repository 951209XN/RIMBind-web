"""Administrator-only local asset lock. No token or automatic model download."""
import json
from pathlib import Path

from backend.predictor import MANIFEST, file_sha256
from backend.memory import FOLDSEEK_SHA256


class Assets:
    def __init__(self, config_path):
        config_path = Path(config_path).resolve()
        config = json.loads(config_path.read_text())
        if config.get('schema') != 'rimbind_serving_local_assets_v1':
            raise RuntimeError('Unsupported asset lock')
        self.paths = {k: (config_path.parent / v).resolve() for k, v in config['paths'].items()}
        expected = {}
        for row in config['files']:
            path = (config_path.parent / row['path']).resolve()
            if path in expected:
                raise RuntimeError('Duplicate file in asset lock')
            if file_sha256(path) != row['sha256']:
                raise RuntimeError('Asset lock checksum mismatch')
            expected[path] = row['sha256']
        frozen = json.loads(MANIFEST.read_text())
        for row in frozen['sscp_checkpoints'] + frozen['training_memory']:
            path = self.paths['checkpoint_root'] / row['archive_path']
            if expected.get(path.resolve()) != row['sha256']:
                raise RuntimeError('Missing original task-specific asset in lock')
        required = {
            self.paths['foldseek']: FOLDSEEK_SHA256,
            self.paths['saprot']/'pytorch_model.bin': frozen['external_assets']['saprot']['weight_sha256'],
            self.paths['esm3_snapshot']/'data/weights/esm3_sm_open_v1.pth': frozen['external_assets']['esm3']['weight_sha256'],
        }
        if any(expected.get(p.resolve()) != h for p, h in required.items()):
            raise RuntimeError('Missing frozen foundation asset in lock')
        for name in ('config.json', 'vocab.txt', 'tokenizer_config.json', 'special_tokens_map.json'):
            if (self.paths['saprot']/name).resolve() not in expected:
                raise RuntimeError('SaProt tokenizer/config is not locked')
        if (self.paths['esm3_snapshot']/'data/weights/esm3_structure_encoder_v0.pth').resolve() not in expected:
            raise RuntimeError('ESM3 structure encoder is not locked')
        for task in ('dna', 'rna'):
            for suffix in ('', '.dbtype', '.index', '_h', '_h.dbtype', '_h.index',
                           '_ss', '_ss.dbtype', '_ss.index', '_ca', '_ca.dbtype', '_ca.index'):
                if Path(str(self.paths['memory_db']/task)+suffix).resolve() not in expected:
                    raise RuntimeError('Incomplete frozen Foldseek database lock')
        self.lock_sha256 = file_sha256(config_path)
        self.manifest = frozen

    def labels_sha(self, task):
        return next(r['sha256'] for r in self.manifest['training_memory'] if r['domain'] == task)
