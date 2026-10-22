"""Trusted prepared-feature inference. No query labels, training or downloads.

This is a computation component, not a statement of deployment acceptance.
Raw uploads must never be deserialized as PyTorch/pickle files.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import torch

from backend.fusion import fuse
from backend.model_loader import load_model_c, source_sha256

MANIFEST = Path(__file__).resolve().parents[1] / 'configs/assets_manifest.json'


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class FrozenPredictor:
    def __init__(self, assets, task, device='cuda'):
        if task not in {'dna', 'rna'}:
            raise ValueError('Task must be dna or rna')
        self.task, self.device = task, torch.device(device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        self.models, self.checkpoints = [], []
        manifest = json.loads(MANIFEST.read_text())
        rows = sorted((r for r in manifest['sscp_checkpoints'] if r['domain'] == task),
                      key=lambda r: r['fold'])
        if [r['fold'] for r in rows] != list(range(5)):
            raise RuntimeError('Exactly the five frozen folds are required')
        for row in rows:
            path = Path(assets) / row['archive_path']
            if file_sha256(path) != row['sha256']:
                raise RuntimeError('Frozen checkpoint SHA256 mismatch')
            # Administrator-supplied, checksum-verified files only, never uploads.
            state = torch.load(path, map_location='cpu', weights_only=False)
            if state.get('variant') != 'C':
                raise RuntimeError('Only frozen variant C is supported')
            model = load_model_c()
            model.load_state_dict(state['state_dict'], strict=True)
            self.models.append(model.to(self.device).eval())
            self.checkpoints.append(row['sha256'])

    @torch.inference_mode()
    def predict(self, saprot, ca, esm3, memory, count, tm):
        n = len(ca)
        vectors = []
        for value, shape in ((saprot, (n, 1280)), (ca, (n, 3)), (esm3, (n, 1536))):
            tensor = torch.as_tensor(value).float()
            if tuple(tensor.shape) != shape or not bool(torch.isfinite(tensor).all()):
                raise ValueError('Invalid or unaligned prepared features')
            vectors.append(tensor.to(self.device))
        if not 1 <= n <= 600:
            raise ValueError('Service residue limit exceeded')
        folds = [torch.sigmoid(model([vectors[0]], [vectors[1]], [vectors[2]])).float().cpu()
                 for model in self.models]
        # Retain the original float32 torch stacking/mean operation order.
        base = torch.stack(folds).mean(0).contiguous().numpy()
        if base.shape != (n,) or not np.isfinite(base).all():
            raise RuntimeError('Invalid model output')
        final = fuse(base, memory, count, tm, self.task)
        return dict(sscp=base, rimbind=final,
                    provenance=dict(task=self.task, rule='FULL' if self.task == 'dna' else 'NO_TEMPLATE_COUNT',
                                    checkpoint_sha256=self.checkpoints, model_source_sha256=source_sha256(),
                                    torch_version=torch.__version__, device=str(self.device)))
