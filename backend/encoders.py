"""Frozen SaProt + ESM3 preprocessing, with local-only pinned ESM3 assets.

The caller owns GPU allocation. This module never downloads or fits weights.
"""
from pathlib import Path

import torch

from vendor.frozen_embeddings import crop


def load_saprot(path, device='cpu'):
    from transformers import EsmModel, EsmTokenizer
    tokenizer = EsmTokenizer.from_pretrained(str(path), local_files_only=True)
    model = EsmModel.from_pretrained(str(path), local_files_only=True).to(device).eval()
    return tokenizer, model


def pin_esm3_snapshot(snapshot):
    # esm 3.1.1's factory otherwise calls snapshot_download without a revision.
    # Constrain only that module's resolver to the administrator-verified local
    # snapshot. Architecture, tokenizer and all weight-loading code stay upstream.
    from esm.utils.constants import esm3 as constants
    snapshot = Path(snapshot).resolve()

    def pinned_snapshot(*args, **kwargs):
        repo = kwargs.get('repo_id', args[0] if args else None)
        if repo != 'EvolutionaryScale/esm3-sm-open-v1':
            raise RuntimeError('Unexpected foundation model repository')
        return str(snapshot)

    resolver = getattr(constants.data_root, '__func__', constants.data_root)
    if hasattr(resolver, 'cache_clear'):
        resolver.cache_clear()
    constants.snapshot_download = pinned_snapshot
    # INFRA_PROVIDER changes upstream data_root to a working-directory lookup.
    import os
    if 'INFRA_PROVIDER' in os.environ:
        raise RuntimeError('INFRA_PROVIDER conflicts with the pinned asset resolver')
    if Path(constants.data_root('esm3')).resolve() != snapshot:
        raise RuntimeError('ESM3 snapshot pinning failed')


def load_esm3(snapshot, device='cpu'):
    from esm.models.esm3 import ESM3
    pin_esm3_snapshot(snapshot)
    return ESM3.from_pretrained('esm3_sm_open_v1', device=torch.device(device)).eval()


@torch.inference_mode()
def saprot_embedding(tokenizer, model, sequence, three_di, device):
    n = len(sequence)
    if len(three_di) != n:
        raise ValueError('3Di/sequence length mismatch')
    text = ''.join(a + b.lower() for a, b in zip(sequence, three_di))
    tokens = tokenizer(text, return_tensors='pt', add_special_tokens=True, truncation=False)
    if tokens['input_ids'].shape[1] != n + 2:
        raise RuntimeError('SaProt token count mismatch')
    tokens = {k: v.to(device) for k, v in tokens.items()}
    result = crop(model(**tokens).last_hidden_state, n).half().contiguous()
    if tuple(result.shape) != (n, 1280) or not torch.isfinite(result).all():
        raise RuntimeError('Invalid SaProt embedding')
    return result


@torch.inference_mode()
def esm3_embedding(model, pdb_path, sequence, device):
    from esm.sdk.api import ESMProtein
    n = len(sequence)
    mode, reason = 'pdb_sequence_structure', None
    try:
        protein = ESMProtein.from_pdb(Path(pdb_path), chain_id='A')
        if len(protein) != n:
            reason = f'pdb_length_{len(protein)}_item_length_{n}'
            protein, mode = ESMProtein(sequence=sequence), 'item_sequence_length_mismatch'
    except Exception as exc:
        reason = f'pdb_failed_{type(exc).__name__}'
        protein, mode = ESMProtein(sequence=sequence), 'item_sequence_pdb_failed'
    if len(protein) != n:
        raise RuntimeError('ESM3 fallback length mismatch')
    encoded = model.encode(protein).to(device)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        output = model(
            sequence_tokens=encoded.sequence.unsqueeze(0) if encoded.sequence is not None else None,
            structure_tokens=encoded.structure.unsqueeze(0) if encoded.structure is not None else None,
            ss8_tokens=encoded.secondary_structure.unsqueeze(0) if encoded.secondary_structure is not None else None,
            sasa_tokens=encoded.sasa.unsqueeze(0) if encoded.sasa is not None else None,
            function_tokens=encoded.function.unsqueeze(0) if encoded.function is not None else None,
            residue_annotation_tokens=encoded.residue_annotations.unsqueeze(0) if encoded.residue_annotations is not None else None,
            structure_coords=encoded.coordinates.unsqueeze(0) if encoded.coordinates is not None else None)
        result = crop(output.embeddings, n).half().contiguous()
    if tuple(result.shape) != (n, 1536) or not torch.isfinite(result).all():
        raise RuntimeError('Invalid ESM3 embedding')
    return result, dict(source_mode=mode, fallback_reason=reason)
