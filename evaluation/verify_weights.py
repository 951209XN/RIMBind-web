#!/usr/bin/env python3
"""Verify the ten frozen model-C checkpoint files against the release manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_schema(state):
    rows = [{"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in sorted(state.items())]
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify(root, check_tensors=False):
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "rimbind_frozen_weights_release_v1" or manifest.get("model_variant") != "C":
        raise ValueError("Unexpected release manifest or model variant")
    rows = manifest["checkpoints"]
    required = {(task, fold) for task in ("dna", "rna") for fold in range(5)}
    if len(rows) != 10 or {(row["task"], row["fold"]) for row in rows} != required:
        raise ValueError("Require exactly five DNA and five RNA checkpoints")
    torch = None
    if check_tensors:
        try:
            import torch
        except ImportError as exc:
            raise ValueError("Install PyTorch to use --check-tensors; checksum verification needs only Python") from exc
    checked = []
    paths = set()
    for row in rows:
        relative = Path(row["path"])
        path = (root / relative).resolve()
        if relative.is_absolute() or root not in path.parents or path in paths:
            raise ValueError("Invalid or duplicate checkpoint path in manifest")
        paths.add(path)
        if path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            raise ValueError(f"Checkpoint size or SHA256 mismatch: {relative}")
        if check_tensors:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            state = checkpoint["state_dict"]
            if checkpoint.get("variant") != "C" or len(state) != row["tensor_count"]:
                raise ValueError(f"Incorrect checkpoint structure: {relative}")
            if not all(isinstance(value, torch.Tensor) and torch.isfinite(value).all().item() for value in state.values()):
                raise ValueError(f"Nonfinite or invalid tensor: {relative}")
            if tensor_schema(state) != row["tensor_schema_sha256"]:
                raise ValueError(f"Tensor names, dtypes or shapes differ: {relative}")
        checked.append(row["path"])
    if {p.resolve() for p in root.rglob("*.pt")} != paths:
        raise ValueError("Unexpected .pt files in the checkpoint directory")
    return {"status": "PASS", "release_id": manifest["release_id"],
            "checkpoint_files": len(checked), "sha256_verified": True,
            "tensor_checks_performed": check_tensors, "paths": checked}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "checkpoints")
    parser.add_argument("--check-tensors", action="store_true", help="Also load tensors safely on CPU; requires PyTorch")
    args = parser.parse_args()
    try:
        result = verify(args.root, args.check_tensors)
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.exit(2, f"Weight verification failed: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
