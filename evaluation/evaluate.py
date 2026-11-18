#!/usr/bin/env python3
"""Evaluate residue probabilities against an explicitly aligned label manifest."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

KEY_COLUMNS = ("dataset", "protein_id", "residue_index")
THRESHOLD = 0.5


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_table(path, required):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        first_line = handle.readline()
        handle.seek(0)
        reader = csv.DictReader(handle, delimiter="\t" if "\t" in first_line else ",")
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)) or not set(required).issubset(fields):
            raise ValueError(f"{Path(path).name}: required columns are {', '.join(required)}")
        rows = {}
        for line, row in enumerate(reader, 2):
            if None in row or any(row.get(column) is None for column in fields):
                raise ValueError(f"{Path(path).name}:{line}: malformed record")
            dataset, protein, index = (row[column].strip() for column in KEY_COLUMNS)
            if not dataset or not protein or not index.isascii() or not index.isdecimal() or int(index) < 1:
                raise ValueError(f"{Path(path).name}:{line}: require dataset, protein_id and positive 1-based residue_index")
            key = (dataset, protein, int(index))
            if key in rows:
                raise ValueError(f"{Path(path).name}:{line}: duplicate residue key {key}")
            rows[key] = row
        if not rows:
            raise ValueError(f"{Path(path).name}: empty table")
        return rows


def measure(labels, probabilities):
    y = np.asarray(labels)
    p = np.asarray(probabilities, dtype=np.float64)
    if y.ndim != 1 or p.shape != y.shape or not len(y):
        raise ValueError("Require nonempty, aligned one-dimensional label and probability arrays")
    if not np.isin(y, (0, 1)).all() or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Labels must be binary and probabilities must be finite in [0, 1]")
    y = y.astype(np.int64)
    order = np.argsort(-p, kind="stable")
    scores, targets = p[order], y[order]
    ends = np.r_[np.flatnonzero(np.diff(scores) != 0), len(scores) - 1]
    cumulative_tp = np.cumsum(targets)[ends]
    cumulative_fp = ends + 1 - cumulative_tp
    positives, negatives = int(y.sum()), int(len(y) - y.sum())
    average_precision = float(np.sum((cumulative_tp / (ends + 1)) * np.diff(np.r_[0, cumulative_tp])) / positives) if positives else 0.0
    auroc = None
    if positives and negatives:
        tpr, fpr = np.r_[0, cumulative_tp / positives], np.r_[0, cumulative_fp / negatives]
        auroc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) * 0.5))
    predicted = p >= THRESHOLD
    tp = int(np.sum((y == 1) & predicted))
    fp = int(np.sum((y == 0) & predicted))
    tn = int(np.sum((y == 0) & ~predicted))
    fn = int(np.sum((y == 1) & ~predicted))
    denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return {
        "n_residues": len(y), "n_positive": positives, "n_negative": negatives,
        "AP": average_precision, "AUROC": auroc,
        "F1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "MCC": (tp * tn - fp * fn) / math.sqrt(denominator) if denominator else 0.0,
        "Precision": tp / (tp + fp) if tp + fp else 0.0,
        "Recall": tp / (tp + fn) if tp + fn else 0.0,
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
    }


def evaluate(labels_path, predictions_path):
    labels = read_table(labels_path, (*KEY_COLUMNS, "y_true"))
    predictions = read_table(predictions_path, (*KEY_COLUMNS, "probability"))
    probabilities = {}
    for key, row in predictions.items():
        if key not in labels:
            raise ValueError(f"Prediction residue is absent from the label manifest: {key}")
        try:
            value = float(row["probability"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid probability at {key}") from exc
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Probability must be finite in [0, 1] at {key}")
        probabilities[key] = value
    datasets, excluded = {}, {}
    for key, row in labels.items():
        mask = row.get("eval_mask", "1").strip()
        if mask not in ("0", "1"):
            raise ValueError(f"eval_mask must be 0 or 1 at {key}")
        if mask == "0":
            excluded[key[0]] = excluded.get(key[0], 0) + 1
            continue
        label = row["y_true"].strip()
        if label not in ("0", "1"):
            raise ValueError(f"An evaluable residue requires a binary y_true at {key}")
        if key not in probabilities:
            raise ValueError(f"Missing prediction for evaluable residue {key}; evaluation aborted")
        bucket = datasets.setdefault(key[0], {"labels": [], "probabilities": [], "proteins": set()})
        bucket["labels"].append(int(label))
        bucket["probabilities"].append(probabilities[key])
        bucket["proteins"].add(key[1])
    if not datasets:
        raise ValueError("The label manifest contains no evaluable residues")
    if set(excluded) - set(datasets):
        raise ValueError("A dataset has no evaluable residues; no metrics were written")
    results = {}
    for dataset, bucket in sorted(datasets.items()):
        results[dataset] = {
            **measure(bucket["labels"], bucket["probabilities"]),
            "n_proteins": len(bucket["proteins"]),
            "n_excluded_residues": excluded.get(dataset, 0),
        }
    return {
        "schema": "rimbind_residue_evaluation_v1",
        "aggregation": "pool all evaluable residues within each dataset",
        "AP_definition": "non-interpolated average precision; tied scores grouped",
        "threshold": THRESHOLD, "threshold_operator": ">=",
        "single_class_policy": "AUROC is null; AP is 0 if no positive labels; undefined precision/recall/F1/MCC are 0",
        "coverage": "all evaluable keys in the supplied label manifest must have a prediction",
        "label_manifest_sha256": sha256(labels_path),
        "predictions_sha256": sha256(predictions_path),
        "datasets": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True, help="CSV/TSV label and evaluation-mask manifest")
    parser.add_argument("--predictions", type=Path, required=True, help="CSV/TSV residue probabilities with matching keys")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON metrics")
    args = parser.parse_args()
    if args.output.resolve() in {args.labels.resolve(), args.predictions.resolve()}:
        parser.error("The output must not overwrite an input file")
    try:
        result = evaluate(args.labels, args.predictions)
    except (ValueError, OSError, csv.Error) as exc:
        parser.exit(2, f"Evaluation failed: {exc}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
