"""Single-query Foldseek adapter using the unchanged frozen projection code."""
from pathlib import Path
import re
import subprocess

import numpy as np
import torch

from backend.predictor import file_sha256
from vendor.frozen_memory_pipeline import COLUMNS, parse_alignments, passes, project

FOLDSEEK_SHA256 = '4464e56b8b1c104f1d595f1ea3529c9793c09d7d199556c322c863779a8a2df0'


def run(command, log):
    with Path(log).open('w') as output:
        subprocess.run([str(s) for s in command], stdout=output, stderr=subprocess.STDOUT,
                       check=True, timeout=180)


def prepare_query(pdb_text, candidate_id, directory, foldseek):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', candidate_id):
        raise ValueError('Unsafe internal query identifier')
    if file_sha256(foldseek) != FOLDSEEK_SHA256:
        raise RuntimeError('Frozen Foldseek SHA256 mismatch')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    source = directory / 'pdb'
    source.mkdir()
    (source / f'{candidate_id}.pdb').write_text(pdb_text)
    db = directory / 'query'
    run([foldseek, 'createdb', source, db, '--threads', '2', '-v', '1'], directory / 'createdb.log')
    run([foldseek, 'lndb', str(db) + '_h', str(db) + '_ss_h', '-v', '1'], directory / 'lndb.log')
    fasta = directory / '3di.fasta'
    run([foldseek, 'convert2fasta', str(db) + '_ss', fasta, '-v', '1'], directory / '3di.log')
    entries = {}
    key = None
    for line in fasta.read_text().splitlines():
        if line.startswith('>'):
            key = line[1:].split()[0]
            entries[key] = ''
        elif line.strip() and key is not None:
            entries[key] += line.strip()
    if set(entries) != {candidate_id}:
        raise RuntimeError('Foldseek did not preserve a single query chain')
    return db, entries[candidate_id]


def retrieve(db, candidate_id, n, task, foldseek, target_db, labels_path, expected_labels_sha):
    if task not in {'dna', 'rna'}:
        raise ValueError('Invalid task')
    if file_sha256(foldseek) != FOLDSEEK_SHA256 or file_sha256(labels_path) != expected_labels_sha:
        raise RuntimeError('Frozen memory asset changed')
    # Only an administrator-provided checksum-verified training-memory file.
    payload = torch.load(labels_path, map_location='cpu', weights_only=False)
    labels = {row['protein_id']: row['labels'].numpy().astype(np.uint8) for row in payload['rows']}
    work = Path(db).parent
    result, tmp, alignments = work / 'result', work / 'search_tmp', work / 'alignments.tsv'
    run([foldseek, 'search', db, target_db, result, tmp, '-s', '9.5', '--max-seqs', '1000',
         '--alignment-type', '2', '--exact-tmscore', '1', '-a', '1', '-e', '10',
         '--threads', '2', '-v', '1'], work / 'search.log')
    run([foldseek, 'convertalis', db, target_db, result, alignments, '--format-output', ','.join(COLUMNS),
         '--exact-tmscore', '1', '--threads', '2', '-v', '1'], work / 'convertalis.log')
    grouped = parse_alignments(alignments)
    if set(grouped) - {candidate_id}:
        raise RuntimeError('Unexpected query identifier in Foldseek output')
    hits = [r for r in grouped.get(candidate_id, []) if r['target'] in labels and passes(r)]
    hits.sort(key=lambda r: (r['alntmscore'], r['qcov'], r['bits']), reverse=True)
    hits = hits[:20]
    sums, rotated, weights = np.zeros(n), np.zeros(n), np.zeros(n)
    count, tm = np.zeros(n, np.int32), np.zeros(n, np.float32)
    for hit in hits:
        project(hit, n, labels[hit['target']], sums, rotated, weights, count, tm)
    score = np.full(n, np.nan, np.float32)
    covered = weights > 0
    score[covered] = (sums[covered] / weights[covered]).astype(np.float32)
    return dict(memory_score=score, template_count=count, best_tm=tm, hits=len(hits),
                alignment_sha256=file_sha256(alignments), query_labels_read=False)
