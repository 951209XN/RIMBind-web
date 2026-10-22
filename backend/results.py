"""Label-free, residue-aligned exports; probabilities are not accuracy metrics."""
import csv
import json
from pathlib import Path

import numpy as np


def write_results(directory, protein, result, extra_provenance):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    n = len(protein.sequence)
    base, final = result['sscp'], result['rimbind']
    if any(np.asarray(v).shape != (n,) or not np.isfinite(v).all() or
           np.any((np.asarray(v) < 0) | (np.asarray(v) > 1)) for v in (base, final)):
        raise ValueError('Invalid result vectors')
    fields = ['residue_index', 'amino_acid', 'chain', 'author_residue_number', 'insertion_code',
              'sscp_probability', 'rimbind_probability', 'predicted_binding_at_0_5']
    table = []
    for i, (aa, key, p, q) in enumerate(zip(protein.sequence, protein.residue_keys, base, final), 1):
        table.append([i, aa, key['chain'], key['author_residue_number'], key['insertion_code'],
                      format(float(p), '.9g'), format(float(q), '.9g'), int(q >= .5)])
    csv_path = directory/'residue_probabilities.csv'
    with csv_path.open('x', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(table)
    lines = []
    for line in protein.normalized_pdb.splitlines():
        if not line.startswith('ATOM  '):
            continue
        i = int(line[22:26])-1
        key = protein.residue_keys[i]
        line = line.ljust(80)
        lines.append(line[:21]+key['chain']+f"{key['author_residue_number']:4d}"+
                     (key['insertion_code'] or ' ')+line[27:60]+f'{float(final[i]):6.2f}'+line[66:80])
    pdb_path = directory/'rimbind_probability_Bfactor.pdb'
    pdb_path.write_text('\n'.join(lines)+'\nEND\n')
    provenance = dict(input_sha256=protein.input_sha256, residues=n, chain=protein.chain,
                      residue_mapping=protein.residue_keys, threshold=.5,
                      bfactor_encoding='RIMBind probability 0 to 1, rounded to two decimals in PDB; CSV has higher precision',
                      metrics_computed=False, query_labels_read=False,
                      **result['provenance'], **extra_provenance)
    provenance_path = directory/'provenance.json'
    provenance_path.write_text(json.dumps(provenance, indent=2)+'\n')
    return table, [str(csv_path), str(pdb_path), str(provenance_path)]
