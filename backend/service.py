"""Real frozen inference coordinator. No laboratory server access or downloads."""
import gc
import importlib.metadata
from pathlib import Path
import re
import shutil
import tempfile
import time
import uuid

import torch

from backend.inputs import parse_pdb
from backend.memory import prepare_query, retrieve
from backend.encoders import load_saprot, load_esm3, saprot_embedding, esm3_embedding
from backend.predictor import FrozenPredictor
from backend.results import write_results


def predict_gpu(assets, protein, three_di, memory, task):
    if not torch.cuda.is_available():
        raise RuntimeError('No real CUDA allocation available')
    device = torch.device('cuda')
    try:
        tokenizer, model = load_saprot(assets.paths['saprot'], device)
        saprot = saprot_embedding(tokenizer, model, protein.sequence, three_di, device)
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        model = load_esm3(assets.paths['esm3_snapshot'], device)
        with tempfile.TemporaryDirectory(prefix='esm3_input_') as directory:
            path = Path(directory)/'query.pdb'
            path.write_text(protein.normalized_pdb)
            esm3, mode = esm3_embedding(model, path, protein.sequence, device)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        predictor = FrozenPredictor(assets.paths['checkpoint_root'], task, device)
        result = predictor.predict(saprot, protein.ca, esm3, memory['memory_score'],
                                   memory['template_count'], memory['best_tm'])
        result['provenance'].update(mode)
        result['provenance']['gpu_name'] = torch.cuda.get_device_name(0)
        result['provenance'].update(transformers_version=importlib.metadata.version('transformers'),
                                    esm_version=importlib.metadata.version('esm'), cuda_version=torch.version.cuda)
        return result
    finally:
        gc.collect()
        torch.cuda.empty_cache()


class Service:
    def __init__(self, assets, results_root):
        self.assets = assets
        self.results_root = Path(results_root).resolve()
        self.results_root.mkdir(parents=True, exist_ok=True)

    def cleanup(self):
        # Only this service's own UUID result directories, no recursive broad glob.
        for directory in self.results_root.iterdir():
            if (directory.is_dir() and not directory.is_symlink() and
                    re.fullmatch(r'[0-9a-f]{32}', directory.name) and
                    time.time() - directory.stat().st_mtime > 3600):
                shutil.rmtree(directory)

    def predict(self, path, chain, sequence, task, accession, gpu_callback):
        self.cleanup()
        if task not in {'dna', 'rna'}:
            raise ValueError('Select DNA or RNA')
        path = Path(path)
        if path.stat().st_size > 2097152:
            raise ValueError('PDB exceeds the 2 MiB limit')
        protein = parse_pdb(path.read_text(), chain, sequence or None)
        accession = accession.strip().lower()
        if accession and not re.fullmatch(r'[0-9][a-z0-9]{3}', accession):
            raise ValueError('Optional PDB accession must contain four alphanumeric characters, starting with a digit')
        job = uuid.uuid4().hex
        candidate = f'{accession or "upload"}_{job}'
        with tempfile.TemporaryDirectory(prefix='rimbind_query_') as directory:
            db, three_di = prepare_query(protein.normalized_pdb, candidate, Path(directory)/'foldseek',
                                         self.assets.paths['foldseek'])
            if len(three_di) != len(protein.sequence):
                raise RuntimeError('3Di sequence length mismatch')
            memory = retrieve(db, candidate, len(protein.sequence), task, self.assets.paths['foldseek'],
                              self.assets.paths['memory_db']/task,
                              self.assets.paths['checkpoint_root']/f'training_memory/{task}.pt',
                              self.assets.labels_sha(task))
            result = gpu_callback(protein, three_di, memory, task)
        table, files = write_results(self.results_root/job, protein, result,
            dict(asset_lock_sha256=self.assets.lock_sha256, memory_hits=memory['hits'],
                 pdb_accession_supplied=accession or None,
                 identity_note='Same-accession exclusion applied when supplied; filename is not trusted as PDB identity.',
                 retention='Service result directories expire after one hour; platform caches have separate cleanup.',
                 job_id=job))
        return table, files
