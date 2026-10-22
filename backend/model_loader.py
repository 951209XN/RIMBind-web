"""Load unchanged model C source without requiring unused E/F-only Mamba."""
import ast
import hashlib
import types
from pathlib import Path

SOURCE=Path(__file__).resolve().parents[1]/'vendor/frozen_model_single.py'
EXPECTED_SOURCE_SHA256='cb7e9b754ef15555d211168d809f078fb424d3dfbff7718833e010dfa5a2819c'

def load_model_class():
    if source_sha256()!=EXPECTED_SOURCE_SHA256:
        raise RuntimeError('Frozen model source SHA256 mismatch')
    tree=ast.parse(SOURCE.read_text())
    # Source uses Mamba only when constructing E/F, never C. Remove that import
    # from the execution AST; source bytes and parameter names remain unchanged.
    imports=[n for n in tree.body if isinstance(n,ast.ImportFrom) and n.module=='mamba_ssm']
    if len(imports)!=1:raise RuntimeError('Unexpected frozen source layout')
    tree.body=[n for n in tree.body if n not in imports]
    namespace={'__name__':'rimbind_frozen_c'}
    exec(compile(tree,str(SOURCE),'exec'),namespace)
    return namespace['SingleNABindModel']

def load_model_c():
    return load_model_class()('C')

def source_sha256():return hashlib.sha256(SOURCE.read_bytes()).hexdigest()
