"""Strict PDB-chain intake. Original residue keys never inferred from position."""
import hashlib
import math
from dataclasses import dataclass

AA=dict(zip('ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL'.split(),'ARNDCQEGHILKMFPSTWYV'))

@dataclass
class ProteinInput:
    sequence: str
    residue_keys: list
    ca: list
    normalized_pdb: str
    input_sha256: str
    chain: str

def parse_pdb(text,chain,sequence=None,max_residues=600):
    raw=text.encode('utf-8')
    if len(raw)>2097152:
        raise ValueError('PDB exceeds the 2 MiB intake limit')
    if not isinstance(chain,str) or len(chain)!=1 or not chain.isalnum():
        raise ValueError('Select one alphanumeric PDB chain identifier')
    residues={}; lines=[]; model_count=0
    for line in text.splitlines():
        if line.startswith('MODEL '):
            model_count+=1
            if model_count>1:break
        if line.startswith('ENDMDL'):break
        if not line.startswith(('ATOM  ','HETATM')) or len(line)<54 or line[21:22]!=chain:continue
        name=line[17:20].strip()
        if line.startswith('HETATM'):
            if name=='MSE':raise ValueError('MSE requires a separately validated conversion; not silently converted')
            continue
        if name not in AA:raise ValueError('Selected chain contains a nonstandard or non-protein residue')
        if line[16:17] not in {' ','A'}:continue
        key=(chain,int(line[22:26]),line[26:27].strip())
        xyz=[float(line[i:i+8]) for i in (30,38,46)]
        if not all(math.isfinite(x) for x in xyz):raise ValueError('Invalid atom coordinates')
        item=residues.setdefault(key,dict(name=name,ca=None,atoms=set(),number=len(residues)+1))
        if item['name']!=name:raise ValueError('Conflicting residue names for the same residue key')
        atom=line[12:16].strip()
        if atom in item['atoms']:raise ValueError('Ambiguous duplicate atom/alternate location')
        item['atoms'].add(atom)
        if atom=='CA':item['ca']=xyz
        padded=line.ljust(80)
        lines.append(padded[:16]+' '+padded[17:21]+'A'+f"{item['number']:4d}"+' '+padded[27:]+'\n')
    if not residues:raise ValueError('No standard protein residues found for the selected chain')
    if len(residues)>max_residues:raise ValueError(f'Initial service supports at most {max_residues} observed residues')
    if any(v['ca'] is None for v in residues.values()):raise ValueError('Every selected residue must have a C-alpha coordinate')
    seq=''.join(AA[v['name']] for v in residues.values())
    if sequence:
        supplied=''.join(x.strip() for x in sequence.splitlines() if not x.startswith('>')).upper()
        if supplied!=seq:raise ValueError('FASTA differs from observed PDB-chain sequence; missing coordinates are not silently remapped')
    keys=[dict(chain=k[0],author_residue_number=k[1],insertion_code=k[2],residue_name=v['name']) for k,v in residues.items()]
    return ProteinInput(seq,keys,[v['ca'] for v in residues.values()],''.join(lines)+'END\n',hashlib.sha256(raw).hexdigest(),chain)
