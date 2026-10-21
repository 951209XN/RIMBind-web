"""Frozen DNA FULL / RNA CR fusion. No training, fitting or test labels."""
import numpy as np

def fuse(base, memory, count, tm, task):
    if task not in {'dna','rna'}:
        raise ValueError('Task must be dna or rna')
    p=np.asarray(base,dtype=np.float64)
    m=np.asarray(memory,dtype=np.float64)
    c=np.asarray(count,dtype=np.float64)
    t=np.asarray(tm,dtype=np.float64)
    if not (p.shape==m.shape==c.shape==t.shape) or p.ndim!=1:
        raise ValueError('One aligned vector per residue is required')
    if not (np.isfinite(p).all() and np.isfinite(c).all() and np.isfinite(t).all()):
        raise ValueError('Non-finite probability, count or TM')
    if np.any((p<0)|(p>1)|(c<0)|(c!=np.floor(c))|(t<0)|(t>1)):
        raise ValueError('Probability/count/TM outside supported range')
    covered=c>0
    if np.any(~np.isfinite(m[covered])) or np.any((m[covered]<0)|(m[covered]>1)):
        raise ValueError('Covered memory must be a finite probability')
    m=np.nan_to_num(m,nan=0.)
    n=covered.astype(float) if task=='rna' else c
    alpha,tau=(.8,4.) if task=='rna' else (.2,1.)
    q=np.clip((n*m+tau*.1)/(n+tau),1e-4,1-1e-4)
    evidence=(np.log(q/(1-q))-np.log(.1/.9))*t**2
    evidence[~covered]=0.
    p=np.clip(p,1e-6,1-1e-6)
    z=np.log(p/(1-p))+alpha*evidence
    return (1/(1+np.exp(-np.clip(z,-30,30)))).astype(np.float32)

def ensemble_logits(fold_logits):
    values=np.asarray(fold_logits,dtype=np.float64)
    if values.ndim!=2 or values.shape[0]!=5 or not np.isfinite(values).all():
        raise ValueError('Exactly five finite aligned fold-logit vectors are required')
    return np.mean(1/(1+np.exp(-np.clip(values,-80,80))),axis=0)
