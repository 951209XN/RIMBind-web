import ast,unittest
from pathlib import Path
import numpy as np
from backend.fusion import fuse,ensemble_logits
from backend.inputs import parse_pdb
ROOT=Path(__file__).resolve().parents[1]

class Contracts(unittest.TestCase):
    def test_rna_count_invariance(self):
        a=fuse([.2,.7],[.8,.2],[1,1],[.8,.9],'rna')
        b=fuse([.2,.7],[.8,.2],[20,3],[.8,.9],'rna')
        np.testing.assert_array_equal(a,b)
    def test_dna_count_effect(self):
        self.assertNotEqual(float(fuse([.2],[.8],[1],[.8],'dna')[0]),float(fuse([.2],[.8],[5],[.8],'dna')[0]))
    def test_no_reference_fallback(self):
        np.testing.assert_allclose(fuse([.2,.7],[np.nan,np.nan],[0,0],[0,0],'rna'),[.2,.7],atol=5e-8)
    def test_original_formula_equivalence(self):
        tree=ast.parse((ROOT/'vendor/frozen_memory_pipeline.py').read_text())
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='evidence')
        env={'np':np};exec(compile(ast.Module(body=[function],type_ignores=[]),'frozen_evidence','exec'),env)
        p=np.array([.12,.5,.8]);m=np.array([.8,.2,.4]);c=np.array([2,0,9]);tm=np.array([.7,0,.8])
        for task,mode,alpha,tau in [('dna','FULL',.2,1),('rna','NO_TEMPLATE_COUNT',.8,4)]:
            e=env['evidence'](m,c,tm,.1,tau,2,mode)
            expected=(1/(1+np.exp(-np.clip(np.log(p/(1-p))+alpha*e,-30,30)))).astype(np.float32)
            np.testing.assert_array_equal(fuse(p,m,c,tm,task),expected)
    def test_ensemble_not_logit_average(self):
        z=np.array([[-3],[0],[1],[2],[4]])
        self.assertNotAlmostEqual(ensemble_logits(z)[0],1/(1+np.exp(-z.mean())),places=3)
    def test_five_folds_required(self):
        with self.assertRaises(ValueError):ensemble_logits([[0],[1]])
    def test_pdb_mapping(self):
        item=parse_pdb((ROOT/'examples/6jhe_A/example.pdb').read_text(),'A')
        self.assertEqual(len(item.sequence),53);self.assertEqual(len(item.residue_keys),53)
        self.assertTrue(item.normalized_pdb.endswith('END\n'))
    def test_wrong_chain(self):
        with self.assertRaises(ValueError):parse_pdb((ROOT/'examples/6jhe_A/example.pdb').read_text(),'Z')
    def test_sequence_mismatch(self):
        with self.assertRaises(ValueError):parse_pdb((ROOT/'examples/6jhe_A/example.pdb').read_text(),'A','AAAA')
    def test_invalid_probability(self):
        with self.assertRaises(ValueError):fuse([2],[.8],[1],[.8],'rna')

if __name__=='__main__':unittest.main()
