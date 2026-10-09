"""显式unittest断言在python -O下仍执行；覆盖数值、图语义和拒绝路径。"""
import argparse,json,unittest
from pathlib import Path
from graph_models import *
from experiment import run
class Checks(unittest.TestCase):
    def test_numeric(self):
        r=run()
        for key in ['chain_total','mrf_total']:self.assertAlmostEqual(r[key],1)
        for key in ['chain_ac_given_b_residual','collider_rs_residual','reverse_max_error','mrf_ac_given_b_residual']:self.assertLess(r[key],1e-12)
        self.assertGreater(r['chain_ac_residual'],.01)
        self.assertGreater(r['collider_rs_given_w_residual'],.01)
        self.assertLess(r['rain_given_wet_sprinkler'],r['rain_given_wet'])
        self.assertEqual(r['mrf_partition'],24)
    def test_structures(self):
        nodes=['A','B','C']; chain=[('A','B'),('B','C')];fork=[('B','A'),('B','C')];collider=[('A','B'),('C','B')]
        for edges in (chain,fork):
            self.assertFalse(d_separated(nodes,edges,'A','C'))
            self.assertTrue(d_separated(nodes,edges,'A','C',['B']))
        self.assertTrue(d_separated(nodes,collider,'A','C'))
        self.assertFalse(d_separated(nodes,collider,'A','C',['B']))
        self.assertEqual(equivalence_signature(nodes,chain),equivalence_signature(nodes,fork))
        self.assertNotEqual(equivalence_signature(nodes,chain),equivalence_signature(nodes,collider))
    def test_descendant_and_multiple_paths(self):
        self.assertFalse(d_separated(['A','B','C','D'],[('A','B'),('C','B'),('B','D')],'A','C',['D']))
        edges=[('A','B'),('B','D'),('A','C'),('C','D')]
        self.assertFalse(d_separated(['A','B','C','D'],edges,'A','D',['B']))
        self.assertTrue(d_separated(['A','B','C','D'],edges,'A','D',['B','C']))
    def test_invalid(self):
        for edges in [[('A','B'),('B','A')],[('A','A')],[('A','Z')]]:
            with self.assertRaises(ValueError):validate_dag(['A','B'],edges)
        with self.assertRaises(ValueError):joint_from_bn(['A'],[],{'A':[1.1]})
        with self.assertRaises(ValueError):joint_from_bn(['A'],[],{'A':[float('nan')]})
        j=joint_from_bn(['A'],[],{'A':[0]})
        with self.assertRaises(ValueError):conditional(j,{'A':0},{'A':1})
    def test_extra_independence(self):
        j=joint_from_bn(['A','B'],[('A','B')],{'A':[.4],'B':[.3,.3]})
        self.assertFalse(d_separated(['A','B'],[('A','B')],'A','B'))
        self.assertLess(ci_residual(j,'A','B'),1e-12)
    def test_cpd_parent_order(self):
        j=joint_from_bn(['R','S','W'],[('S','W'),('R','W')],{'R':[.2],'S':[.3],'W':[.01,.8,.9,.99]})
        self.assertAlmostEqual(conditional(j,{'W':1},{'R':1,'S':0}),.9)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='outputs/test-result.json');a=p.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'optimized':not __debug__},indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
