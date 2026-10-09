"""unittest的显式检查保留在普通和优化解释器两种模式。"""
import argparse,json,unittest
from itertools import permutations
from pathlib import Path
from exact_inference import *
from experiment import run
class Checks(unittest.TestCase):
    def test_reference(self):
        r=run();self.assertLess(r['max_method_error'],1e-12);self.assertLess(r['star_order_difference'],1e-10)
        self.assertEqual(r['star_good_peak_entries'],4);self.assertEqual(r['star_bad_peak_entries'],128)
        self.assertAlmostEqual(r['evidence_probability'],.40432);self.assertAlmostEqual(r['star_partition'],4096)
    def test_all_orders(self):
        fs=chain_factors([.6,.4],[[.85,.15],[.25,.75]],4,[.1,.9])
        for e in (0,1):
            base,z=enumerate_query(fs,'X0',{'E':e})
            for order in permutations(['X1','X2','X3']):
                value,mass,_=eliminate(fs,'X0',{'E':e},order)
                self.assertAlmostEqual(value.table[(1,)],base.table[(1,)]);self.assertAlmostEqual(mass,z)
    def test_axis_alignment(self):
        f=Factor(('A','B'),{(0,0):1,(0,1):2,(1,0):3,(1,1):4})
        g=Factor(('B','A'),{(0,0):5,(0,1):6,(1,0):7,(1,1):8})
        self.assertEqual(multiply([f,g]).table[(1,0)],18)
        self.assertEqual(sum_out(f,'A').table,{(0,):4,(1,):6})
        self.assertEqual(restrict(f,{'B':1}).table,{(0,):2,(1,):4})
    def test_invalid(self):
        for scope,table in [(('A','A'),{}),(('A',),{(0,):1}),((),{():-1}),((),{():float('nan')})]:
            with self.assertRaises(ValueError):Factor(scope,table)
        fs=chain_factors([.6,.4],[[.85,.15],[.25,.75]],3,[.1,.9])
        for order in [['X1'],['X1','X1'],['X0','X1']]:
            with self.assertRaises(ValueError):eliminate(fs,'X0',{'E':1},order)
        for evidence in [{'E':2},{'Z':1},{'X0':1}]:
            with self.assertRaises(ValueError):eliminate(fs,'X0',evidence)
    def test_zero_evidence(self):
        fs=chain_factors([.6,.4],[[.85,.15],[.25,.75]],2,[0,0])
        for method in (eliminate,enumerate_query):
            with self.assertRaises(ValueError):method(fs,'X0',{'E':1})
        with self.assertRaises(ValueError):chain_sum_product([.6,.4],[[.85,.15],[.25,.75]],2,[0,0])
    def test_length_one_and_scaling(self):
        params=([.6,.4],[[.85,.15],[.25,.75]],1,[.1,.9]);m,z,_,_=chain_sum_product(*params)
        self.assertAlmostEqual(m[0][1],.36/.42)
        fs=chain_factors(*params);a,za,_=eliminate(fs,'X0',{'E':1})
        scaled=[Factor(f.scope,{k:7*v for k,v in f.table.items()}) if i==0 else f for i,f in enumerate(fs)]
        b,zb,_=eliminate(scaled,'X0',{'E':1});self.assertAlmostEqual(a.table[(1,)],b.table[(1,)]);self.assertAlmostEqual(zb,7*za)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='outputs/test-result.json');a=p.parse_args();r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks));out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps({'tests_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'optimized':not __debug__},indent=2)+'\n');raise SystemExit(0 if r.wasSuccessful() else 1)
