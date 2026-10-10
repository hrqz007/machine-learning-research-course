"""Explicit unittest assertions remain active under python -O."""
from pathlib import Path
import argparse, io, json, unittest
import numpy as np
from numpy_core import *
from experiment import gradient_audit, hand_example, run

class BackpropTests(unittest.TestCase):
    def setUp(self):
        self.X=np.array([[.3,-.7],[1.2,.4],[-.5,.9]])
        self.y=np.array([0,1,0]); self.p=initialize(8502,3)
    def test_all_parameters_centered_difference(self):
        _,g,_=loss_and_grad(self.X,self.y,self.p);n=finite_difference(self.X,self.y,self.p)
        self.assertEqual(sum(v.size for v in n.values()),17)
        for k in g: np.testing.assert_allclose(g[k],n[k],rtol=1e-6,atol=1e-8)
    def test_batch_reduction_and_duplicate(self):
        a=gradient_audit();self.assertLess(a['sum_mean_scaling_max_abs'],1e-12)
        self.assertLess(a['duplicated_batch_mean_max_abs'],1e-12)
        self.assertTrue(a['missing_batch_division_detected'])
    def test_shapes(self):
        _,g,c=loss_and_grad(self.X,self.y,self.p)
        for k in g:self.assertEqual(g[k].shape,self.p[k].shape)
        self.assertEqual(c['dX'].shape,self.X.shape)
        self.assertEqual(c['dS'].shape,(3,2))
    def test_hand_forward_relu(self):
        r=hand_example();np.testing.assert_allclose(r['cache']['Z'],[[2.,1.5],[-.5,2.5]])
        np.testing.assert_allclose(r['cache']['S'],[[.05,.2],[-.65,.8]],atol=1e-15)
        self.assertEqual(r['cache']['dZ'][1][0],0.)
    def test_softmax_translation_invariant(self):
        l,g,c=loss_and_grad(self.X,self.y,self.p)
        q={k:v.copy() for k,v in self.p.items()};q['b2']+=10000
        ll,gg,cc=loss_and_grad(self.X,self.y,q)
        self.assertAlmostEqual(l,ll,places=10)
        np.testing.assert_allclose(c['P'],cc['P'],atol=1e-12)
        np.testing.assert_allclose(c['dS'].sum(axis=1),0,atol=1e-15)
    def test_invalid_shapes_and_labels(self):
        for X,y in [(self.X,self.y[:,None]),(self.X[:0],self.y[:0]),(self.X,np.array([0,2,0])),(self.X,self.y.astype(float))]:
            with self.assertRaises(ValueError):loss_and_grad(X,y,self.p)
    def test_finite_difference_does_not_mutate(self):
        old={k:v.copy() for k,v in self.p.items()};finite_difference(self.X,self.y,self.p)
        for k in old:np.testing.assert_array_equal(old[k],self.p[k])
    def test_input_gradient(self):
        _,_,c=loss_and_grad(self.X,self.y,self.p);e=1e-5
        for idx in np.ndindex(self.X.shape):
            plus=self.X.copy();minus=self.X.copy();plus[idx]+=e;minus[idx]-=e
            n=(loss_and_grad(plus,self.y,self.p)[0]-loss_and_grad(minus,self.y,self.p)[0])/(2*e)
            self.assertAlmostEqual(c['dX'][idx],n,places=8)
    def test_training_and_test_holdout(self):
        r=run()['training'];self.assertLess(r['final_train']['loss'],r['initial_train']['loss']*.2)
        self.assertGreater(r['final_test']['accuracy'],.93)
    def test_descent_for_small_step(self):
        l,g,_=loss_and_grad(self.X,self.y,self.p)
        self.assertLess(loss_and_grad(self.X,self.y,sgd_step(self.p,g,1e-3))[0],l)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='outputs/test-result.json');args=ap.parse_args()
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackpropTests))
    obj=dict(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),successful=result.wasSuccessful(),optimization=__import__('sys').flags.optimize,log=stream.getvalue())
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(obj,indent=2)+'\n')
    print(stream.getvalue());raise SystemExit(0 if result.wasSuccessful() else 1)
if __name__=='__main__':main()
