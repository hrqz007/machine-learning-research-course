"""True unittest assertions remain active under python -O."""
import argparse
from pathlib import Path
import json
import time
import unittest
import numpy as np
import experiment as ex

class InitializationTests(unittest.TestCase):
    def test_activation_values_and_zero_convention(self):
        np.testing.assert_array_equal(ex.activate(np.array([-1.,0.,2.]),'relu'),[0,0,2])
        np.testing.assert_array_equal(ex.derivative(np.array([-1.,0.,2.]),'relu'),[0,0,1])
        self.assertEqual(float(ex.activate(np.array([0.]),'sigmoid')[0]),.5)
        self.assertEqual(float(ex.derivative(np.array([0.]),'sigmoid')[0]),.25)
    def test_activation_derivatives_finite_difference(self):
        z=np.array([-2.,-.4,.3,1.2]); eps=1e-6
        for kind in ('relu','sigmoid','tanh'):
            numerical=(ex.activate(z+eps,kind)-ex.activate(z-eps,kind))/(2*eps)
            np.testing.assert_allclose(numerical,ex.derivative(z,kind),rtol=1e-7,atol=1e-9)
    def test_second_moment_is_not_variance(self):
        z=np.array([-3.,-1.,1.,3.]); a=ex.activate(z,'relu'); m=ex.moments(a)
        self.assertEqual(m['second_moment'],2.5)
        self.assertEqual(m['variance'],1.5)
        self.assertAlmostEqual(m['second_moment'],m['variance']+m['mean']**2)
        self.assertEqual(m['second_moment'],ex.moments(z)['second_moment']/2)
    def test_scales_and_paired_random_draws(self):
        self.assertAlmostEqual(ex.weight_std(64,64,'he')**2,2/64)
        self.assertAlmostEqual(ex.weight_std(64,32,'xavier')**2,2/96)
        wh,_=ex.initialize_classifier([4,6,3],87,'he')
        wx,_=ex.initialize_classifier([4,6,3],87,'xavier')
        np.testing.assert_allclose(wh[0]/ex.weight_std(4,6,'he'),wx[0]/ex.weight_std(4,6,'xavier'))
        np.testing.assert_array_equal(wh[-1],wx[-1])
    def test_input_vjp_against_finite_difference(self):
        rng=np.random.default_rng(31); x=rng.normal(size=(2,3)); w=[rng.normal(size=(3,4))*.2,rng.normal(size=(4,2))*.2];b=[np.zeros(4),np.zeros(2)];u=rng.normal(size=(2,2))
        a,z=ex.forward(x,w,b,'tanh');g,_=ex.input_vjp(w,z,u,'tanh')
        num=np.zeros_like(x);eps=1e-6
        for index in np.ndindex(x.shape):
            xp=x.copy();xm=x.copy();xp[index]+=eps;xm[index]-=eps
            num[index]=(np.sum(ex.forward(xp,w,b,'tanh')[0][-1]*u)-np.sum(ex.forward(xm,w,b,'tanh')[0][-1]*u))/(2*eps)
        np.testing.assert_allclose(g[0],num,atol=1e-10,rtol=1e-7)
    def test_classifier_gradient_all_parameters(self):
        x=np.array([[.3,-.8],[1.1,.5],[-.2,.6]]); y=np.array([0,1,2]);w,b=ex.initialize_classifier([2,4,3],731,'he');b[0]+=0.13
        loss,gw,gb,_=ex.loss_and_grad(x,y,w,b);eps=1e-6
        for parameters,gradients in ((w,gw),(b,gb)):
            for p,g in zip(parameters,gradients):
                for index in np.ndindex(p.shape):
                    old=p[index];p[index]=old+eps;lp=ex.loss_and_grad(x,y,w,b)[0]
                    p[index]=old-eps;lm=ex.loss_and_grad(x,y,w,b)[0];p[index]=old
                    self.assertAlmostEqual(g[index],(lp-lm)/(2*eps),delta=2e-7)
    def test_softmax_cross_entropy_extreme_logits(self):
        x=np.array([[1.],[1.]])
        loss,gw,gb,acc=ex.loss_and_grad(x,np.array([0,1]),[np.array([[1000.,-1000.]])],[np.zeros(2)])
        self.assertTrue(np.isfinite(loss));self.assertEqual(loss,1000.0)
        self.assertTrue(np.isfinite(gw[0]).all());self.assertEqual(acc,.5)
    def test_residual_vjp_and_identity_limit(self):
        rng=np.random.default_rng(43);x=rng.normal(size=(2,3));u=rng.normal(size=x.shape);w=[rng.normal(size=(3,3))*.2 for _ in range(3)]
        a,g=ex.residual_vjp(x,w,u,.4);num=np.zeros_like(x);eps=1e-6
        for i in np.ndindex(x.shape):
            xp=x.copy();xm=x.copy();xp[i]+=eps;xm[i]-=eps
            num[i]=(np.sum(ex.residual_vjp(xp,w,u,.4)[0][-1]*u)-np.sum(ex.residual_vjp(xm,w,u,.4)[0][-1]*u))/(2*eps)
        np.testing.assert_allclose(g,num,atol=1e-9,rtol=1e-7)
        aa,gg=ex.residual_vjp(x,w,u,0);np.testing.assert_array_equal(aa[-1],x);np.testing.assert_array_equal(gg,u)
    def test_real_data_provenance_and_standardization(self):
        x,y,d=ex.load_wine(); self.assertEqual(x.shape,(30,13));self.assertEqual(len(set(d['row_ids'])),30)
        np.testing.assert_array_equal(np.bincount(y),[10,10,10])
        np.testing.assert_allclose(x.mean(0),0,atol=1e-13);np.testing.assert_allclose(x.std(0),1,atol=1e-13)
    def test_reproducibility(self):
        self.assertEqual(ex.probe(depth=2,seed=8701),ex.probe(depth=2,seed=8701))
    def test_depth_initialization_contrast(self):
        small=ex.probe(depth=24,mode='small');he=ex.probe(depth=24,mode='he')
        self.assertLess(small['gradient_rms_ratio'],1e-20)
        self.assertGreater(he['gradient_rms_ratio'],small['gradient_rms_ratio']*1e10)
    def test_true_training_updates_and_memorization(self):
        r=ex.train_tiny(seed=8701,mode='he')
        self.assertTrue(r['memorization_pass']);self.assertLess(r['final_loss'],r['history'][0]['loss']/50)
        self.assertGreater(r['history'][0]['first_weight_grad_norm'],0)
    def test_small_initialization_negative_control(self):
        r=ex.train_tiny(seed=8701,mode='small',steps=100)
        self.assertGreater(r['final_loss'],1.0)
    def test_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):ex.weight_std(0,32,'he')
        with self.assertRaises(ValueError):ex.activate(np.ones(2),'unknown')
        with self.assertRaises(ValueError):ex.probe(depth=0)
        with self.assertRaises(ValueError):ex.loss_and_grad(np.ones((2,2)),np.array([0.,1.]),[np.ones((2,3))],[np.zeros(3)])

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',default='outputs/test-result.json');args=parser.parse_args()
    started=time.perf_counter();suite=unittest.defaultTestLoader.loadTestsFromTestCase(InitializationTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    output={'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'successful':result.wasSuccessful(),'optimization':__import__('sys').flags.optimize,'seconds':time.perf_counter()-started}
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(output,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
