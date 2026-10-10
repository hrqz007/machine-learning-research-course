"""使用unittest而不是可被python -O移除的assert。"""
from pathlib import Path
import argparse,json,unittest
import numpy as np
from samplers import *
from experiment import run
class Checks(unittest.TestCase):
    def test_analytic(self):
        m,v=analytic_posterior([.8,1.2,.9,1.4,1.1,.7,1.3,1.])
        self.assertAlmostEqual(m,8.4/8.25);self.assertAlmostEqual(v,1/8.25)
    def test_log_density_ratio(self):
        y=[.8,1.2];m,v=analytic_posterior(y)
        self.assertAlmostEqual(log_posterior(1.3,y)-log_posterior(.2,y),-.5*((1.3-m)**2-(.2-m)**2)/v)
    def test_reproducible_and_rejections(self):
        target=lambda x:-x*x/2
        a,r=metropolis(target,0,12,1000,4);b,_=metropolis(target,0,12,1000,4)
        np.testing.assert_array_equal(a,b);self.assertTrue(np.any(np.diff(a)==0));self.assertLess(r,.3)
    def test_mh_normal(self):
        a,_=metropolis(lambda x:-x*x/2,5,2,30000,16);a=a[2000:]
        self.assertLess(abs(a.mean()),.06);self.assertLess(abs(a.var()-1),.09)
    def test_diagnostics(self):
        rng=np.random.default_rng(8);a=rng.normal(size=(4,6000))
        self.assertLess(split_rhat(a),1.02);self.assertGreater(ess(a[0]),3500)
        bad=a+np.arange(4)[:,None]*3;self.assertGreater(split_rhat(bad),2)
        self.assertEqual(split_rhat(np.ones((2,12))),float('inf'))
    def test_gibbs_covariance(self):
        g=gibbs_gaussian(.8,30000,3)[2000:]
        np.testing.assert_allclose(np.cov(g.T),[[1,.8],[.8,1]],atol=.06)
    def test_invalid_inputs(self):
        for call in [lambda:metropolis(lambda x:0.,0,0,10,1),lambda:analytic_posterior([]),lambda:gibbs_gaussian(1,100,1),lambda:autocorrelation([1]*10),lambda:split_rhat([[1,2]]),lambda:analytic_posterior([1],noise=float('nan')),lambda:analytic_posterior([1],prior_sd=float('inf')),lambda:gibbs_gaussian(.5,3.2,1),lambda:gibbs_gaussian(.5,3,1,start=(0,float('nan'))),lambda:importance_normal([1],2,float('inf'),1)]:
            with self.assertRaises(ValueError):call()
    def test_full_experiment_contrasts(self):
        r=run();self.assertLess(abs(r['mh']['balanced']['mean_error']),.02)
        self.assertLess(r['mh']['balanced']['split_rhat'],1.02)
        self.assertGreater(r['mh']['tiny']['split_rhat'],2)
        self.assertGreater(r['importance_sampling']['2.0']['weight_ess'],r['importance_sampling']['0.2']['weight_ess']*20)
        self.assertGreater(r['gibbs']['0.2']['ess_x'],r['gibbs']['0.95']['ess_x']*5)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='outputs/tests.json');a=p.parse_args()
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps({'tests':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'successful':r.wasSuccessful()},indent=2)+'\n')
    if not r.wasSuccessful():raise SystemExit(1)
