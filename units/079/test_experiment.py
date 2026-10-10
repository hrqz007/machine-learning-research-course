"""独立手算、恒等式、边界与可重复性测试；不依赖会被-O删除的assert。"""
import argparse,json,tempfile,unittest
from pathlib import Path
import numpy as np
from variational import *
from generate_data import models,generate
from experiment import run
class Checks(unittest.TestCase):
    def test_analytic_posterior(self):
        p=posterior(**models()['correlated'])
        np.testing.assert_allclose(p['mean'],[4/9,4/9],atol=1e-14)
        np.testing.assert_allclose(p['covariance'],np.array([[5,-4],[-4,5]])/9,atol=1e-14)
        self.assertAlmostEqual(p['log_evidence'],-.5*(np.log(2*np.pi*2.25)+1/2.25))
    def test_first_two_sweeps(self):
        p=posterior(**models()['correlated']);q=coordinate_mean_field(p['precision'],p['natural_mean'],max_sweeps=2)
        np.testing.assert_allclose(q['trace'][2]['mean'],[.8,.16]);np.testing.assert_allclose(q['trace'][4]['mean'],[.672,.2624]);self.assertFalse(q['converged'])
    def test_decomposition_and_monotonicity(self):
        r=run();self.assertTrue(r['converged']);self.assertLess(r['max_decomposition_error'],1e-12);self.assertGreater(r['min_elbo_increment'],-1e-12);self.assertLess(r['max_mean_error'],1e-10)
        self.assertAlmostEqual(r['kl_q_p'],-.5*np.log(.36));self.assertAlmostEqual(r['variance_ratio'][0],.36)
    def test_arbitrary_full_covariance_identity(self):
        model=models()['correlated'];p=posterior(**model)
        for mean,C in [([2.,-1.],[[2.,.3],[.3,.7]]),([0.,0.],[[1.,0.],[0.,1.]])]:
            self.assertAlmostEqual(elbo(**model,q_mean=mean,q_cov=C)+gaussian_kl(mean,C,p['mean'],p['covariance']),p['log_evidence'],places=12)
    def test_direction_and_predictive(self):
        r=run();self.assertAlmostEqual(r['forward_kl_optimum']['variance'][0],5/9)
        self.assertGreater(r['forward_kl_optimum']['kl_q_p'],r['kl_q_p']);self.assertAlmostEqual(r['linear_functionals']['sum']['exact']['variance'],2/9)
        self.assertAlmostEqual(r['linear_functionals']['sum']['mean_field']['variance'],.4);self.assertAlmostEqual(r['linear_functionals']['difference']['exact']['variance'],2.)
        self.assertAlmostEqual(r['linear_functionals']['future_sum_observation']['exact']['variance'],17/36)
    def test_independent_and_general_dimension(self):
        self.assertAlmostEqual(run()['independent_control']['kl_q_p'],0.,places=12)
        P=np.array([[4.,1.,0.],[1.,3.,.5],[0.,.5,2.]]);h=np.array([2.,-1.,.5]);q=coordinate_mean_field(P,h)
        np.testing.assert_allclose(q['mean'],np.linalg.solve(P,h),atol=1e-10);np.testing.assert_allclose(q['variance'],1/np.diag(P))
    def test_invalid_matrices(self):
        for P in [[[1.,2.],[2.,1.]],[[1.,1.],[0.,1.]],[[1.,0.],[0.,0.]],[[float('nan'),0.],[0.,1.]]]:
            with self.assertRaises(ValueError):coordinate_mean_field(P,[0.,0.])
        for kwargs in [{'tol':0},{'tol':float('nan')},{'max_sweeps':0},{'max_sweeps':True},{'initial_mean':[1.]},{'initial_variance':[-1.,1.]}]:
            with self.assertRaises(ValueError):coordinate_mean_field(np.eye(2),[0.,0.],**kwargs)
    def test_invalid_model(self):
        for change in [{'A':[[1.]]},{'x':[float('inf')]},{'noise_cov':[[0.]]},{'prior_cov':[[1.,2.],[2.,1.]]}]:
            with self.assertRaises(ValueError):posterior(**{**models()['correlated'],**change})
        with self.assertRaises(ValueError):gaussian_kl([0.],[[1.]],[0.,0.],np.eye(2))
        with self.assertRaises(ValueError):linear_summary([0.,0.],np.eye(2),[1.,1.],-1)
    def test_reproducibility(self):
        self.assertEqual(run(),run())
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
            self.assertEqual(generate(a),generate(b))
            self.assertEqual((Path(a)/'correlated.json').read_bytes(),(Path(b)/'correlated.json').read_bytes())
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='outputs/test-result.json');a=p.parse_args();r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks));out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps({'tests_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'optimized':not __debug__},indent=2)+'\n');raise SystemExit(0 if r.wasSuccessful() else 1)
