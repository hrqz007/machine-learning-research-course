"""概率正确性、置换不识别、数据职责及数值极端值的独立核验。"""
import argparse,copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from latent_project import *
from generate_data import make_data,write_data
from experiment import run

class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.result=run()
    def test_independent_density(self):
        w=[.3,.7];mu=[[-1.,0.],[1.,2.]];cov=[[[1.,.2],[.2,.7]],[[.5,0],[0,1.2]]];x=np.array([[0,0],[1,1],[-5,3]])
        r,lp=responsibilities(x,w,mu,cov)
        p=np.column_stack([w[k]*multivariate_normal.pdf(x,mean=mu[k],cov=cov[k]) for k in range(2)])
        np.testing.assert_allclose(lp,np.log(p.sum(axis=1)),atol=1e-12)
        np.testing.assert_allclose(r,p/p.sum(axis=1)[:,None],atol=1e-12)
    def test_label_permutation(self):
        w=np.array([.3,.7]);mu=np.array([[-1.,0.],[1.,2.]]);cov=np.array([np.eye(2),2*np.eye(2)]);x=[[0,0],[2,1]]
        r,lp=responsibilities(x,w,mu,cov);rp,lpp=responsibilities(x,w[::-1],mu[::-1],cov[::-1])
        np.testing.assert_allclose(lp,lpp);np.testing.assert_allclose(r,rp[:,::-1])
    def test_single_component(self):
        r,lp=responsibilities([[0,0]], [1],[[0,0]],[np.eye(2)])
        self.assertEqual(r[0,0],1);self.assertAlmostEqual(lp[0],-np.log(2*np.pi))
    def test_extreme_tail_finite(self):
        r,lp=responsibilities([[1e4,-1e4]],[.4,.6],[[0,0],[1,1]],[np.eye(2),np.eye(2)])
        self.assertTrue(np.isfinite(lp).all());self.assertAlmostEqual(r.sum(),1,places=8)
    def test_hard_gap_identity(self):
        joint=np.array([[-10,-11],[-3,-3],[-1000,-1000]])
        gap=hard_variational_gap(joint);r=np.exp(joint-logsumexp(joint,axis=1)[:,None])
        np.testing.assert_allclose(gap,-np.log(r.max(axis=1)),atol=1e-12)
        self.assertAlmostEqual(gap[1],np.log(2));self.assertTrue((gap>=0).all())
    def test_matching(self):
        true=[[0,0],[1,2],[-1,3]];got=[[-1,3],[0,0],[1,2]]
        self.assertEqual(match_means(got,true)['matched_mean_rmse'],0)
        with self.assertRaises(ValueError):match_means([[0,0]],true)
    def test_invalid_parameters(self):
        cases=[([-.1,1.1],[[0],[1]],[[[1]],[[1]]]),([.2,.2],[[0],[1]],[[[1]],[[1]]]),([1],[[0]],[[[-1]]]),([1],[[0]],[[[np.nan]]])]
        for w,m,c in cases:
            with self.assertRaises(ValueError):responsibilities([[0]],w,m,c)
        with self.assertRaises(ValueError):responsibilities([[0,1]],[1],[[0]],[[[1]]])
        with self.assertRaises(ValueError):responsibilities([[0,1]],[1],[[0,1]],[[[1,.5],[0,1]]])
    def test_data_sizes_and_disjoint(self):
        d=make_data()['wine'];sets=[set(d[k]['row_id']) for k in d]
        self.assertEqual(sum(map(len,sets)),178);self.assertEqual(len(set.union(*sets)),178)
        self.assertEqual([len(d[k]['x']) for k in ['train','validation','test']],[106,36,36])
    def test_selection_only_validation(self):
        for name in ['synthetic','wine','negative']:
            rows=self.result[name]['candidates'];best=max(rows,key=lambda r:r['validation_log_density'])
            self.assertEqual(best['k'],self.result[name]['selected_k'])
    def test_real_labels_not_training(self):
        d=copy.deepcopy(make_data());d['wine']['test']['cultivar']=[0]*36
        with patch('experiment.make_data',return_value=d):r=run()
        self.assertEqual(r['wine']['candidates'],self.result['wine']['candidates'])
        self.assertEqual(r['wine']['test_log_density'],self.result['wine']['test_log_density'])
        self.assertNotEqual(r['wine']['external_cultivar_ari'],self.result['wine']['external_cultivar_ari'])
    def test_predictive_reproducibility(self):
        rng=np.random.default_rng(1);x=rng.normal(size=(100,2));m=fit_model(x,1)
        a,_=predictive_check(m,x,seed=4,replicates=20);b,_=predictive_check(m,x,seed=4,replicates=20);self.assertEqual(a,b)
        with self.assertRaises(ValueError):predictive_check(m,x,replicates=1)
    def test_nonconstant_statistics(self):
        with self.assertRaises(ValueError):diagnostic_statistics(np.ones((10,2)))
    def test_sampler_moments(self):
        model=fit_model(np.random.default_rng(2).normal(size=(200,2)),1)
        x=draw_mixture(model,20000,np.random.default_rng(4))
        np.testing.assert_allclose(x.mean(axis=0),model.means_[0],atol=.03)
    def test_reproducible_bytes(self):
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
            self.assertEqual(write_data(a),write_data(b));self.assertEqual((Path(a)/'dataset.json').read_bytes(),(Path(b)/'dataset.json').read_bytes())
    def test_conjugate_posterior(self):
        mean,var=normal_mean_posterior([1.,2.,3.])
        self.assertAlmostEqual(var,1/3.25);self.assertAlmostEqual(mean,6/3.25)
        with self.assertRaises(ValueError):normal_mean_posterior([],prior_variance=4)
        with self.assertRaises(ValueError):normal_mean_posterior([1],noise_variance=0)
    def test_shared_parameter_predictive_variance(self):
        r=self.result['conjugate_posterior_predictive']
        self.assertAlmostEqual(r['posterior_variance'],1/600.25)
        self.assertAlmostEqual(r['full_predictive_mean_variance_exact'],1/600.25+1/400)
        self.assertAlmostEqual(r['variance_ratio_exact'],1+400/600.25)
        self.assertLess(abs(r['full_predictive_mean_variance_mc']/r['full_predictive_mean_variance_exact']-1),.08)
        self.assertLess(abs(r['plugin_mean_variance_mc']/r['plugin_mean_variance_exact']-1),.08)
        self.assertGreater(r['full_predictive_mean_variance_mc']/r['plugin_mean_variance_mc'],1.5)
    def test_synthetic_truth_and_null(self):
        r=self.result
        self.assertEqual(r['synthetic']['selected_k'],3);self.assertGreater(r['synthetic']['truth_ari'],.9)
        self.assertEqual(r['negative']['selected_k'],1)
        for name in ['synthetic','wine','negative']:
            self.assertLess(r[name]['responsibility_max_error'],1e-10);self.assertLess(r[name]['log_density_max_error'],1e-10)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='outputs/test-result.json');a=p.parse_args()
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks));out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'tests_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'optimized':not __debug__},indent=2)+'\n');raise SystemExit(0 if r.wasSuccessful() else 1)
