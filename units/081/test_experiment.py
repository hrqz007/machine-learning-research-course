"""非平凡数值/协议/异常输入测试；普通与-O模式使用同一套unittest。"""
import argparse,copy,hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.stats import multivariate_normal
from anomaly import *
from generate_data import make_data,write_data
from experiment import run

class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.result=run()
    def test_gaussian_independent_reference(self):
        x=np.array([[0.,1.],[1.,2.],[2.,-.5],[3.,1.]])
        m=GaussianScore(.2).fit(x);q=np.array([[.5,.8],[9.,-10.]])
        expected=-multivariate_normal.logpdf(q,mean=m.mean_,cov=m.covariance_)
        np.testing.assert_allclose(m.score(q),expected,rtol=1e-12,atol=1e-12)
    def test_gaussian_singular_regularization(self):
        m=GaussianScore(.01).fit(np.ones((10,3)))
        self.assertTrue(np.isfinite(m.score([[1,1,1],[2,2,2]])).all())
        self.assertGreater(m.score([[2,2,2]])[0],m.score([[1,1,1]])[0])
    def test_knn_hand_calculation(self):
        m=KNNRadius(2).fit([[0],[2],[5]])
        np.testing.assert_allclose(m.score([[1],[4]]),[1,2])
    def test_knn_batch_and_permutation(self):
        rng=np.random.default_rng(3);x=rng.normal(size=(30,2));q=rng.normal(size=(600,2));m=KNNRadius(3).fit(x)
        np.testing.assert_allclose(m.score(q),KNNRadius(3).fit(x[::-1]).score(q),atol=1e-14)
    def test_rank_hand(self):
        c=calibrate([5,1,4,2,3],alpha=.34)
        self.assertEqual(c.rank,4);self.assertEqual(c.threshold,4)
        self.assertEqual(c.alarms([3,4,5]).tolist(),[False,False,True])
    def test_exact_rank_exchangeability(self):
        # 固定5个不同分数，每个依次当新正常点，穷举留一秩，误报恰1/5。
        values=np.arange(5.);alarms=[]
        for i in range(5):alarms.append(bool(calibrate(np.delete(values,i),.2).alarms([values[i]])[0]))
        self.assertEqual(sum(alarms),1)
    def test_ties_conservative(self):
        c=calibrate([1]*19,.05);self.assertEqual(c.alarms([1]*5).sum(),0)
    def test_too_few_calibration_points(self):
        c=calibrate([1,2],.05);self.assertTrue(math.isinf(c.threshold));self.assertEqual(c.marginal_bound,0)
        self.assertFalse(c.alarms([1e100])[0])
    def test_invalid_inputs(self):
        for a in [[],[np.nan],[np.inf],[[1,2]]]:
            with self.assertRaises(ValueError):calibrate(a)
        for alpha in [0,1,-.1,np.nan,True]:
            with self.assertRaises(ValueError):calibrate([1,2],alpha)
        for k in [0,-1,2.5,True]:
            with self.assertRaises(ValueError):KNNRadius(k)
        with self.assertRaises(ValueError):KNNRadius(3).fit([[1],[2]])
        with self.assertRaises(ValueError):GaussianScore(0)
        with self.assertRaises(ValueError):GaussianScore().score([[1]])
        with self.assertRaises(ValueError):GaussianScore().fit([[1],[2]]).score([[1,2]])
    def test_library_score_sign(self):
        x=np.random.default_rng(4).normal(size=(50,2));q=np.array([[0,0],[20,20]])
        for kind in ['lof','ocsvm']:
            model=LibraryScore(kind).fit(x)
            np.testing.assert_allclose(model.score(q),-model.estimator_.score_samples(q))
            self.assertGreater(model.score(q)[1],model.score(q)[0])
    def test_counts_and_wilson(self):
        r=evaluate([0,1,2],[2,3],calibrate([0,1,2,3],.4))
        self.assertEqual(r['false_positives'],0);self.assertEqual(r['true_positives'],1)
        lo,hi=wilson(0,43);self.assertEqual(lo,0);self.assertGreater(hi,.05)
        with self.assertRaises(ValueError):wilson(2,1)
    def test_split_disjoint_and_real_negatives(self):
        d=make_data()['digits'];sets=[set(d[k]['row_id']) for k in d]
        self.assertEqual(sum(map(len,sets)),len(set.union(*sets)))
        for name in ['train','calibration','normal_test']:self.assertEqual(set(d[name]['digit']),{0})
        self.assertEqual(set(d['anomaly_test']['digit']),{6});self.assertEqual(len(d['normal_test']['x']),43)
    def test_test_anomalies_cannot_select_threshold(self):
        d=copy.deepcopy(make_data())
        for name in ['synthetic','digits']:
            d[name]['anomaly_test']['x']=(np.asarray(d[name]['anomaly_test']['x'])+10).tolist()
        with patch('experiment.make_data',return_value=d):changed=run()
        for name in ['synthetic','digits']:
            for method in make_models():self.assertEqual(self.result[name]['methods'][method]['threshold'],changed[name]['methods'][method]['threshold'])
    def test_reproducible_bytes(self):
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
            ma=write_data(a);mb=write_data(b);self.assertEqual(ma,mb)
            self.assertEqual((Path(a)/'dataset.json').read_bytes(),(Path(b)/'dataset.json').read_bytes())
    def test_frozen_mechanism(self):
        rows=self.result['synthetic']['methods']
        self.assertLess(rows['gaussian']['recall_by_kind']['bridge'],.1)
        self.assertGreater(rows['lof']['recall_by_kind']['bridge'],.8)
        for v in rows.values():self.assertAlmostEqual(v['marginal_fpr_bound'],.05)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='outputs/test-result.json');a=p.parse_args()
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks));out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'tests_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'optimized':not __debug__},indent=2)+'\n');raise SystemExit(0 if r.wasSuccessful() else 1)
