"""训练信息边界、输入拒绝、确定性和错误传播的显式unittest验收。"""
import argparse,json,tempfile,unittest
from pathlib import Path
import numpy as np
from supervision import *
from generate_data import generate,make_data
from experiment import run,select_labels,stress_experiment
class Checks(unittest.TestCase):
    def test_sigmoid_extremes(self):
        np.testing.assert_allclose(sigmoid([-1000.,0.,1000.]),[0.,.5,1.])
    def test_supervised_fit(self):
        X=np.array([[-2.],[-1.],[1.],[2.]]);model=fit_logistic(X,[0,0,1,1]);self.assertTrue(model['converged']);self.assertGreater(evaluate(model,X,[0,0,1,1])['accuracy'],.99)
        self.assertTrue(all(b<=a+1e-12 for a,b in zip(model['loss_trace'],model['loss_trace'][1:])))
    def test_no_repeated_pseudolabels(self):
        fit=self_train([[-1.],[1.]],[0,1],[[-3.],[-2.],[2.],[3.]],threshold=.7)
        ids=[i for h in fit['history'] for i in h['new_ids']];self.assertEqual(len(ids),len(set(ids)));self.assertEqual(int(fit['accepted'].sum()),len(ids))
    def test_no_acceptance(self):
        fit=self_train([[-1.],[1.]],[0,1],[[0.],[0.]],threshold=.99);self.assertFalse(fit['accepted'].any());self.assertEqual(fit['history'][0]['reason'],'no_new_confident_points')
        np.testing.assert_allclose(fit['model']['coef'],fit['models'][0]['coef'])
    def test_hidden_truth_not_training_argument(self):
        import inspect
        self.assertNotIn('y_u',inspect.signature(self_train).parameters);self.assertNotIn('test',inspect.signature(self_train).parameters)
        Xl=[[-1.],[1.]];yl=[0,1];U=[[-2.],[2.]];hidden=np.array([0,1]);a=self_train(Xl,yl,U);hidden[:]=1-hidden;b=self_train(Xl,yl,U)
        np.testing.assert_array_equal(a['model']['coef'],b['model']['coef']);self.assertEqual(a['history'],b['history'])
    def test_weak_abstention_tie_duplicate(self):
        np.testing.assert_array_equal(majority_vote([[-1,-1],[0,1],[1,1],[0,-1]]),[-1,-1,1,0])
        p=independent_label_probability([[1,0]],[.8,.8]);d=independent_label_probability([[1,1,0]],[.8,.8,.8]);self.assertAlmostEqual(p[0],.5);self.assertAlmostEqual(d[0],.8)
    def test_stress(self):
        r=stress_experiment();self.assertEqual(r['accepted'],120);self.assertEqual(r['wrong_accepted'],120);self.assertEqual(r['stages'][-1]['accuracy'],0.)
    def test_invalid_inputs(self):
        cases=[([],[]),([[1.],[2.]],[0,0]),([[float('nan')],[2.]],[0,1]),([[1.],[2.]],[0,2])]
        for X,y in cases:
            with self.assertRaises(ValueError):fit_logistic(X,y)
        for kwargs in [{'l2':0},{'weights':[0.,1.]},{'tol':0},{'max_iter':True}]:
            with self.assertRaises(ValueError):fit_logistic([[-1.],[1.]],[0,1],**kwargs)
        for kwargs in [{'threshold':.5},{'threshold':1.1},{'pseudo_weight':0},{'rounds':0}]:
            with self.assertRaises(ValueError):self_train([[-1.],[1.]],[0,1],[[0.]],**kwargs)
        with self.assertRaises(ValueError):majority_vote([[2]])
        with self.assertRaises(ValueError):independent_label_probability([[1]],[.5])
    def test_label_subsets(self):
        _,truth=make_data(7);a,_,_=select_labels(truth['pool_y'],.02,0,7);b,_,_=select_labels(truth['pool_y'],.1,0,7);self.assertTrue(set(a).issubset(set(b)));self.assertEqual(len(a),4);self.assertEqual(len(b),24)
    def test_reproducibility_and_protocol(self):
        a=run();self.assertEqual(a,run());self.assertEqual(len(a['runs']),45);self.assertEqual(sum(r['status']=='single_observed_class' for r in a['runs']),1)
        json.dumps(a,allow_nan=False)
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:self.assertEqual(generate(a),generate(b))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='outputs/test-result.json');a=p.parse_args();r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks));out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps({'tests_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'optimized':not __debug__},indent=2)+'\n');raise SystemExit(0 if r.wasSuccessful() else 1)
