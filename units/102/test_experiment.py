import unittest,json,math
from lm_common import *
class TestExperiment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parts=corpus();cls.tok=Tokenizer(cls.parts['train'],24);cls.results=json.loads((ROOT/'outputs/results.json').read_text())
    def test_split(self):
        a,b,c=[set(self.parts[k]) for k in ['train','validation','test']]
        self.assertEqual([len(a),len(b),len(c)],[1800,400,776]);self.assertFalse(a&b or a&c or b&c)
    def test_tokenizer(self):
        for d in self.parts['test']:
            ids=self.tok.encode(d);self.assertNotIn(3,ids);self.assertEqual(self.tok.decode(ids),d)
    def test_alignment(self):
        docs=self.parts['train'][:2];x,y=batch(docs,self.tok)
        for i,d in enumerate(docs):
            s=self.tok.encode(d);self.assertEqual(x[i,:len(s)-1].tolist(),s[:-1]);self.assertEqual(y[i,:len(s)-1].tolist(),s[1:])
        self.assertEqual(int((y!=0).sum()),sum(len(self.tok.encode(d))-1 for d in docs))
    def test_parameter_formula(self):
        v=len(self.tok.symbols)
        for h in [12,24,48]:self.assertEqual(sum(p.numel() for p in LM(v,h).parameters()),6*h*h+2*v*h+6*h+v)
    def test_grid(self):
        runs=self.results['runs'];self.assertEqual(len(runs),18)
        self.assertEqual({(r['width'],r['unique_train_docs'],r['seed']) for r in runs},{(w,d,s) for w in [12,24,48] for d in [120,1800] for s in [7,19,31]})
    def test_budgets(self):
        for r in self.results['runs']:
            self.assertEqual(r['steps'],80);self.assertEqual(r['history'][-1]['step'],80)
            self.assertEqual(r['history'][-1]['tokens'],r['processed_target_tokens'])
            self.assertEqual(r['compute_proxy_6ND'],6*r['parameters']*r['processed_target_tokens'])
            self.assertTrue(math.isfinite(r['validation']['nll']))
            self.assertLess(r['validation']['nll'],r['initial_validation']['nll'])
    def test_validation_selection(self):
        key=min(self.results['setting_mean_validation_nll'],key=self.results['setting_mean_validation_nll'].get)
        self.assertEqual(key,str(self.results['selected_setting']['width'])+'-'+str(self.results['selected_setting']['unique_docs']))

if __name__=='__main__':unittest.main()
