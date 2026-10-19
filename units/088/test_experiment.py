"""使用unittest断言，python -O不会跳过验收。"""
import unittest,tempfile
from pathlib import Path
import torch
from experiment import setup,accumulation,train,rate
class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):setup()
    def test_accumulation(self):
        r=accumulation();self.assertLess(r['correct_max_abs_error'],1e-6);self.assertGreater(r['unscaled_max_abs_error'],1e-3)
    def test_resume(self):
        full,_,_=train('warmup');_,c,_=train('warmup',stop=57)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'c.pt';torch.save(c,p);c=torch.load(p,weights_only=True)
        right,_,_=train('warmup',state=c);wrong,_,_=train('warmup',state=c,weights_only=True)
        diff=lambda m:max(float((p.detach()-q.detach()).abs().max()) for p,q in zip(full.parameters(),m.parameters()))
        self.assertLess(diff(right),1e-7);self.assertGreater(diff(wrong),1e-6)
    def test_schedule(self):
        self.assertAlmostEqual(rate(0),.002);self.assertAlmostEqual(rate(9),.02);self.assertAlmostEqual(rate(10),.02)
        self.assertGreater(rate(119),.002);self.assertLess(rate(119),rate(10))
if __name__=='__main__':unittest.main()
