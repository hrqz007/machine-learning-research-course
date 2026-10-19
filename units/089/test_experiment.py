import unittest
import torch
from experiment import axis_probe,dropout_probe,data
class Tests(unittest.TestCase):
    def test_axes(self):
        r=axis_probe();self.assertGreater(r['bn_train_companion_difference'],1)
        self.assertLess(r['ln_companion_difference'],1e-6);self.assertLess(r['bn_eval_companion_difference'],1e-6)
        self.assertGreater(r['bn_eval_shift_difference'],1);self.assertLess(r['ln_shift_difference'],1e-6)
    def test_dropout(self):
        r=dropout_probe();self.assertAlmostEqual(r['train_mean'],3,delta=.03);self.assertAlmostEqual(r['train_var'],3,delta=.05);self.assertEqual(r['eval_max_error'],0)
    def test_training_only_standardization(self):
        x,y,vx,vy=data();self.assertEqual(tuple(x.shape),(64,8));self.assertEqual(len(vy),256)
        self.assertLess(float(x.mean(0).abs().max()),1e-6);self.assertTrue(bool(torch.isfinite(vx).all()))
if __name__=='__main__':unittest.main()
