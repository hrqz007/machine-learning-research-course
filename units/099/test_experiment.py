import unittest
import torch
from experiment import attention,run_checks
class Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.c=run_checks()
 def test_reference(self):self.assertLess(self.c['framework_output_error'],1e-12)
 def test_manual_gradients(self):self.assertTrue(all(v<1e-12 for v in self.c['manual_gradient_errors_QKV']))
 def test_finite_difference(self):self.assertLess(self.c['finite_difference_q_error'],1e-8)
 def test_causal_future(self):self.assertLess(self.c['future_perturbation_masked_error'],1e-12)
 def test_unmasked_counterexample(self):self.assertGreater(self.c['future_perturbation_unmasked_change'],.01)
 def test_padding(self):self.assertLess(self.c['padding_append_error'],1e-12)
 def test_probability(self):self.assertLess(self.c['row_sum_error'],1e-12);self.assertEqual(self.c['blocked_weight_max'],0.)
 def test_all_blocked_rejected(self):
  with self.assertRaises(ValueError):attention(torch.ones(1,2),torch.ones(1,2),torch.ones(1,3),torch.zeros(1,1,dtype=torch.bool))
 def test_cross_attention_shape(self):
  out,a=attention(torch.ones(2,5,3),torch.ones(2,7,3),torch.ones(2,7,4));self.assertEqual(tuple(out.shape),(2,5,4));self.assertEqual(tuple(a.shape),(2,5,7))
if __name__=='__main__':unittest.main()
