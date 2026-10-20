import unittest
import torch
from experiment import Block,checks,sinusoidal
class Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.c=checks()
 def test_forward_reference(self):self.assertLess(self.c['reference_output_error'],1e-12)
 def test_input_gradient(self):self.assertLess(self.c['reference_input_gradient_error'],1e-12)
 def test_parameter_gradients(self):self.assertLess(self.c['reference_parameter_gradient_error'],1e-11)
 def test_equivariance(self):self.assertLess(self.c['permutation_equivariance_error'],1e-12)
 def test_pool_invariance(self):self.assertLess(self.c['no_position_pool_change'],1e-12)
 def test_positions_break_symmetry(self):self.assertGreater(self.c['with_position_pool_change'],1e-5)
 def test_causal_future(self):self.assertLess(self.c['causal_future_error'],1e-12)
 def test_parameters(self):self.assertEqual(self.c['parameters_measured'],self.c['parameters_formula'])
 def test_position_shape(self):self.assertEqual(tuple(sinusoidal(10,8).shape),(10,8));self.assertTrue(torch.isfinite(sinusoidal(10,8)).all())
 def test_all_blocked(self):
  with self.assertRaises(ValueError):Block()(torch.zeros(1,2,8),torch.zeros(2,2,dtype=torch.bool))
if __name__=='__main__':unittest.main()
