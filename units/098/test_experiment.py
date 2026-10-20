import unittest
import torch
from experiment import make_data,masked_loss,checks,Memory
class Tests(unittest.TestCase):
 def test_cell_reference(self):self.assertLess(checks()['lstm_cell_max_error'],1e-12)
 def test_masked_loss_padding(self):self.assertLess(checks()['loss_padding_error'],1e-12)
 def test_valid_state_padding(self):self.assertLess(checks()['valid_readout_padding_error'],1e-12)
 def test_wrong_last_state_counterexample(self):self.assertGreater(checks()['wrong_last_readout_change'],1e-5)
 def test_empty_mask(self):
  with self.assertRaises(ValueError):masked_loss(torch.zeros(1,2,3),torch.zeros(1,2,dtype=torch.long),torch.zeros(1,2,dtype=torch.bool))
 def test_reproducibility(self):
  x,y=make_data(10,4,99);xx,yy=make_data(10,4,99);self.assertTrue(torch.equal(x,xx));self.assertTrue(torch.equal(y,yy))
 def test_finite_gradient(self):
  x,y=make_data(8,6,1)
  for kind in ('RNN','GRU','LSTM'):
   m=Memory(kind);m(x).sum().backward();self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
if __name__=='__main__':unittest.main()
