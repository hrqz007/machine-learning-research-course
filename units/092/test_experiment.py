import unittest
import torch
from experiment import data,SmallCNN,count_macs
class TestCNN(unittest.TestCase):
    def test_data(self):
        a,y=data(20,1);b,z=data(20,1)
        self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(y,z));self.assertEqual(a.shape,(20,1,12,12));self.assertEqual(int(y.sum()),10)
    def test_shapes_and_gradient(self):
        torch.set_num_threads(1)
        for residual in [False,True]:
            m=SmallCNN(residual);x,y=data(4,2);out=m(x)
            self.assertEqual(out.shape,(4,2));out.square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
            self.assertEqual(sum(p.numel() for p in m.parameters()),1266)
        self.assertEqual(count_macs(),176272)
    def test_identity_path(self):
        a=SmallCNN(False);b=SmallCNN(True);b.load_state_dict(a.state_dict())
        for model in [a,b]:
            torch.nn.init.zeros_(model.conv2.weight);torch.nn.init.zeros_(model.conv2.bias)
        x,_=data(3,7)
        # Zero residual branch still preserves a nonzero stem signal.
        self.assertGreater(float((a(x)-b(x)).abs().max().detach()),0.)
if __name__=='__main__':unittest.main()
