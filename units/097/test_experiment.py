"""Small invariant tests: unittest assertions remain active under python -O."""

import unittest
import numpy as np
import torch
import experiment as e


class TestMechanism(unittest.TestCase):
    def test_bptt_finite_difference(self):
        x = [0.2, -0.1, 0.4]
        w = 0.5
        eps = 1e-6
        loss, g, h = e.scalar_bptt(x, 0.7, w)
        fd = (e.scalar_bptt(x, 0.7, w + eps)[0] - e.scalar_bptt(x, 0.7, w - eps)[0]) / (
            2 * eps
        )
        self.assertAlmostEqual(g, fd, places=8)

    def test_zero_recurrent_memory(self):
        loss, g, h = e.scalar_bptt([1.0, 0.0], 1.0, 0.0)
        self.assertEqual(h[-1], 0.0)

    def test_reset_batch_independence(self):
        torch.manual_seed(0)
        m = e.Memory()
        x, y = e.memory_data(4, 6, 1)
        with torch.no_grad():
            a = m(x)
            b = torch.cat([m(x[i : i + 1]) for i in range(4)])
        self.assertTrue(torch.allclose(a, b, atol=1e-6))

    def test_truncate_forward_gradient(self):
        torch.manual_seed(0)
        m = e.Memory()
        x, y = e.memory_data(4, 6, 1)
        a = m(x)
        b = m(x, truncate=2)
        self.assertTrue(torch.equal(a, b))
        e.F.cross_entropy(a, y).backward()
        full = m.hidden.weight.grad.clone()
        m.zero_grad()
        e.F.cross_entropy(m(x, truncate=2), y).backward()
        self.assertGreater(float((full - m.hidden.weight.grad).abs().max()), 1e-8)

    def test_parameter_count(self):
        self.assertEqual(sum(p.numel() for p in e.Memory().parameters()), 98)

    def test_first_bit_label(self):
        x, y = e.memory_data(10, 6, 1)
        self.assertTrue(torch.equal((x[:, 0, 0] > 0).long(), y))


if __name__ == "__main__":
    unittest.main()
