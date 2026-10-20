"""Small invariant tests: unittest assertions remain active under python -O."""

import unittest
import numpy as np
import torch
import experiment as e


class TestMechanism(unittest.TestCase):
    def test_collapsed_loss(self):
        self.assertAlmostEqual(
            float(e.contrastive(torch.ones(16, 8))), np.log(15), places=6
        )

    def test_temperature_validation(self):
        with self.assertRaises(ValueError):
            e.contrastive(torch.ones(4, 2), 0)
        with self.assertRaises(ValueError):
            e.contrastive(torch.ones(3, 2))

    def test_pair_permutation(self):
        torch.manual_seed(1)
        z = torch.randn(8, 3)
        p = torch.tensor([2, 0, 3, 1])
        zz = torch.cat([z[:4][p], z[4:][p]])
        self.assertAlmostEqual(
            float(e.contrastive(z)), float(e.contrastive(zz)), places=5
        )

    def test_correct_pairs_better(self):
        a = torch.eye(4)
        z = torch.cat([a, a])
        bad = torch.cat([a, a.roll(1, 0)])
        self.assertLess(float(e.contrastive(z)), float(e.contrastive(bad)))

    def test_finite_backward(self):
        z = torch.randn(8, 4, requires_grad=True)
        e.contrastive(z).backward()
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_probe_does_not_mutate_encoder(self):
        m = e.Encoder()
        x, y = e.images(8, 1)
        old = [p.clone() for p in m.parameters()]
        with torch.no_grad():
            h = m(x)
        e.train(torch.nn.Linear(8, 2), h, y, 3)
        for a, b in zip(old, m.parameters()):
            self.assertTrue(torch.equal(a, b))


if __name__ == "__main__":
    unittest.main()
