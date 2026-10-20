"""Small invariant tests: unittest assertions remain active under python -O."""

import unittest
import numpy as np
import torch
import experiment as e


class TestMechanism(unittest.TestCase):
    def test_manual_gradient(self):
        r = e.gradcheck()
        self.assertLess(r["finite_difference_max_error"], 1e-8)
        self.assertLess(r["autograd_max_error"], 1e-12)

    def test_zero_input_gradient(self):
        loss, g = e.manual_loss_grad(np.ones(2), np.zeros((3, 2)), np.ones(3))
        np.testing.assert_array_equal(g, np.zeros(2))
        self.assertAlmostEqual(loss, 0.5)

    def test_one_sample_gradient(self):
        loss, g = e.manual_loss_grad(np.zeros(2), np.array([[1.0, 0.0]]), np.ones(1))
        np.testing.assert_array_equal(g, [-1.0, 0.0])

    def test_balanced_splits(self):
        for n in [128, 300]:
            x, y = e.images(n, 1)
            self.assertEqual(y.sum().item(), n // 2)

    def test_prediction_shape(self):
        m = e.Vision()
        x, y = e.images(7, 1)
        self.assertEqual(tuple(m(x).shape), (7, 2))


if __name__ == "__main__":
    unittest.main()
