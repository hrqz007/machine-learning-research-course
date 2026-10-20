"""Small invariant tests: unittest assertions remain active under python -O."""

import unittest
import numpy as np
import torch
import experiment as e


class TestMechanism(unittest.TestCase):
    def test_iou_known(self):
        self.assertAlmostEqual(e.iou([0, 0, 2, 2], [1, 0, 3, 2]), 1 / 3)

    def test_iou_disjoint_identical(self):
        self.assertEqual(e.iou([0, 0, 1, 1], [2, 2, 3, 3]), 0)
        self.assertEqual(e.iou([0, 0, 1, 1], [0, 0, 1, 1]), 1)

    def test_iou_invalid(self):
        with self.assertRaises(ValueError):
            e.iou([0, 0, 0, 1], [0, 0, 1, 1])

    def test_duplicate_detector(self):
        x, y = e.images(4, 11)
        r = e.audit(x, torch.cat([x[:1], x[:1] + 0.001]))
        self.assertEqual(r["exact_count"], 1)
        self.assertEqual(r["near_count_rmse_lt_0_03"], 2)

    def test_frozen_encoder_unchanged(self):
        torch.manual_seed(1)
        m = e.Vision()
        x, y = e.images(8, 1)
        for p in m.encoder.parameters():
            p.requires_grad_(False)
        old = [p.clone() for p in m.encoder.parameters()]
        e.train(m, x, y, 3)
        for a, b in zip(old, m.encoder.parameters()):
            self.assertTrue(torch.equal(a, b))

    def test_independent_data(self):
        x, y = e.images(8, 1)
        z, t = e.images(8, 2)
        self.assertFalse(torch.equal(x, z))
        self.assertEqual(y.sum().item(), 4)


if __name__ == "__main__":
    unittest.main()
