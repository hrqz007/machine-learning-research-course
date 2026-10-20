"""Small invariant tests: unittest assertions remain active under python -O."""

import unittest
import numpy as np
import torch
import experiment as e


class TestMechanism(unittest.TestCase):
    def test_lookup_equivalence(self):
        torch.manual_seed(0)
        w = torch.randn(7, 3)
        ids = torch.tensor([2, 2, 5])
        self.assertTrue(
            torch.equal(w[ids], torch.nn.functional.one_hot(ids, 7).float() @ w)
        )

    def test_repeated_gradient(self):
        m = torch.nn.Embedding(5, 3)
        m(torch.tensor([1, 1, 3])).sum().backward()
        self.assertTrue(torch.equal(m.weight.grad[1], torch.full((3,), 2.0)))
        self.assertTrue(torch.equal(m.weight.grad[3], torch.ones(3)))
        self.assertTrue(torch.equal(m.weight.grad[0], torch.zeros(3)))

    def test_window_edges(self):
        got = e.pairs([["a", "b", "c"]], {"a": 0, "b": 1, "c": 2}).tolist()
        self.assertEqual(got, [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]])

    def test_no_cross_sentence(self):
        got = e.pairs([["a", "b"], ["c", "d"]], dict(zip("abcd", range(4)))).tolist()
        self.assertEqual(got, [[0, 1], [1, 0], [2, 3], [3, 2]])

    def test_corpus_count(self):
        self.assertEqual(len(e.make_corpus()), 242)


if __name__ == "__main__":
    unittest.main()
