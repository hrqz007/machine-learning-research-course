import unittest
import torch
from experiment import integrity_probe,run
class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def test_integrity(self):
        r=integrity_probe();self.assertTrue(r['correct_pairing_check']);self.assertFalse(r['wrong_pairing_check'])
        self.assertGreater(r['gradient_without_step'],0);self.assertEqual(r['parameter_change_without_step'],0)
    def test_tiny_fit_and_random_memory(self):
        for mode in ['clean','shuffled_labels']:
            r=run(mode);self.assertLess(r['history'][-1]['loss'],.01);self.assertEqual(r['history'][-1]['train_acc'],1)
    def test_no_step(self):
        r=run('no_step');self.assertEqual(r['history'][0]['loss'],r['history'][-1]['loss'])
        self.assertGreater(r['history'][-1]['gradient_norm'],0)
    def test_group_totals(self):
        r=run();self.assertEqual(sum(g['n'] for g in r['groups'].values()),256)
        self.assertAlmostEqual(sum(g['errors'] for g in r['groups'].values())/256,1-r['validation_acc'])
if __name__=='__main__':unittest.main()
