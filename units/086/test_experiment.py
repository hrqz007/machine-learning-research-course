"""PyTorch is mandatory: absence is an error, never a skipped acceptance."""
from pathlib import Path
import argparse, copy, io, json, unittest
import numpy as np
import torch
from experiment import *

class FrameworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
    def test_forward_gradient_and_step_parity(self):
        a=parity_audit();self.assertEqual(a['checked_scalar_parameters'],17)
        self.assertLess(a['loss_abs_error'],1e-12);self.assertLess(a['logits_max_abs'],1e-12)
        for section in ['gradient_max_abs','step_max_abs']:
            for v in a[section].values():self.assertLess(v,1e-12)
    def test_gradient_accumulation(self):
        a=accumulation_audit();self.assertEqual(a['first'],30.)
        self.assertEqual(a['second_without_zero'],60.);self.assertEqual(a['after_reset'],30.)
    def test_zero_grad_none(self):
        model=TwoLayer();opt=torch.optim.SGD(model.parameters(),lr=.1)
        model(torch.ones(2,2,dtype=DTYPE)).sum().backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))
        opt.zero_grad(set_to_none=True);self.assertTrue(all(p.grad is None for p in model.parameters()))
    def test_dataset_dataloader_shapes_and_tail(self):
        loaders=make_loaders(load_data());batches=list(loaders['train'])
        self.assertEqual(sum(len(y) for _,y in batches),323)
        self.assertEqual(batches[-1][0].shape,(3,2));self.assertEqual(batches[-1][1].dtype,torch.long)
        self.assertEqual(batches[-1][0].dtype,DTYPE)
    def test_registered_parameters(self):
        model=TwoLayer(3);self.assertEqual(sum(p.numel() for p in model.parameters()),17)
        self.assertEqual(len(model.state_dict()),4)
    def test_evaluation_does_not_update_parameters_or_grads(self):
        model=TwoLayer();old=copy.deepcopy(model.state_dict());evaluate(model,make_loaders(load_data())['validation'])
        for k,v in model.state_dict().items():self.assertTrue(torch.equal(v,old[k]))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
    def test_eval_no_grad_separate(self):
        a=state_audit();self.assertTrue(a['dropout_eval_all_ones']);self.assertTrue(a['eval_alone_requires_grad'])
        self.assertFalse(a['eval_with_no_grad_requires_grad']);self.assertGreater(a['dropout_train_zero_fraction'],.3)
    def test_detach_breaks_path(self):
        x=torch.tensor(2.,requires_grad=True);z=(x*x).detach()
        self.assertFalse(z.requires_grad)
        with self.assertRaises(RuntimeError):z.backward()
    def test_unequal_batch_weighting(self):
        model=TwoLayer();set_parameters(model,initialize());data=load_data()
        out=evaluate(model,make_loaders(data)['validation']);full=loss_and_grad(*data['validation'],initialize())[0]
        self.assertAlmostEqual(out['loss'],full,places=13)
    def test_training_checkpoint_and_heldout(self):
        a=run()['training'];self.assertLess(a['final']['train']['loss'],.03)
        self.assertGreater(a['final']['test']['accuracy'],.95)
        self.assertLess(a['weighted_validation_vs_full_numpy_abs'],1e-12)
        self.assertGreaterEqual(a['best_epoch'],1);self.assertLessEqual(a['best_epoch'],180)
    def test_input_shape_rejected(self):
        with self.assertRaises(ValueError):ArrayDataset(np.zeros((3,3)),np.zeros(3,dtype=int))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='outputs/test-result.json');args=ap.parse_args()
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FrameworkTests))
    obj=dict(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),successful=result.wasSuccessful(),optimization=__import__('sys').flags.optimize,torch=torch.__version__,log=stream.getvalue())
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(obj,indent=2)+'\n')
    print(stream.getvalue());raise SystemExit(0 if result.wasSuccessful() else 1)
if __name__=='__main__':main()
