"""Assertions use unittest methods and remain active under python -O."""
import argparse, io, json, math, tempfile, unittest
from pathlib import Path
from experiment import affine, relu, fixed_xor, alternative_xor, collapse, parameter_count, interpolate_square, generate_data, run

class RepresentationTests(unittest.TestCase):
    def test_affine_hand_computation(self):
        self.assertEqual(affine([2,-1],[[1,2],[-1,3]],[.5,-2]),[.5,-7])
    def test_shapes_rejected(self):
        for x,w,b in [([1],[[1,2]],[0]),([1],[[1]],[]),([], [[]],[0])]:
            with self.assertRaises(ValueError): affine(x,w,b)
    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError): affine([math.nan],[[1]],[0])
    def test_relu(self): self.assertEqual([relu(x) for x in [-2,0,3]],[0,0,3])
    def test_fixed_xor_all_rows(self):
        for x,y in [([0,0],0),([0,1],1),([1,0],1),([1,1],0)]:
            with self.subTest(x=x): self.assertEqual(fixed_xor(x)[1],y)
    def test_hidden_hand_values(self): self.assertEqual(fixed_xor([1,1]),([2.,1.],0.))
    def test_removed_activation(self): self.assertEqual([fixed_xor(x,False)[1] for x in [[0,0],[0,1],[1,0],[1,1]]],[2,1,1,0])
    def test_collapse_hand(self):
        w,b=collapse([[1,2],[3,4]],[5,6],[[7,8]],[9]); self.assertEqual(w,[[31,46]]); self.assertEqual(b,[92])
    def test_collapse_invalid(self):
        with self.assertRaises(ValueError): collapse([[1,2]],[0],[[1,2]],[0])
    def test_parameter_counts(self): self.assertEqual(parameter_count([2,2,1]),9); self.assertEqual(parameter_count([2,4,3,1]),31)
    def test_parameter_count_invalid(self):
        for widths in [[2],[2,0,1],[2,1.5],[True,2]]:
            with self.assertRaises(ValueError): parameter_count(widths)
    def test_extension_not_identified(self):
        self.assertEqual(fixed_xor([.5,.5])[1],1); self.assertEqual(alternative_xor([.5,.5]),0)
    def test_scores_not_probabilities(self): self.assertEqual(fixed_xor([2,2])[1],-2)
    def test_square_knots_and_midpoints(self):
        for n in [2,4,8,16]:
            for k in range(n+1): self.assertAlmostEqual(interpolate_square(k/n,n),(k/n)**2)
            for k in range(n):
                x=(k+.5)/n; self.assertAlmostEqual(interpolate_square(x,n)-x*x,1/(4*n*n))
    def test_data_reproducible(self):
        with tempfile.TemporaryDirectory() as d: self.assertEqual(generate_data(d),generate_data(d))
    def test_full_run(self):
        r=run(); self.assertEqual(r['fixed_accuracy'],1); self.assertEqual(r['finite_linear_search']['best_accuracy'],.75); self.assertLess(r['affine_collapse_max_abs_error'],1e-12)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',default='outputs/test-result.json'); a=p.parse_args()
    s=io.StringIO(); r=unittest.TextTestRunner(stream=s,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RepresentationTests)); print(s.getvalue())
    target=Path(a.out); target.parent.mkdir(parents=True,exist_ok=True); target.write_text(json.dumps(dict(tests=r.testsRun,failures=len(r.failures),errors=len(r.errors),successful=r.wasSuccessful(),optimized=not __debug__),indent=2)+'\n')
    raise SystemExit(0 if r.wasSuccessful() else 1)
if __name__=='__main__': main()
