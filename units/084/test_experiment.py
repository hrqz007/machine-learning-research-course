"""unittest, including shared operands and optimized-interpreter verification."""
import argparse,io,json,math,unittest
from pathlib import Path
from autodiff import Value,Dual,finite_difference
from experiment import composite, composite_float, vector_function, run

class AutodiffTests(unittest.TestCase):
    def test_add(self):
        x,y=Value(2),Value(3); z=x+y; z.backward(); self.assertEqual((z.data,x.grad,y.grad),(5,1,1))
    def test_product(self):
        x,y=Value(2),Value(3); z=x*y; z.backward(); self.assertEqual((z.data,x.grad,y.grad),(6,3,2))
    def test_repeated_operand(self):
        x=Value(3); z=x*x; z.backward(); self.assertEqual(x.grad,6)
    def test_shared_diamond(self):
        x,y=Value(2),Value(3); t=x*y; z=t*t+t; z.backward(); self.assertEqual((z.data,t.grad,x.grad,y.grad),(42,13,39,26))
    def test_three_branches(self):
        x=Value(2); t=x*x; z=t+t+t; z.backward(); self.assertEqual(x.grad,12)
    def test_topology_unique(self):
        x=Value(2); t=x*x; z=t*t+t; order=z.backward(); self.assertEqual(len(order),len(set(order))); self.assertEqual(len(order),4)
    def test_leaf_output(self):
        x=Value(4); x.backward(); self.assertEqual(x.grad,1)
    def test_repeat_resets(self):
        x=Value(3); z=x*x+x; z.backward(); z.backward(); self.assertEqual(x.grad,7)
    def test_seed(self):
        x=Value(3); z=x*x+x; z.backward(2); self.assertEqual(x.grad,14)
    def test_seed_zero(self):
        x=Value(3); z=x*x; z.backward(0); self.assertEqual(x.grad,0)
    def test_readonly_data(self):
        with self.assertRaises(AttributeError): Value(2).data=3
    def test_relu_sides_and_kink(self):
        for p,g in [(-1,0),(0,0),(1,1)]:
            x=Value(p); x.relu().backward(); self.assertEqual(x.grad,g)
    def test_sin_exp_log(self):
        for method,p,g in [('sin',.7,math.cos(.7)),('exp',.7,math.exp(.7)),('log',2,.5)]:
            x=Value(p); getattr(x,method)().backward(); self.assertAlmostEqual(x.grad,g)
    def test_domains(self):
        with self.assertRaises(ValueError): Value(0).log()
        with self.assertRaises(ValueError): Value(float('nan'))
        with self.assertRaises(ValueError): Value(2)**.5
        with self.assertRaises(ValueError): finite_difference(lambda x:x,[1],0)
    def test_operators(self):
        x=Value(2); z=3-x+2*x+x**3; z.backward(); self.assertEqual(z.data,13); self.assertEqual(x.grad,13)
    def test_composite_all_methods(self):
        for a,b in [(.7,-1.2),(1.1,.3),(-.9,.8),(0,-.5),(2,-2)]:
            x,y=Value(a),Value(b); z=composite(x,y); z.backward(); finite=finite_difference(composite_float,[a,b])
            self.assertAlmostEqual(x.grad,finite[0],places=8); self.assertAlmostEqual(y.grad,finite[1],places=8)
            self.assertAlmostEqual(x.grad,composite(Dual(a,1),Dual(b,0)).tangent)
            self.assertAlmostEqual(y.grad,composite(Dual(a,0),Dual(b,1)).tangent)
    def test_vector_products(self):
        x,y=Value(2),Value(3); a,b=vector_function(x,y); z=2*a-b; z.backward(); self.assertEqual([x.grad,y.grad],[2,3])
        self.assertEqual([v.tangent for v in vector_function(Dual(2,1),Dual(3,-1))],[1,3])
    def test_long_graph_no_recursion(self):
        x=Value(1); z=x
        for _ in range(2500): z=z+1
        z.backward(); self.assertEqual((z.data,x.grad),(2501,1))
    def test_full_run(self):
        r=run(); self.assertEqual(r['shared_node']['dx'],39); self.assertLess(max(v['max_finite_abs_error'] for v in r['gradient_checks']),1e-8)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',default='outputs/test-result.json'); a=p.parse_args(); s=io.StringIO()
    r=unittest.TextTestRunner(stream=s,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AutodiffTests)); print(s.getvalue()); target=Path(a.out); target.parent.mkdir(parents=True,exist_ok=True); target.write_text(json.dumps(dict(tests=r.testsRun,failures=len(r.failures),errors=len(r.errors),successful=r.wasSuccessful(),optimized=not __debug__),indent=2)+'\n'); raise SystemExit(0 if r.wasSuccessful() else 1)
if __name__=='__main__':main()
