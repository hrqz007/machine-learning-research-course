import unittest
import numpy as np
from experiment import corr2d,backward,run
class TestConvolution(unittest.TestCase):
    def test_hand(self):
        x=np.arange(1,10.).reshape(1,1,3,3); w=np.array([1.,0,0,-1]).reshape(1,1,2,2)
        np.testing.assert_array_equal(corr2d(x,w,np.zeros(1)),np.full((1,1,2,2),-4.))
    def test_overlap(self):
        x=np.ones((1,1,3,3));w=np.ones((1,1,2,2))
        dx,dw,db=backward(x,w,np.ones((1,1,2,2)))
        np.testing.assert_array_equal(dx[0,0],[[1,2,1],[2,4,2],[1,2,1]])
        np.testing.assert_array_equal(dw,np.full_like(w,4));self.assertEqual(db[0],4)
    def test_framework(self):
        r=run()
        for a in r['audits']:
            for name,value in a['errors'].items():self.assertLess(value,1e-7,name)
        self.assertLess(r['circular_equivariance_max_error'],1e-12)
        self.assertEqual(r['rf_nonzero_pixels'],[9,25])
if __name__=='__main__':unittest.main()
