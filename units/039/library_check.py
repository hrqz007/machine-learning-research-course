"""Actual fixed-version NumPy/SciPy checks, independent from our forward code."""
from pathlib import Path
from decimal import Decimal as D, localcontext
import json
import math
import numpy as np
import scipy
from scipy.special import logsumexp, log_expit, expit
from scipy.optimize import check_grad
import experiment as e


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def run():
    data, s = e.load_inputs()  # Entire raw contract before library calculations.
    require(np.__version__ == '2.3.5' and scipy.__version__ == '1.17.0', 'fixed library versions required')
    t = np.array([-2000.,-1000.,-40.,-2.,0.,2.,40.,1000.,2000.],dtype=np.float64)
    ours = np.array([e.softplus_scalar(float(z)) for z in t])
    np_answer = np.logaddexp(0., t)
    scipy_answer = -log_expit(-t)
    mat = np.column_stack((np.zeros_like(t),t))
    lse = logsumexp(mat, axis=1, keepdims=False)
    keep = logsumexp(mat, axis=1, keepdims=True)
    # LSE max-shift is overflow-safe but may lose small corrections when max=0.
    # Test broad absolute tolerance, and separately retain the -40 tail test.
    require(np.allclose(ours,np_answer,rtol=1e-15,atol=0),'numpy softplus parity')
    require(np.allclose(ours,scipy_answer,rtol=1e-15,atol=0),'scipy log_expit parity')
    require(np.allclose(ours,lse,rtol=1e-15,atol=5e-16),'scipy logsumexp absolute parity')
    require(keep.shape==(len(t),1) and lse.shape==(len(t),),'axis/keepdims shape')
    decimal_rows=[]
    with localcontext() as ctx:
        ctx.prec=1100
        for z,actual in zip(t,ours):
            dz=D(str(float(z)))
            value=(D(1)+dz.exp()).ln()
            decimal_rows.append({'t':float(z),'decimal_softplus':str(value),'float64':float(actual),
                                 'correctly_rounded_reference_float':float(value),
                                 'absolute_error':str(abs(D(float(actual))-value))})
            expected=float(value)
            require(abs(float(actual)-expected)<=max(math.ulp(expected)*2,0.),'Decimal reference within two ulps')
    design=np.array([[1.,row[1]] for row in data],dtype=np.float64)
    y=np.array([row[2] for row in data],dtype=np.float64)
    th=np.array(s['fd_parameters'],dtype=np.float64)
    def f(v):
        r=design@v-y
        return float(np.dot(r,r)/(2*len(data)))
    def g(v):
        return design.T@(design@v-y)/len(data)
    err=float(check_grad(f,g,th,epsilon=1e-6,direction='all'))
    # check_grad uses FORWARD differences: exact quadratic error=h*diag(H)/2.
    H=design.T@design/len(data)
    expected_err=float(np.linalg.norm(1e-6*np.diag(H)/2))
    require(abs(err-expected_err)<2e-10,'check_grad semantics, not central difference')
    pair_cost=e.Cost()
    own=e.square_forward(data,th.tolist(),pair_cost)
    require(np.allclose(g(th),own['gradient'],atol=1e-15,rtol=0),'independent vectorized paired gradient')
    wrong_residual=(design@th)[:,None]-y
    wrong_grad=design.T@wrong_residual.mean(axis=1)/len(data)
    wrong=e.square_forward(data,th.tolist(),e.Cost(),'broadcast')
    require(wrong_residual.shape==(4,4) and np.allclose(wrong_grad,wrong['gradient'],atol=1e-15,rtol=0),'actual NumPy broadcast semantics')
    stress=[]
    for theta in s['stress_parameters']:
        z=design@np.array(theta)
        signed=(1.-2*y)*z
        lib_losses=-log_expit(-signed)
        lib_local=(1.-2*y)*expit(signed)
        own=e.logistic_forward(data,theta,e.Cost())
        require(np.allclose(lib_losses,[r['loss'] for r in own['rows']],rtol=1e-15,atol=0),'same-row logistic loss')
        require(np.allclose(design.T@lib_local/len(data),own['gradient'],rtol=1e-15,atol=1e-16),'same-row logistic gradient')
        stress.append({'theta':theta,'losses':lib_losses.tolist(),'gradient':(design.T@lib_local/len(data)).tolist()})
    # Max-shift with finite inputs; single term dominates but -1000 row stays finite.
    matrix=np.array([[1000.,1001.,999.],[-1000.,-1001.,-999.],[40.,0.,-40.]],dtype=np.float64)
    shifted=np.array([float(max(a))+math.log(sum(math.exp(float(v-max(a))) for v in a)) for a in matrix])
    library=logsumexp(matrix,axis=1)
    require(np.array_equal(shifted,library),'independent shifted LSE')
    info=np.finfo(np.float64)
    require(info.eps==2.**-52 and info.smallest_subnormal>0 and info.smallest_normal>info.smallest_subnormal,'float64 properties')
    return {'float64_info':{key:float(getattr(info,key)) for key in ('eps','max','smallest_normal','smallest_subnormal')},'status':'passed','numpy':np.__version__,'scipy':scipy.__version__,'dtype':'float64',
            'softplus_inputs':t.tolist(),'numpy_logaddexp':np_answer.tolist(),
            'scipy_negative_log_expit':scipy_answer.tolist(),'scipy_logsumexp_axis1':lse.tolist(),
            'keepdims_shape':list(keep.shape),'decimal_precision':1100,
            'decimal_checks':decimal_rows,'max_softplus_error':float(np.max(abs(ours-np_answer))),
            'check_grad_forward_difference_error':err,'quadratic_predicted_forward_error':expected_err,
            'broadcast_shape':list(wrong_residual.shape),'broadcast_gradient':wrong_grad.tolist(),
            'stress_same_data':stress,'lse_matrix':matrix.tolist(),'lse_result':library.tolist(),
            'limits':['check_grad is forward difference; it is not our central checker.',
                      'Finite-input binary labels only; weighted BCE and nonfinite LSE are not claimed.',
                      'Library agreement is evidence for tested semantics, not proof for all inputs.']}


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);a=p.parse_args()
    r=run()
    if a.output is not None:e.atomic_json(a.output,r)
    print(json.dumps({k:r[k] for k in ('status','numpy','scipy','check_grad_forward_difference_error','max_softplus_error')},sort_keys=True))
