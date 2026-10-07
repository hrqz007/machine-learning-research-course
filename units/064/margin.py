"""Primal soft-margin optimizer and geometric quantities, without assert gates."""
import numpy as np
from scipy.optimize import minimize
from common import require,finite_matrix

def validate(X,y,C):
    X=finite_matrix(X);y=np.asarray(y,dtype=float)
    require(y.ndim==1 and len(y)==len(X) and set(y)=={-1.,1.},'both labels -1 and +1 required')
    require(np.isfinite(C) and C>0,'C must be finite and positive');return X,y

def objective(X,y,w,b,C):
    X,y=validate(X,y,C);w=np.asarray(w,dtype=float)
    require(w.shape==(X.shape[1],) and np.isfinite(w).all() and np.isfinite(b),'invalid w or b')
    xi=np.maximum(0,1-y*(X@w+b));return float(.5*(w@w)+C*xi.sum())

def solve_primal(X,y,C):
    """Minimize 0.5||w||² + C sum(xi), with explicit linear constraints."""
    X,y=validate(X,y,C);n,d=X.shape
    # z contains d weights, an unpenalized intercept, then n nonnegative slacks.
    A=np.c_[y[:,None]*X,y,np.eye(n)]
    def fun(z):return .5*np.dot(z[:d],z[:d])+C*np.sum(z[d+1:])
    def jac(z):return np.r_[z[:d],0.,np.full(n,C)]
    z0=np.r_[np.zeros(d+1),np.ones(n)]
    sol=minimize(fun,z0,jac=jac,bounds=[(None,None)]*(d+1)+[(0,None)]*n,
        constraints={'type':'ineq','fun':lambda z:A@z-1,'jac':lambda z:A},
        method='SLSQP',options={'ftol':1e-10,'maxiter':2000})
    require(sol.success,'primal optimization failed: '+sol.message)
    w=sol.x[:d];b=float(sol.x[d]);xi=sol.x[d+1:];m=y*(X@w+b)
    require(np.min(m+xi-1)>-1e-7 and np.min(xi)>-1e-8,'primal feasibility failed')
    return {'w':w.tolist(),'b':b,'slack':xi.tolist(),'objective':float(sol.fun),'iterations':int(sol.nit),
            'max_constraint_violation':float(max(0,1-np.min(m+xi))), 'hinge_slack_difference':float(np.max(np.abs(xi-np.maximum(0,1-m))))}

def geometry(X,y,w,b):
    X=finite_matrix(X);y=np.asarray(y);w=np.asarray(w);norm=float(np.linalg.norm(w));require(norm>0,'zero normal has no geometric margin')
    m=y*(X@w+b)
    return {'functional_margin':m.tolist(),'signed_distance':(m/norm).tolist(),'slack':np.maximum(0,1-m).tolist(),
            'norm_w':norm,'canonical_half_width':1/norm,'canonical_full_width':2/norm,'misclassified':int((m<0).sum()),
            'within_margin':int((m<1-1e-7).sum())}
