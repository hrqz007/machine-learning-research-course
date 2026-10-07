"""Transparent small C-SVC dual QP, KKT audits and bias interval recovery."""
import numpy as np
from scipy.optimize import minimize
from common import require,finite_matrix

def polynomial_features(X):
    X=finite_matrix(X);require(X.shape[1]==2,'two columns needed for this explicit map')
    return np.c_[X[:,0]**2,np.sqrt(2)*X[:,0]*X[:,1],X[:,1]**2]

def gram(X,Z=None):
    X=finite_matrix(X);Z=X if Z is None else finite_matrix(Z)
    require(X.shape[1]==Z.shape[1],'matching feature dimensions needed');return (X@Z.T)**2

def validate_gram(K):
    K=np.asarray(K,dtype=float);require(K.ndim==2 and K.shape[0]==K.shape[1] and len(K)>0,'nonempty square Gram required')
    require(np.isfinite(K).all(),'finite Gram required');require(np.allclose(K,K.T,atol=1e-10,rtol=1e-10),'Gram must be symmetric')
    eig=np.linalg.eigvalsh((K+K.T)/2);require(eig.min()>=-1e-8*max(1,np.linalg.norm(K,2)),'Gram must be PSD within tolerance');return K

def recover_bias(alpha,y,K,C,tol=1e-6):
    """Use free SVs, or derive an interval if every coefficient is at a bound."""
    tol=min(tol,C/10)
    alpha=np.asarray(alpha);y=np.asarray(y);g=K@(alpha*y);free=(alpha>tol)&(alpha<C-tol)
    if free.any():return float(np.mean(y[free]-g[free])),{'method':'free_support_vectors','free_count':int(free.sum()),'spread':float(np.ptp(y[free]-g[free]))}
    low=[];high=[]
    for ai,yi,gi in zip(alpha,y,g):
        if ai<=tol:
            (low if yi==1 else high).append(float(yi-gi))
        elif ai>=C-tol:
            (high if yi==1 else low).append(float(yi-gi))
        else:raise RuntimeError('unclassified coefficient')
    lo=max(low) if low else -np.inf;hi=min(high) if high else np.inf
    require(lo<=hi+1e-5,'KKT bias interval is empty')
    b=(lo+hi)/2 if np.isfinite([lo,hi]).all() else (lo if np.isfinite(lo) else hi)
    require(np.isfinite(b),'unbounded bias recovery')
    return float(b),{'method':'bound_interval','free_count':0,'lower':float(lo),'upper':float(hi)}

def solve_dual(K,y,C):
    K=validate_gram(K);y=np.asarray(y,dtype=float);require(y.shape==(len(K),) and set(y)=={-1.,1.},'both -1 and +1 required');require(np.isfinite(C) and C>0,'positive finite C required')
    Q=y[:,None]*K*y[None,:]
    fun=lambda a:.5*a@Q@a-np.sum(a)
    jac=lambda a:Q@a-np.ones(len(a))
    sol=minimize(fun,np.zeros(len(y)),jac=jac,bounds=[(0,C)]*len(y),
        constraints={'type':'eq','fun':lambda a:y@a,'jac':lambda a:y},method='SLSQP',options={'ftol':1e-12,'maxiter':2000})
    require(sol.success,'dual optimization failed: '+sol.message);a=sol.x;b,bias=recover_bias(a,y,K,C)
    scores=K@(a*y)+b;m=y*scores;xi=np.maximum(0,1-m);norm2=float((a*y)@K@(a*y));P=.5*norm2+C*xi.sum();D=a.sum()-.5*norm2
    audit={'primal_value':float(P),'dual_value':float(D),'gap':float(P-D),'equality_residual':float(abs(y@a)),
        'box_violation':float(max(0,-a.min(),a.max()-C)),
        'margin_complementarity':float(np.max(np.abs(a*(m-1+xi)))),
        'slack_complementarity':float(np.max(np.abs((C-a)*xi)))}
    require(audit['gap']>=-1e-7 and audit['gap']<1e-4,'duality gap too large')
    require(max(audit[k] for k in ['equality_residual','box_violation','margin_complementarity','slack_complementarity'])<1e-4,'KKT audit failed')
    return {'alpha':a.tolist(),'b':b,'bias_recovery':bias,'scores':scores.tolist(),'functional_margin':m.tolist(),'slack':xi.tolist(),'norm_squared':norm2,'iterations':int(sol.nit),'audit':audit}
