"""一维Gaussian混合：责任度、约束M步、观测似然。"""
import numpy as np
from scipy.special import logsumexp
from common import require

def log_joint(x,weights,means,variances):
    x=np.asarray(x,dtype=float);w=np.asarray(weights,dtype=float);m=np.asarray(means,dtype=float);v=np.asarray(variances,dtype=float)
    require(x.ndim==1 and len(x)>0 and np.isfinite(x).all(),'finite one-dimensional observations')
    require(w.ndim==m.ndim==v.ndim==1 and len(w)==len(m)==len(v)>0,'aligned components')
    require(np.isfinite(w).all() and np.isfinite(m).all() and np.isfinite(v).all(),'finite parameters')
    require((w>0).all() and np.isclose(w.sum(),1) and (v>0).all(),'positive weights and variances; weights sum to one')
    return np.log(w)[None,:]-.5*(np.log(2*np.pi*v)[None,:]+(x[:,None]-m[None,:])**2/v[None,:])  # n×K个对数联合密度。

def expectation(x,w,m,v):
    joint=log_joint(x,w,m,v)  # 在log空间计算，避免极小密度乘积下溢。
    normalizer=logsumexp(joint,axis=1)  # 每行把K个可能来源加起来。
    responsibility=np.exp(joint-normalizer[:,None])  # 条件概率，每行之和为1。
    return responsibility,float(normalizer.sum())

def maximization(x,r,floor=1e-4):
    x=np.asarray(x,dtype=float);r=np.asarray(r,dtype=float)
    require(np.isfinite(floor) and floor>0,'positive finite variance floor')
    require(x.ndim==1 and r.ndim==2 and r.shape[0]==len(x) and r.shape[1]>0,'aligned x and responsibilities')
    require(np.isfinite(x).all() and np.isfinite(r).all() and (r>=0).all() and np.allclose(r.sum(axis=1),1),'valid responsibilities')
    counts=r.sum(axis=0)  # 有效样本数允许是小数。
    require((counts>1e-14).all(),'numerically empty component; restart required')
    weights=counts/len(x);means=(r*x[:,None]).sum(axis=0)/counts
    variance=(r*(x[:,None]-means[None,:])**2).sum(axis=0)/counts
    return weights,means,np.maximum(variance,floor)  # 对v>=floor的一维精确约束最优解。

def em(x,means,floor=1e-4,max_iter=300,tol=1e-9):
    x=np.asarray(x,dtype=float);m=np.asarray(means,dtype=float);k=len(m)
    require(max_iter>0 and tol>=0,'positive iterations and nonnegative tolerance')
    w=np.ones(k)/k;v=np.full(k,max(float(np.var(x)),floor))  # 相同初始尺度便于比较初始化。
    r,ll=expectation(x,w,m,v);history=[ll];parameters=[{'weights':w.tolist(),'means':m.tolist(),'variances':v.tolist()}];converged=False
    for iteration in range(max_iter):
        w,m,v=maximization(x,r,floor)  # 固定责任度后更新全部参数。
        r,new_ll=expectation(x,w,m,v)  # 用新参数重新计算真实观测似然。
        require(new_ll>=history[-1]-1e-8,'likelihood decreased beyond rounding tolerance')
        history.append(new_ll);parameters.append({'weights':w.tolist(),'means':m.tolist(),'variances':v.tolist()})
        if abs(new_ll-ll)<tol:converged=True;break
        ll=new_ll
    return {'weights':w.tolist(),'means':m.tolist(),'variances':v.tolist(),'history':history,'parameters':parameters,'converged':converged,'iterations':len(history)-1}
