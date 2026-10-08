"""可读的 Lloyd K-means：确定性平局、D² 初始化、重启和空簇日志。
空簇仅移动没有分配样本的中心，不抢走已分配样本，因此两步下降证明仍成立。
"""
import numpy as np
from common import finite_matrix, require

def positive_int(value, name):
    require(isinstance(value,(int,np.integer)) and not isinstance(value,(bool,np.bool_)) and value>0,name+' must be a positive integer')
    return int(value)

def squared_distances(X, C):
    X=finite_matrix(X);C=finite_matrix(C)
    require(X.shape[1]==C.shape[1],'feature dimensions differ')
    with np.errstate(over='ignore', invalid='ignore'):
        D=np.sum((X[:,None,:]-C[None,:,:])**2,axis=2)
    require(np.isfinite(D).all(),'squared distances overflow; rescale data')
    return D

def initialize(X,k,rng,method):
    """Basic k-means++, not scikit-learn's greedy multiple-trial variant."""
    if method=='random':return X[rng.choice(len(X),k,replace=False)].copy()
    require(method=='k-means++','unknown initialization')
    chosen=[int(rng.integers(len(X)))];C=[X[chosen[0]].copy()]
    while len(C)<k:
        d=squared_distances(X,np.asarray(C)).min(axis=1)
        if d.max()==0:
            # 重复数据可能不足 k 个不同位置；从尚未用过的行选择。
            idx=next(i for i in range(len(X)) if i not in chosen)
        else:
            w=d/d.max();idx=int(rng.choice(len(X),p=w/w.sum()))
        chosen.append(idx);C.append(X[idx].copy())
    return np.asarray(C)

def lloyd(X,centers,max_iter=100,tol=1e-10):
    X=finite_matrix(X);C=finite_matrix(centers).copy()
    positive_int(max_iter,'max_iter');require(np.isfinite(tol) and tol>=0,'tol must be finite and nonnegative')
    D=squared_distances(X,C);labels=D.argmin(axis=1);history=[];converged=False
    for step in range(max_iter):
        before=float(D[np.arange(len(X)),labels].sum());new=C.copy();empty=[]
        counts=np.bincount(labels,minlength=len(C))
        for j in range(len(C)):
            if counts[j]:new[j]=X[labels==j].mean(axis=0)
            else:empty.append(j)
        # 用旧分配的最大残差点依次替换空中心；对固定 labels 的目标无影响。
        order=np.argsort(-D[np.arange(len(X)),labels],kind='stable')
        for rank,j in enumerate(empty):new[j]=X[order[rank%len(X)]]
        updated=squared_distances(X,new)
        after_update=float(updated[np.arange(len(X)),labels].sum())
        next_labels=updated.argmin(axis=1);after_assign=float(updated[np.arange(len(X)),next_labels].sum())
        history.append({'iteration':step+1,'before':before,'after_update':after_update,'after_assign':after_assign,'empty_clusters':empty,'centers_before':C.tolist(),'centers_after':new.tolist(),'labels_before':labels.tolist(),'labels_after':next_labels.tolist()})
        stable=np.array_equal(labels,next_labels);shift=float(np.linalg.norm(new-C))
        C=new;labels=next_labels;D=updated
        # 只用中心变化小来停可能留下非均值中心，因此同时要求分配稳定。
        if stable and shift<=tol:converged=True;break
    return {'centers':C,'labels':labels,'inertia':float(D[np.arange(len(X)),labels].sum()),'history':history,'converged':converged,'iterations':len(history),'effective_k':len(np.unique(labels))}

def fit(X,k,*,init='k-means++',n_init=10,seed=71,max_iter=100,tol=1e-10):
    X=finite_matrix(X);k=positive_int(k,'k');n_init=positive_int(n_init,'n_init');require(k<=len(X),'k exceeds sample count')
    positive_int(max_iter,'max_iter');require(np.isfinite(tol) and tol>=0,'tol must be finite and nonnegative')
    explicit=not isinstance(init,str)
    if explicit:
        C=finite_matrix(init);require(C.shape==(k,X.shape[1]),'initial center shape differs');require(n_init==1,'explicit init requires n_init=1')
    else:require(init in ['random','k-means++'],'unknown initialization')
    runs=[]
    # 第 r 次重启只使用 seed+r；增加重启预算不会改变前面的候选解。
    for r in range(n_init):
        C0=C.copy() if explicit else initialize(X,k,np.random.default_rng(seed+r),init)
        model=lloyd(X,C0,max_iter,tol);model['restart']=r;runs.append(model)
    best=min(runs,key=lambda m:m['inertia']);best['restart_inertias']=[m['inertia'] for m in runs]
    best['restart_converged']=[m['converged'] for m in runs]
    return best

def predict(model,X):return squared_distances(X,model['centers']).argmin(axis=1)
