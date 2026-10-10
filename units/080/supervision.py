"""透明的二分类监督基线、固定伪标签自训练与弱标签聚合。

训练函数只接收允许使用的标签。隐藏真值由experiment单独审计，从不传给伪标签过程。
"""
import numpy as np


def features(value,name='X'):
    X=np.asarray(value,dtype=float)
    if X.ndim!=2 or X.shape[1]<1 or not np.all(np.isfinite(X)):raise ValueError(name+' must be a finite matrix')
    return X.copy()


def labels(value,n):
    y=np.asarray(value)
    if y.shape!=(n,) or not np.all(np.isin(y,[0,1])):raise ValueError('labels must be binary with matching length')
    return y.astype(float)


def sigmoid(z):
    z=np.asarray(z,dtype=float);out=np.empty_like(z);positive=z>=0
    out[positive]=1/(1+np.exp(-z[positive]));exp=np.exp(z[~positive]);out[~positive]=exp/(1+exp)
    return out


def fit_logistic(X,y,weights=None,l2=.05,max_iter=80,tol=1e-10):
    """带L2惩罚的Newton法，回溯保证目标不增；截距不惩罚。
    权重归一化到总和1，所以增加伪标签会改变真实标签的相对权重。
    """
    X=features(X);y=labels(y,len(X))
    if len(X)<2 or len(np.unique(y))!=2:raise ValueError('training needs both classes and at least two rows')
    if not np.isscalar(l2) or not np.isfinite(l2) or l2<=0:raise ValueError('l2 must be positive')
    if isinstance(max_iter,bool) or not isinstance(max_iter,int) or max_iter<1 or not np.isfinite(tol) or tol<=0:raise ValueError('invalid optimizer controls')
    w=np.ones(len(X)) if weights is None else np.asarray(weights,dtype=float)
    if w.shape!=(len(X),) or not np.all(np.isfinite(w)) or np.any(w<=0):raise ValueError('weights must be finite and positive')
    w=w/w.sum();design=np.column_stack([np.ones(len(X)),X]);beta=np.zeros(design.shape[1]);penalty=np.diag([0.]+[l2]*X.shape[1])
    def loss(b):
        z=design@b
        return float(np.sum(w*(np.logaddexp(0,z)-y*z))+.5*b@penalty@b)
    trace=[loss(beta)];converged=False
    for step in range(max_iter):
        p=sigmoid(design@beta);gradient=design.T@(w*(p-y))+penalty@beta
        if np.max(np.abs(gradient))<tol:converged=True;break
        curvature=w*p*(1-p);hessian=design.T@(curvature[:,None]*design)+penalty
        try:direction=np.linalg.solve(hessian+1e-12*np.eye(len(beta)),gradient)
        except np.linalg.LinAlgError as error:raise ValueError('optimizer linear system failed') from error
        rate=1.
        for _ in range(50):
            candidate=beta-rate*direction
            if loss(candidate)<=trace[-1]-1e-4*rate*gradient@direction+1e-15:break
            rate*=.5
        else:raise RuntimeError('line search failed')
        beta=candidate;trace.append(loss(beta))
    # Explicit status allows callers to refuse an unfinished model.
    return {'coef':beta,'converged':converged,'iterations':len(trace)-1,'loss_trace':trace}


def predict_proba(model,X):
    X=features(X);b=np.asarray(model['coef'],dtype=float)
    if b.shape!=(X.shape[1]+1,) or not np.all(np.isfinite(b)):raise ValueError('model dimension mismatch')
    return sigmoid(np.column_stack([np.ones(len(X)),X])@b)


def self_train(X_l,y_l,X_u,threshold=.8,pseudo_weight=.5,rounds=5,l2=.05):
    """已接纳的伪标签固定且不重复计入；全程没有隐藏标签参数。"""
    L=features(X_l);U=features(X_u)
    if U.shape[1]!=L.shape[1]:raise ValueError('labeled and unlabeled dimensions differ')
    y=labels(y_l,len(L))
    if not np.isfinite(threshold) or not .5<threshold<=1:raise ValueError('threshold must be in (0.5,1]')
    if not np.isfinite(pseudo_weight) or pseudo_weight<=0:raise ValueError('pseudo_weight must be positive')
    if isinstance(rounds,bool) or not isinstance(rounds,int) or rounds<1:raise ValueError('rounds must be a positive integer')
    accepted=np.zeros(len(U),dtype=bool);pseudo=np.full(len(U),-1,dtype=int);history=[];models=[]
    model=fit_logistic(L,y,l2=l2)
    if not model['converged']:raise RuntimeError('supervised optimizer did not converge')
    models.append(model)
    for r in range(1,rounds+1):
        prob=predict_proba(model,U);confidence=np.maximum(prob,1-prob)
        new=np.flatnonzero((~accepted)&(confidence>=threshold))
        if not len(new):
            history.append({'round':r,'new_ids':[],'new_labels':[],'new_confidence':[],'accepted_total':int(accepted.sum()),'reason':'no_new_confident_points'});break
        pseudo[new]=(prob[new]>=.5).astype(int);accepted[new]=True
        ix=np.flatnonzero(accepted);trainX=np.vstack([L,U[ix]]);trainy=np.r_[y,pseudo[ix]];weights=np.r_[np.ones(len(L)),np.full(len(ix),pseudo_weight)]
        model=fit_logistic(trainX,trainy,weights,l2=l2)
        if not model['converged']:raise RuntimeError('pseudo-label optimizer did not converge')
        history.append({'round':r,'new_ids':new.tolist(),'new_labels':pseudo[new].tolist(),'new_confidence':confidence[new].tolist(),'accepted_total':int(accepted.sum()),'reason':'refit'})
        models.append(model)
    return {'model':model,'models':models,'history':history,'pseudo_labels':pseudo,'accepted':accepted}


def majority_vote(L):
    L=np.asarray(L)
    if L.ndim!=2 or L.shape[1]<1 or not np.all(np.isin(L,[-1,0,1])):raise ValueError('weak labels must be -1 (abstain), 0, or 1')
    zero=(L==0).sum(axis=1);one=(L==1).sum(axis=1)
    return np.where(one>zero,1,np.where(zero>one,0,-1))


def independent_label_probability(L,accuracies,prior=.5):
    """教学聚合器使用给定、非学习的准确率及条件独立假设。
    还假设二类对称错误、忽略无信息弃权；不声称从无标签数据识别准确率。
    """
    L=np.asarray(L);majority_vote(L);a=np.asarray(accuracies,dtype=float)
    if a.shape!=(L.shape[1],) or not np.all(np.isfinite(a)) or np.any((a<=.5)|(a>=1)):raise ValueError('accuracies must be in (0.5,1)')
    if not np.isfinite(prior) or not 0<prior<1:raise ValueError('prior must be in (0,1)')
    logodds=np.full(len(L),np.log(prior/(1-prior)))
    for j in range(L.shape[1]):
        # 弃权贡献0；如果缺失机制依赖类别，本简化式不成立。
        logodds+=np.where(L[:,j]==-1,0,2*L[:,j]-1)*np.log(a[j]/(1-a[j]))
    return sigmoid(logodds)


def evaluate(model,X,y):
    X=features(X);y=labels(y,len(X))
    if not len(X):raise ValueError('evaluation set is empty')
    p=predict_proba(model,X);pred=p>=.5
    z=np.column_stack([np.ones(len(X)),X])@model['coef']
    return {'accuracy':float(np.mean(pred==y)),'brier':float(np.mean((p-y)**2)),'log_loss':float(np.mean(np.logaddexp(0,z)-y*z))}
