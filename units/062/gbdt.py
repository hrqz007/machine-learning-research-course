"""Small gradient-boosted regression trees, entirely NumPy.
Squared loss: one half squared error. Logistic loss: raw score is full logit.
Newton logistic leaf correction is explicit; optional gradient-only mode exists.
"""
from dataclasses import dataclass
import numpy as np
from common import finite_matrix, require

def sigmoid(F):
    F=np.asarray(F,dtype=float)
    out=np.empty_like(F);positive=F>=0
    out[positive]=1/(1+np.exp(-F[positive]))
    z=np.exp(F[~positive]);out[~positive]=z/(1+z)
    return out

def loss(y,F,kind):
    if kind=='squared':return float(.5*np.mean((y-F)**2))
    if kind=='logistic':return float(np.mean(np.logaddexp(0,F)-y*F))
    raise ValueError('unknown loss')

def negative_gradient(y,F,kind):
    if kind=='squared':return y-F
    if kind=='logistic':return y-sigmoid(F)
    raise ValueError('unknown loss')

@dataclass
class Node:
    value: float
    indices: np.ndarray
    feature: int=-1
    threshold: float=0.
    left: object=None
    right: object=None

class RegressionTree:
    def __init__(self,max_depth=1,min_samples_leaf=3):
        require(type(max_depth) is int and max_depth>=0,'max_depth must be nonnegative integer')
        require(type(min_samples_leaf) is int and min_samples_leaf>0,'min_samples_leaf must be positive integer')
        self.max_depth=max_depth;self.min_samples_leaf=min_samples_leaf
    def fit(self,X,y):
        X=finite_matrix(X);y=np.asarray(y,dtype=float)
        require(y.shape==(len(X),) and np.isfinite(y).all(),'invalid regression targets')
        self.n_features_in_=X.shape[1];self.leaves_=[]
        def grow(indices,depth):
            target=y[indices];mean=float(target.mean())
            node=Node(mean,indices.copy())
            best=float(np.sum((target-mean)**2));split=None
            if depth<self.max_depth:
                for j in range(X.shape[1]):
                    vals=np.unique(X[indices,j])
                    for a,b in zip(vals[:-1],vals[1:]):
                        t=a/2+b/2
                        if not a<=t<b:t=a
                        mask=X[indices,j]<=t
                        if min(mask.sum(),(~mask).sum())<self.min_samples_leaf:continue
                        left,right=target[mask],target[~mask]
                        sse=float(np.sum((left-left.mean())**2)+np.sum((right-right.mean())**2))
                        if sse<best-1e-14:best=sse;split=(j,float(t),indices[mask],indices[~mask])
            if split is None:self.leaves_.append(node)
            else:
                j,t,left,right=split;node.feature=j;node.threshold=t
                node.left=grow(left,depth+1);node.right=grow(right,depth+1)
            return node
        self.root_=grow(np.arange(len(X)),0)
        return self
    def predict(self,X):
        require(hasattr(self,'root_'),'tree is not fitted')
        X=finite_matrix(X);require(X.shape[1]==self.n_features_in_,'wrong feature count')
        def one(row):
            node=self.root_
            while node.feature!=-1:
                node=node.left if row[node.feature]<=node.threshold else node.right
            return node.value
        return np.array([one(row) for row in X])

class GBDT:
    def __init__(self,loss_kind='squared',n_estimators=40,learning_rate=.1,
                 max_depth=1,min_samples_leaf=3,logistic_update='newton'):
        require(loss_kind in ('squared','logistic'),'unknown loss')
        require(type(n_estimators) is int and n_estimators>0,'n_estimators must be positive integer')
        require(np.isscalar(learning_rate) and np.isfinite(learning_rate)
                and 0<learning_rate<=1,'learning_rate must be in (0,1]')
        require(logistic_update in ('newton','gradient'),'unknown leaf update')
        RegressionTree(max_depth,min_samples_leaf) # validate shared parameters
        self.loss_kind=loss_kind;self.n_estimators=n_estimators;self.learning_rate=float(learning_rate)
        self.max_depth=max_depth;self.min_samples_leaf=min_samples_leaf;self.logistic_update=logistic_update
    def fit(self,X,y):
        X=finite_matrix(X);y=np.asarray(y,dtype=float)
        require(y.shape==(len(X),) and np.isfinite(y).all(),'invalid y')
        if self.loss_kind=='logistic':
            require(np.isin(y,[0,1]).all() and len(np.unique(y))==2,'logistic y must contain both 0 and 1')
            p=y.mean();self.init_=float(np.log(p/(1-p)))
        else:self.init_=float(y.mean())
        self.n_features_in_=X.shape[1];self.trees_=[];self.history_=[]
        F=np.full(len(y),self.init_);self.losses_=[loss(y,F,self.loss_kind)]
        for m in range(self.n_estimators):
            residual=negative_gradient(y,F,self.loss_kind)
            tree=RegressionTree(self.max_depth,self.min_samples_leaf).fit(X,residual)
            if self.loss_kind=='logistic' and self.logistic_update=='newton':
                p=sigmoid(F);hessian=p*(1-p)
                for leaf in tree.leaves_:
                    idx=leaf.indices;den=float(hessian[idx].sum())
                    leaf.value=float(residual[idx].sum()/den) if den>1e-12 else 0.
            direction=tree.predict(X);step=self.learning_rate;previous=self.losses_[-1]
            # Damped line search makes the didactic descent contract explicit.
            # Normal data do not trigger it; record every accepted step.
            for _ in range(40):
                candidate=F+step*direction;current=loss(y,candidate,self.loss_kind)
                if np.isfinite(current) and current<=previous+1e-12:break
                step*=.5
            else:raise RuntimeError('line search failed to reduce training loss')
            self.trees_.append((tree,step));F=candidate;self.losses_.append(current)
            self.history_.append({'round':m+1,'step':step,'pseudo_residual':residual.tolist(),
                'direction':direction.tolist(),'score':F.tolist(),'loss':current,
                'leaf_values':[leaf.value for leaf in tree.leaves_]})
        return self
    def staged_decision_function(self,X):
        require(hasattr(self,'trees_'),'model is not fitted');X=finite_matrix(X)
        require(X.shape[1]==self.n_features_in_,'wrong feature count')
        F=np.full(len(X),self.init_)
        for tree,step in self.trees_:
            F+=step*tree.predict(X);yield F.copy()
    def decision_function(self,X):
        require(hasattr(self,'trees_'),'model is not fitted');X=finite_matrix(X)
        require(X.shape[1]==self.n_features_in_,'wrong feature count')
        F=np.full(len(X),self.init_)
        for stage in self.staged_decision_function(X):F=stage
        return F
    def predict(self,X):
        F=self.decision_function(X)
        return F if self.loss_kind=='squared' else (F>=0).astype(int)
    def predict_proba(self,X):
        require(self.loss_kind=='logistic','probabilities only exist for logistic loss')
        p=sigmoid(self.decision_function(X));return np.column_stack([1-p,p])
