"""Transparent small-data Euclidean kNN, intentionally not an optimized index.
Tied distances keep the training row order. Binary vote ties predict class 0.
Exact-zero distance neighbors receive all weight; at most the selected k are used.
"""
import numpy as np
from common import finite_matrix,require
class KNN:
    def __init__(self,k=5,weights='uniform',task='classification'):
        require(isinstance(k,(int,np.integer)) and not isinstance(k,(bool,np.bool_)) and k>=1,'k must be a positive integer')
        require(weights in ['uniform','distance'],'weights must be uniform or distance')
        require(task in ['classification','regression'],'task must be classification or regression')
        self.k=int(k);self.weights=weights;self.task=task
    def fit(self,X,y):
        X=finite_matrix(X);y=np.asarray(y,dtype=float)
        require(y.ndim==1 and len(y)==len(X),'y must be a vector matching X rows')
        require(np.isfinite(y).all(),'y must be finite')
        require(self.k<=len(X),'k exceeds training rows')
        if self.task=='classification':require(set(y)<={0.,1.},'this teaching classifier supports binary labels 0 and 1')
        self.X=X.copy();self.y=y.copy();return self
    def neighbors(self,Q):
        require(hasattr(self,'X'),'fit must be called before prediction')
        Q=finite_matrix(Q);require(Q.shape[1]==self.X.shape[1],'query feature count differs from training')
        indices=[];distances=[]
        # One query at a time limits temporary memory to O(n*d), rather than O(q*n*d).
        for q in Q:
            d=np.sqrt(np.sum((self.X-q)**2,axis=1))
            require(np.isfinite(d).all(),'distance overflow; rescale input to ordinary numeric ranges')
            ix=np.argsort(d,kind='stable')[:self.k];indices.append(ix);distances.append(d[ix])
        return np.array(indices),np.array(distances)
    def predict_value(self,Q):
        ix,d=self.neighbors(Q);w=np.ones_like(d)
        if self.weights=='distance':
            zero=d==0;rows=zero.any(axis=1)
            w[rows]=zero[rows].astype(float)
            # Normalize with nearest distance: same as 1/d after row normalization.
            w[~rows]=d[~rows,:1]/d[~rows]
        w/=w.sum(axis=1,keepdims=True)
        return np.sum(w*self.y[ix],axis=1)
    def predict_proba(self,Q):
        require(self.task=='classification','predict_proba is only for classification')
        p=self.predict_value(Q);return np.column_stack([1-p,p])
    def predict(self,Q):
        p=self.predict_value(Q)
        return (p>.5).astype(int) if self.task=='classification' else p
