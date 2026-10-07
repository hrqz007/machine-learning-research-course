"""A small transparent bootstrap ensemble; base trees come from sklearn.

We own bootstrap membership, probability aggregation and out-of-bag masks.
The implementation is intentionally restricted to finite binary 0/1 data.
"""
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from common import finite_matrix,require

def positive_probability(model,X):
    """A bootstrap can contain one class only. Align columns explicitly."""
    q=model.predict_proba(X)
    return q[:,list(model.classes_).index(1)] if 1 in model.classes_ else np.zeros(len(X))

class BootstrapEnsemble:
    def __init__(self,n_estimators=80,max_features=None,min_samples_leaf=3,seed=60060):
        for name,value in [('n_estimators',n_estimators),('min_samples_leaf',min_samples_leaf)]:
            require(isinstance(value,(int,np.integer)) and not isinstance(value,bool) and value>=1,name+' must be a positive integer')
        require(max_features is None or (isinstance(max_features,(int,np.integer)) and not isinstance(max_features,bool) and max_features>=1),'max_features must be None or a positive integer')
        self.n_estimators=int(n_estimators);self.max_features=max_features;self.min_samples_leaf=int(min_samples_leaf);self.seed=seed
    def fit(self,X,y):
        X=finite_matrix(X);y=np.asarray(y)
        require(y.ndim==1 and len(y)==len(X) and np.isin(y,[0,1]).all(),'y must be aligned binary 0/1 labels')
        require(self.max_features is None or self.max_features<=X.shape[1],'max_features exceeds input columns')
        self.n_features_=X.shape[1];self.n_samples_=len(y);rng=np.random.default_rng(self.seed)
        # Draw all samples before tree seeds. Prefixes of this fixed ensemble
        # can be compared without retraining or changing bootstrap membership.
        self.samples_=rng.integers(0,len(y),size=(self.n_estimators,len(y)))
        self.seeds_=rng.integers(0,2**31-1,size=self.n_estimators)
        self.trees_=[];self.train_probabilities_=[];self.oob_mask_=np.zeros((self.n_estimators,len(y)),dtype=bool)
        for b,ids in enumerate(self.samples_):
            tree=DecisionTreeClassifier(min_samples_leaf=self.min_samples_leaf,max_features=self.max_features,random_state=int(self.seeds_[b])).fit(X[ids],y[ids])
            self.trees_.append(tree);self.train_probabilities_.append(positive_probability(tree,X))
            self.oob_mask_[b]=np.bincount(ids,minlength=len(y))==0
        self.train_probabilities_=np.asarray(self.train_probabilities_);return self
    def _prefix(self,B):
        require(hasattr(self,'trees_'),'fit before predict')
        B=self.n_estimators if B is None else B
        require(isinstance(B,(int,np.integer)) and not isinstance(B,bool) and 1<=B<=self.n_estimators,'invalid ensemble prefix')
        return int(B)
    def individual_probabilities(self,X,B=None):
        B=self._prefix(B);X=finite_matrix(X);require(X.shape[1]==self.n_features_,'feature count mismatch')
        return np.array([positive_probability(t,X) for t in self.trees_[:B]])
    def predict_probability(self,X,B=None):return self.individual_probabilities(X,B).mean(axis=0)
    def oob(self,B=None):
        B=self._prefix(B);mask=self.oob_mask_[:B];counts=mask.sum(axis=0);valid=counts>0
        probability=np.full(self.n_samples_,np.nan)
        probability[valid]=(self.train_probabilities_[:B]*mask).sum(axis=0)[valid]/counts[valid]
        return probability,counts
