"""Independent SciPy basis/QR and actual sklearn Ridge/GridSearchCV references."""
import numpy as np
from scipy.special import eval_legendre
from scipy.linalg import lstsq
from sklearn.base import BaseEstimator,RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

def basis(x,d):return np.column_stack([eval_legendre(j,np.asarray(x,dtype=float)) for j in range(d+1)])
def qr_fit(x,y,c):
    A=basis(x,c['degree']);n=len(x);pen=np.diag([0.]+[1.]*c['degree']);theta,_,rank,_=lstsq(np.vstack([A,np.sqrt(n*c['penalty'])*pen]),np.r_[y,np.zeros(c['degree']+1)],lapack_driver='gelsy')
    if rank!=c['degree']+1:raise RuntimeError('independent QR rank failure')
    return theta

class ReferenceRidge(RegressorMixin,BaseEstimator):
    def __init__(self,degree=1,penalty=0.):self.degree=degree;self.penalty=penalty
    def fit(self,X,y):
        x=np.asarray(X)[:,0];self.n_features_in_=1
        if self.degree==0:self.coefficients_=np.array([np.mean(y)])
        else:
            A=basis(x,self.degree)[:,1:];model=Ridge(alpha=len(x)*self.penalty,fit_intercept=True,solver='svd').fit(A,y);self.coefficients_=np.r_[model.intercept_,model.coef_]
        return self
    def predict(self,X):return basis(np.asarray(X)[:,0],self.degree)@self.coefficients_

def negative_sse(model,X,y):return -float(np.sum((model.predict(X)-y)**2))
def actual_grid_search(x,y,folds,candidates):
    grid=[{'degree':[c['degree']],'penalty':[c['penalty']]} for c in candidates];cv=[(f['training_indices'],f['validation_indices']) for f in folds]
    search=GridSearchCV(ReferenceRidge(),grid,scoring=negative_sse,cv=cv,refit=True,n_jobs=1,error_score='raise').fit(np.asarray(x)[:,None],y)
    # sklearn averages fold SSE; convert back to pooled per-observation MSE.
    return {'selected_index':int(search.best_index_),'pooled_candidate_MSE':(-search.cv_results_['mean_test_score']*len(folds)/len(x)).tolist(),'coefficients':search.best_estimator_.coefficients_.tolist()}
