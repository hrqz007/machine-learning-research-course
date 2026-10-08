"""精确邻域指标；排除自身，固定稳定排序，显式限制k<n/2。"""
import numpy as np
from scipy.spatial.distance import cdist,pdist
from scipy.stats import spearmanr
from common import require,finite_matrix

def ordering(X):
    X=finite_matrix(X);D=cdist(X,X);np.fill_diagonal(D,np.inf);return np.argsort(D,axis=1,kind='stable')

def validate(X,Z,k):
    X=finite_matrix(X);Z=finite_matrix(Z);n=len(X)
    require(len(Z)==n and n>=3,'matching datasets with at least3 samples')
    require(isinstance(k,(int,np.integer)) and not isinstance(k,bool) and 1<=k<n/2,'k must satisfy1<=k<n/2')
    return X,Z

def trust(X,Z,k=10):
    X,Z=validate(X,Z,k);n=len(X);order=ordering(X);near=ordering(Z)[:,:k];ranks=np.empty_like(order);ranks[np.arange(n)[:,None],order]=np.arange(1,n+1)[None,:]
    # 新出现的二维近邻若原空间排名超过k，会按超出多少惩罚。
    penalty=np.maximum(ranks[np.arange(n)[:,None],near]-k,0).sum()
    return float(1-2*penalty/(n*k*(2*n-3*k-1)))

def recall(X,Z,k=10):
    X,Z=validate(X,Z,k);a=ordering(X)[:,:k];b=ordering(Z)[:,:k]
    return float(np.mean([len(set(x)&set(y))/k for x,y in zip(a,b)]))

def distance_rank_correlation(X,Z):
    X=finite_matrix(X);Z=finite_matrix(Z);require(len(X)==len(Z),'matching sample count')
    a=pdist(X);b=pdist(Z);require(np.std(a)>0 and np.std(b)>0,'nonconstant pair distances required')
    return float(spearmanr(a,b).statistic)
