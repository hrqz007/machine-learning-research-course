"""从零PCA：只在fit学习训练均值，行向量为样本，components每行为一个主轴。"""
import numpy as np
from common import require,finite_matrix

def fit(X,k,method='svd'):
    X=finite_matrix(X);n,d=X.shape
    require(n>=2,'at least two training rows')
    require(isinstance(k,(int,np.integer)) and not isinstance(k,bool) and 1<=k<=min(n-1,d),'invalid component count')
    require(method in ['svd','eigh'],'method must be svd or eigh')
    mean=X.mean(axis=0);centered=X-mean
    if method=='svd':
        _,s,vt=np.linalg.svd(centered,full_matrices=False);values=s*s/(n-1);axes=vt
    else:
        values,vectors=np.linalg.eigh(centered.T@centered/(n-1));order=np.argsort(values)[::-1];values=np.maximum(values[order],0);axes=vectors[:,order].T
    # 固定符号便于阅读；数学对象其实是子空间，不要求各库选同一符号。
    for row in axes:
        if row[np.argmax(abs(row))]<0:row*=-1
    total=float(np.sum(centered**2)/(n-1));ratio=values/total if total>0 else np.zeros_like(values)
    return {'mean':mean,'components':axes[:k].copy(),'eigenvalues':values,'explained_variance_ratio':ratio,'n_samples':n,'n_features':d,'method':method}

def transform(model,X):
    X=finite_matrix(X);require(X.shape[1]==model['n_features'],'feature dimension mismatch')
    return (X-model['mean'])@model['components'].T

def inverse_transform(model,Z):
    Z=finite_matrix(Z);require(Z.shape[1]==len(model['components']),'score dimension mismatch')
    return Z@model['components']+model['mean']

def reconstruct(model,X):return inverse_transform(model,transform(model,X))
