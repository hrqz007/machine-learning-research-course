"""PPCA闭式解、后验均值与Gaussian观测似然。"""
import numpy as np
from common import finite_matrix,require

def ppca(X,q):
    X=finite_matrix(X);n,d=X.shape;require(isinstance(q,int) and 0<q<d,'q integer in 1..d-1')
    mu=X.mean(axis=0);centered=X-mu  # 各列减去训练均值，不看审计集。
    covariance=centered.T@centered/n  # 极大似然协方差用n，而非无偏估计n-1。
    values,vectors=np.linalg.eigh(covariance)  # 对称矩阵的正交特征分解。
    order=np.argsort(values)[::-1];values=values[order];vectors=vectors[:,order]
    noise=float(values[q:].mean())  # 被舍弃方向的平均方差估计各向同性噪声。
    require(noise>1e-12,'positive residual variance required for this teaching implementation')
    W=vectors[:,:q]*np.sqrt(np.maximum(values[:q]-noise,0))[None,:]  # 每列缩放自己的主方向。
    return mu,W,noise,values,vectors

def posterior(X,mu,W,noise):
    X=finite_matrix(X);W=finite_matrix(W);mu=np.asarray(mu,dtype=float)
    require(X.shape[1]==W.shape[0] and mu.shape==(X.shape[1],),'compatible dimensions')
    require(noise>0 and np.isfinite(noise) and np.isfinite(mu).all(),'valid mean and noise')
    M=W.T@W+noise*np.eye(W.shape[1])  # q×q后验精度的缩放版本。
    scores=np.linalg.solve(M,W.T@(X-mu).T).T  # 求解方程，避免显式矩阵求逆。
    reconstructed=scores@W.T+mu  # 后验无噪声信号的均值。
    return scores,reconstructed

def log_density(X,mu,C):
    X=finite_matrix(X);C=np.asarray(C,dtype=float);mu=np.asarray(mu,dtype=float)
    require(mu.shape==(X.shape[1],) and np.isfinite(mu).all(),'finite compatible mean')
    require(C.shape==(X.shape[1],X.shape[1]) and np.isfinite(C).all() and np.allclose(C,C.T),'finite symmetric covariance')
    L=np.linalg.cholesky(C)  # 若协方差非正定，这一步明确失败。
    residual=np.linalg.solve(L,(X-mu).T)  # 白化残差，得到Mahalanobis距离。
    return -.5*(X.shape[1]*np.log(2*np.pi)+2*np.log(np.diag(L)).sum()+np.sum(residual**2,axis=0))

def projector(W):
    W=finite_matrix(W)  # 拒绝非矩阵或非有限输入。
    U,s,_=np.linalg.svd(W,full_matrices=False)  # 奇异值识别有效方向而非盲取全部列。
    threshold=np.finfo(float).eps*max(W.shape)*s[0]  # 与矩阵规模相关的数值秩门槛。
    Q=U[:,s>threshold]  # 零矩阵得到零列，秩亏矩阵不增加虚假方向。
    return Q@Q.T  # 投影矩阵消除基的旋转、缩放与符号自由度。
