"""有限维GP回归。物理噪声、数值jitter和潜在函数方差显式分开。"""
import numpy as np
from scipy.linalg import cholesky, solve_triangular, cho_solve
from common import finite_matrix, require


def kernel(X, Z, length_scale=1., signal_variance=1.):
    """RBF协方差，返回len(X)乘len(Z)的矩阵。"""
    X = finite_matrix(X)
    Z = finite_matrix(Z)
    require(X.shape[1] == Z.shape[1], 'feature dimension mismatch')
    require(np.isfinite(length_scale) and length_scale > 0,
            'length_scale must be positive')
    require(np.isfinite(signal_variance) and signal_variance > 0,
            'signal_variance must be positive')
    d2 = np.maximum(
        np.sum(X * X, axis=1)[:, None]
        + np.sum(Z * Z, axis=1)[None, :] - 2 * X @ Z.T,
        0.)
    return signal_variance * np.exp(-d2 / (2 * length_scale ** 2))


def fit(X, y, length_scale=1., noise_variance=.0324,
        jitter=1e-10, signal_variance=1.):
    """零均值、已知固定超参数的精确GP，不在此函数中调参。"""
    X = finite_matrix(X)
    y = np.asarray(y, dtype=float)
    require(y.shape == (len(X),) and np.isfinite(y).all(),
            'y must be a finite vector matching X')
    require(np.isfinite(noise_variance) and noise_variance >= 0,
            'noise_variance must be nonnegative')
    require(np.isfinite(jitter) and jitter >= 0,
            'jitter must be nonnegative')
    K = kernel(X, X, length_scale, signal_variance)
    C = K + (noise_variance + jitter) * np.eye(len(X))
    # 不悄悄增加jitter。奇异时保留异常，由调用者选择新值并记录。
    L = cholesky(C, lower=True)
    a = cho_solve((L, True), y)
    # log|C| = 2 sum(log(diag(L)))。避免先求逆或乘很小的行列式。
    fit_term = float(-.5 * y @ a)
    logdet_term = float(-np.log(np.diag(L)).sum())
    constant = float(-len(X) / 2 * np.log(2 * np.pi))
    return {
        'X': X.copy(), 'y': y.copy(), 'L': L, 'a': a,
        'length_scale': length_scale, 'signal_variance': signal_variance,
        'noise_variance': noise_variance, 'jitter': jitter, 'C': C,
        'lml': fit_term + logdet_term + constant,
        'lml_parts': [fit_term, logdet_term, constant]}


def predict(model, Z, observation=False):
    """默认返回潜在均值和完整协方差；观测预测仅另加物理噪声。"""
    Z = finite_matrix(Z)
    cross = kernel(model['X'], Z, model['length_scale'],
                   model['signal_variance'])
    mean = cross.T @ model['a']
    # cross是n乘q；解L V = cross后，V^T V是q乘q信息修正项。
    v = solve_triangular(model['L'], cross, lower=True)
    cov = (kernel(Z, Z, model['length_scale'], model['signal_variance'])
           - v.T @ v)
    cov = (cov + cov.T) / 2
    # 只裁舍入级负对角；不能用裁剪隐藏实质性的负方差错误。
    scale = max(1., model['signal_variance'])
    require(np.min(np.diag(cov)) >= -1e-10 * scale,
            'material negative posterior variance')
    diagonal = np.diag_indices_from(cov)
    cov[diagonal] = np.maximum(cov[diagonal], 0.)
    if observation:
        # jitter是训练数值扰动，不解释成下一次传感器噪声。
        cov = cov + model['noise_variance'] * np.eye(len(Z))
    return mean, cov
