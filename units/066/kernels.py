"""从零核矩阵与核岭回归。阅读时先核对每个数组的行列含义。"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from common import finite_matrix, require


def rbf(X, Z, gamma=1.):
    """返回形状(len(X), len(Z))的RBF交叉核矩阵。"""
    X = finite_matrix(X)
    Z = finite_matrix(Z)
    require(X.shape[1] == Z.shape[1], 'feature dimension mismatch')
    require(np.isfinite(gamma) and gamma > 0, 'gamma must be positive')
    # ||x-z||² = ||x||² + ||z||² - 2 x^T z。
    # maximum只去除浮点舍入造成的极小负平方距离。
    d2 = np.maximum(
        np.sum(X * X, axis=1)[:, None]
        + np.sum(Z * Z, axis=1)[None, :] - 2 * X @ Z.T,
        0.)
    return np.exp(-gamma * d2)


def polynomial(X, Z, gamma=1., coef0=1., degree=2):
    """用标准非负参数和整数次数构造半正定多项式核。"""
    X = finite_matrix(X)
    Z = finite_matrix(Z)
    require(X.shape[1] == Z.shape[1], 'feature dimension mismatch')
    require(isinstance(degree, (int, np.integer))
            and not isinstance(degree, bool) and degree >= 1,
            'degree must be a positive integer')
    require(np.isfinite(gamma) and gamma >= 0
            and np.isfinite(coef0) and coef0 >= 0,
            'gamma and coef0 must be nonnegative')
    K = (gamma * (X @ Z.T) + coef0) ** degree
    require(np.isfinite(K).all(),
            'kernel overflow; scale inputs or reduce degree')
    return K


def krr_fit(X, y, alpha=.5, gamma=1.):
    """未平均平方误差目标；alpha对应(K + alpha I)中的对角量。"""
    X = finite_matrix(X)
    y = np.asarray(y, dtype=float)
    require(y.shape == (len(X),) and np.isfinite(y).all(),
            'y shape or finite check failed')
    require(np.isfinite(alpha) and alpha > 0, 'alpha must be positive')
    K = rbf(X, X, gamma)
    # alpha > 0使半正定K变为正定，先分解再解方程，不显式求逆。
    a = cho_solve(cho_factor(K + alpha * np.eye(len(X)), lower=True), y)
    return {'X': X.copy(), 'coefficient': a, 'gamma': gamma, 'alpha': alpha}


def krr_predict(model, Z):
    """新点行乘训练系数；核岭一般保留全部训练点。"""
    return rbf(Z, model['X'], model['gamma']) @ model['coefficient']
