"""Original, offline two-layer classifier. All conventions are row-batched."""
from pathlib import Path
import json
import numpy as np

PARAM_KEYS = ('W1', 'b1', 'W2', 'b2')

def initialize(seed=8501, hidden=8):
    rng = np.random.default_rng(seed)
    return dict(W1=rng.normal(0, .7, (2, hidden)), b1=np.zeros(hidden),
                W2=rng.normal(0, .5, (hidden, 2)), b2=np.zeros(2))

def validate(X, y, p):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    if X.ndim != 2 or not len(X):
        raise ValueError('X must be a nonempty (batch, features) matrix')
    if y.shape != (len(X),) or not np.issubdtype(y.dtype, np.integer):
        raise ValueError('y must be an integer vector of length batch')
    if set(p) != set(PARAM_KEYS):
        raise ValueError('four named parameters are required')
    if p['W1'].ndim != 2 or p['W2'].ndim != 2:
        raise ValueError('weights must be matrices')
    d, h = p['W1'].shape
    if X.shape[1] != d or p['b1'].shape != (h,) or p['W2'].shape[0] != h:
        raise ValueError('first layer shape mismatch')
    c = p['W2'].shape[1]
    if c < 2 or p['b2'].shape != (c,) or (y < 0).any() or (y >= c).any():
        raise ValueError('class or second layer shape mismatch')
    if not np.isfinite(X).all() or any(not np.isfinite(v).all() for v in p.values()):
        raise ValueError('finite inputs and parameters required')
    return X, y

def loss_and_grad(X, y, p, reduction='mean', activation='tanh'):
    X, y = validate(X, y, p)
    if reduction not in ('mean', 'sum') or activation not in ('tanh', 'relu'):
        raise ValueError('unsupported reduction or activation')
    Z = X @ p['W1'] + p['b1']
    H = np.tanh(Z) if activation == 'tanh' else np.maximum(Z, 0)
    S = H @ p['W2'] + p['b2']
    shifted = S - S.max(axis=1, keepdims=True)
    logP = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    P = np.exp(logP)
    losses = -logP[np.arange(len(y)), y]
    loss = float(losses.mean() if reduction == 'mean' else losses.sum())
    dS = P.copy()
    dS[np.arange(len(y)), y] -= 1
    if reduction == 'mean':
        dS /= len(y)
    dH = dS @ p['W2'].T
    dZ = dH * ((1 - H * H) if activation == 'tanh' else (Z > 0))
    grads = dict(W1=X.T @ dZ, b1=dZ.sum(axis=0),
                 W2=H.T @ dS, b2=dS.sum(axis=0))
    cache = dict(X=X, Z=Z, H=H, S=S, P=P, dS=dS, dH=dH,
                 dZ=dZ, dX=dZ @ p['W1'].T, losses=losses)
    return loss, grads, cache

def sgd_step(p, grads, lr):
    if not np.isfinite(lr) or lr <= 0:
        raise ValueError('positive finite learning rate required')
    # All derivatives have already been evaluated at the SAME old parameters.
    return {k: p[k] - lr * grads[k] for k in PARAM_KEYS}

def finite_difference(X, y, p, epsilon=1e-5, activation='tanh'):
    if epsilon <= 0:
        raise ValueError('epsilon must be positive')
    result = {}
    q = {k: v.copy() for k, v in p.items()}
    for key in PARAM_KEYS:
        g = np.zeros_like(q[key])
        for index in np.ndindex(q[key].shape):
            old = q[key][index]
            q[key][index] = old + epsilon
            plus = loss_and_grad(X, y, q, activation=activation)[0]
            q[key][index] = old - epsilon
            minus = loss_and_grad(X, y, q, activation=activation)[0]
            q[key][index] = old
            g[index] = (plus - minus) / (2 * epsilon)
        result[key] = g
    return result

def load_data(path=None):
    path = Path(path) if path else Path(__file__).parent / 'data/dataset.json'
    obj = json.loads(path.read_text())
    return {name: (np.asarray(v['X'], dtype=np.float64), np.asarray(v['y'], dtype=np.int64))
            for name, v in obj['splits'].items()}

def metrics(X, y, p):
    loss, _, c = loss_and_grad(X, y, p)
    return dict(loss=loss, accuracy=float(np.mean(c['S'].argmax(axis=1) == y)))
