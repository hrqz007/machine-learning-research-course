"""Readable exhaustive shallow binary tree. Numeric X; y is binary or real.

This teaching implementation uses float64 and deterministic ties. It deliberately
omits missing-value routing, sample weights, pruning and efficient prefix scans.
"""
from numbers import Integral
import numpy as np
from common import finite_matrix, require

CRITERIA = ('gini', 'entropy', 'squared_error')
GAIN_EPS = 1e-12

def labels(y, n=None, criterion='gini'):
    require(criterion in CRITERIA, 'unknown criterion')
    y = np.asarray(y, dtype=float)
    require(y.ndim == 1 and y.size > 0, 'y must be a nonempty vector')
    require(n is None or len(y) == n, 'X and y length mismatch')
    require(np.isfinite(y).all(), 'y must be finite')
    if criterion != 'squared_error':
        require(np.isin(y, [0, 1]).all(), 'classification labels must be 0 or 1')
    return y

def impurity(y, criterion='gini'):
    y = labels(y, criterion=criterion)
    if criterion == 'squared_error':
        with np.errstate(over='ignore', invalid='ignore'):
            value = float(np.mean((y-y.mean())**2))
        require(np.isfinite(value), 'regression variance overflow; rescale y')
        return value
    p = float(y.mean())
    if criterion == 'gini':
        return 2*p*(1-p)
    if p == 0 or p == 1:
        return 0.0
    return float(-p*np.log2(p)-(1-p)*np.log2(1-p))

def _integer(value, name, minimum):
    require(isinstance(value, Integral) and not isinstance(value, bool)
            and value >= minimum, f'{name} must be an integer >= {minimum}')

def midpoint(a, b):
    """A separating threshold in [a,b), even at float64 extremes."""
    t = float(a/2 + b/2)
    if t < a or t >= b:
        t = float(a)
    return t

def candidates(X, y, criterion='gini', min_samples_leaf=1):
    """Enumerate every feasible distinct-value split; gain is LOCAL reduction."""
    X = finite_matrix(X); y = labels(y, len(X), criterion)
    _integer(min_samples_leaf, 'min_samples_leaf', 1)
    n = len(y); parent = impurity(y, criterion); result = []
    for j in range(X.shape[1]):
        values = np.unique(X[:, j])
        for a, b in zip(values[:-1], values[1:]):
            t = midpoint(a, b); left = X[:, j] <= t; nl = int(left.sum())
            if min(nl, n-nl) < min_samples_leaf:
                continue
            il = impurity(y[left], criterion); ir = impurity(y[~left], criterion)
            weighted = nl/n*il + (n-nl)/n*ir
            result.append(dict(feature=j, threshold=t, n_left=nl, n_right=n-nl,
                               left_impurity=il, right_impurity=ir,
                               weighted_impurity=float(weighted), gain=float(parent-weighted)))
    return result

def best_split(X, y, criterion='gini', min_samples_leaf=1):
    rows = candidates(X, y, criterion, min_samples_leaf)
    if not rows:
        return None
    # Stable order: lower feature then lower threshold. Gains within 1e-12 tie.
    best = rows[0]
    for row in rows[1:]:
        if row['gain'] > best['gain'] + GAIN_EPS:
            best = row
    return best

class ShallowTree:
    def __init__(self, criterion='gini', max_depth=2, min_samples_leaf=1):
        require(criterion in CRITERIA, 'unknown criterion')
        _integer(max_depth, 'max_depth', 0)
        require(max_depth <= 20, 'teaching implementation supports max_depth <= 20')
        _integer(min_samples_leaf, 'min_samples_leaf', 1)
        self.criterion = criterion; self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)

    def fit(self, X, y):
        X = finite_matrix(X); y = labels(y, len(X), self.criterion)
        self.n_features_in_ = X.shape[1]; self.nodes_ = []
        def grow(ids, depth):
            target = y[ids]; node_id = len(self.nodes_); value = float(target.mean())
            node = dict(id=node_id, depth=depth, n=len(ids), value=value,
                        impurity=impurity(target, self.criterion),
                        train_indices=[int(i) for i in ids], feature=None,
                        threshold=None, gain=0.0, left=None, right=None)
            self.nodes_.append(node)
            if depth >= self.max_depth or len(ids) < 2*self.min_samples_leaf or node['impurity'] <= GAIN_EPS:
                return node_id
            split = best_split(X[ids], target, self.criterion, self.min_samples_leaf)
            if split is None or split['gain'] <= GAIN_EPS:
                return node_id
            node.update(feature=split['feature'], threshold=split['threshold'], gain=split['gain'])
            mask = X[ids, split['feature']] <= split['threshold']
            node['left'] = grow(ids[mask], depth+1)
            node['right'] = grow(ids[~mask], depth+1)
            return node_id
        grow(np.arange(len(X)), 0)
        return self

    def apply(self, X):
        require(hasattr(self, 'nodes_'), 'fit before predicting')
        X = finite_matrix(X)
        require(X.shape[1] == self.n_features_in_, 'feature count mismatch')
        output = []
        for row in X:
            node = self.nodes_[0]
            while node['feature'] is not None:
                branch = 'left' if row[node['feature']] <= node['threshold'] else 'right'
                node = self.nodes_[node[branch]]
            output.append(node['id'])
        return np.asarray(output, dtype=int)

    def predict_proba(self, X):
        require(self.criterion != 'squared_error', 'regression tree has no class probabilities')
        p = np.asarray([self.nodes_[i]['value'] for i in self.apply(X)])
        return np.column_stack([1-p, p])

    def predict(self, X):
        values = np.asarray([self.nodes_[i]['value'] for i in self.apply(X)])
        # Equality p=0.5 selects class 0, the smaller class index.
        return values if self.criterion == 'squared_error' else (values > 0.5).astype(int)

    def state(self):
        require(hasattr(self, 'nodes_'), 'fit before exporting')
        return dict(criterion=self.criterion, max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    n_features=self.n_features_in_, nodes=self.nodes_)
