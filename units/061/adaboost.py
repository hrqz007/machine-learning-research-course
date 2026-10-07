"""Binary discrete AdaBoost, deliberately small and readable.
Labels are exactly {-1,+1}; every weak rule is an axis-aligned stump.
No sklearn objects are used here. Public validation uses exceptions, not assert.
"""
from dataclasses import dataclass
import numpy as np
from common import finite_matrix, require

@dataclass(frozen=True)
class Stump:
    feature: int       # -1 represents a constant predictor
    threshold: float
    polarity: int      # left predicts polarity, right predicts -polarity
    def predict(self, X):
        if self.feature == -1:
            return np.full(len(X), self.polarity, dtype=int)
        return np.where(X[:, self.feature] <= self.threshold,
                        self.polarity, -self.polarity)

def midpoint(a, b):
    t = a / 2 + b / 2
    return float(t if a <= t < b else a)

def best_stump(X, y, weights):
    """Minimize weighted 0-1 error, not weighted Gini impurity.
    Stable ties: constant -1, constant +1, feature, threshold, polarity -1/+1.
    """
    candidates = [Stump(-1, 0.0, -1), Stump(-1, 0.0, 1)]
    for j in range(X.shape[1]):
        values = np.unique(X[:, j])
        for a, b in zip(values[:-1], values[1:]):
            for polarity in (-1, 1):
                candidates.append(Stump(j, midpoint(a, b), polarity))
    best, error = None, float('inf')
    for stump in candidates:
        current = float(weights @ (stump.predict(X) != y))
        if current < error - 1e-15:
            best, error = stump, current
    return best, error

class AdaBoost:
    def __init__(self, n_estimators=40, learning_rate=1.0):
        require(type(n_estimators) is int and n_estimators > 0,
                'n_estimators must be a positive integer')
        require(np.isscalar(learning_rate) and np.isfinite(learning_rate)
                and 0 < learning_rate <= 1, 'learning_rate must be in (0,1]')
        self.n_estimators = n_estimators
        self.learning_rate = float(learning_rate)

    def fit(self, X, y):
        X = finite_matrix(X)
        y = np.asarray(y, dtype=float)
        require(y.shape == (len(X),), 'y must be a 1D vector matching X')
        require(np.isin(y, [-1, 1]).all(), 'labels must be -1 or +1')
        self.n_features_in_ = X.shape[1]
        self.stumps_, self.alphas_, self.history_ = [], [], []
        self.stop_reason_ = 'max_rounds'
        # log weights protect normalization against multiplicative underflow.
        logw = np.full(len(y), -np.log(len(y)))
        F = np.zeros(len(y))
        self.initial_weights_ = np.exp(logw)
        for m in range(self.n_estimators):
            weights = np.exp(logw)
            stump, error = best_stump(X, y, weights)
            if error >= .5 - 1e-15:
                self.stop_reason_ = 'no_edge'
                break
            # Perfect weak learners have an infinite theoretical coefficient.
            # A documented finite cap is used, then training stops immediately.
            clipped = max(error, 1e-15)
            alpha = self.learning_rate * .5 * np.log((1-clipped)/clipped)
            prediction = stump.predict(X)
            updated = logw - alpha*y*prediction
            maxlog = np.max(updated)
            logZ = maxlog + np.log(np.exp(updated-maxlog).sum())
            logw = updated-logZ
            F += alpha*prediction
            self.stumps_.append(stump)
            self.alphas_.append(float(alpha))
            self.history_.append({'round':m+1, 'error':error, 'alpha':float(alpha),
                'Z':float(np.exp(logZ)), 'weights_before':weights.tolist(),
                'weights_after':np.exp(logw).tolist(), 'score':F.tolist(),
                'train_error':float(np.mean(np.where(F>=0,1,-1)!=y)),
                'exp_loss':float(np.exp(-y*F).mean())})
            if error <= 1e-15:
                self.stop_reason_ = 'perfect_stump'
                break
        return self

    def staged_decision_function(self, X):
        require(hasattr(self,'stumps_'), 'model is not fitted')
        X = finite_matrix(X)
        require(X.shape[1] == self.n_features_in_, 'wrong feature count')
        F = np.zeros(len(X))
        for stump, alpha in zip(self.stumps_, self.alphas_):
            F += alpha*stump.predict(X)
            yield F.copy()

    def decision_function(self, X):
        # Validate even if no useful stump was found.
        require(hasattr(self,'stumps_'), 'model is not fitted')
        X = finite_matrix(X)
        require(X.shape[1] == self.n_features_in_, 'wrong feature count')
        F = np.zeros(len(X))
        for stage in self.staged_decision_function(X): F = stage
        return F

    def predict(self, X):
        # An exactly zero vote is assigned +1, consistently everywhere.
        return np.where(self.decision_function(X) >= 0, 1, -1)
