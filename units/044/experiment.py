"""ML044: explicit Poisson log-link GLM, count comparison, offset and dispersion.

Input scope is educational binary64, not an arbitrary-precision GLM library.
All runtime guards are explicit exceptions and remain active under python -O.
"""
from pathlib import Path
import argparse, csv, hashlib, json, math, os, platform, tempfile, warnings
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '1')
import numpy as np
from scipy.special import gammaln
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance
import scipy, sklearn
ROOT = Path(__file__).resolve().parent


def real_array(value, name):
    a = np.asarray(value)
    if a.dtype.kind not in 'iuf' or a.dtype.kind == 'b':
        raise ValueError(name + ' must contain real numbers, not strings, bools, or complex values')
    a = np.asarray(a, dtype=float)
    if not np.all(np.isfinite(a)):
        raise ValueError(name + ' must be finite')
    return a


def validate_data(X, y, exposure):
    X, y, exposure = (real_array(v, n) for v, n in [(X, 'X'), (y, 'y'), (exposure, 'exposure')])
    if X.ndim != 2 or not (1 <= X.shape[0] <= 100000) or not (1 <= X.shape[1] <= 20):
        raise ValueError('X must have 1..100000 rows and 1..20 columns')
    n = len(X)
    if y.shape != (n,) or exposure.shape != (n,):
        raise ValueError('y and exposure must be aligned one-dimensional vectors')
    if np.any(np.abs(X) > 10):
        raise ValueError('educational scope: abs(X) <= 10')
    if np.any((y < 0) | (y > 1e6) | (y != np.floor(y))):
        raise ValueError('count y must be integer in [0, 1000000]')
    if np.any((exposure < 1e-6) | (exposure > 1e6)):
        raise ValueError('exposure must lie in [1e-6, 1e6]')
    return X, y, exposure


def state(beta, X, y, exposure):
    X, y, exposure = validate_data(X, y, exposure)
    beta = real_array(beta, 'beta')
    if beta.shape != (X.shape[1],):
        raise ValueError('beta shape must match X columns')
    eta = X @ beta + np.log(exposure)
    if np.any(~np.isfinite(eta)) or np.any(np.abs(eta) > 30):
        raise ValueError('educational scope: finite log mean eta in [-30, 30]')
    mu = np.exp(eta)
    n = len(y)
    # Retain log(y!), even though optimization may drop this parameter-free term.
    loss = mu - y * eta + gammaln(y + 1)
    local_mu = 1 - y / mu
    local_eta = mu - y
    contributions = local_eta[:, None] * X / n
    H_contrib = np.einsum('i,ij,ik->ijk', mu / n, X, X)
    return {'beta': beta.tolist(), 'eta': eta.tolist(), 'mu': mu.tolist(),
            'loss': loss.tolist(), 'mean_loss': float(np.mean(loss)),
            'd_loss_d_mu': local_mu.tolist(), 'd_mu_d_eta': mu.tolist(),
            'd_loss_d_eta': local_eta.tolist(), 'd_eta_d_beta': X.tolist(),
            'mean_gradient_contributions': contributions.tolist(),
            'gradient': contributions.sum(axis=0).tolist(),
            'hessian_contributions': H_contrib.tolist(),
            'hessian': H_contrib.sum(axis=0).tolist()}


def deviance_rows(y, mu):
    y, mu = real_array(y, 'y'), real_array(mu, 'mu')
    if y.ndim != 1 or mu.shape != y.shape or len(y) == 0:
        raise ValueError('deviance needs aligned nonempty vectors')
    if np.any(y < 0) or np.any(mu <= 0):
        raise ValueError('deviance requires y >= 0 and mu > 0')
    # For y>0 write y log(y/mu)-y+mu = y*(t-log1p(t)), t=(mu-y)/y.
    # A short series avoids cancellation near t=0. Zero y contributes 2 mu.
    result = 2 * mu.copy()
    positive = y > 0
    t = (mu[positive] - y[positive]) / y[positive]
    # Use each row's own branch, so a large ratio in one row cannot make
    # a different nearly saturated row lose its cancellation protection.
    far_left = t < -0.5
    v = np.empty_like(t)
    v[far_left] = (np.log(y[positive][far_left]) - np.log(mu[positive][far_left]) + t[far_left])
    v[~far_left] = t[~far_left] - np.log1p(t[~far_left])
    small = np.abs(t) < 1e-4
    ts = t[small]
    v[small] = ts**2 * (0.5 + ts * (-1/3 + ts * (1/4 + ts * (-1/5 + ts / 6))))
    d = y[positive] * v
    result[positive] = 2 * d
    return result


def fit_poisson(X, y, exposure, tol=1e-9, max_updates=100):
    X, y, exposure = validate_data(X, y, exposure)
    if not isinstance(max_updates, int) or isinstance(max_updates, bool) or not 0 <= max_updates <= 1000:
        raise ValueError('max_updates must be an integer in [0,1000]')
    if isinstance(tol, (bool, np.bool_)) or not np.isscalar(tol) or not np.isfinite(tol) or not 1e-12 <= tol <= 1e-3:
        raise ValueError('tol must be in [1e-12,1e-3]')
    if not np.all(X[:, 0] == 1):
        raise ValueError('fit requires an explicit first column of ones for the intercept')
    if np.linalg.matrix_rank(X) != X.shape[1]:
        raise ValueError('fit requires full column rank')
    if np.sum(y) == 0:
        raise ValueError('all-zero counts have no finite intercept MLE; not fit here')
    beta = np.zeros(X.shape[1]); history = []; status = 'max_updates'
    for k in range(max_updates + 1):
        s = state(beta, X, y, exposure)
        g = np.asarray(s['gradient']); H = np.asarray(s['hessian'])
        norm = float(np.linalg.norm(g))
        history.append({'update': k, 'beta': beta.tolist(), 'mean_loss': s['mean_loss'], 'gradient_l2': norm})
        if norm <= tol:
            status = 'gradient_tolerance'; break
        if k == max_updates:
            break
        step = np.linalg.solve(H, g)
        directional = float(g @ step)
        if not np.isfinite(directional) or directional <= 0:
            status = 'invalid_newton_direction'; break
        accepted = False
        # Armijo and numerical scope rejection; do not clamp eta or alter target.
        for j in range(41):
            alpha = 0.5 ** j
            proposal = beta - alpha * step
            try:
                new = state(proposal, X, y, exposure)
            except ValueError:
                continue
            if new['mean_loss'] <= s['mean_loss'] - 1e-4 * alpha * directional:
                beta = proposal; accepted = True
                history[-1]['accepted_step_size'] = alpha
                break
        if not accepted:
            status = 'line_search_failed'; break
    return {'status': status, 'updates': len(history) - 1, 'state': s, 'history': history}


def metrics(y, pred):
    y, pred = real_array(y, 'y'), real_array(pred, 'pred')
    if y.shape != pred.shape or y.ndim != 1 or len(y) == 0:
        raise ValueError('metrics need aligned nonempty vectors')
    bad = int(np.count_nonzero(pred <= 0))
    return {'count_mse': float(np.mean((pred-y)**2)), 'count_mae': float(np.mean(np.abs(pred-y))),
            'nonpositive_predictions': bad,
            'mean_poisson_deviance': None if bad else float(np.mean(deviance_rows(y, pred))),
            'deviance_reason': 'undefined for nonpositive predictions; no clipping or row exclusion' if bad else None}


def verify_data():
    expected = json.loads((ROOT/'data_integrity.json').read_text())
    for name, digest in expected.items():
        path = ROOT/name
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('data integrity mismatch: ' + name)


def read_data():
    verify_data()
    rows = list(csv.DictReader((ROOT/'data/simulated.csv').open()))
    x = np.array([float(r['x']) for r in rows]); e = np.array([float(r['exposure']) for r in rows])
    truth = np.array([float(r['true_mean']) for r in rows]); train = np.array([r['split']=='train' for r in rows])
    return rows, np.column_stack([np.ones(len(x)), x]), e, truth, train


def scientific_run():
    rows, X, e, truth, train = read_data()
    p = json.loads((ROOT/'data/protocol.json').read_text())
    tiny_rows = list(csv.DictReader((ROOT/'data/tiny.csv').open()))
    Xt = np.array([[1., float(r['x'])] for r in tiny_rows]); yt = np.array([float(r['y']) for r in tiny_rows])
    et = np.array([float(r['exposure']) for r in tiny_rows]); beta = np.zeros(2); ledger = []
    for k in range(p['tiny_updates'] + 1):
        s = state(beta, Xt, yt, et); s['update'] = k; ledger.append(s)
        if k < p['tiny_updates']:
            beta = beta - p['tiny_learning_rate'] * np.asarray(s['gradient'])
    tiny_fit = fit_poisson(Xt, yt, et)
    cases = {}
    for label, field in [('poisson', 'poisson_y'), ('overdispersed', 'overdispersed_y')]:
        y = np.array([float(r[field]) for r in rows]); Xtr, ytr, etr = X[train], y[train], e[train]
        fitted = fit_poisson(Xtr, ytr, etr, p['newton_gradient_tolerance'], p['newton_max_updates'])
        no_offset = fit_poisson(Xtr, ytr, np.ones(len(ytr)), p['newton_gradient_tolerance'], p['newton_max_updates'])
        # This count-scale least-squares model honors exposure as mu=e*(b+w*x).
        linear_beta, _, rank, singular = np.linalg.lstsq(etr[:,None]*Xtr, ytr, rcond=None)
        pred = {'poisson_offset': e*np.exp(X@np.array(fitted['state']['beta'])),
                'poisson_no_offset': np.exp(X@np.array(no_offset['state']['beta'])),
                'linear_exposure': e*(X@linear_beta)}
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            ref = PoissonRegressor(alpha=0.0, fit_intercept=True, solver='newton-cholesky', tol=1e-11, max_iter=100)
            ref.fit(Xtr[:,1:], ytr/etr, sample_weight=etr)
        rb = np.r_[ref.intercept_, ref.coef_]
        ref_g = np.linalg.norm(state(rb, Xtr, ytr, etr)['gradient'])
        mu_train = pred['poisson_offset'][train]
        pearson = float(np.sum((ytr-mu_train)**2/mu_train)/(len(ytr)-X.shape[1]))
        cases[label] = {'poisson_fit': fitted, 'no_offset_fit': no_offset,
            'linear_beta': linear_beta.tolist(), 'linear_rank': int(rank), 'linear_singular_values': singular.tolist(),
            'sklearn_beta': rb.tolist(), 'sklearn_gradient_l2_original_count_objective': float(ref_g),
            'sklearn_max_abs_beta_difference': float(np.max(np.abs(rb-np.array(fitted['state']['beta'])))),
            'sklearn_warnings': [{'category': w.category.__name__, 'message': str(w.message)} for w in captured],
            'pearson_dispersion_train': pearson,
            'raw_train_mean': float(ytr.mean()), 'raw_train_variance_ddof1': float(ytr.var(ddof=1)),
            'train_metrics': {k: metrics(y[train], v[train]) for k,v in pred.items()},
            'test_metrics': {k: metrics(y[~train], v[~train]) for k,v in pred.items()},
            'predictions': {k: v.tolist() for k,v in pred.items()}}
    # Independent distribution and metric functions at all three hand states.
    cross = []
    for s in ledger:
        mu = np.array(s['mu']); d = deviance_rows(yt,mu)
        cross.append({'update': s['update'],
                      'scipy_logpmf_max_abs_error': float(np.max(np.abs(np.array(s['loss'])+poisson.logpmf(yt,mu)))),
                      'sklearn_mean_deviance_error': float(abs(d.mean()-mean_poisson_deviance(yt,mu)))})
    return {'protocol': p, 'tiny_ledger': ledger, 'tiny_fit': tiny_fit,
            'cases': cases, 'distribution_reference': cross,
            'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__}}


def write_json(out, payload):
    out = Path(out).expanduser().resolve()
    if out.suffix != '.json': raise ValueError('output must have .json suffix')
    if out.is_relative_to(ROOT) and not out.is_relative_to(ROOT/'outputs'):
        raise ValueError('inside this unit, generated JSON may only go in outputs/')
    for p in ROOT.rglob('*'):
        if p.is_file() and not p.is_relative_to(ROOT/'outputs') and out.exists() and os.path.samefile(p,out):
            raise ValueError('output aliases a delivered file')
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()
    out.parent.mkdir(parents=True,exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.'+out.name+'.', dir=out.parent)
    try:
        with os.fdopen(fd,'wb') as f: f.write(encoded)
        os.replace(temp,out)
    finally:
        if os.path.exists(temp): os.unlink(temp)
    return str(out)

if __name__ == '__main__':
    a=argparse.ArgumentParser(); a.add_argument('--out', default=str(ROOT/'outputs/result.json'))
    dest=write_json(a.parse_args().out, scientific_run()); print(dest)
