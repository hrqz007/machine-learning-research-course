"""Unit 021: joint laws, covariance and Gaussian conditioning.

Offline synthetic teaching models. Every public numerical API validates the
whole input before arithmetic or random sampling. Rows are samples/states;
columns are coordinates. No covariance projection, jitter, or silent symmetry
repair is performed. See lab.md for the intentionally conservative float64
range and the explicit probability-rounding audit.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import numbers
import os
from pathlib import Path
import sys
import tempfile
from fractions import Fraction
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
MAX_ENTRIES = 2_000_000
MAX_SAMPLES = 200_000
MAX_DIM = 32
INPUT_MAX = 1e100
INPUT_MIN = float(np.finfo(float).tiny)
PROB_TOL = 8 * np.finfo(float).eps
PD_REL_MIN = 1e-12


def require(condition: bool, message: str) -> None:
    """An executable check retained under python -O."""
    if not condition:
        raise ValueError(message)


def _integer(value: Any, name: str, lower: int, upper: int) -> int:
    require(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)),
            f"{name} must be an integer, not bool")
    require(lower <= value <= upper, f"{name} outside [{lower}, {upper}]")
    return int(value)


def _array(value: Any, name: str, ndim: int) -> np.ndarray:
    """Reject bad leaves before NumPy can coerce a mixed list to a new dtype."""
    count = 0

    def visit(item: Any) -> None:
        nonlocal count
        if isinstance(item, np.ndarray):
            require(item.dtype.kind in 'iuf', f"{name} requires real numeric dtype, not {item.dtype.kind}")
            require(item.size <= MAX_ENTRIES, f"{name} is too large")
            for child in item.flat:
                visit(child)
        elif isinstance(item, (list, tuple)):
            require(len(item) <= MAX_ENTRIES, f"{name} is too large")
            for child in item:
                visit(child)
        else:
            count += 1
            require(count <= MAX_ENTRIES, f"{name} is too large")
            require(isinstance(item, numbers.Real) and not isinstance(item, (bool, np.bool_)),
                    f"{name} requires real numbers; bool, strings, complex and objects are rejected")
            try:
                number = float(item)
            except (OverflowError, ValueError, TypeError) as exc:
                raise ValueError(f"{name} cannot be represented as float64") from exc
            require(math.isfinite(number), f"{name} must be finite")
            require(abs(number) <= INPUT_MAX, f"{name} magnitude exceeds implementation range")
            require(number == 0 or abs(number) >= INPUT_MIN,
                    f"{name} nonzero magnitude is below implementation range")
            require(number != 0 or item == 0, f"{name} conversion underflow")

    visit(value)
    try:
        result = np.array(value, dtype=float, copy=True)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must have a rectangular shape") from exc
    require(result.ndim == ndim, f"{name} must have {ndim} dimensions")
    require(result.size > 0, f"{name} cannot be empty")
    return result


def _finite(value: Any, name: str) -> Any:
    require(bool(np.all(np.isfinite(value))), f"nonfinite intermediate: {name}")
    return value


def _multiply(a: Any, b: Any, name: str) -> np.ndarray:
    """NumPy elementwise products expose overflow and meaningful underflow."""
    try:
        with np.errstate(over='raise', under='raise', invalid='raise', divide='raise'):
            result = np.multiply(a, b)
    except FloatingPointError as exc:
        raise ValueError(f"unrepresentable intermediate: {name}") from exc
    return _finite(result, name)


def _divide(a: Any, b: Any, name: str) -> np.ndarray:
    try:
        with np.errstate(over='raise', under='raise', invalid='raise', divide='raise'):
            result = np.divide(a, b)
    except FloatingPointError as exc:
        raise ValueError(f"unrepresentable intermediate: {name}") from exc
    return _finite(result, name)


def _sum(values: Any, name: str) -> float:
    try:
        result = math.fsum(np.ravel(values))
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"unrepresentable sum: {name}") from exc
    require(math.isfinite(result), f"nonfinite sum: {name}")
    return result


def _dot(a: np.ndarray, b: np.ndarray, name: str) -> float:
    return _sum(_multiply(a, b, name), name)


def _matmul(a: np.ndarray, b: np.ndarray, name: str) -> np.ndarray:
    require(a.ndim == b.ndim == 2 and a.shape[1] == b.shape[0], f"shape mismatch: {name}")
    out = np.empty((a.shape[0], b.shape[1]))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = _dot(a[i], b[:, j], name)
    return _finite(out, name)


def _gram(rows: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Compute both symmetric entries from the same centered product sum."""
    d = rows.shape[1]
    result = np.empty((d, d))
    for i in range(d):
        for j in range(i, d):
            products = _multiply(rows[:, i], rows[:, j], 'Gram products')
            if weights is not None:
                products = _multiply(products, weights, 'weighted Gram products')
            result[i, j] = result[j, i] = _sum(products, 'Gram sum')
    require(bool(np.all(np.diag(result) >= 0)), 'negative diagonal after Gram computation')
    return result


def probabilities(values: Any) -> tuple[np.ndarray, dict[str, float]]:
    """Validate a near-unit float representation, then explicitly canonicalize.

    Only totals within 8 float64 eps of one are accepted. The returned audit
    records the input total and maximum adjustment. Negative/above-one weights
    are never accepted; arbitrary weights are never silently normalized.
    Reusing the returned probabilities obeys the same tolerance.
    """
    p = _array(values, 'probabilities', 1)
    require(bool(np.all((p >= 0) & (p <= 1))), 'each probability must lie in [0,1]')
    total = _sum(p, 'probability total')
    require(abs(total - 1) <= PROB_TOL, 'probability total differs from 1 beyond rounding tolerance')
    used = _divide(p, total, 'probability canonicalization')
    return used, {'input_total': total, 'tolerance': float(PROB_TOL),
                  'maximum_adjustment': float(np.max(np.abs(used - p))),
                  'used_total': _sum(used, 'canonical probability total')}


def _states(values: Any, weights: Any) -> tuple[np.ndarray, np.ndarray, dict]:
    data = _array(values, 'states', 2)
    require(data.shape[1] <= MAX_DIM, 'too many coordinates')
    p, audit = probabilities(weights)
    require(data.shape[0] == p.size, 'one probability is needed per state')
    return data, p, audit


def finite_moments(values: Any, weights: Any) -> dict[str, Any]:
    """Exact finite-model formula in float64; never a sample covariance estimator."""
    data, p, audit = _states(values, weights)
    # Full state and probability validation has already completed. Preserve an
    # exactly constant coordinate, rather than creating rounding deviations in
    # a weighted sum. Zero-weight and excluded rows are never skipped here.
    constant = np.all(data == data[0], axis=0)
    mean = np.array([data[0, j] if constant[j] else _dot(data[:, j], p, 'weighted mean')
                     for j in range(data.shape[1])])
    centered = _finite(data - mean, 'centered states')
    centered[:, constant] = 0.0
    covariance = _gram(centered, p)
    return {'mean': mean, 'covariance': covariance, 'probabilities': p, 'probability_audit': audit}


def conditional_finite(values: Any, weights: Any, given_column: int, given_value: Any) -> dict[str, Any]:
    """Condition on an exactly matching finite state coordinate.

    All states, including excluded and zero-mass states, are validated first.
    Returned moments include the now-constant conditioned coordinate.
    """
    data, p, audit = _states(values, weights)
    full = finite_moments(data, p)  # validates products on the entire support before selection
    column = _integer(given_column, 'given_column', 0, data.shape[1] - 1)
    query = _array([given_value], 'given_value', 1)[0]
    mask = data[:, column] == query
    mass = _sum(p[mask], 'condition mass')
    require(mass > 0, 'conditioning event has zero model probability')
    selected = _divide(p[mask], mass, 'conditional probabilities')
    result = finite_moments(data[mask], selected)
    result.update({'states': data[mask], 'event_probability': mass,
                   'input_probability_audit': audit, 'unconditional_mean': full['mean']})
    return result


def total_variance(values: Any, weights: Any, target_column: int, group_column: int) -> dict[str, Any]:
    data, p, audit = _states(values, weights)
    target = _integer(target_column, 'target_column', 0, data.shape[1] - 1)
    group = _integer(group_column, 'group_column', 0, data.shape[1] - 1)
    overall = finite_moments(data, p)
    entries = []
    within_terms, between_terms = [], []
    for label in np.unique(data[:, group]):
        mask = data[:, group] == label
        mass = _sum(p[mask], 'group mass')
        if mass == 0:
            continue  # every state was already validated; a null group has no conditional law here
        conditional = conditional_finite(data, p, group, label)
        mean = conditional['mean'][target]
        variance = conditional['covariance'][target, target]
        delta = mean - overall['mean'][target]
        within_terms.append(float(_multiply(mass, variance, 'within contribution')))
        between_terms.append(float(_multiply(mass, _multiply(delta, delta, 'between square'), 'between contribution')))
        entries.append({'label': float(label), 'probability': mass, 'mean': float(mean), 'variance': float(variance)})
    return {'total': float(overall['covariance'][target, target]),
            'within': _sum(within_terms, 'within total'), 'between': _sum(between_terms, 'between total'),
            'groups': entries, 'probability_audit': audit}


def sample_covariance(values: Any, ddof: int = 0) -> dict[str, Any]:
    """Rows n x d; ddof=0 empirical distribution, ddof=1 iid unbiased convention."""
    data = _array(values, 'sample rows', 2)
    n, d = data.shape
    require(d <= MAX_DIM, 'too many coordinates')
    correction = _integer(ddof, 'ddof', 0, 1)
    require(n > correction, 'sample count must exceed ddof')
    # Validate all rows and ddof before inspecting constant columns. Divide
    # variable columns first; constants keep their exact input value and avoid
    # a needless division that might underflow or create a false residual.
    constant = np.all(data == data[0], axis=0)
    mean = np.array([data[0, j] if constant[j] else
                     _sum(_divide(data[:, j], n, 'sample mean scaling'), 'sample mean')
                     for j in range(d)])
    centered = _finite(data - mean, 'sample centering')
    centered[:, constant] = 0.0
    cov = _divide(_gram(centered), n - correction, 'sample covariance normalization')
    return {'n': n, 'd': d, 'ddof': correction, 'mean': mean, 'covariance': cov}


def covariance_matrix(value: Any, strictly_positive: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Validate full exact symmetry before eigenvalue routines read a triangle.

    No negative computed eigenvalue is repaired. Numerical near-singularity is
    deliberately rejected in PD APIs: lambda_min > 1e-12 * lambda_max.
    This is a computation-domain condition, not a mathematical PD definition.
    """
    require(type(strictly_positive) is bool, 'strictly_positive must be bool')
    cov = _array(value, 'covariance', 2)
    require(cov.shape[0] == cov.shape[1] and cov.shape[0] <= MAX_DIM, 'covariance must be square of dimension 1..32')
    require(np.array_equal(cov, cov.T), 'covariance must be exactly symmetric; no automatic symmetrization')
    require(bool(np.all((cov == 0) | (np.abs(cov) >= 1e-100))),
            'nonzero covariance entries below 1e-100 exceed this linear-algebra interface range')
    require(bool(np.all(np.diag(cov) >= 0)), 'covariance diagonal must be nonnegative')
    try:
        eigenvalues = np.linalg.eigvalsh(cov)
    except np.linalg.LinAlgError as exc:
        raise ValueError('covariance eigendecomposition failed') from exc
    _finite(eigenvalues, 'covariance eigenvalues')
    require(bool(np.all(eigenvalues >= 0)), 'negative computed covariance eigenvalue; no PSD repair')
    if strictly_positive:
        scale = float(eigenvalues[-1])
        require(scale > 0 and eigenvalues[0] > PD_REL_MIN * scale,
                'strict PD numerical interface requires lambda_min/lambda_max > 1e-12')
    return cov, eigenvalues


def linear_moments(mean: Any, covariance: Any, transform: Any, shift: Any) -> dict[str, np.ndarray]:
    mu = _array(mean, 'mean', 1)
    cov, _ = covariance_matrix(covariance)
    a = _array(transform, 'transform', 2)
    b = _array(shift, 'shift', 1)
    require(cov.shape == (mu.size, mu.size), 'mean/covariance dimension mismatch')
    require(a.shape[1] == mu.size and a.shape[0] == b.size and b.size <= MAX_DIM, 'transform/shift dimension mismatch')
    new_mean = _finite(_matmul(a, mu[:, None], 'linear mean')[:, 0] + b, 'shifted mean')
    # The bilinear form is symmetric mathematically. Compute each pair once,
    # as in _gram; this is an evaluation order, not a repair of an input matrix.
    new_cov = np.empty((a.shape[0], a.shape[0]))
    for i in range(a.shape[0]):
        for j in range(i, a.shape[0]):
            right = _matmul(cov, a[j, :, None], 'Sigma a_j')[:, 0]
            new_cov[i, j] = new_cov[j, i] = _dot(a[i], right, 'a_i Sigma a_j')
    _finite(new_cov, 'transformed covariance')
    require(bool(np.all(np.diag(new_cov) >= 0)), 'negative transformed variance due to numerical cancellation')
    return {'mean': new_mean, 'covariance': new_cov}


def correlation(covariance: Any) -> np.ndarray:
    cov, _ = covariance_matrix(covariance)
    diagonal = np.diag(cov)
    require(bool(np.all(diagonal > 0)), 'correlation is undefined for a zero-variance coordinate')
    sd = np.sqrt(diagonal)
    denom = _multiply(sd[:, None], sd[None, :], 'standard deviation product')
    result = _divide(cov, denom, 'correlation')
    require(bool(np.all(np.abs(result) <= 1 + PROB_TOL)), 'computed correlation outside [-1,1] beyond roundoff')
    return result  # no clipping; diagonal may differ from 1 by ordinary rounding


def _gaussian_model(mean: Any, covariance: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = _array(mean, 'mean', 1)
    cov, _ = covariance_matrix(covariance, strictly_positive=True)
    require(cov.shape == (mu.size, mu.size), 'mean/covariance dimension mismatch')
    try:
        factor = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError as exc:
        raise ValueError('Cholesky failed without modification') from exc
    _finite(factor, 'Cholesky factor')
    reconstructed = _gram(factor.T)
    require(bool(np.allclose(reconstructed, cov, rtol=2e-12, atol=0)), 'Cholesky reconstruction failed')
    return mu, cov, factor


def _sampling_controls(n: Any, seed: Any) -> tuple[int, int]:
    return _integer(n, 'n', 1, MAX_SAMPLES), _integer(seed, 'seed', 0, 2**32 - 1)


def sample_finite(values: Any, weights: Any, n: int = 12000, seed: int = 2101) -> dict[str, Any]:
    data, p, audit = _states(values, weights)
    finite_moments(data, p)  # entire model, not only states reached by the seed
    count, random_seed = _sampling_controls(n, seed)
    require(count * data.shape[1] <= MAX_ENTRIES, 'sample array too large')
    rng = np.random.default_rng(random_seed)
    rows = data[rng.choice(data.shape[0], size=count, p=p)]
    return {'rows': rows, 'probability_audit': audit, 'n': count, 'seed': random_seed}


def sample_gaussian_factor(mean: Any, factor: Any, n: int = 12000, seed: int = 2102) -> dict[str, Any]:
    """Construct mu + B epsilon, including singular laws, without a PDF claim."""
    mu = _array(mean, 'mean', 1)
    b = _array(factor, 'factor', 2)
    require(b.shape[0] == mu.size and max(b.shape) <= MAX_DIM, 'factor dimension mismatch or too large')
    count, random_seed = _sampling_controls(n, seed)
    require(count * b.shape[1] <= MAX_ENTRIES and count * mu.size <= MAX_ENTRIES, 'sample array too large')
    covariance = _gram(b.T)  # all factors checked before creating a random stream
    # Guard numerical scale before RNG. The sampling API chooses a conservative
    # dynamic range; float64 cannot meaningfully add microscopic noise to 1e100.
    sd = np.sqrt(np.diag(covariance))
    require(bool(np.all((sd == 0) | (sd >= np.abs(mu) * 1e-12))),
            'noise scale too small relative to mean for this teaching sampler')
    rng = np.random.default_rng(random_seed)
    noise = rng.standard_normal((count, b.shape[1]))
    # Checked vectorized multiply per latent coordinate, rather than unchecked BLAS.
    centered = np.zeros((count, mu.size))
    for k in range(b.shape[1]):
        centered = _finite(centered + _multiply(noise[:, k, None], b[:, k], 'Gaussian transform'), 'Gaussian accumulation')
    rows = _finite(centered + mu, 'Gaussian shift')
    return {'rows': rows, 'mean': mu, 'covariance': covariance, 'factor': b, 'n': count, 'seed': random_seed}


def sample_gaussian(mean: Any, covariance: Any, n: int = 12000, seed: int = 2102) -> dict[str, Any]:
    mu, cov, factor = _gaussian_model(mean, covariance)
    result = sample_gaussian_factor(mu, factor, n, seed)
    result['covariance'] = cov
    return result


def gaussian_logpdf(points: Any, mean: Any, covariance: Any) -> np.ndarray:
    """Log density for numerical strictly PD input; points are n x d."""
    pts = _array(points, 'points', 2)
    mu, cov, lower = _gaussian_model(mean, covariance)
    require(pts.shape[1] == mu.size, 'point dimension mismatch')
    centered = _finite(pts - mu, 'density centering')
    try:
        whitened = np.linalg.solve(lower, centered.T).T
    except np.linalg.LinAlgError as exc:
        raise ValueError('density solve failed') from exc
    _finite(whitened, 'whitened points')
    squares = _multiply(whitened, whitened, 'Mahalanobis squares')
    norms = np.array([_sum(row, 'Mahalanobis norm') for row in squares])
    logdet = 2 * _sum(np.log(np.diag(lower)), 'log determinant')
    return _finite(-0.5 * (mu.size * math.log(2 * math.pi) + logdet + norms), 'log density')


def gaussian_pdf(points: Any, mean: Any, covariance: Any) -> np.ndarray:
    logs = gaussian_logpdf(points, mean, covariance)
    require(bool(np.all(logs >= math.log(np.finfo(float).tiny))),
            'density would be subnormal or underflow; use gaussian_logpdf')
    require(bool(np.all(logs <= math.log(np.finfo(float).max))), 'density overflow; use gaussian_logpdf')
    try:
        with np.errstate(over='raise', under='raise', invalid='raise'):
            return _finite(np.exp(logs), 'density exponential')
    except FloatingPointError as exc:
        raise ValueError('unrepresentable density; use gaussian_logpdf') from exc


def gaussian_conditional(mean: Any, covariance: Any, observed_indices: Any, observed_values: Any) -> dict[str, Any]:
    """Condition remaining coordinates on observed ones, strictly PD only.

    Solve linear systems; do not invert Sigma_BB. The Schur correction is a
    whitened Gram product, so off-diagonal pairs are computed once, not repaired.
    """
    mu, cov, _ = _gaussian_model(mean, covariance)
    require(isinstance(observed_indices, (list, tuple, np.ndarray)), 'observed_indices must be a sequence')
    if isinstance(observed_indices, np.ndarray):
        require(observed_indices.ndim == 1 and observed_indices.dtype.kind in 'iu', 'indices must be integer vector')
    ids = [_integer(i, 'observed index', 0, mu.size - 1) for i in observed_indices]
    require(0 < len(ids) < mu.size and len(set(ids)) == len(ids), 'observe a unique, nonempty, proper subset')
    given = _array(observed_values, 'observed_values', 1)
    require(given.size == len(ids), 'one observed value per index')
    remaining = [i for i in range(mu.size) if i not in ids]
    aa = cov[np.ix_(remaining, remaining)]
    ab = cov[np.ix_(remaining, ids)]
    bb = cov[np.ix_(ids, ids)]
    delta = _finite(given - mu[ids], 'conditional displacement')
    try:
        coefficients = np.linalg.solve(bb, ab.T).T
        lower_b = np.linalg.cholesky(bb)
        whitened = np.linalg.solve(lower_b, ab.T)
    except np.linalg.LinAlgError as exc:
        raise ValueError('conditional linear solve failed') from exc
    _finite(coefficients, 'conditional coefficients')
    _finite(whitened, 'conditional whitened block')
    conditional_mean = _finite(mu[remaining] + _matmul(coefficients, delta[:, None], 'conditional mean')[:, 0], 'conditional mean sum')
    conditional_covariance = _finite(aa - _gram(whitened), 'Schur complement')
    covariance_matrix(conditional_covariance, strictly_positive=True)
    residual_cross = _finite(ab - _matmul(coefficients, bb, 'residual cross covariance'), 'residual cross covariance')
    return {'remaining_indices': remaining, 'mean': conditional_mean, 'covariance': conditional_covariance,
            'coefficients': coefficients, 'residual_cross_covariance': residual_cross}


def triangle_density(points: Any) -> np.ndarray:
    pts = _array(points, 'triangle points', 2)
    require(pts.shape[1] == 2, 'triangle points must be n x 2')
    x, y = pts[:, 0], pts[:, 1]
    return np.where((y >= 0) & (y <= x) & (x <= 1), 2.0, 0.0)


def triangle_conditional(x: Any) -> dict[str, float]:
    value = float(_array([x], 'triangle x', 1)[0])
    require(0 < value <= 1, 'triangle conditional requires 0 < x <= 1; x=0 is not a 0/0 formula')
    square = float(_multiply(value, value, 'triangle square'))
    return {'low': 0.0, 'high': value, 'mean': float(_divide(value, 2, 'triangle mean')),
            'variance': float(_divide(square, 12, 'triangle variance')),
            'density': float(_divide(1, value, 'triangle conditional density'))}


def sample_triangle(n: int = 12000, seed: int = 2103) -> np.ndarray:
    count, random_seed = _sampling_controls(n, seed)
    # Sorting two independent uniforms gives density 2 on the triangular half.
    rng = np.random.default_rng(random_seed)
    values = rng.random((count, 2))
    return np.column_stack((np.max(values, axis=1), np.min(values, axis=1)))


def _csv_rows(path: Any, fields: list[str]) -> list[dict[str, str]]:
    with Path(path).open('r', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == fields, 'CSV header must exactly match the documented schema')
        rows = list(reader)
    require(0 < len(rows) <= MAX_SAMPLES, 'CSV requires a bounded nonempty set of rows')
    require(all(set(row) == set(fields) and all(isinstance(row[key], str) and row[key].strip() for key in fields) for row in rows),
            'CSV contains missing, extra or blank fields')
    return rows


def load_states(path: Any = None) -> dict[str, Any]:
    rows = _csv_rows(ROOT / 'data/joint_states.csv' if path is None else path,
                     ['scenario_id', 'probability_numerator', 'probability_denominator', 'x', 'z'])
    labels = [row['scenario_id'] for row in rows]
    require(len(set(labels)) == len(labels), 'scenario_id must be unique')
    probs, states = [], []
    for row in rows:
        try:
            numerator, denominator = int(row['probability_numerator']), int(row['probability_denominator'])
            require(denominator > 0 and 0 <= numerator <= denominator, 'invalid rational CSV probability')
            probs.append(Fraction(numerator, denominator))
            states.append([float(row['x']), float(row['z'])])
        except (ValueError, OverflowError, ZeroDivisionError) as exc:
            raise ValueError('invalid numeric state CSV cell') from exc
    require(sum(probs) == 1, 'CSV rational probabilities must sum exactly to 1')
    result = finite_moments(states, probs)
    result.update({'states': _array(states, 'CSV states', 2), 'rational_probabilities': probs, 'labels': labels})
    return result


def load_gaussian(path: Any = None) -> dict[str, np.ndarray]:
    rows = _csv_rows(ROOT / 'data/gaussian_model.csv' if path is None else path,
                     ['coordinate', 'mean', 'covariance_x', 'covariance_y'])
    require([row['coordinate'] for row in rows] == ['x', 'y'], 'Gaussian CSV rows must be x then y')
    try:
        mean = [float(row['mean']) for row in rows]
        covariance = [[float(row['covariance_x']), float(row['covariance_y'])] for row in rows]
    except (ValueError, OverflowError) as exc:
        raise ValueError('invalid numeric Gaussian CSV cell') from exc
    mu, cov, factor = _gaussian_model(mean, covariance)
    return {'mean': mu, 'covariance': cov, 'factor': factor}


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def generate_experiment(states_path: Any = None, gaussian_path: Any = None, n: int = 12000) -> tuple[dict, dict[str, np.ndarray]]:
    """Validate BOTH input files and controls before creating any RNG or output."""
    state = load_states(states_path)
    gaussian = load_gaussian(gaussian_path)
    count = _integer(n, 'n', 1, MAX_SAMPLES)
    # This CLI is tied to the written lesson. Use reusable APIs for changed models.
    require(state['labels'] == list('ABCDEF') and np.array_equal(state['states'], [[0,0],[0,0],[1,1],[1,1],[2,1],[4,1]])
            and state['rational_probabilities'] == [Fraction(1,6)] * 6, 'CLI lesson state model was changed; use APIs with new reference calculations')
    require(np.array_equal(gaussian['mean'], [1,-2]) and np.array_equal(gaussian['covariance'], [[4,3],[3,9]]),
            'CLI lesson Gaussian model was changed; use APIs with new reference calculations')
    rows = {}
    rows['finite'] = sample_finite(state['states'], state['probabilities'], count, 2101)['rows']
    rows['gaussian'] = sample_gaussian(gaussian['mean'], gaussian['covariance'], count, 2102)['rows']
    rows['triangle'] = sample_triangle(count, 2103)
    rng = np.random.default_rng(2104)
    u = rng.choice([-1., 0., 1.], count)
    rows['nonlinear'] = np.column_stack((u, u**2))
    rng = np.random.default_rng(2105)
    g = rng.standard_normal(count)
    signs = rng.choice([-1., 1.], count)
    rows['normal_marginals'] = np.column_stack((g, signs * g))
    rows['independent_normal'] = sample_gaussian([0,0], [[1,0],[0,1]], count, 2106)['rows']
    conditional = gaussian_conditional(gaussian['mean'], gaussian['covariance'], [0], [5])
    rows['conditional_x5'] = sample_gaussian(conditional['mean'], conditional['covariance'], count, 2107)['rows']
    rows['singular'] = sample_gaussian_factor([0,0], [[1],[2]], count, 2108)['rows']
    moments = {key: sample_covariance(value) for key, value in rows.items()}
    transform = linear_moments(state['mean'], state['covariance'], [[1,1],[1,-1]], [0,0])
    report = {'unit': '021', 'synthetic': True, 'n_each': count,
              'seeds': dict(zip(rows, range(2101,2109))), 'python': sys.version.split()[0], 'numpy': np.__version__,
              'rows_are_samples': True, 'sample_ddof': 0,
              'finite_model': {key: state[key] for key in ['mean','covariance','probability_audit']},
              'finite_transform': transform, 'total_variance': total_variance(state['states'], state['probabilities'], 0, 1),
              'gaussian_model': gaussian, 'gaussian_conditional_x5': conditional,
              'triangle_theory': {'mean': [2/3,1/3], 'covariance': [[1/18,1/36],[1/36,1/18]], 'within_y_given_x':1/24, 'between_y_given_x':1/72},
              'empirical': moments,
              'interpretation': 'Finite simulations are not equality, a convergence proof, or an independence test. Gaussian normal-marginal trap obeys |G|=|H| by construction.'}
    return to_jsonable(report), rows


def write_results(out: Any, report: dict, samples: dict[str, np.ndarray]) -> None:
    """Serialize completely before touching outputs; replace each file atomically.

    Invalid inputs never reach this function through main. Existing files are
    preserved on validation/serialization failure. This is not a transaction
    across all files if an OS/storage failure occurs during multiple replaces.
    """
    payloads = {'experiment_report.json': json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n'}
    for name, rows in samples.items():
        buffer = io.StringIO(newline='')
        writer = csv.writer(buffer, lineterminator='\n')
        writer.writerow([f'coordinate_{j}' for j in range(rows.shape[1])])
        writer.writerows([[format(float(value), '.17g') for value in row] for row in rows])
        payloads[name + '_samples.csv'] = buffer.getvalue()
    destination = Path(out)
    destination.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        temporary = None
        try:
            with tempfile.NamedTemporaryFile('w', encoding='utf-8', newline='', dir=destination, prefix='.pending-', delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(payload)
            os.replace(temporary, destination / name)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'outputs')
    parser.add_argument('--states', type=Path, default=ROOT / 'data/joint_states.csv')
    parser.add_argument('--gaussian', type=Path, default=ROOT / 'data/gaussian_model.csv')
    parser.add_argument('--n', type=int, default=12000)
    args = parser.parse_args()
    report, rows = generate_experiment(args.states, args.gaussian, args.n)
    write_results(args.out, report, rows)
    print(json.dumps({'status':'passed', 'n_each':report['n_each'], 'seeds':report['seeds'],
                      'finite_covariance':report['finite_model']['covariance'],
                      'gaussian_empirical':report['empirical']['gaussian'],
                      'conditional_x5':report['gaussian_conditional_x5']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
