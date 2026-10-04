"""028: auditable, bounded one-variable least squares teaching experiment.

All external fields are checked before local RNG creation or output replacement.
Only modest, finite binary64-range inputs are supported; see README.md.
Exact rational arithmetic certifies stored-value centering and collinearity in this small bounded experiment.
"""
from pathlib import Path
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import argparse
import csv
import json
import math
import os
import re
import sys
import tempfile
import numpy as np

ROOT = Path(__file__).resolve().parent
MIN_NORMAL = sys.float_info.min
TOKEN = re.compile(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z')
CASES = ('hand', 'curved', 'influential', 'perfect', 'constant_x')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def real(value, name='value', low=-1e6, high=1e6):
    permitted = type(value) in (int, float, np.float16, np.float32, np.float64) or isinstance(value, np.integer)
    require(permitted and not isinstance(value, (bool, np.bool_)), name+' requires a supported real scalar')
    require(low <= value <= high, name+' is nonfinite or outside the supported range')
    require(value == 0 or abs(value) >= 1e-100, name+' nonzero magnitude is below 1e-100')
    result = float(value)
    require(math.isfinite(result) and (value == 0 or result != 0), name+' conversion loses a nonzero value')
    return result


def integer(value, name, low, high):
    require(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)), name+' requires an integer')
    require(low <= value <= high, name+' is outside the supported range')
    return int(value)


def vector(values, name='values', minimum=1, maximum=2000):
    require(isinstance(values, (list, tuple, np.ndarray)), name+' requires a list, tuple or numeric array')
    if isinstance(values, np.ndarray):
        require(values.ndim == 1, name+' requires shape (n,)')
        require(values.dtype.kind in 'iuf' and values.dtype.itemsize <= 8, name+' rejects bool/object/complex/extended arrays')
    require(minimum <= len(values) <= maximum, name+' length outside supported range')
    return [real(v, name+'['+str(i)+']') for i, v in enumerate(values)]


def pair(x, y):
    x = vector(x, 'x')
    y = vector(y, 'y')
    require(len(x) == len(y), 'x and y must have equal lengths')
    return x, y


def finite(value, name):
    require(math.isfinite(value), name+' overflow/nonfinite result')
    return value


def product(a, b, name='product'):
    q = finite(a*b, name)
    require(a == 0 or b == 0 or abs(q) >= MIN_NORMAL, name+' nonzero product became zero/subnormal')
    return q


def sumsq(values, name='sum of squares'):
    squares = [product(v, v, name) for v in values]
    return finite(math.fsum(squares), name)


def mean(values):
    # Do not subtract an anchor: [-1, 1, 1e-100] has nonzero mean.
    if all(v == values[0] for v in values):
        return values[0]
    total = math.fsum(values)
    result = total/len(values)
    require(total == 0 or abs(result) >= MIN_NORMAL, 'mean underflow')
    return finite(result, 'mean')


def _stored_center(values):
    """Preserve the original fsum mean, plus its true stored-value remainder.

    Subtracting an already rounded mean can lose half an ulp on adjacent inputs.
    Summing those rounded differences is not a reliable mean correction either
    for mixed-scale inputs. Small exact binary rationals resolve both cases.
    """
    m = mean(values)
    exact_values = [Fraction(v) for v in values]
    exact_mean = sum(exact_values, Fraction(0))/len(values)
    remainder = exact_mean-Fraction(m)
    offset = float(remainder)
    require(remainder == 0 or abs(offset) >= MIN_NORMAL, 'mean remainder underflow')
    centered = []
    for value in exact_values:
        exact_d = value-exact_mean
        d = float(exact_d)
        require(exact_d == 0 or abs(d) >= MIN_NORMAL, 'centered difference underflow')
        centered.append(finite(d, 'centered difference'))
    return m, offset, centered, exact_mean


def centered_moments(x, y):
    mx, ox, dx, _ = _stored_center(x)
    my, oy, dy, _ = _stored_center(y)
    # Residual correction only improves the moment sums. Every downstream
    # quantity uses the same correctly centered differences and mean remainder.
    sx, sy = math.fsum(dx), math.fsum(dy)
    xx = sumsq(dx, 'x spread')-product(sx, sx, 'x centering correction')/len(x)
    yy = sumsq(dy, 'y spread')-product(sy, sy, 'y centering correction')/len(y)
    xy = math.fsum(product(a, b, 'cross product') for a, b in zip(dx, dy))-product(sx, sy, 'cross correction')/len(x)
    constant_x = all(v == x[0] for v in x)
    constant_y = all(v == y[0] for v in y)
    if constant_x:
        xx = 0.0
    else:
        require(MIN_NORMAL <= xx < math.inf, 'nonconstant x spread is zero/subnormal/nonfinite')
    if constant_y:
        yy = 0.0
    else:
        require(MIN_NORMAL <= yy < math.inf, 'nonconstant y spread is zero/subnormal/nonfinite')
    return mx, my, dx, dy, xx, finite(xy, 'Sxy'), yy, constant_x, ox, oy


def exact_collinear(x, y):
    """Certificate for the supplied binary values, not pre-rounding physical data."""
    xq, yq = [Fraction(v) for v in x], [Fraction(v) for v in y]
    j = next((j for j in range(1, len(x)) if xq[j] != xq[0]), None)
    if j is None:
        return all(v == yq[0] for v in yq)
    dx, dy = xq[j]-xq[0], yq[j]-yq[0]
    return all((v-yq[0])*dx == (t-xq[0])*dy for t, v in zip(xq, yq))


def fit_line(x, y):
    x, y = pair(x, y)
    mx, my, dx, dy, xx, xy, yy, constant_x, ox, oy = centered_moments(x, y)
    b = 0.0 if constant_x else finite(xy/xx, 'slope')
    require(xy == 0 or abs(b) >= MIN_NORMAL, 'nonzero slope underflow')
    # This bounded implementation declines very large coefficients rather than
    # pretending every rescaled near-singular problem is accurately supported.
    require(abs(b) <= 1e12, 'slope magnitude exceeds the teaching limit 1e12')
    a = finite(math.fsum([my, oy, -product(b, mx), -product(b, ox)]), 'intercept')
    require(abs(a) <= 1e12, 'intercept magnitude exceeds the teaching limit 1e12')
    residuals = [math.fsum([v, -product(b, d)]) for v, d in zip(dy, dx)]
    fitted = [math.fsum([my, oy, product(b, d)]) for d in dx]
    sse = sumsq(residuals, 'residual square')
    exact_zero = exact_collinear(x, y)
    require(sse > 0 or exact_zero, 'rounded zero SSE without an exact collinearity certificate')
    mse = sse/len(x)
    require(sse == 0 or mse >= MIN_NORMAL, 'MSE underflow')
    explained = sumsq([product(b, d) for d in dx], 'explained square')
    orthogonal_one = math.fsum(residuals)
    orthogonal_x = math.fsum(product(d, e) for d, e in zip(dx, residuals))
    # Exact data collinearity is distinct from a rounded residual SSE.
    variance_mle = None if exact_zero else mse
    leverage = [1/len(x)]*len(x) if constant_x else [1/len(x)+product(d, d)/xx for d in dx]
    return {'n':len(x), 'intercept':a, 'slope':b, 'x_mean':mx, 'y_mean':my,
            'x_mean_offset':ox, 'y_mean_offset':oy, 'Sxx':xx, 'Sxy':xy, 'Syy':yy, 'fitted':fitted, 'residuals':residuals,
            'sse':sse, 'mse':mse, 'explained_ss':explained,
            'decomposition_error':yy-explained-sse,
            'residual_sum':orthogonal_one, 'centered_x_residual_sum':orthogonal_x,
            'unique_parameters':not constant_x, 'constant_x':constant_x,
            'exact_minimum_sse_zero':exact_zero,
            'variance_mle':variance_mle,
            'variance_mle_status':'no_finite_maximum_sigma_to_zero' if exact_zero else 'positive_interior',
            'leverage':leverage,
            'r_squared':None if yy == 0 else 1-sse/yy}


def evaluate_line(x, y, a, b):
    x, y = pair(x, y)
    a, b = real(a, 'intercept'), real(b, 'slope')
    predictions = [math.fsum([a, product(b, t)]) for t in x]
    residuals = [math.fsum([v, -a, -product(b, t)]) for t, v in zip(x, y)]
    sse = sumsq(residuals, 'candidate residual square')
    mse = sse/len(x)
    require(sse == 0 or mse >= MIN_NORMAL, 'candidate MSE underflow')
    return {'intercept':a, 'slope':b, 'fitted':predictions, 'sse':sse, 'mse':mse}


def gaussian_nll(x, y, a, b, sigma):
    # Validate even sigma on a perfect-fit input; no early-return bypass.
    x, y = pair(x, y)
    a, b = real(a, 'intercept'), real(b, 'slope')
    sigma = real(sigma, 'sigma', .01, 1000)
    sse = evaluate_line(x, y, a, b)['sse']
    return finite(len(x)*(.5*math.log(2*math.pi)+math.log(sigma))+sse/(2*sigma*sigma), 'NLL')


def prediction_diagnostics(x, y, query, sigma):
    x, y = pair(x, y)
    query = vector(query, 'prediction_x', 1, 100)
    sigma = real(sigma, 'sigma', .01, 1000)
    model = fit_line(x, y)
    require(model['unique_parameters'], 'prediction away from constant x is unidentified')
    answers = []
    exact_x_mean = _stored_center(x)[3]
    for t in query:
        d = float(Fraction(t)-exact_x_mean)
        mu = math.fsum([model['y_mean'], model['y_mean_offset'], product(model['slope'], d)])
        factor = 1/len(x)+product(d, d)/model['Sxx']
        se_mean = sigma*math.sqrt(factor)
        require(MIN_NORMAL <= se_mean < math.inf, 'prediction standard error underflow/overflow')
        answers.append({'x':t, 'prediction':mu, 'outside_training_range':not min(x) <= t <= max(x),
                        'conditional_mean_se_if_model_true':se_mean,
                        'new_observation_sd_if_model_true':sigma*math.sqrt(1+factor)})
    return answers



def check_notebook_baseline(data_path=None, config_path=None):
    """Fixed teaching figures require the documented default input files."""
    import hashlib
    paths = {'cases.csv': Path(data_path or ROOT/'data/cases.csv'),
             'model_spec.json': Path(config_path or ROOT/'data/model_spec.json')}
    expected = {'cases.csv': '96a1b9f704756b4981931e45948e1b730f5849dba9e7f7d919c55ba9c2eae92d', 'model_spec.json': '9f87bb2bccf08762b86eb302d75d0e5aa01f5ca26eb648e50378624b01865693'}
    for name, path in paths.items():
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected[name],
                'Notebook requires unchanged default '+name+'; use the configurable script for custom experiments')
    return True


def number_token(token):
    require(type(token) is str and len(token) <= 64 and TOKEN.fullmatch(token) is not None, 'invalid numeric token')
    try:
        d = Decimal(token)
    except InvalidOperation as exc:
        raise ValueError('invalid decimal token') from exc
    require(d.is_finite() and abs(d) <= Decimal('1e6'), 'numeric token outside finite teaching range')
    require(d == 0 or abs(d) >= Decimal('1e-100'), 'nonzero numeric token below 1e-100')
    return real(float(d), 'numeric token')


def reject_constant(token):
    raise ValueError('nonstandard JSON constant: '+token)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key: '+key)
        result[key] = value
    return result


def read_config(path):
    cfg = json.loads(Path(path).read_text(encoding='utf-8'), parse_float=number_token,
                     parse_int=int, parse_constant=reject_constant, object_pairs_hook=unique_object)
    expected = {'known_sigma','seed','simulation_repeats','prediction_x','candidate_lines','focus_case'}
    require(type(cfg) is dict and set(cfg) == expected, 'config requires exactly the documented fields')
    sigma = real(cfg['known_sigma'], 'known_sigma', .01, 1000)
    seed = integer(cfg['seed'], 'seed', 0, 2**32-1)
    repeats = integer(cfg['simulation_repeats'], 'simulation_repeats', 10, 2000)
    queries = vector(cfg['prediction_x'], 'prediction_x', 1, 100)
    lines = cfg['candidate_lines']
    require(type(lines) is list and 1 <= len(lines) <= 100, 'candidate_lines requires 1..100 lines')
    clean = []
    for i, row in enumerate(lines):
        require(type(row) is dict and set(row) == {'intercept','slope'}, 'candidate line fields incorrect')
        clean.append({'intercept':real(row['intercept'], 'candidate intercept'), 'slope':real(row['slope'], 'candidate slope')})
    require(type(cfg['focus_case']) is str and cfg['focus_case'] in CASES, 'invalid focus_case')
    return {'known_sigma':sigma, 'seed':seed, 'simulation_repeats':repeats,
            'prediction_x':queries, 'candidate_lines':clean, 'focus_case':cfg['focus_case']}


def read_cases(path):
    groups = {name:{'ids':[], 'x':[], 'y':[]} for name in CASES}
    seen = set()
    with Path(path).open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == ['case','id','x','y'], 'CSV header must be case,id,x,y')
        for row in reader:
            require(set(row) == {'case','id','x','y'} and all(v is not None for v in row.values()), 'CSV malformed row')
            name, key = row['case'], row['id']
            require(name in groups and re.fullmatch(r'[A-Z][0-9]{2}', key) is not None, 'CSV case/id invalid')
            require(key not in seen, 'duplicate CSV id')
            seen.add(key)
            # Check every case, including rows after the focused case.
            x, y = number_token(row['x']), number_token(row['y'])
            groups[name]['ids'].append(key)
            groups[name]['x'].append(x)
            groups[name]['y'].append(y)
            require(len(groups[name]['x']) <= 2000, 'CSV case too large')
    for name, values in groups.items():
        require(len(values['x']) >= 2, name+' requires at least two observations')
        pair(values['x'], values['y'])
    return groups


def compile_report(data_path, config_path):
    cfg = read_config(config_path)
    groups = read_cases(data_path)
    fits = {name:fit_line(v['x'], v['y']) for name, v in groups.items()}
    focus = groups[cfg['focus_case']]
    comparisons = [evaluate_line(focus['x'], focus['y'], row['intercept'], row['slope']) for row in cfg['candidate_lines']]
    hand = groups['hand']
    predictions = prediction_diagnostics(hand['x'], hand['y'], cfg['prediction_x'], cfg['known_sigma'])
    nll = [gaussian_nll(focus['x'], focus['y'], row['intercept'], row['slope'], cfg['known_sigma']) for row in cfg['candidate_lines']]
    # No user data remain unvalidated when the RNG is created.
    rng = np.random.default_rng(cfg['seed'])
    x = np.array(hand['x'])
    dx = np.array(_stored_center(hand['x'])[2]); sxx = fits['hand']['Sxx']
    noise = rng.normal(0, cfg['known_sigma'], size=(cfg['simulation_repeats'], len(x)))
    slopes = 1.2+(noise@dx)/sxx
    simulated = {'model':'Y=0.6+1.2*x+iid Normal(0,known_sigma^2), fixed hand x',
                 'repeats':cfg['simulation_repeats'], 'seed':cfg['seed'],
                 'slope_average':float(np.mean(slopes)), 'slope_sample_sd':float(np.std(slopes, ddof=1)),
                 'theoretical_slope_sd':cfg['known_sigma']/math.sqrt(sxx)}
    return {'unit':'028', 'synthetic_teaching_data':True, 'config':cfg, 'fits':fits,
            'candidate_comparisons':comparisons, 'candidate_gaussian_nll':nll,
            'hand_predictions':predictions, 'simulation':simulated,
            'limits':'Small synthetic CPU example. Residual patterns and conditional uncertainty do not establish causality or extrapolation validity.'}


def atomic_json(report, target):
    payload = (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n').encode('utf-8')
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.'+target.name+'.', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        os.replace(temporary, target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def run(data_path=ROOT/'data/cases.csv', config_path=ROOT/'data/model_spec.json', output=ROOT/'outputs/report.json'):
    report = compile_report(data_path, config_path)
    atomic_json(report, output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'data/cases.csv')
    parser.add_argument('--config', type=Path, default=ROOT/'data/model_spec.json')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/report.json')
    args = parser.parse_args()
    report = run(args.data, args.config, args.output)
    hand = report['fits']['hand']
    print(json.dumps({'unit':'028','intercept':hand['intercept'],'slope':hand['slope'],'SSE':hand['sse'],'MSE':hand['mse']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
