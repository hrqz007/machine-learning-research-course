"""032: full-batch NumPy GD and coordinate scaling, for bounded teaching data.

No sklearn optimizer or hidden closed-form iterate substitution. See README for
strict input, numerical, trace, and output contracts. Python -O is supported.
"""
from pathlib import Path
from fractions import Fraction
from decimal import Decimal, InvalidOperation
import argparse, csv, hashlib, json, math, os, re, sys, tempfile
import numpy as np
ROOT = Path(__file__).resolve().parent
MIN_NORMAL = sys.float_info.min
TOKEN = re.compile(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z')
CASES = ('hand', 'correlated', 'constant')
BASELINE = {'data/cases.csv': 'de64df1dfab241112053173f095370f4d37619e4c0a20725a483779f300218e6', 'data/query.csv': '88e9395811cd148a0fb1c37fc3927f508ec273935b64975c954739e068bf2dbe', 'data/model_spec.json': 'a449e68e1f4f21414934053e3612c1c7f8caf6865e630fef3ef4c9aa36cf81e4'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def real(v, name='value', low=-1e6, high=1e6):
    permitted = type(v) in (int, float, np.float16, np.float32, np.float64) or isinstance(v, np.integer)
    require(permitted and not isinstance(v, (bool, np.bool_)), name+' requires ordinary real scalar')
    require(low <= v <= high, name+' outside finite supported range')
    require(v == 0 or abs(v) >= 1e-100, name+' nonzero magnitude below 1e-100')
    r = float(v)
    require(math.isfinite(r) and (v == 0 or r != 0), name+' cast loses nonzero value')
    if isinstance(v, (int, np.integer)):
        require(int(r) == int(v), name+' integer cast is not exact')
    return r


def integer(v, name, low, high):
    require(isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)), name+' requires integer')
    require(low <= v <= high, name+' outside range')
    return int(v)


def array(values, name, ndim):
    require(isinstance(values, (list, tuple, np.ndarray)), name+' requires list, tuple, ndarray')
    if isinstance(values, np.ndarray):
        require(values.ndim == ndim, name+' wrong dimension')
        require(values.dtype.kind in 'iuf' and values.dtype.itemsize <= 8, name+' rejects bool/object/complex/extended array')
    if ndim == 1:
        require(1 <= len(values) <= 200, name+' wrong length')
        return np.array([real(v, name) for v in values], dtype=np.float64)
    require(2 <= len(values) <= 200, name+' requires 2..200 rows')
    rows = [array(row, name+' row', 1) for row in values]
    width = len(rows[0])
    require(1 <= width <= 8 and all(len(row) == width for row in rows), name+' ragged or too wide')
    return np.stack(rows)


def data_pair(A, y):
    A, y = array(A, 'A', 2), array(y, 'y', 1)
    require(A.shape[0] == y.size, 'A/y sample count mismatch')
    return A, y


def checked(v, name):
    a = np.asarray(v, dtype=np.float64)
    require(np.isfinite(a).all(), name+' nonfinite computed result')
    require(np.all((a == 0) | (np.abs(a) >= MIN_NORMAL)), name+' computed subnormal unsupported')
    return a


def product(a, b, name):
    z = float(a)*float(b)
    require(math.isfinite(z) and (a == 0 or b == 0 or abs(z) >= MIN_NORMAL), name+' overflow/underflow')
    return z


def norm(v):
    return math.sqrt(math.fsum(product(z, z, 'norm square') for z in v))


def design(X):
    X = array(X, 'X', 2)
    require(X.shape[1] <= 7, 'at most 7 features with intercept')
    return np.column_stack((np.ones(len(X)), X))


def _state(A, y, theta):
    # Every multiplication is guarded, including BLAS calls which do not always
    # signal underflow via np.errstate. Products are computed once for checking;
    # NumPy @ remains the actual full-batch prediction/accumulation path.
    for row in A:
        for aj, tj in zip(row, theta):
            product(aj, tj, 'prediction product')
    prediction = checked(A @ theta, 'prediction')
    residual = checked(prediction-y, 'residual')
    squares = [product(z, z, 'residual square') for z in residual]
    loss = math.fsum(squares)/(2*len(A))
    require(not any(squares) or loss >= MIN_NORMAL, 'loss division underflow')
    for row, r in zip(A, residual):
        for aj in row:
            product(aj, r, 'gradient product')
    gradient = checked((A.T @ residual)/len(A), 'gradient')
    # Zero loss must be zero on the actual stored values, not a cancellation
    # artifact such as (1 + 2^-54) rounding back to 1.
    if loss == 0:
        for row, yi in zip(A, y):
            exact = sum((Fraction(float(a))*Fraction(float(t)) for a, t in zip(row, theta)), Fraction())-Fraction(float(yi))
            require(exact == 0, 'rounded zero loss hides a nonzero exact stored residual')
    return {'theta': theta.tolist(), 'prediction': prediction.tolist(), 'residual': residual.tolist(),
            'loss': loss, 'gradient': gradient.tolist(), 'gradient_norm': norm(gradient)}


def loss_gradient(A, y, theta):
    A, y = data_pair(A, y)
    theta = array(theta, 'theta', 1)
    require(theta.size == A.shape[1], 'theta requires shape (p,)')
    return _state(A, y, theta)


def exact_reference(A, y):
    """Independent exact normal-equation elimination on stored float values.

    Full-rank coefficient reference only. Singular systems report exact rank,
    never pretend a particular free-variable solution is uniquely identified.
    """
    A, y = data_pair(A, y)
    F = [[Fraction(float(v)) for v in row] for row in A]
    Y = [Fraction(float(v)) for v in y]
    n, p = A.shape
    G = [[sum((row[j]*row[k] for row in F), Fraction()) for k in range(p)] for j in range(p)]
    h = [sum((row[j]*yi for row, yi in zip(F, Y)), Fraction()) for j in range(p)]
    M = [row.copy()+[b] for row, b in zip(G, h)]
    rank = 0
    for col in range(p):
        pivot = next((r for r in range(rank, p) if M[r][col]), None)
        if pivot is None:
            continue
        M[rank], M[pivot] = M[pivot], M[rank]
        lead = M[rank][col]
        M[rank] = [z/lead for z in M[rank]]
        for r in range(p):
            if r != rank:
                z = M[r][col]
                M[r] = [a-z*b for a, b in zip(M[r], M[rank])]
        rank += 1
    result = {'rank_exact': rank, 'hessian_exact': [[str(z/n) for z in row] for row in G]}
    if rank != p:
        return dict(result, theta=None, theta_exact=None, loss=None, loss_exact=None)
    beta = [M[j][-1] for j in range(p)]
    rs = [sum((a*b for a, b in zip(row, beta)), Fraction())-yi for row, yi in zip(F, Y)]
    loss = sum((r*r for r in rs), Fraction())/(2*n)
    for value in beta+[loss]:
        f = float(value)
        require(math.isfinite(f) and abs(f) <= 1e12 and (value == 0 or abs(f) >= MIN_NORMAL), 'reference outside supported float display range')
    return dict(result, theta=[float(b) for b in beta], theta_exact=[str(b) for b in beta], loss=float(loss), loss_exact=str(loss))


def geometry(A, y):
    A, y = data_pair(A, y)
    reference = exact_reference(A, y)
    for row in A:
        for a in row:
            for b in row:
                product(a, b, 'Hessian product')
    H = checked(A.T @ A/len(A), 'Hessian')
    eig = checked(np.linalg.eigvalsh(H), 'eigenvalues')
    positive = reference['rank_exact'] == A.shape[1] and eig[0] > 0
    return {'hessian': H.tolist(), 'eigenvalues': eig.tolist(),
            'condition_number': float(eig[-1]/eig[0]) if positive else None,
            'stable_step_upper': float(2/eig[-1]) if eig[-1] > 0 else None,
            'reference': reference}


def fit_scaler(X):
    """Training-only ddof=0; exact stored centering avoids adjacent-float bug."""
    X = array(X, 'X_train', 2)
    mu, scales, zeros = [], [], []
    for col in X.T:
        F = [Fraction(float(v)) for v in col]
        mean = sum(F, Fraction())/len(F)
        var = sum(((v-mean)**2 for v in F), Fraction())/len(F)
        center = float(mean)
        variance = float(var)
        require(var == 0 or variance >= MIN_NORMAL, 'positive variance underflow')
        scale = 1.0 if var == 0 else math.sqrt(variance)
        mu.append(center); scales.append(scale); zeros.append(var == 0)
    # Mean is retained exactly as a rational string so transforming new data
    # uses the same learned map even if the mean is between adjacent floats.
    means = [str(sum((Fraction(float(v)) for v in col), Fraction())/len(col)) for col in X.T]
    return {'mean': mu, 'mean_exact': means, 'scale': scales, 'zero_variance': zeros}


def _scaler(scaler, p):
    require(type(scaler) is dict and set(scaler) == {'mean', 'mean_exact', 'scale', 'zero_variance'}, 'bad scaler fields')
    m, s = array(scaler['mean'], 'scaler mean', 1), array(scaler['scale'], 'scaler scale', 1)
    require(m.size == s.size == p, 'scaler feature mismatch')
    require(np.all(s > 0), 'scaler scale must be positive')
    require(type(scaler['zero_variance']) is list and len(scaler['zero_variance']) == p and all(type(v) is bool for v in scaler['zero_variance']), 'bad zero-variance flags')
    require(type(scaler['mean_exact']) is list and len(scaler['mean_exact']) == p, 'bad exact mean list')
    exact = []
    for text, mean in zip(scaler['mean_exact'], m):
        require(type(text) is str and len(text) <= 1000 and re.fullmatch(r'-?\d+(?:/[1-9]\d*)?', text) is not None, 'bad exact mean')
        f = Fraction(text)
        require(float(f) == mean, 'inconsistent scaler mean')
        exact.append(f)
    return m, s, exact


def transform(X, scaler):
    X = array(X, 'X', 2)
    _, s, exact = _scaler(scaler, X.shape[1])
    Z = np.empty_like(X)
    for i, row in enumerate(X):
        for j, value in enumerate(row):
            q = (Fraction(float(value))-exact[j])/Fraction(float(s[j]))
            z = float(q)
            require(math.isfinite(z) and abs(z) <= 1e6 and (q == 0 or abs(z) >= MIN_NORMAL), 'scaled feature outside supported range')
            Z[i,j] = z
    return Z


def to_raw(theta_scaled, scaler):
    gamma = array(theta_scaled, 'theta_scaled', 1)
    _, s, means = _scaler(scaler, len(gamma)-1)
    b = [Fraction(float(gamma[j+1]))/Fraction(float(s[j])) for j in range(len(s))]
    a = Fraction(float(gamma[0]))-sum((u*v for u, v in zip(means, b)), Fraction())
    values = [a]+b
    result = checked([float(v) for v in values], 'raw coefficient')
    require(all(v == 0 or q != 0 for v, q in zip(values, result)), 'coefficient mapping underflow')
    return result


def to_scaled(theta_raw, scaler):
    beta = array(theta_raw, 'theta_raw', 1)
    _, scales, means = _scaler(scaler, len(beta)-1)
    b = [Fraction(float(v)) for v in beta[1:]]
    alpha = Fraction(float(beta[0]))+sum((u*v for u,v in zip(means,b)), Fraction())
    values = [alpha]+[Fraction(float(s))*v for s,v in zip(scales,b)]
    result = checked([float(v) for v in values], 'scaled coefficient')
    require(all(v == 0 or q != 0 for v,q in zip(values,result)), 'coefficient mapping underflow')
    return result


def path_spec(spec, p):
    keys = {'initial','learning_rate','gradient_tolerance','max_steps','parameter_limit'}
    require(type(spec) is dict and set(spec) == keys, 'bad path fields')
    initial = array(spec['initial'], 'initial', 1)
    require(initial.size == p, 'initial shape mismatch')
    eta = real(spec['learning_rate'], 'learning_rate', 0, 2)
    tol = real(spec['gradient_tolerance'], 'gradient_tolerance', 0, 1)
    steps = integer(spec['max_steps'], 'max_steps', 0, 10000)
    limit = real(spec['parameter_limit'], 'parameter_limit', 1, 1e6)
    require(np.all(np.abs(initial) <= limit), 'initial exceeds parameter limit')
    return initial, eta, tol, steps, limit


def trace_path(A, y, spec):
    A, y = data_pair(A, y)
    theta, eta, tol, budget, limit = path_spec(spec, A.shape[1])
    history = []
    status = None
    for t in range(budget+1):
        state = _state(A, y, theta)
        state['iteration'] = t
        history.append(state)
        if state['gradient_norm'] <= tol:
            status = 'gradient_tolerance_met'; break
        if t == budget:
            status = 'budget_exhausted'; break
        g = np.array(state['gradient'])
        update = checked([product(eta, v, 'update product') for v in g], 'update')
        candidate = checked(theta-update, 'candidate')
        if np.any(np.abs(candidate) > limit):
            status = 'parameter_limit_reached'; break
        if np.array_equal(candidate, theta):
            status = 'stagnated_without_tolerance'; break
        if len(history) > 1 and np.array_equal(candidate, np.array(history[-2]['theta'])):
            status = 'two_cycle_without_tolerance'; break
        theta = candidate
    return {'status': status, 'success': status == 'gradient_tolerance_met',
            'updates_completed': len(history)-1, 'history': history}


def token(text):
    require(type(text) is str and len(text) <= 120 and TOKEN.fullmatch(text) is not None, 'bad numeric token')
    try:
        d = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError('bad decimal token') from exc
    require(d.is_finite() and abs(d) <= Decimal('1e6'), 'token nonfinite/outside range')
    require(d == 0 or abs(d) >= Decimal('1e-100'), 'nonzero token underflow/below floor')
    f = float(d)
    require(d == 0 or f != 0, 'token underflow')
    return f


def unique_pairs(pairs):
    obj = {}
    for k,v in pairs:
        require(k not in obj, 'duplicate JSON key: '+k)
        obj[k] = v
    return obj


def json_integer(text):
    value = token(text)
    out = int(text)
    require(int(value) == out, 'JSON integer cast loses precision')
    return out


def load_json(path):
    def reject(s):
        raise ValueError('nonstandard JSON constant '+s)
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_float=token,
                      parse_int=json_integer, parse_constant=reject, object_pairs_hook=unique_pairs)


def read_csv(path, header):
    with Path(path).open(encoding='utf-8', newline='') as f:
        rows = list(csv.reader(f))
    require(rows and rows[0] == header, 'CSV header mismatch')
    require(2 <= len(rows) <= 601 and all(len(r) == len(header) for r in rows[1:]), 'CSV row count/width')
    return rows[1:]


def load_inputs(data_path=None, query_path=None, config_path=None):
    rows = read_csv(data_path or ROOT/'data/cases.csv', ['case','x1','x2','y'])
    data = {key: {'X': [], 'y': []} for key in CASES}
    for row in rows:
        require(row[0] in CASES, 'unknown case')
        numbers = [token(s) for s in row[1:]]
        data[row[0]]['X'].append(numbers[:2]); data[row[0]]['y'].append(numbers[2])
    query = [[token(s) for s in row] for row in read_csv(query_path or ROOT/'data/query.csv', ['x1','x2'])]
    config = load_json(config_path or ROOT/'data/model_spec.json')
    return validate_inputs(data, query, config)


def validate_inputs(data, query, config):
    require(type(data) is dict and set(data) == set(CASES), 'case set mismatch')
    clean = {}
    for name, case in data.items():
        require(type(case) is dict and set(case) == {'X','y'}, 'case fields mismatch')
        X, y = data_pair(case['X'], case['y'])
        require(X.shape[1] == 2, 'file experiments require exactly two features')
        clean[name] = {'X':X, 'y':y}
    query = array(query, 'query', 2)
    require(query.shape[1] == 2, 'query feature count')
    keys = {'objective','initial','gradient_tolerance','max_steps','parameter_limit','paths'}
    require(type(config) is dict and set(config) == keys and config['objective'] == 'SSE/(2n)', 'config fields/objective mismatch')
    require(type(config['paths']) is list and 1 <= len(config['paths']) <= 12, 'paths count')
    names = set()
    for p in config['paths']:
        require(type(p) is dict and set(p) == {'name','coordinates','learning_rate'}, 'path record fields')
        require(type(p['name']) is str and re.fullmatch(r'[a-z][a-z0-9_]{0,35}',p['name']) is not None and p['name'] not in names, 'path name invalid/duplicate')
        names.add(p['name'])
        require(p['coordinates'] in ('raw','scaled'), 'unknown coordinates')
        spec = {k:config[k] for k in ('initial','gradient_tolerance','max_steps','parameter_limit')}
        spec['learning_rate'] = p['learning_rate']
        path_spec(spec, 3)
    # Validation completes for all cases, queries, and final paths before
    # reference, eigendecomposition, fitting, iterations, or output begins.
    return clean, query, config


def compile_report(data, query, config):
    data, query, config = validate_inputs(data, query, config)
    case_reports = {}
    for name, case in data.items():
        X, y = case['X'], case['y']
        scaler = fit_scaler(X)
        Z = transform(X, scaler)
        case_reports[name] = {'raw':geometry(design(X), y), 'scaled':geometry(design(Z),y), 'scaler':scaler}
    X, y = data['hand']['X'], data['hand']['y']
    scaler = case_reports['hand']['scaler']
    raw_A, scaled_A = design(X), design(transform(X,scaler))
    reference = case_reports['hand']['raw']['reference']
    require(reference['theta'] is not None, 'hand experiment requires full rank')
    theta_star = np.array(reference['theta'])
    paths = {}
    for p in config['paths']:
        spec = {k:config[k] for k in ('initial','gradient_tolerance','max_steps','parameter_limit')}
        spec['learning_rate'] = p['learning_rate']
        scaled = p['coordinates'] == 'scaled'
        if scaled:
            # Same initial prediction function, expressed in scaled coordinates.
            spec['initial'] = to_scaled(spec['initial'], scaler).tolist()
        run = trace_path(scaled_A if scaled else raw_A, y, spec)
        for state in run['history']:
            beta = to_raw(state['theta'],scaler) if scaled else np.array(state['theta'])
            error = beta-theta_star
            state['theta_raw'] = beta.tolist()
            state['parameter_error_raw'] = norm(error)
            # Nonnegative anchor-based gap avoids loss - loss* cancellation.
            state['quadratic_gap_raw'] = norm(raw_A @ error)**2/(2*len(y))
            state['gradient_norm_raw'] = loss_gradient(raw_A,y,beta)['gradient_norm']
        beta = np.array(run['history'][-1]['theta_raw'])
        run.update(coordinates=p['coordinates'], learning_rate=p['learning_rate'],
                   query_prediction=(design(query)@beta).tolist())
        paths[p['name']] = run
    return {'objective':'SSE/(2n)', 'cases':case_reports, 'paths':paths,
            'query':query.tolist(), 'query_scaled_train_only':transform(query,scaler).tolist(),
            'query_prediction_reference':(design(query)@theta_star).tolist()}


def check_notebook_baseline():
    for relative, digest in BASELINE.items():
        require(hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() == digest,
                'Notebook fixed teaching baseline changed: '+relative)
    require(len(BASELINE) == 3, 'baseline hash set missing')
    return load_inputs()


def run(data_path=None, query_path=None, config_path=None, output=None):
    path = Path(output) if output is not None else ROOT/'outputs/report.json'
    sources = [Path(data_path or ROOT/'data/cases.csv'),Path(query_path or ROOT/'data/query.csv'),Path(config_path or ROOT/'data/model_spec.json'),Path(__file__)]
    resolved = path.resolve()
    require(resolved not in [p.resolve() for p in sources], 'output may not overwrite source input')
    require(not resolved.is_relative_to(ROOT) or resolved.is_relative_to(ROOT/'outputs'), 'package files are protected; use outputs/ or an external directory')
    inputs = load_inputs(data_path,query_path,config_path)
    report = compile_report(*inputs)
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w',encoding='utf-8',dir=path.parent,prefix='.report-',delete=False) as f:
            temporary = Path(f.name); f.write(payload)
        os.replace(temporary,path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data'); parser.add_argument('--query'); parser.add_argument('--config'); parser.add_argument('--output')
    args = parser.parse_args()
    report = run(args.data,args.query,args.config,args.output)
    print(json.dumps({k:{q:v[q] for q in ('status','updates_completed')} for k,v in report['paths'].items()},ensure_ascii=False))


if __name__ == '__main__':
    main()
