"""Unit 039: executable diagnostic evidence, including deliberately wrong paths.

No numerical work or output occurs at import. Public runners validate complete
raw inputs first. A fault injection is labelled, never used as the reference.
"""
from pathlib import Path
from dataclasses import dataclass, asdict
import argparse
import copy
import csv
from decimal import Decimal, InvalidOperation
import io
import json
import math
import os
import tempfile
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
KEYS = {'schema_version', 'initial_parameters', 'learning_rate', 'max_steps',
        'gradient_tolerance', 'step_tolerance', 'loss_tolerance', 'fd_steps',
        'fd_parameters', 'stress_parameters'}


def real(value, name, *, lower=None, upper=None, raw=True):
    # Reject bool/string/object before conversion. -O does not remove guards.
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(name + ': expected a real numeric scalar, not bool/string')
    try:
        v = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(name + ': unrepresentable number') from exc
    if v == 0 and value != 0:
        raise ValueError(name + ': nonzero lost during float conversion')
    if not math.isfinite(v):
        raise ValueError(name + ': nonfinite')
    if raw and v != 0 and abs(v) < 1e-150:
        raise ValueError(name + ': raw nonzero value is outside supported input scale')
    if (lower is not None and v < lower) or (upper is not None and v > upper):
        raise ValueError(name + ': outside declared domain')
    return v


def sequence(value, name, size=None):
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise TypeError(name + ': expected sequence')
    if isinstance(value, np.ndarray) and value.ndim != 1:
        raise ValueError(name + ': expected one-dimensional shape')
    if size is not None and len(value) != size:
        raise ValueError(name + ': wrong length')
    return value


def vector(value, name, size=2, bound=1e6, raw=True):
    seq = sequence(value, name, size)
    return [real(v, name + '[' + str(i) + ']', lower=-bound, upper=bound, raw=raw) for i, v in enumerate(seq)]


def validate(data, config, observer=None):
    """Validate ALL raw fields before ndarray creation, callbacks or calculations."""
    if not isinstance(config, dict) or set(config) != KEYS:
        raise ValueError('config: exact key set required')
    if type(config['schema_version']) is not int or config['schema_version'] != 1:
        raise ValueError('schema_version must be integer 1')
    if type(config['max_steps']) is not int or not 1 <= config['max_steps'] <= 1000:
        raise ValueError('max_steps must be integer in [1,1000]')
    s = dict(config)
    s['initial_parameters'] = vector(s['initial_parameters'], 'initial_parameters', bound=10)
    s['learning_rate'] = real(s['learning_rate'], 'learning_rate', lower=1e-12, upper=1000)
    for k in ('gradient_tolerance', 'step_tolerance', 'loss_tolerance'):
        s[k] = real(s[k], k, lower=1e-16, upper=1e-2)
    s['fd_parameters'] = vector(s['fd_parameters'], 'fd_parameters', bound=10)
    fs = sequence(s['fd_steps'], 'fd_steps')
    if not 2 <= len(fs) <= 32:
        raise ValueError('fd_steps: length in [2,32]')
    s['fd_steps'] = [real(h, 'fd_steps', lower=1e-16, upper=.1) for h in fs]
    if len(set(s['fd_steps'])) != len(s['fd_steps']):
        raise ValueError('fd_steps must not repeat')
    sp = sequence(s['stress_parameters'], 'stress_parameters')
    if not 1 <= len(sp) <= 8:
        raise ValueError('stress_parameters: length in [1,8]')
    s['stress_parameters'] = [vector(t, 'stress_parameters', bound=2000) for t in sp]
    if not isinstance(data, (list, tuple, np.ndarray)):
        raise TypeError('data: expected rows')
    if isinstance(data, np.ndarray) and (data.ndim != 2 or data.shape[1] != 3):
        raise ValueError('data: expected (n,3) with id,x,y')
    if not 2 <= len(data) <= 100:
        raise ValueError('data: n in [2,100]')
    rows = []
    for i, row in enumerate(data):
        row = sequence(row, 'row', 3)
        ident = real(row[0], 'id', lower=1, upper=100)
        if ident != i + 1:
            raise ValueError('ids must be unique consecutive 1..n in row order')
        x = real(row[1], 'x', lower=-100, upper=100)
        y = real(row[2], 'y', lower=0, upper=1)
        if y not in (0., 1.):
            raise ValueError('this teaching contract requires binary y')
        rows.append([ident, x, y])
    if observer is not None and not callable(observer):
        raise TypeError('observer must be callable or None')
    return rows, s


def strict_json(text):
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result:
                raise ValueError('duplicate JSON key: ' + k)
            result[k] = v
        return result
    def floating(token):
        d = Decimal(token)
        f = float(d)
        if not math.isfinite(f) or (d != 0 and (f == 0 or abs(f) < 1e-150)):
            raise ValueError('JSON number exceeds supported raw scale')
        return f
    def constant(token):
        raise ValueError('nonstandard JSON numeric constant: ' + token)
    return json.loads(text, object_pairs_hook=pairs, parse_float=floating, parse_constant=constant)


def read_decimal(token, name):
    try:
        d = Decimal(token)
    except InvalidOperation as exc:
        raise ValueError(name + ': invalid decimal token') from exc
    if not d.is_finite():
        raise ValueError(name + ': nonfinite token')
    f = float(d)
    if not math.isfinite(f) or (d != 0 and (f == 0 or abs(f) < 1e-150)):
        raise ValueError(name + ': raw decimal underflow/overflow')
    return f


def load_inputs(data_path=None, config_path=None):
    dp = ROOT/'data/observations.csv' if data_path is None else Path(data_path)
    cp = ROOT/'data/experiment_config.json' if config_path is None else Path(config_path)
    config = strict_json(cp.read_text(encoding='utf-8'))
    reader = csv.reader(io.StringIO(dp.read_text(encoding='utf-8')))
    if next(reader, None) != ['id', 'x', 'y']:
        raise ValueError('CSV header must be id,x,y')
    rows = []
    for row in reader:
        if len(row) != 3:
            raise ValueError('CSV must have exactly three fields on every row')
        rows.append([read_decimal(v, 'CSV') for v in row])
    return validate(rows, config)


@dataclass
class Cost:
    objective_calls: int = 0
    prediction_rows: int = 0
    loss_elements: int = 0
    gradient_calls: int = 0
    attempted_steps: int = 0
    committed_steps: int = 0


def internal_theta(theta):
    # This is an INTERNAL representability guard, not the initial [-10,10] domain.
    return vector(theta, 'internal theta', bound=1e100, raw=False)


def square_forward(data, theta, cost, objective='paired'):
    """Prevalidated rows only. Counts increments where work actually starts."""
    if objective not in ('paired', 'broadcast'):
        raise ValueError('unknown objective')
    theta = internal_theta(theta)
    cost.objective_calls += 1
    predictions = []
    for _, x, _ in data:
        cost.prediction_rows += 1
        p = theta[0] + theta[1] * x
        if not math.isfinite(p):
            raise FloatingPointError('nonfinite prediction')
        predictions.append(p)
    n = len(data)
    rows = []
    g = [0., 0.]
    loss = 0.
    cost.gradient_calls += 1
    if objective == 'paired':
        for row, p in zip(data, predictions):
            ident, x, y = row
            cost.loss_elements += 1
            r = p - y
            half = .5*r*r
            contribution = [r/n, r*x/n]
            if not all(math.isfinite(v) for v in [half, *contribution]):
                raise FloatingPointError('nonfinite local loss or gradient')
            g = [g[j] + contribution[j] for j in range(2)]
            loss += half/n
            rows.append({'id': int(ident), 'x': x, 'y': y, 'prediction': p,
                         'residual': r, 'half_square': half, 'mean_loss_contribution': half/n,
                         'd_half_square_d_residual': r, 'd_residual_d_prediction': 1.,
                         'd_prediction_d_theta': [1., x], 'mean_gradient_contribution': contribution})
        shape = [n]
    else:
        # Deliberate faulty cross-pairing. The formula and gradient are consistent
        # with EACH OTHER, but they no longer represent paired observations.
        matrix = []
        for i, p in enumerate(predictions):
            x = data[i][1]
            rr = []
            for _, _, y in data:
                cost.loss_elements += 1
                r = p-y
                half = .5*r*r
                gc = [r/(n*n), r*x/(n*n)]
                if not all(math.isfinite(v) for v in [half, *gc]):
                    raise FloatingPointError('nonfinite broadcast cell')
                loss += half/(n*n)
                g = [g[j]+gc[j] for j in range(2)]
                rr.append(r)
            matrix.append(rr)
        rows = matrix
        shape = [n, n]
    if not all(math.isfinite(v) for v in [loss, *g]):
        raise FloatingPointError('nonfinite aggregate')
    return {'theta': theta, 'objective': objective, 'residual_shape': shape,
            'rows': rows, 'loss': loss, 'gradient': g}


def norm(v):
    return math.hypot(*v)


def _run_path(data, config, fault='none', observer=None):
    if fault not in ('none', 'sign', 'broadcast', 'tiny_step', 'large_step'):
        raise ValueError('unknown injected fault')
    s = config
    theta = s['initial_parameters'][:]
    objective = 'broadcast' if fault == 'broadcast' else 'paired'
    eta = 1e-12 if fault == 'tiny_step' else 10. if fault == 'large_step' else s['learning_rate']
    c = Cost()
    old = square_forward(data, theta, c, objective)
    initial = copy.deepcopy(old)
    trace = []
    status = 'budget_exhausted'
    for step in range(s['max_steps']):
        true_g = old['gradient']
        if norm(true_g) <= s['gradient_tolerance']:
            status = 'gradient_tolerance'
            break
        c.attempted_steps += 1
        used_g = [-v for v in true_g] if fault == 'sign' else true_g[:]
        delta = [-eta*v for v in used_g]
        trial = [theta[j]+delta[j] for j in range(2)]
        actual_delta = [trial[j]-theta[j] for j in range(2)]
        record = {'attempt': step+1, 'old': old, 'used_gradient': used_g,
                  'learning_rate': eta, 'proposed_update': delta, 'update': actual_delta, 'trial_theta': trial,
                  'gradient_norm': norm(true_g), 'parameter_norm': norm(theta), 'update_norm': norm(actual_delta),
                  'relative_update': norm(actual_delta)/max(1., norm(theta)),
                  'directional_derivative': sum(true_g[j]*actual_delta[j] for j in range(2)),
                  'accepted': False}
        try:
            new = square_forward(data, trial, c, objective)
        except (FloatingPointError, OverflowError, ValueError) as exc:
            record['numeric_failure'] = str(exc)
            trace.append(record)
            status = 'numerical_failure'
            break
        record['new'] = new
        decrease = old['loss']-new['loss']
        record['loss_decrease'] = decrease
        record['relative_loss_change'] = abs(decrease)/max(1., abs(old['loss']))
        # A diagnostic fail-stop, not a line search and not a proof of correctness.
        if decrease < -1e-14*max(1., abs(old['loss'])):
            trace.append(record)
            status = 'loss_increase_rejected'
            if observer is not None:
                observer(copy.deepcopy(record))
            break
        record['accepted'] = True
        c.committed_steps += 1
        trace.append(record)
        theta, old = trial, new
        if observer is not None:
            observer(copy.deepcopy(record))
        if norm(new['gradient']) <= s['gradient_tolerance']:
            status = 'gradient_tolerance'
            break
        if record['relative_update'] <= s['step_tolerance'] or record['relative_loss_change'] <= s['loss_tolerance']:
            status = 'stagnation_gradient_large'
            break
    return {'label': fault, 'objective': objective, 'status': status,
            'initial': initial, 'trace': trace, 'final': old, 'cost': asdict(c)}


def run_path(data, config, fault='none', observer=None):
    rows, s = validate(data, config, observer)
    if type(fault) is not str or fault not in ('none', 'sign', 'broadcast', 'tiny_step', 'large_step'):
        raise ValueError('unknown injected fault')
    return _run_path(rows, s, fault, observer)


def softplus_scalar(t):
    # Finite t is guaranteed by callers; exp has nonpositive argument.
    return max(t, 0.) + math.log1p(math.exp(-abs(t)))


def sigmoid_scalar(t):
    if t >= 0:
        return 1./(1.+math.exp(-t))
    e = math.exp(t)
    return e/(1.+e)


def logistic_forward(data, theta, cost):
    theta = internal_theta(theta)
    cost.objective_calls += 1
    rows = []
    loss = 0.
    g = [0., 0.]
    n = len(data)
    cost.gradient_calls += 1
    for ident, x, y in data:
        cost.prediction_rows += 1
        z = theta[0]+theta[1]*x
        if not math.isfinite(z):
            raise FloatingPointError('nonfinite logit')
        sign = 1.-2.*y
        t = sign*z
        cost.loss_elements += 1
        ell = softplus_scalar(t)
        local = sign*sigmoid_scalar(t)
        gc = [local/n, local*x/n]
        loss += ell/n
        g = [g[j]+gc[j] for j in range(2)]
        rows.append({'id': int(ident), 'x': x, 'y': y, 'logit': z, 'signed_logit': t,
                     'loss': ell, 'd_loss_d_logit': local, 'mean_gradient_contribution': gc,
                     'exp_tail_underflowed': math.exp(-abs(t)) == 0.})
    if not all(math.isfinite(v) for v in [loss, *g]):
        raise FloatingPointError('nonfinite logistic aggregate')
    return {'theta': theta, 'rows': rows, 'loss': loss, 'gradient': g}


def _central(data, theta, h, objective):
    c = Cost()
    f = logistic_forward if objective == 'logistic' else lambda d, t, c: square_forward(d, t, c, objective)
    fd, effective = [], []
    for j in range(2):
        plus = theta[:]
        minus = theta[:]
        plus[j] += h
        minus[j] -= h
        if plus[j] == theta[j] or minus[j] == theta[j]:
            fd.append(None)
            effective.append([plus[j]-theta[j], theta[j]-minus[j]])
            continue
        fp = f(data, plus, c)['loss']
        fm = f(data, minus, c)['loss']
        fd.append((fp-fm)/(2*h))
        effective.append([plus[j]-theta[j], theta[j]-minus[j]])
    return {'gradient': fd, 'effective_displacements': effective, 'cost': asdict(c)}


def gradient_checks(data, s):
    th = s['fd_parameters']
    c = Cost()
    good = square_forward(data, th, c)['gradient']
    wrong = square_forward(data, th, c, 'broadcast')['gradient']
    rows = []
    for h in s['fd_steps']:
        fd = _central(data, th, h, 'paired')
        bd = _central(data, th, h, 'broadcast')
        lf = logistic_forward(data, th, c)
        ld = _central(data, th, h, 'logistic')
        def error(a, b):
            return None if any(v is None for v in b) else norm([a[j]-b[j] for j in range(2)])/max(1., norm(a), norm(b))
        rows.append({'h': h, 'paired': fd, 'broadcast': bd, 'logistic': ld,
                     'paired_error': error(good, fd['gradient']),
                     'sign_error': error([-v for v in good], fd['gradient']),
                     'broadcast_vs_intended_error': error(wrong, fd['gradient']),
                     'broadcast_self_error': error(wrong, bd['gradient']),
                     'logistic_error': error(lf['gradient'], ld['gradient'])})
    return {'theta': th, 'paired_gradient': good, 'broadcast_gradient': wrong,
            'table': rows, 'reference_evaluation_cost': asdict(c)}


def finite_json(value):
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return 'NaN' if math.isnan(value) else 'Infinity' if value > 0 else '-Infinity'
    if isinstance(value, np.ndarray):
        return finite_json(value.tolist())
    if isinstance(value, list):
        return [finite_json(v) for v in value]
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    return value


def stability_checks(data, s):
    cases = []
    c = Cost()
    naive_calls = 0
    naive_elements = 0
    for th in s['stress_parameters']:
        good = logistic_forward(data, th, c)
        z = np.array([row['logit'] for row in good['rows']], dtype=np.float64)
        y = np.array([row[2] for row in data], dtype=np.float64)
        # Intentionally wrong numerical representations; never average nonfinite
        # entries by nanmean. Strings preserve exact failure category in JSON.
        with np.errstate(over='ignore', under='ignore', divide='ignore', invalid='ignore'):
            naive_calls += 1
            naive_elements += len(data)
            naive_softplus = np.log(1.+np.exp(z))-y*z
            naive_calls += 1
            naive_elements += len(data)
            p = 1./(1.+np.exp(-z))
            naive_probability = -y*np.log(p)-(1.-y)*np.log(1.-p)
            naive_calls += 1
            naive_elements += len(data)
            subtractive_stable = np.logaddexp(0., z)-y*z
        cases.append({'theta': th, 'stable': good,
                      'naive_exp_then_log': finite_json(naive_softplus),
                      'naive_probability_logs': finite_json(naive_probability),
                      'stable_softplus_but_subtractive': subtractive_stable.tolist(),
                      'naive_nonfinite_rows': [i+1 for i,v in enumerate(naive_probability) if not math.isfinite(float(v))],
                      'status': 'numerical_representation_failure' if not np.all(np.isfinite(naive_probability)) else 'finite_but_cancellation_possible'})
    return {'same_input_rows': True, 'stable_cost': asdict(c),
            'naive_evaluation_calls': naive_calls, 'naive_loss_elements': naive_elements,
            'cases': cases}


def _report(data, s, observer=None):
    paths = {name: _run_path(data, s, name, observer) for name in ('none','sign','broadcast','tiny_step','large_step')}
    gc = gradient_checks(data, s)
    stability = stability_checks(data, s)
    intended_cost = Cost()
    intended_finals = {name: square_forward(data, path['final']['theta'], intended_cost) for name,path in paths.items()}
    probe = min(gc['table'], key=lambda row: abs(math.log10(row['h'])+5))
    def first(name, key):
        trace = paths[name]['trace']
        return trace[0].get(key) if trace else None
    conclusions = [
        {'fault': 'sign', 'evidence': {'check_h': probe['h'], 'sign_error_vs_intended': probe['sign_error'], 'first_directional_derivative': first('sign','directional_derivative'), 'path_status': paths['sign']['status']}, 'interpretation': 'A nonzero gradient with reversed sign gives an uphill local update; zero gradients cannot expose this sign error.', 'action': 'Repair sign before tuning learning rate; compare at nonstationary points.'},
        {'fault': 'broadcast', 'evidence': {'residual_shape': paths['broadcast']['initial']['residual_shape'], 'check_h': probe['h'], 'error_vs_intended': probe['broadcast_vs_intended_error'], 'self_error': probe['broadcast_self_error'], 'own_final_loss': paths['broadcast']['final']['loss'], 'intended_final_loss': intended_finals['broadcast']['loss']}, 'interpretation': 'Cross-pairing defines a different objective; matching its own finite differences cannot establish intended row pairing.', 'action': 'Restore row pairing and require residual shape exactly (n,).'},
        {'fault': 'naive_probability_logs', 'evidence': {'nonfinite_row_ids_by_case': [case['naive_nonfinite_rows'] for case in stability['cases']], 'case_statuses': [case['status'] for case in stability['cases']]}, 'interpretation': 'Compare each case with the finite signed-softplus result, including representable small tails.', 'action': 'Keep logits; do not use nanmean or clip away the evidence.'},
        {'fault': 'tiny_step', 'evidence': {'path_status': paths['tiny_step']['status'], 'first_gradient_norm': first('tiny_step','gradient_norm'), 'first_relative_update': first('tiny_step','relative_update')}, 'interpretation': 'A small step with a large gradient is stagnation, not stationarity.', 'action': 'Check update scale and representability together with gradient norm.'},
        {'fault': 'large_step', 'evidence': {'path_status': paths['large_step']['status'], 'first_directional_derivative': first('large_step','directional_derivative'), 'first_loss_decrease': first('large_step','loss_decrease')}, 'interpretation': 'A downhill local direction can still increase the loss after a large finite step.', 'action': 'After checking gradient and objective, reduce learning rate or use justified step selection.'}]
    return {'unit': '039', 'runtime': {'python': sys.version.split()[0], 'numpy': np.__version__, 'dtype': 'binary64', 'float_mantissa_bits': sys.float_info.mant_dig},
            'semantics': {'parameter_order': ['intercept_b','slope_w'], 'primary_objective': 'mean of paired half squared residuals', 'stability_objective': 'separate mean binary signed-softplus loss on same rows', 'row_order': 'input id order; no shuffling'},
            'data': data, 'configuration': s, 'paths': paths,
            'gradient_checks': gc, 'stability': stability, 'diagnosis': conclusions,
            'intended_final_evaluations': intended_finals, 'intended_final_evaluation_cost': asdict(intended_cost),
            'scope': 'Synthetic deterministic diagnostic experiment; no generalization claim.'}


def run_experiment(data, config, observer=None):
    rows, s = validate(data, config, observer)
    return _report(rows, s, observer)


def atomic_json(path, value):
    """Commit a complete strict JSON report; failure preserves prior file."""
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)+'\n'
    path = Path(path)
    if not path.parent.is_dir():
        raise ValueError('output parent must already exist')
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError('output must be a regular file path')
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, prefix='.unit039-', suffix='.tmp', delete=False) as f:
            tmp = Path(f.name)
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path)
    p.add_argument('--config', type=Path)
    p.add_argument('--output', type=Path, help='existing parent; atomically replace this report only')
    a = p.parse_args()
    data, config = load_inputs(a.data, a.config)
    # Validate destination before computations and protect all shipped inputs/code.
    if a.output is not None:
        dest = a.output.resolve()
        protected_names = ['README.md','lecture.md','lecture.pdf','lab.md','lab.pdf',
                           'answers.md','answers.pdf','experiment.py','experiment.ipynb',
                           'audit.py','library_check.py','build_assets.py','requirements.txt',
                           'environment.yml','source-checks.json','verification.json',
                           'test-result.json','experiment-result.json','library-result.json',
                           'data/README.md','data/observations.csv','data/experiment_config.json']
        forbidden = {(ROOT/name).resolve() for name in protected_names}
        forbidden.update(q.resolve() for q in (ROOT/'figures').glob('*.png'))
        if dest in forbidden or dest in {q.resolve() for q in (a.data, a.config) if q is not None}:
            raise ValueError('refuse overwriting a source/artifact input')
        if not a.output.parent.is_dir() or a.output.is_symlink() or (a.output.exists() and not a.output.is_file()):
            raise ValueError('invalid output path')
    report = run_experiment(data, config)
    if a.output is not None:
        atomic_json(a.output, report)
    reference = report['paths']['none']
    committed = [row['new'] for row in reference['trace'] if row['accepted']]
    second = committed[1] if len(committed) >= 2 else None
    print(json.dumps({'statuses': {k: v['status'] for k,v in report['paths'].items()},
                      'completed_updates': reference['cost']['committed_steps'],
                      'final_theta': reference['final']['theta'],
                      'final_loss': reference['final']['loss'],
                      'two_round_theta': second['theta'] if second else None,
                      'two_round_loss': second['loss'] if second else None}, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
