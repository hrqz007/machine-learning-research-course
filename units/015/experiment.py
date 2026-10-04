"""第015讲 曲率与局部近似。固定合成数据的小CPU离线实验。

python experiment.py 在脚本所在目录写入 outputs；核心计算仅依赖 NumPy。
所有数值诊断均为有限 float64 核验，不代替光滑性和极值定理的证明。
"""
from pathlib import Path
from numbers import Real
from fractions import Fraction as F
from functools import wraps
import csv
import json
import sys
import numpy as np

BASE = Path(__file__).resolve().parent


def scalar(value, name='value'):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(name + ' 必须是实数标量，不接受布尔、字符串或复数')
    try:
        result = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError(name + ' 超出浮点范围') from error
    if not np.isfinite(result):
        raise ValueError(name + ' 必须有限')
    return result


def real_array(values, name='array'):
    def check(value):
        if isinstance(value, np.ndarray):
            if value.dtype.kind not in 'iuf':
                raise ValueError(name + ' 拒绝布尔、字符串、复数和object数组')
        elif isinstance(value, (list, tuple)):
            for item in value:
                check(item)
        else:
            scalar(value, name)
    check(values)
    try:
        result = np.asarray(values, dtype=float)
    except (ValueError, OverflowError) as error:
        raise ValueError(name + ' 不能构成实数组') from error
    if not np.all(np.isfinite(result)):
        raise ValueError(name + ' 必须全部有限')
    return result


def vector(values, name='theta'):
    result = real_array(values, name)
    if result.shape != (2,):
        raise ValueError(name + ' 必须形状(2,)，不接收行列矩阵')
    return result


def data_vectors(x, y):
    x, y = real_array(x, 'x'), real_array(y, 'y')
    if x.ndim != 1 or x.size == 0 or y.shape != x.shape:
        raise ValueError('x,y 必须同形且非空的一维数组，禁止目标广播')
    return x, y


def checked(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            try:
                return function(*args, **kwargs)
            except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as error:
                raise ValueError(function.__name__ + ' 的计算超出有限可处理范围') from error
    return wrapper


def load_data():
    with (BASE / 'data/observations.csv').open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['record_id', 'x', 'y']:
            raise ValueError('CSV 表头错误')
        rows = list(reader)
    ids = [r['record_id'] for r in rows]
    if any(not item for item in ids) or len(set(ids)) != len(ids):
        raise ValueError('record_id 必须非空且唯一')
    x, y = data_vectors([float(r['x']) for r in rows], [float(r['y']) for r in rows])
    if not np.array_equal(x, [0, 1, 2]) or not np.array_equal(y, [1, 2, 2]):
        raise ValueError('本讲的已知答案要求固定三条记录，不能静默替换数据')
    return x, y


@checked
def loss(theta, x, y):
    a, b = vector(theta)
    x, y = data_vectors(x, y)
    residual = a*x + b*b - y
    return scalar(np.mean(residual**2), 'loss')


@checked
def gradient(theta, x, y):
    a, b = vector(theta)
    x, y = data_vectors(x, y)
    residual = a*x + b*b - y
    return vector([2*np.mean(residual*x), 4*b*np.mean(residual)], 'gradient')


@checked
def hessian(theta, x, y):
    a, b = vector(theta)
    x, y = data_vectors(x, y)
    residual = a*x + b*b - y
    return real_array([[2*np.mean(x*x), 4*b*np.mean(x)],
                       [4*b*np.mean(x), 8*b*b + 4*np.mean(residual)]], 'Hessian')


@checked
def expanded_loss(theta):
    """独立多项式实现，仅对应固定 CSV；不能替代任意数据的 loss。"""
    a, b = vector(theta)
    return scalar((5/3)*a*a + 2*a*b*b - 4*a + b**4 - (10/3)*b*b + 3)


@checked
def expanded_hessian(theta):
    a, b = vector(theta)
    return real_array([[10/3, 4*b], [4*b, 4*a + 12*b*b - 20/3]])


@checked
def symmetric_matrix(H):
    H = real_array(H, 'H')
    if H.shape != (2, 2):
        raise ValueError('H 必须形状(2,2)')
    scale = max(1., float(np.max(np.abs(H))))
    if np.max(np.abs(H-H.T)) > 1e-12*scale:
        raise ValueError('H 不是给定容差内的对称矩阵；拒绝静默忽略一个三角区')
    return H


@checked
def quadratic_form(H, displacement):
    H, h = symmetric_matrix(H), vector(displacement, 'h')
    return scalar(h @ H @ h, 'quadratic form')


@checked
def taylor(theta, displacement, x, y, order=2):
    if isinstance(order, (bool, np.bool_)) or order not in (1, 2):
        raise ValueError('order 只能为1或2')
    theta, h = vector(theta), vector(displacement, 'h')
    value = loss(theta, x, y) + gradient(theta, x, y) @ h
    if order == 2:
        value += .5*quadratic_form(hessian(theta, x, y), h)
    return scalar(value, 'Taylor approximation')


@checked
def exact_remainder(theta, displacement, x, y):
    """原模型精确二阶余项；数值上避免两个接近函数值相减。"""
    _, b = vector(theta)
    da, db = vector(displacement, 'h')
    x, y = data_vectors(x, y)
    return scalar(2*np.mean(x)*da*db**2 + 4*b*db**3 + db**4)


@checked
def finite_hessian(theta, x, y, step=1e-5):
    """逐列差分梯度，保留原始不对称误差，不用对称化掩盖错误。"""
    theta, step = vector(theta), scalar(step, 'step')
    if step <= 0:
        raise ValueError('step 必须正')
    denominator = scalar(2*step, 'denominator')
    result = np.empty((2, 2))
    for j in range(2):
        plus, minus = theta.copy(), theta.copy()
        plus[j] += step
        minus[j] -= step
        if plus[j] == theta[j] or minus[j] == theta[j]:
            raise ValueError('采样位移在当前尺度不可分辨')
        result[:, j] = (gradient(plus, x, y)-gradient(minus, x, y))/denominator
    return real_array(result, 'finite Hessian')


@checked
def classify(grad, H, grad_tol=1e-9, eig_tol=1e-10):
    """返回数值候选类别；exact-stationary 是使用数学判据的另一个前提。"""
    grad, H = vector(grad, 'gradient'), symmetric_matrix(H)
    grad_tol, eig_tol = scalar(grad_tol), scalar(eig_tol)
    if grad_tol < 0 or eig_tol < 0:
        raise ValueError('容差不能负')
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    eigenvalues, eigenvectors = real_array(eigenvalues), real_array(eigenvectors)
    threshold = scalar(eig_tol*max(1., float(np.max(np.abs(eigenvalues)))))
    pos, neg = eigenvalues > threshold, eigenvalues < -threshold
    if np.all(pos):
        curvature = 'positive_definite'
    elif np.all(neg):
        curvature = 'negative_definite'
    elif np.any(pos) and np.any(neg):
        curvature = 'indefinite'
    else:
        curvature = 'near_semidefinite_or_zero'
    stationary = bool(np.max(np.abs(grad)) <= grad_tol)
    conclusion = 'nonstationary' if not stationary else {
        'positive_definite': 'strict_local_minimum_candidate',
        'negative_definite': 'strict_local_maximum_candidate',
        'indefinite': 'saddle_candidate',
        'near_semidefinite_or_zero': 'inconclusive'}[curvature]
    return {'gradient_within_tolerance': stationary, 'curvature': curvature,
            'conclusion': conclusion, 'eigenvalues': eigenvalues.tolist(),
            'eigenvectors_columns': eigenvectors.tolist(),
            'eigenvalue_threshold': threshold,
            'warning': '数值候选；不能证明梯度精确为零或邻域满足C2条件'}


def rotated_quadratic():
    angle = np.pi/6
    Q = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    return Q @ np.diag([1., 16.]) @ Q.T, Q


@checked
def quadratic_descent(H, initial, step, iterations=20):
    H, point, step = symmetric_matrix(H), vector(initial), scalar(step)
    if step <= 0 or isinstance(iterations, bool) or not isinstance(iterations, int) or not 0 <= iterations <= 10000:
        raise ValueError('要求正步长及0到10000的整数步数')
    points = [point.copy()]
    for _ in range(iterations):
        point = vector(point-step*(H@point))
        points.append(point.copy())
    return np.array(points)


def exact_fraction_checks():
    x, y = [F(0), F(1), F(2)], [F(1), F(2), F(2)]
    a, b = F(1), F(1, 2)
    r = [a*v+b*b-w for v, w in zip(x, y)]
    L = sum(v*v for v in r)/3
    g = [2*sum(v*w for v, w in zip(r, x))/3, 4*b*sum(r)/3]
    H = [[2*sum(v*v for v in x)/3, 4*b*sum(x)/3],
         [4*b*sum(x)/3, 8*b*b+4*sum(r)/3]]
    assert L == F(19, 48) and g == [F(-1, 6), F(-5, 6)]
    assert H == [[F(10, 3), F(2)], [F(2), F(1, 3)]]
    u = [F(3, 5), F(4, 5)]
    coefficients = [sum(v*w for v, w in zip(g, u)),
                    sum(u[i]*H[i][j]*u[j] for i in range(2) for j in range(2))/2,
                    2*u[0]*u[1]**2+4*b*u[1]**3, u[1]**4]
    assert coefficients == [F(-23, 30), F(5, 3), F(224, 125), F(256, 625)]
    return {'loss': str(L), 'gradient': list(map(str, g)),
            'Hessian': [[str(v) for v in row] for row in H],
            'direction_coefficients_orders_1_to_4': list(map(str, coefficients))}


def error_sweep(x, y):
    theta, u = np.array([1., .5]), np.array([.6, .8])
    cancel = np.array([-1., 1.])/np.sqrt(2)
    rows = []
    for t in np.logspace(-1, -12, 23):
        h = t*u
        truth = loss(theta+h, x, y)
        p1, p2 = taylor(theta, h, x, y, 1), taylor(theta, h, x, y, 2)
        r2 = exact_remainder(theta, h, x, y)
        r1 = .5*quadratic_form(hessian(theta, x, y), h)+r2
        # cancellation direction has exact cubic coefficient zero; use t^4/4
        rows.append({'step': float(t), 'error1_subtraction': abs(truth-p1),
                     'error2_subtraction': abs(truth-p2),
                     'error1_polynomial': abs(r1), 'error2_polynomial': abs(r2),
                     'r1_over_t2': abs(r1)/t**2, 'r2_over_t3': abs(r2)/t**3,
                     'cancel_error_polynomial': float(t**4/4),
                     'cancel_error_subtraction': abs(loss(theta+t*cancel, x, y)-taylor(theta, t*cancel, x, y))})
    return rows


def run_all(write=True):
    x, y = load_data()
    exact = exact_fraction_checks()
    theta = np.array([1., .5])
    L, g, H = loss(theta, x, y), gradient(theta, x, y), hessian(theta, x, y)
    assert np.allclose(g, [-1/6, -5/6], atol=1e-14, rtol=0)
    assert np.allclose(H, [[10/3, 2], [2, 1/3]], atol=1e-14, rtol=0)
    assert np.allclose(np.linalg.eigvalsh(H), [-2/3, 13/3], atol=1e-14, rtol=0)
    audit = []
    for a in [-.5, 0., .5, 1., 1.5]:
        for b in [-1., -.5, 0., .5, 1.]:
            point = np.array([a, b]); h = np.array([.013, -.021])
            finite = finite_hessian(point, x, y)
            actual = hessian(point, x, y)
            assert np.allclose(actual, expanded_hessian(point), atol=2e-14, rtol=0)
            assert abs(loss(point, x, y)-expanded_loss(point)) < 2e-14
            assert np.allclose(finite, actual, atol=2e-8, rtol=2e-8)
            rem = loss(point+h, x, y)-taylor(point, h, x, y)
            assert abs(rem-exact_remainder(point, h, x, y)) < 2e-14
            audit.append({'a': a, 'b': b, 'hessian_max_abs_error': float(np.max(np.abs(finite-actual))),
                          'raw_symmetry_error': float(np.max(np.abs(finite-finite.T)))})
    classifications = {'base_nonstationary': classify(g, H)}
    points = {'stationary_saddle': [6/5, 0],
              'minimum_positive_b': [.5, np.sqrt(7/6)],
              'minimum_negative_b': [.5, -np.sqrt(7/6)]}
    for name, point in points.items():
        classifications[name] = classify(gradient(point, x, y), hessian(point, x, y))
        classifications[name]['point'] = point
        classifications[name]['loss'] = loss(point, x, y)
    assert classifications['base_nonstationary']['conclusion'] == 'nonstationary'
    assert classifications['stationary_saddle']['conclusion'] == 'saddle_candidate'
    assert abs(loss(points['stationary_saddle'], x, y)-3/5) < 1e-14
    assert np.allclose(hessian(points['stationary_saddle'], x, y), np.diag([10/3, -28/15]), atol=1e-14)
    for name in ['minimum_positive_b', 'minimum_negative_b']:
        assert classifications[name]['conclusion'] == 'strict_local_minimum_candidate'
        assert abs(classifications[name]['loss']-1/18) < 1e-14
    assert classify([0, 0], [[-2, 0], [0, -4]])['conclusion'] == 'strict_local_maximum_candidate'
    assert classify([0, 0], [[2, 0], [0, 0]])['conclusion'] == 'inconclusive'
    assert classify([0, 0], [[0, 0], [0, 0]])['conclusion'] == 'inconclusive'
    assert classify([0, 0], [[1, 0], [0, -1e-12]])['conclusion'] == 'inconclusive'
    A, Q = rotated_quadratic()
    assert np.allclose(Q.T@Q, np.eye(2), atol=1e-14)
    assert np.allclose(np.linalg.eigvalsh(A), [1, 16], atol=1e-13)
    eigenvalues, vectors = np.linalg.eigh(A)
    assert np.allclose(A@vectors, vectors*eigenvalues, atol=1e-13)
    assert np.allclose(vectors.T@vectors, np.eye(2), atol=1e-13)
    initial = Q@np.array([1., 1.])
    stable = quadratic_descent(A, initial, .1, 20)
    unstable = quadratic_descent(A, initial, .15, 20)
    assert np.allclose(Q.T@stable[1], [.9, -.6], atol=1e-13)
    assert np.allclose(Q.T@unstable[1], [.85, -1.4], atol=1e-13)
    values = np.array([.5*quadratic_form(A, point) for point in stable])
    assert np.all(np.diff(values) < 0)
    assert .5*quadratic_form(A, unstable[-1]) > .5*quadratic_form(A, unstable[0])
    # Independent completed-square proof verified at the same finite grid.
    for item in audit:
        a, b = item['a'], item['b']; A0, C0 = a-.5, b*b-7/6
        certificate = 1/18+(5/3)*(A0+3*C0/5)**2+(2/5)*C0*C0
        assert abs(loss([a, b], x, y)-certificate) < 2e-14
    boundaries = []
    def rejects(name, function):
        try:
            function()
        except ValueError:
            boundaries.append({'case': name, 'status': 'rejected_as_expected'})
        else:
            raise AssertionError('未拒绝非法输入: '+name)
    bad_vectors = [[True, .5], ['1', .5], [1+0j, .5], [1, float('nan')], [1, float('inf')],
                   [[1, .5]], [[1], [.5]], [1], [1, 2, 3], np.array([1, .5], dtype=object)]
    for index, item in enumerate(bad_vectors):
        rejects('invalid_theta_'+str(index), lambda item=item: loss(item, x, y))
    for index, (xx, yy) in enumerate([([], []), ([1, 2], [1]), ([1, 2], [[1], [2]]), ([True], [1]), ([1], [float('nan')])]):
        rejects('invalid_data_'+str(index), lambda xx=xx, yy=yy: loss(theta, xx, yy))
    for value in [0, -1, float('nan'), float('inf'), True, 1e-20, 1e308]:
        rejects('invalid_difference_step_'+str(value), lambda value=value: finite_hessian(theta, x, y, value))
    rejects('overflow_loss', lambda: loss([1e308, 1e308], x, y))
    rejects('overflow_hessian', lambda: hessian([1, 1e308], x, y))
    rejects('overflow_remainder', lambda: exact_remainder(theta, [1e308, 1e308], x, y))
    rejects('asymmetric_H', lambda: classify([0, 0], [[1, 9], [0, 1]]))
    rejects('nonsquare_H', lambda: classify([0, 0], [[1, 0]]))
    rejects('nonfinite_H', lambda: classify([0, 0], [[1, 0], [0, float('inf')]]))
    rejects('negative_gradient_tolerance', lambda: classify([0, 0], np.eye(2), grad_tol=-1))
    rejects('negative_eigen_tolerance', lambda: classify([0, 0], np.eye(2), eig_tol=-1))
    rejects('overflow_eigen_tolerance', lambda: classify([0, 0], np.eye(2)*1e100, eig_tol=1e308))
    rejects('invalid_order', lambda: taylor(theta, [0, 0], x, y, 3))
    rejects('bool_order', lambda: taylor(theta, [0, 0], x, y, True))
    rejects('negative_descent_step', lambda: quadratic_descent(A, initial, -1))
    rejects('invalid_iteration_count', lambda: quadratic_descent(A, initial, .1, True))
    assert loss([0, 0], [0], [0]) == 0
    assert np.array_equal(hessian([0, 0], [0], [0]), np.zeros((2, 2)))
    assert taylor(theta, [0, 0], x, y) == L
    assert quadratic_form(H, [0, 0]) == 0
    boundaries.append({'case': 'zero_increment_and_single_sample', 'status': 'accepted_as_expected'})
    rows = error_sweep(x, y)
    fd_sweep = []
    for step in np.logspace(-1, -16, 16):
        try:
            hh = finite_hessian(theta, x, y, float(step))
            fd_sweep.append({'step': float(step), 'status': 'ok',
                             'max_abs_error': float(np.max(np.abs(hh-H)))})
        except ValueError as error:
            fd_sweep.append({'step': float(step), 'status': 'rejected', 'reason': str(error)})
    report = {'unit': '015', 'status': 'passed', 'python': sys.version.split()[0],
              'numpy': np.__version__, 'data': {'x': x.tolist(), 'y': y.tolist(), 'synthetic': True},
              'fraction_checks': exact, 'base': {'loss': L, 'gradient': g.tolist(), 'Hessian': H.tolist()},
              'analytic_fd_points': len(audit), 'max_hessian_difference_error': max(v['hessian_max_abs_error'] for v in audit),
              'classifications': classifications,
              'quadratic_descent': {'eigenvalues': [1, 16], 'stable_step': .1,
                                    'stable_final_loss': float(values[-1]), 'unstable_step': .15,
                                    'unstable_final_loss': .5*quadratic_form(A, unstable[-1])},
              'boundary_case_count': len(boundaries), 'boundary_cases': boundaries,
              'error_sweep': rows, 'finite_hessian_sweep': fd_sweep,
              'limits': ['有限点数值核验不能证明可微或全局最优', '实验不是泛化性能评价']}
    if write:
        out = BASE/'outputs'; out.mkdir(exist_ok=True)
        (out/'experiment_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        with (out/'approximation_errors.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        with (out/'hessian_audit.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(audit[0])); writer.writeheader(); writer.writerows(audit)
    return report


if __name__ == '__main__':
    report = run_all()
    print(json.dumps({key: report[key] for key in ['unit', 'status', 'python', 'numpy', 'fraction_checks',
          'analytic_fd_points', 'max_hessian_difference_error', 'boundary_case_count', 'quadratic_descent']},
          ensure_ascii=False, indent=2))
