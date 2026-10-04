"""第016讲 凸性与约束的几何。离线、固定合成数据、float64 教学实验。

python experiment.py 在本目录 outputs 中重建报告；核心计算只需 NumPy。
有限网格、数值秩和浮点容差都是诊断，不替代正文的全域证明。
"""
from pathlib import Path
from fractions import Fraction as Q
from numbers import Real
from functools import wraps
import csv
import json
import sys
import numpy as np

BASE = Path(__file__).resolve().parent
ATOL = 1e-10
WEIGHT_ATOL = 1e-12


def scalar(value, name='value'):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(name + ' 必须是实数标量，不接受布尔、字符串或复数')
    try:
        out = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError(name + ' 超出 float64 范围') from error
    if not np.isfinite(out):
        raise ValueError(name + ' 必须有限')
    return out


def real_array(values, name='array'):
    """先审原类型，后转 float64；拒绝被隐式转成数字的混合布尔/字符串。"""
    def visit(value):
        if isinstance(value, np.ndarray):
            if value.dtype.kind not in 'iuf':
                raise ValueError(name + ' 拒绝布尔、字符串、复数和 object 数组')
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        else:
            scalar(value, name)
    visit(values)
    try:
        out = np.asarray(values, dtype=np.float64)
    except (ValueError, OverflowError, TypeError) as error:
        raise ValueError(name + ' 不能构成实数组') from error
    if not np.all(np.isfinite(out)):
        raise ValueError(name + ' 必须全部有限')
    return out


def vector(values, name='vector', size=None):
    out = real_array(values, name)
    if out.ndim != 1 or out.size == 0 or (size is not None and out.size != size):
        raise ValueError(name + ' 必须是指定长度的非空一维数组，禁止广播')
    return out


def checked(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            try:
                return function(*args, **kwargs)
            except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as error:
                raise ValueError(function.__name__ + ' 超出本教学实现的有限数值范围') from error
    return wrapper


@checked
def squared_distance(x, y):
    x = vector(x, 'x')
    y = vector(y, 'y', x.size)
    delta = vector(x-y, 'distance displacement')
    return scalar(np.dot(delta, delta), 'squared distance')


@checked
def project_interval(value, lower, upper):
    value, lower, upper = (scalar(a, n) for a, n in
                            [(value, 'value'), (lower, 'lower'), (upper, 'upper')])
    if lower > upper:
        raise ValueError('区间下界不能大于上界')
    return scalar(min(upper, max(lower, value)), 'projection')


@checked
def project_box(values, lower, upper):
    values = vector(values, 'values')
    lower = vector(lower, 'lower', values.size)
    upper = vector(upper, 'upper', values.size)
    if np.any(lower > upper):
        raise ValueError('每个坐标的下界都不能大于上界')
    return vector(np.clip(values, lower, upper), 'projection', values.size)


@checked
def project_ball(values, center, radius):
    values = vector(values, 'values')
    center = vector(center, 'center', values.size)
    radius = scalar(radius, 'radius')
    if radius < 0:
        raise ValueError('radius 必须非负')
    if radius == 0:
        return center.copy()
    delta = vector(values-center, 'ball displacement')
    distance = scalar(np.linalg.norm(delta), 'ball norm')
    # 极小非零向量的平方和可能下溢；不可把它误称为精确零。
    if distance == 0 and np.any(delta != 0):
        raise ValueError('球距离下溢，无法可靠判定内外')
    if distance <= radius:
        return values.copy()
    scale = scalar(radius/distance, 'ball radial scale')
    if scale == 0:
        raise ValueError('球径向缩放系数下溢，不能返回球心充当投影')
    result = vector(center+scale*delta, 'projection', values.size)
    if np.array_equal(result, center):
        raise ValueError('球心相加吞掉非零投影位移，请缩放问题')
    # 在本教学尺度内核对几何，不宣称任意尺度稳定。
    projected_delta = vector(result-center, 'projected displacement')
    residual = scalar(np.linalg.norm(projected_delta), 'projected norm')
    if residual == 0 and np.any(projected_delta != 0):
        raise ValueError('投影后球范数下溢，无法可靠核对边界')
    if residual > radius + ATOL*max(1.0, radius):
        raise ValueError('球投影可行残差超出容差')
    return result


def normal_data(values, normal, rhs):
    values = vector(values, 'values')
    normal = vector(normal, 'normal', values.size)
    rhs = scalar(rhs, 'rhs')
    norm2 = scalar(np.dot(normal, normal), 'normal norm squared')
    if norm2 == 0:
        raise ValueError('法向量为零或其长度在 float64 中无法解析')
    violation = scalar(np.dot(normal, values)-rhs, 'linear residual')
    return values, normal, rhs, norm2, violation


@checked
def project_hyperplane(values, normal, rhs):
    values, normal, rhs, norm2, violation = normal_data(values, normal, rhs)
    result = vector(values-(violation/norm2)*normal, 'projection', values.size)
    residual = abs(scalar(np.dot(normal, result)-rhs, 'projected residual'))
    if residual > ATOL*max(1.0, abs(rhs)):
        raise ValueError('超平面可行残差超出教学容差，请先缩放问题')
    return result


@checked
def project_halfspace(values, normal, rhs):
    values, normal, rhs, norm2, violation = normal_data(values, normal, rhs)
    if violation <= 0:
        return values.copy()
    return project_hyperplane(values, normal, rhs)


@checked
def project_affine(values, matrix, rhs):
    """只接收1≤m≤n、数值满行秩的小矩阵；拒绝冗余/矛盾行。

    满行秩 A 的 Az=b 总有解。lstsq 的返回残差数组可能为空，
    因而必须自己计算 A@p-b，不能把空数组解释为零误差。
    """
    values = vector(values, 'values')
    matrix = real_array(matrix, 'matrix')
    if matrix.ndim != 2 or matrix.shape[1] != values.size:
        raise ValueError('matrix 必须形状(m,n)，n 等于向量长度')
    m, n = matrix.shape
    if not 1 <= m <= n:
        raise ValueError('本接口要求 1<=m<=n 的独立等式')
    rhs = vector(rhs, 'rhs', m)
    target = vector(rhs-matrix@values, 'correction target', m)
    correction, unused_residuals, rank, singular_values = np.linalg.lstsq(matrix, target, rcond=None)
    if rank != m:
        raise ValueError('matrix 不是数值满行秩；本接口拒绝冗余或矛盾等式')
    vector(singular_values, 'singular values', m)
    correction = vector(correction, 'correction', n)
    result = vector(values+correction, 'projection', n)
    residual = vector(matrix@result-rhs, 'affine residual', m)
    threshold = ATOL*max(1.0, scalar(np.linalg.norm(rhs), 'rhs norm'))
    if scalar(np.linalg.norm(residual), 'residual norm') > threshold:
        raise ValueError('仿射投影可行残差超出教学容差，请检查尺度')
    return result


def weighted_points(points, weights):
    points = real_array(points, 'points')
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] == 0:
        raise ValueError('points 必须是非空(m,n)数组；标量点也写成(m,1)')
    weights = vector(weights, 'weights', points.shape[0])
    if np.any(weights < 0):
        raise ValueError('凸组合权重不能为负')
    total = scalar(np.sum(weights), 'weight sum')
    if abs(total-1.0) > WEIGHT_ATOL:
        raise ValueError('权重总和必须在绝对容差1e-12内等于1，不自动归一化')
    return points, weights


@checked
def convex_combination(points, weights):
    points, weights = weighted_points(points, weights)
    return vector(weights@points, 'weighted point', points.shape[1])


@checked
def jensen_diagnostic(function, points, weights):
    """有限点诊断。检查数值；不判断用户函数是否全域凸或定义域是否凸。"""
    points, weights = weighted_points(points, weights)
    average = convex_combination(points, weights)  # 必须在任何 callback 之前有限。
    left = scalar(function(average.copy()), 'function at average')
    values = vector([scalar(function(point.copy()), 'function at point') for point in points],
                    'function values', len(points))
    right = scalar(weights@values, 'weighted function values')
    gap = scalar(right-left, 'Jensen gap')
    return {'average': average.tolist(), 'left': left, 'right': right, 'gap': gap,
            'holds_within_tolerance': bool(gap >= -ATOL), 'weight_sum': float(np.sum(weights))}


@checked
def line_samples(function, base, direction, steps):
    """沿直线采样，以演示可行方向或一维截面；不自动保证采样点可行。"""
    base = vector(base, 'base')
    direction = vector(direction, 'direction', base.size)
    steps = vector(steps, 'steps')
    # 先完整构造并检查所有采样点，之后才允许调用用户函数。
    samples = real_array(base[None, :]+steps[:, None]*direction[None, :], 'line samples')
    values = vector([scalar(function(point.copy()), 'callback return') for point in samples],
                    'sample values', len(steps))
    return samples, values


@checked
def certificate_values(values, candidate, feasible_points):
    values = vector(values, 'values')
    candidate = vector(candidate, 'candidate', values.size)
    feasible_points = real_array(feasible_points, 'feasible_points')
    if feasible_points.ndim != 2 or feasible_points.shape[1] != values.size or len(feasible_points) == 0:
        raise ValueError('feasible_points 必须为非空(k,n)数组')
    # 名称说明用途；是否真的可行须由调用者独立检查。
    return vector((feasible_points-candidate)@(values-candidate), 'certificate values', len(feasible_points))


def load_data():
    with (BASE/'data/points2d.csv').open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['point_id', 'z1', 'z2']:
            raise ValueError('points2d.csv 表头错误')
        rows = list(reader)
    names = [r['point_id'] for r in rows]
    points = real_array([[float(r['z1']), float(r['z2'])] for r in rows], 'points CSV')
    if names != ['main', 'inside', 'negative', 'origin'] or not np.array_equal(points, [[3,2],[.5,.5],[-1,3],[0,0]]):
        raise ValueError('本讲已知答案要求保留固定点表；修改数据需另建实验')
    with (BASE/'data/affine_system.csv').open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['row_id', 'a1', 'a2', 'a3', 'rhs']:
            raise ValueError('affine_system.csv 表头错误')
        rows = list(reader)
    matrix = real_array([[float(r[k]) for k in ['a1','a2','a3']] for r in rows], 'A CSV')
    rhs = vector([float(r['rhs']) for r in rows], 'b CSV')
    if [r['row_id'] for r in rows] != ['E1','E2'] or not np.array_equal(matrix, [[1,1,0],[0,1,1]]) or not np.array_equal(rhs, [1,0]):
        raise ValueError('仿射例要求保留固定两条等式')
    return names, points, matrix, rhs


def require(condition, message='课堂验收失败'):
    if not condition:
        raise AssertionError(message)


def fraction_checks():
    """分数算术独立于浮点投影实现，用已知几何构造检查结论。"""
    v = [Q(3), Q(2)]
    box = [min(Q(2), max(Q(0), v[0])), min(Q(1), max(Q(0), v[1]))]
    line_lambda = (v[0]+v[1]-Q(1))/Q(2)
    line = [z-line_lambda for z in v]
    # 由等式写一般可行点 (1-s,s,-s)。距离导数 6s+2=0。
    s = -Q(1,3)
    affine = [1-s, s, -s]
    affine_residual = [a-b for a,b in zip([Q(3),Q(2),Q(1)], affine)]
    affine_d2 = sum(r*r for r in affine_residual)
    affine_dot = affine_residual[0]-affine_residual[1]+affine_residual[2]
    weights, x = [Q(1,2),Q(1,4),Q(1,4)], [Q(1),Q(2),Q(4)]
    mean = sum(w*a for w,a in zip(weights,x))
    rhs = sum(w*a*a for w,a in zip(weights,x))
    require(box == [Q(2), Q(1)] and line == [Q(1), Q(0)], '验收条件未满足')
    require(affine == [Q(4,3),Q(-1,3),Q(1,3)] and affine_d2 == Q(26,3) and affine_dot == 0, '验收条件未满足')
    require(mean == 2 and rhs == Q(11,2) and rhs-mean*mean == Q(3,2), '验收条件未满足')
    return {'box': [str(z) for z in box], 'line': [str(z) for z in line],
            'affine': [str(z) for z in affine], 'affine_squared_distance': str(affine_d2),
            'affine_residual_dot_null_direction': str(affine_dot),
            'jensen_mean': str(mean), 'jensen_left': str(mean*mean),
            'jensen_right': str(rhs), 'jensen_gap': str(rhs-mean*mean)}


def boundary_checks():
    results = []
    def reject(name, call):
        try:
            call()
        except ValueError as error:
            results.append({'case': name, 'status': 'rejected_as_expected', 'message': str(error)})
        else:
            raise AssertionError(name + ' 未拒绝')
    bad_vectors = [[], [[3,2]], [[3],[2]], [True,2], ['3',2], [3+0j,2],
                   np.array([3,2],dtype=object), [np.nan,2], [np.inf,2], [3]]
    for index, value in enumerate(bad_vectors):
        reject(f'invalid_box_input_{index}', lambda value=value: project_box(value,[0,0],[2,1]))
    reject('box_broadcast_lower', lambda: project_box([3,2],[0],[2,1]))
    reject('box_reversed_bounds', lambda: project_box([3,2],[0,2],[2,1]))
    reject('interval_bool', lambda: project_interval(True,0,1))
    reject('interval_reversed', lambda: project_interval(0,1,0))
    reject('interval_nonfinite', lambda: project_interval(np.inf,0,1))
    for radius in [-1, True, np.inf, '2']:
        reject('invalid_radius_'+str(radius), lambda radius=radius: project_ball([3,2],[0,0],radius))
    reject('ball_wrong_center_shape', lambda: project_ball([3,2],[0],2))
    reject('ball_displacement_overflow', lambda: project_ball([1e308],[-1e308],2))
    reject('ball_norm_overflow', lambda: project_ball([1e200,1e200],[0,0],2))
    reject('ball_norm_underflow', lambda: project_ball([1e-250],[0],1))
    reject('ball_scale_underflow', lambda: project_ball([1e150],[0],1e-200))
    reject('ball_projected_norm_underflow', lambda: project_ball([1],[0],1e-200))
    reject('ball_center_addition_loss', lambda: project_ball([2e100],[1e100],1e-50))
    reject('distance_overflow', lambda: squared_distance([1e200],[0]))
    for name, call in [
        ('hyperplane_zero_normal', lambda: project_hyperplane([3,2],[0,0],1)),
        ('halfspace_zero_normal', lambda: project_halfspace([3,2],[0,0],1)),
        ('normal_wrong_shape', lambda: project_hyperplane([3,2],[[1,1]],1)),
        ('normal_mixed_bool', lambda: project_hyperplane([3,2],[1,True],1)),
        ('normal_squared_overflow', lambda: project_hyperplane([3,2],[1e200,1e200],1)),
        ('normal_squared_underflow', lambda: project_hyperplane([3],[1e-250],1)),
        ('linear_residual_overflow', lambda: project_hyperplane([1e308],[2],0)),
        ('rhs_bool', lambda: project_hyperplane([3,2],[1,1],True)),
        ('affine_wrong_matrix_shape', lambda: project_affine([3,2,1],[1,1,0],[1])),
        ('affine_wrong_rhs_shape', lambda: project_affine([3,2,1],[[1,1,0],[0,1,1]],[[1],[0]])),
        ('affine_empty_equations', lambda: project_affine([3,2],np.empty((0,2)),[])),
        ('affine_too_many_equations', lambda: project_affine([3],[[1],[2]],[1,2])),
        ('affine_redundant_rows', lambda: project_affine([3,2,1],[[1,1,0],[2,2,0]],[1,2])),
        ('affine_inconsistent_rows', lambda: project_affine([3,2,1],[[1,1,0],[2,2,0]],[1,3])),
        ('affine_nonfinite_matrix', lambda: project_affine([3,2],[[np.inf,1]],[1])),
        ('affine_object_matrix', lambda: project_affine([3,2],np.array([[1,1]],dtype=object),[1])),
        ('affine_intermediate_overflow', lambda: project_affine([1e308],[[2]],[0])),
        ('weights_negative', lambda: convex_combination([[0],[1],[2]],[1,1,-1])),
        ('weights_zero_total', lambda: convex_combination([[0],[1]],[0,0])),
        ('weights_wrong_total', lambda: convex_combination([[0],[1]],[.2,.2])),
        ('weights_outside_tolerance', lambda: convex_combination([[0],[1]],[.5,.5+2e-12])),
        ('weights_wrong_shape', lambda: convex_combination([[0],[1]],[[.5],[.5]])),
        ('weights_bool', lambda: convex_combination([[0],[1]],[True,0])),
        ('empty_points', lambda: convex_combination(np.empty((0,2)),[])),
        ('scalar_points_need_column', lambda: convex_combination([1,2],[.5,.5])),
        ('jensen_nonfinite_return', lambda: jensen_diagnostic(lambda x: np.inf,[[1],[2]],[.5,.5])),
        ('jensen_nonscalar_return', lambda: jensen_diagnostic(lambda x: [1],[[1],[2]],[.5,.5])),
        ('jensen_bool_return', lambda: jensen_diagnostic(lambda x: True,[[1],[2]],[.5,.5])),
        ('line_nonfinite_return', lambda: line_samples(lambda x: np.nan,[0],[1],[0,1])),
        ('line_bad_step_shape', lambda: line_samples(lambda x: 0,[0],[1],[[0,1]])),
        ('certificate_wrong_shape', lambda: certificate_values([3,2],[2,1],[1,1])),
        ('certificate_overflow', lambda: certificate_values([1e308],[0],[[1e308]])),
    ]:
        reject(name, call)
    calls = []
    def forbidden_callback(x):
        calls.append(x.copy())
        return 0.0
    reject('line_nonfinite_samples_before_callback',
           lambda: line_samples(forbidden_callback,[1e308],[1e308],[0,2]))
    require(calls == [], '非有限中间采样点必须在任何 callback 前拒绝')
    # 近1权重在容差内，但不归一化；同号巨大值加权可能溢出，仍须先拒绝。
    reject('jensen_nonfinite_mean_before_callback',
           lambda: jensen_diagnostic(forbidden_callback,[[np.finfo(float).max],[np.finfo(float).max]],
                                      [.5,.5+5e-13]))
    require(calls == [], '非有限平均点必须在任何 callback 前拒绝')
    valid = {
        'interval_singleton': project_interval(3,1,1) == 1,
        'box_singleton_coordinate': np.array_equal(project_box([3,2],[1,0],[1,1]),[1,1]),
        'ball_inside': np.array_equal(project_ball([.5,.5],[0,0],2),[.5,.5]),
        'ball_center': np.array_equal(project_ball([1,-1],[1,-1],2),[1,-1]),
        'ball_zero_radius': np.array_equal(project_ball([4,3],[1,-1],0),[1,-1]),
        'halfspace_inside': np.array_equal(project_halfspace([0,0],[1,1],1),[0,0]),
        'halfspace_boundary': np.array_equal(project_halfspace([1,0],[1,1],1),[1,0]),
        'one_weight': np.array_equal(convex_combination([[1],[4]],[0,1]),[4]),
        'zero_weight': jensen_diagnostic(lambda x: float(x[0]**2),[[1],[4]],[1,0])['gap'] == 0,
        'zero_direction': np.array_equal(line_samples(lambda x: float(x[0]),[2],[0],[0,1])[1],[2,2]),
    }
    for name, passed in valid.items():
        require(passed, name)
        results.append({'case':name,'status':'accepted_as_expected'})
    return results


@checked
def run_experiment(write_outputs=True):
    names, points, matrix, rhs = load_data()
    lower, upper, center, radius = np.array([0.,0.]), np.array([2.,1.]), np.zeros(2), 2.
    specs = {
        'box': (lambda z: project_box(z,lower,upper)),
        'ball': (lambda z: project_ball(z,center,radius)),
        'hyperplane': (lambda z: project_hyperplane(z,[1,1],1)),
        'halfspace': (lambda z: project_halfspace(z,[1,1],1)),
    }
    rows = []
    for name, v in zip(names, points):
        for set_name, projector in specs.items():
            p = projector(v)
            if set_name == 'box':
                violation = float(max(0, np.max(lower-p),np.max(p-upper)))
            elif set_name == 'ball':
                violation = float(max(0,np.linalg.norm(p)-radius))
            elif set_name == 'hyperplane':
                violation = float(abs(p.sum()-1))
            else:
                violation = float(max(0,p.sum()-1))
            require(violation <= ATOL, '验收条件未满足')
            require(np.allclose(projector(p),p,rtol=0,atol=ATOL), '验收条件未满足')
            rows.append({'point_id':name,'set':set_name,'v1':v[0],'v2':v[1],
                         'p1':p[0],'p2':p[1],'squared_distance':squared_distance(v,p),
                         'objective_half_squared_distance':.5*squared_distance(v,p),
                         'feasibility_violation':violation})
    v = points[0]
    expectations = {'box':[2,1],'ball':np.array([6,4])/np.sqrt(13),
                    'hyperplane':[1,0],'halfspace':[1,0]}
    for set_name, expected in expectations.items():
        require(np.allclose(specs[set_name](v),expected,rtol=0,atol=ATOL), '验收条件未满足')
    interval_cases = [{'v':x,'projection':project_interval(x,0,1)} for x in [-2,0,.4,1,3]]
    p3 = project_affine([3,2,1],matrix,rhs)
    require(np.allclose(p3,[4/3,-1/3,1/3],rtol=0,atol=ATOL), '验收条件未满足')
    null = np.array([1.,-1.,1.])
    samples, distances = line_samples(lambda z: squared_distance([3,2,1],z),p3,null,[-2,-1,0,1,2])
    analytic_distances = 26/3+3*np.array([-2,-1,0,1,2])**2
    require(np.allclose(distances,analytic_distances,rtol=0,atol=ATOL), '验收条件未满足')
    require(np.max(abs(samples@matrix.T-rhs)) <= ATOL, '验收条件未满足')
    jensen = jensen_diagnostic(lambda z: squared_distance(z,[0]),[[1],[2],[4]],[.5,.25,.25])
    require(jensen['average']==[2] and jensen['left']==4 and jensen['right']==5.5 and jensen['gap']==1.5, '验收条件未满足')
    nonconvex = jensen_diagnostic(lambda z: float(z[0]**4-z[0]**2),
                                 [[-1/np.sqrt(2)],[1/np.sqrt(2)]],[.5,.5])
    require(not nonconvex['holds_within_tolerance'] and np.isclose(nonconvex['gap'],-.25,rtol=0,atol=ATOL), '验收条件未满足')
    shifted = project_ball([4,3],[1,-1],2)
    require(np.allclose(shifted,[11/5,3/5],rtol=0,atol=ATOL), '验收条件未满足')
    require(np.isclose(squared_distance([4,3],shifted),9,rtol=0,atol=ATOL), '验收条件未满足')
    exact = np.array([np.sqrt(3),1.])
    box_ball = specs['ball'](specs['box'](v))
    ball_box = specs['box'](specs['ball'](v))
    candidates = {'true_intersection':exact,'box_then_ball':box_ball,'ball_then_box':ball_box}
    intersection_rows = []
    for label,p in candidates.items():
        require(np.all(p >= lower-ATOL) and np.all(p <= upper+ATOL) and np.linalg.norm(p)<=radius+ATOL, '验收条件未满足')
        intersection_rows.append({'candidate':label,'p1':float(p[0]),'p2':float(p[1]),
                                  'squared_distance':squared_distance(v,p),
                                  'certificate_at_true_projection':float(np.dot(v-p,exact-p))})
    require(np.isclose(squared_distance(v,exact),13-6*np.sqrt(3),rtol=0,atol=ATOL), '验收条件未满足')
    require(np.isclose(squared_distance(v,box_ball),17-32/np.sqrt(5),rtol=0,atol=ATOL), '验收条件未满足')
    require(np.isclose(squared_distance(v,ball_box),166/13-36/np.sqrt(13),rtol=0,atol=ATOL), '验收条件未满足')
    require(np.allclose(v-exact,(np.sqrt(3)-1)*exact+(2-np.sqrt(3))*np.array([0,1]),rtol=0,atol=ATOL), '验收条件未满足')
    require(all(row['squared_distance']>intersection_rows[0]['squared_distance'] for row in intersection_rows[1:]), '验收条件未满足')
    require(all(row['certificate_at_true_projection']>0 for row in intersection_rows[1:]), '验收条件未满足')
    # 证书抽查和网格最小值各司其职；都不是无限集合证明。
    xx, yy = np.meshgrid(np.linspace(0,2,101),np.linspace(0,1,101))
    box_grid = np.column_stack([xx.ravel(),yy.ravel()])
    theta = np.linspace(0,2*np.pi,121)
    radii = np.linspace(0,2,21)
    ball_grid = np.concatenate([np.column_stack([r*np.cos(theta),r*np.sin(theta)]) for r in radii])
    ts = np.linspace(-3,3,121)
    line_grid = np.column_stack([ts,1-ts])
    half_grid = np.concatenate([line_grid-s*np.array([1,1]) for s in np.linspace(0,2,11)])
    intersection_grid = box_grid[np.sum(box_grid**2,axis=1)<=4+1e-14]
    candidate_sets = {'box':box_grid,'ball':ball_grid,'hyperplane':line_grid,'halfspace':half_grid,
                      'intersection':intersection_grid}
    grids = []
    for label,grid in candidate_sets.items():
        p = exact if label == 'intersection' else specs[label](v)
        values = certificate_values(v,p,grid)
        require(np.max(values) <= ATOL, '验收条件未满足')
        grid_distances = np.sum((grid-v)**2,axis=1)
        gap = float(np.min(grid_distances)-squared_distance(v,p))
        require(gap >= -ATOL, '验收条件未满足')
        grids.append({'set':label,'candidate_count':len(grid),'max_certificate_inner_product':float(np.max(values)),
                      'grid_best_minus_analytic_d2':gap})
    # 同一投影映射的非扩张性，覆盖固定4x4输入对。
    nonexpansive = []
    for label,projector in specs.items():
        largest = max(squared_distance(projector(x),projector(y))-squared_distance(x,y)
                      for x in points for y in points)
        require(largest <= ATOL, '验收条件未满足')
        nonexpansive.append({'set':label,'ordered_pairs':len(points)**2,
                            'max_projected_minus_original_squared_distance':largest})
    report = {
        'unit':'016','python':sys.version.split()[0],'numpy':np.__version__,
        'data':'原创固定无量纲合成点与线性等式；无随机抽样',
        'tolerances':{'geometry_absolute':ATOL,'weight_sum_absolute':WEIGHT_ATOL,
                      'weights_normalized':False,'interpretation':'权重容差仅处理浮点输入；定理仍要求精确和为1'},
        'main_projections':[row for row in rows if row['point_id']=='main'],
        'projection_case_count':len(rows),'interval_cases':interval_cases,
        'affine':{'projection':p3.tolist(),'residual':(matrix@p3-rhs).tolist(),
                  'squared_distance':squared_distance([3,2,1],p3),
                  'residual_dot_null':float(np.dot(np.array([3,2,1])-p3,null)),
                  'line_squared_distances':distances.tolist()},
        'jensen':jensen,'nonconvex_counterexample':nonconvex,
        'shifted_ball':{'projection':shifted.tolist(),'squared_distance':squared_distance([4,3],shifted)},
        'intersection':intersection_rows,'finite_grid_diagnostics':grids,
        'nonexpansive_diagnostics':nonexpansive,'fraction_checks':fraction_checks(),
        'boundary_checks':boundary_checks(),
        'limits':['有限候选未证明全域凸性或全域最优性','float64小规模教学接口，拒绝无法解析的尺度',
                  'lstsq数值满行秩不等于精确代数秩证明','未测试真实任务、GPU或大型优化器']}
    if write_outputs:
        out = BASE/'outputs'
        out.mkdir(exist_ok=True)
        (out/'experiment-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        for filename, table in [('projection-results.csv',rows),('intersection-comparison.csv',intersection_rows),
                                ('grid-diagnostics.csv',grids)]:
            with (out/filename).open('w',encoding='utf-8',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    return report


if __name__ == '__main__':
    result = run_experiment()
    print(json.dumps({'status':'passed','projection_cases':result['projection_case_count'],
                      'boundary_cases':len(result['boundary_checks']),'jensen_gap':result['jensen']['gap'],
                      'affine_squared_distance':result['affine']['squared_distance'],
                      'intersection':result['intersection'],'outputs':'outputs/'},ensure_ascii=False,indent=2))
