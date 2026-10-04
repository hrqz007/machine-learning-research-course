"""第012讲：特征值、SVD 与低秩近似的离线可复现实验。

运行：python experiment.py
仅依赖 NumPy。所有数值为原创小矩阵，无联网、随机抽样或大型下载。
浮点核验采用明确容差；断言核查实现，不替代正文中的一般证明。
"""
from pathlib import Path
from fractions import Fraction
import csv
import json
import sys
import numpy as np

BASE = Path(__file__).resolve().parent
ATOL = 1e-11
RTOL = 1e-11


def real_matrix(values):
    """本讲接口只接收非空、有限的二维实矩阵；不静默丢弃虚部。"""
    # 先检查原始叶子，不能让 np.asarray 先把混合列表中的 False 变成 0、
    # 把数字字符串变成数值。这里仅接收 Python/NumPy 整数与浮点实数。
    def check_numeric_leaves(value):
        if isinstance(value, np.ndarray):
            if value.dtype.kind not in "iuf":
                raise ValueError('数组必须为整数或浮点实数类型；拒绝布尔、字符串、复数与 object')
        elif isinstance(value, (list, tuple)):
            for item in value:
                check_numeric_leaves(item)
        elif isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, float, np.integer, np.floating)):
            raise ValueError('元素必须为整数或浮点实数；拒绝布尔、字符串、复数与 object')
    check_numeric_leaves(values)
    raw = np.asarray(values)
    a = np.asarray(raw, dtype=float)
    if a.ndim != 2 or min(a.shape, default=0) == 0:
        raise ValueError('需要非空二维矩阵')
    if not np.all(np.isfinite(a)):
        raise ValueError('矩阵必须全部为有限数值')
    return a


def symmetric_eigh(values):
    """先检查对称，再调用 eigh；eigh 本身不替你核验两个三角区域相同。"""
    a = real_matrix(values)
    if a.shape[0] != a.shape[1]:
        raise ValueError('特征分解需要方阵')
    # 此课程包装器严格要求输入已对称。测量误差下的容差应由任务另定。
    if not np.array_equal(a, a.T):
        raise ValueError('需要实对称输入')
    return np.linalg.eigh(a)


def rank_from_s(s, shape, rtol=None):
    """返回数值秩与绝对阈值。它不是任意精度的精确秩判定。"""
    if rtol is None:
        rtol = max(shape) * np.finfo(float).eps
    if not np.isfinite(rtol) or rtol < 0:
        raise ValueError('相对阈值必须有限且非负')
    tol = float(rtol * s[0])
    return int(np.count_nonzero(s > tol)), tol


def truncated_svd(values, k):
    """保留前 k 个奇异值，0 <= k <= min(m,n)，返回重构及误差核验。"""
    a = real_matrix(values)
    p = min(a.shape)
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)):
        raise ValueError('k 必须为整数，不能用 True/False 代替')
    if k < 0 or k > p:
        raise ValueError('k 超出允许范围')
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    # u[:, :k] 是 m×k；乘 s[:k] 按列缩放；vt[:k, :] 是 k×n。
    ak = (u[:, :k] * s[:k]) @ vt[:k, :]
    residual = a - ak
    fro_actual = float(np.linalg.norm(residual, ord='fro'))
    spectral_actual = float(np.linalg.norm(residual, ord=2))
    fro_expected = float(np.sqrt(np.sum(s[k:] ** 2)))
    spectral_expected = float(s[k]) if k < p else 0.0
    total = float(np.sum(s ** 2))
    energy = float(np.sum(s[:k] ** 2) / total) if total > 0 else None
    scale = max(1.0, float(s[0]))
    assert np.isclose(fro_actual, fro_expected, atol=ATOL * scale, rtol=RTOL)
    assert np.isclose(spectral_actual, spectral_expected, atol=ATOL * scale, rtol=RTOL)
    return ak, {'k': int(k), 'frobenius_actual': fro_actual,
                'frobenius_expected': fro_expected, 'spectral_actual': spectral_actual,
                'spectral_expected': spectral_expected, 'energy_fraction': energy}


def main_matrices():
    b = np.array([[2., 1.], [1., 2.]])
    a = np.array([[2., 1.], [1., 2.], [1., -1.]])
    q = np.array([[1., 1.], [1., -1.]]) / np.sqrt(2.)
    u = np.column_stack((np.array([1., 1., 0.]) / np.sqrt(2.),
                         np.array([1., -1., 2.]) / np.sqrt(6.),
                         np.array([1., -1., -1.]) / np.sqrt(3.)))
    sigma = np.array([[3., 0.], [0., np.sqrt(3.)], [0., 0.]])
    return b, a, q, u, sigma


def exact_hand_check():
    """有理数部分用 Fraction 精确检查，不用容差凑出 3/4。"""
    f = Fraction
    a = [[f(2), f(1)], [f(1), f(2)], [f(1), f(-1)]]
    a1 = [[f(3, 2), f(3, 2)], [f(3, 2), f(3, 2)], [f(0), f(0)]]
    total = sum(x*x for row in a for x in row)
    loss = sum((a[i][j]-a1[i][j])**2 for i in range(3) for j in range(2))
    assert total == 12 and loss == 3 and (total-loss)/total == f(3, 4)
    # B 的特征多项式 λ²−4λ+3；检验手算的两个根。
    assert all(l*l - 4*l + 3 == 0 for l in [f(1), f(3)])
    return {'B_eigenvalues': [1, 3], 'A_gram': [[6, 3], [3, 6]],
            'A_singular_values': ['3', 'sqrt(3)'], 'rank1_frobenius_squared': str(loss),
            'total_frobenius_squared': str(total), 'rank1_energy_fraction': '3/4'}


def decomposition_checks():
    b, a, q, u, sigma = main_matrices()
    w, z = symmetric_eigh(b)
    assert np.allclose(w, [1, 3], atol=ATOL, rtol=RTOL)
    assert np.allclose(b @ z, z * w, atol=ATOL, rtol=RTOL)
    assert np.allclose(z.T @ z, np.eye(2), atol=ATOL, rtol=RTOL)
    assert np.allclose(b, (z * w) @ z.T, atol=ATOL, rtol=RTOL)
    assert np.allclose(a, u @ sigma @ q.T, atol=ATOL, rtol=RTOL)
    assert np.allclose(u.T @ u, np.eye(3), atol=ATOL, rtol=RTOL)
    assert np.allclose(a.T @ u[:, 2], np.zeros(2), atol=ATOL, rtol=RTOL)
    assert np.allclose(a.T @ a, [[6, 3], [3, 6]], atol=ATOL, rtol=RTOL)
    ul, s, vt = np.linalg.svd(a, full_matrices=False)
    assert np.allclose(s, [3, np.sqrt(3)], atol=ATOL, rtol=RTOL)
    # 成对换号：U 的一列与 Vt 的相同行同时乘 −1，矩阵不变。
    uf, vtf = ul.copy(), vt.copy()
    uf[:, 0] *= -1
    vtf[0, :] *= -1
    assert np.allclose((uf * s) @ vtf, a, atol=ATOL, rtol=RTOL)
    wrong = (uf * s) @ vt  # 只改一侧：这次真的改变了矩阵。
    assert not np.allclose(wrong, a, atol=ATOL, rtol=RTOL)
    # 重复奇异值 2,2：同一个二维子空间换正交基。
    d = np.diag([2., 2., 0.5])
    angle = np.pi / 5
    r = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    h = np.eye(3); h[:2, :2] = r
    assert np.allclose(h @ d @ h.T, d, atol=ATOL, rtol=RTOL)
    p_old = np.eye(3)[:, :2] @ np.eye(3)[:, :2].T
    p_new = h[:, :2] @ h[:, :2].T
    assert np.allclose(p_new, p_old, atol=ATOL, rtol=RTOL)
    assert not np.allclose(h[:, :2], np.eye(3)[:, :2], atol=ATOL, rtol=RTOL)
    return {'B_eigenvalues_ascending': w.tolist(), 'A_singular_values_descending': s.tolist(),
            'A_reconstruction_frobenius': float(np.linalg.norm(a - (ul*s)@vt, ord='fro')),
            'A_U_orthogonality_frobenius': float(np.linalg.norm(ul.T@ul-np.eye(2), ord='fro')),
            'paired_sign_reconstruction_frobenius': float(np.linalg.norm(a-(uf*s)@vtf, ord='fro')),
            'one_sided_sign_error_frobenius': float(np.linalg.norm(a-wrong, ord='fro')),
            'repeated_block_projector_difference': float(np.linalg.norm(p_old-p_new, ord='fro'))}


def hadamard(n):
    """构造 ±1 的正交条纹，不是训练出的语义特征。n 须为 2 的整数次幂。"""
    h = np.ones((1, 1))
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    if h.shape[0] != n:
        raise ValueError('n 必须是 2 的整数次幂')
    return h


def synthetic_field():
    # 32×32；常量背景和五组条纹的强度分别是这六个奇异值。
    h = hadamard(32) / np.sqrt(32.)
    left = h[:, [0, 1, 2, 4, 8, 16]]
    right = h[:, [0, 2, 8, 1, 16, 4]]
    s = np.array([16., 8., 4., 2., .5, .25])
    # 各项均为二进制有理数；显式缩放用整数符号矩阵避免制造舍入噪声。
    signs = hadamard(32)
    field = (signs[:, [0,1,2,4,8,16]] * (s/32.)) @ signs[:, [0,2,8,1,16,4]].T
    assert np.allclose(field, (left*s)@right.T, atol=ATOL, rtol=RTOL)
    assert field.min() >= 0 and field.max() <= 1
    return field, s


def condition_checks():
    rows = []
    for exponent in [0, 10, 20, 30]:
        eps = 2. ** (-exponent)
        c = np.diag([1., eps])
        cond = float(np.linalg.cond(c, p=2))
        gram_cond = float(np.linalg.cond(c.T@c, p=2))
        assert np.isclose(cond, 1/eps, atol=ATOL, rtol=RTOL)
        assert np.isclose(gram_cond, cond**2, atol=ATOL, rtol=RTOL)
        rows.append({'exponent': exponent, 'epsilon': eps, 'condition_2': cond,
                     'gram_condition_2': gram_cond})
    # 满行秩宽矩阵：返回的两个奇异值全非零，仍有输入零空间。
    wide = np.array([[1.,0.,0.], [0.,1.,0.]])
    s = np.linalg.svd(wide, compute_uv=False)
    assert np.allclose(s, [1,1]) and np.linalg.cond(wide)==1.
    assert np.array_equal(wide @ np.array([0.,0.,1.]), np.zeros(2))
    # 近共线：直接 SVD 与先形成 Gram 的平方根路线实际比较。
    gram_rows=[]
    for exponent in [10, 20, 26, 30]:
        eps=2.**(-exponent)
        c=np.array([[1.,1.],[0.,eps]])
        sd=np.linalg.svd(c,compute_uv=False)
        lam=np.linalg.eigvalsh(c.T@c)
        # 只记录，不把负值或零偷偷修成预期答案。
        inferred=np.sqrt(lam) if np.all(lam>=0) else None
        gram_rows.append({'exponent':exponent,'direct_singular_values':sd.tolist(),
                          'gram_eigenvalues':lam.tolist(),
                          'sqrt_gram_eigenvalues':None if inferred is None else inferred.tolist()})
    return {'diagonal_cases':rows,'wide_condition_reported':float(np.linalg.cond(wide)),
            'wide_nullity':1,'near_collinear_gram_audit':gram_rows}


def definiteness_and_uniqueness_checks():
    cases = {'positive_definite': np.array([[2., -1.], [-1., 2.]]),
             'semidefinite_not_definite': np.diag([3., 0.]),
             'positive_entries_indefinite': np.array([[1., 2.], [2., 1.]])}
    values = {name: symmetric_eigh(a)[0].tolist() for name, a in cases.items()}
    assert np.allclose(values['positive_definite'], [1, 3])
    assert np.allclose(values['semidefinite_not_definite'], [0, 3])
    assert np.allclose(values['positive_entries_indefinite'], [-1, 3])
    d = 2*np.eye(2)
    first = np.diag([2., 0.]); second = np.diag([0., 2.])
    assert np.linalg.norm(d-first, ord='fro') == np.linalg.norm(d-second, ord='fro') == 2.
    assert not np.array_equal(first, second)
    # 即使奇异值不同，谱范数最优解也可不唯一。
    spectral_errors = [float(np.linalg.norm(np.diag([3.,1.])-np.diag([c,0.]),ord=2))
                       for c in [2.,3.,4.]]
    assert spectral_errors == [1.,1.,1.]
    # 剪切 H 的唯一特征值为 1，H-I 的零空间只有一维。
    h=np.array([[1.,1.],[0.,1.]])
    assert np.linalg.matrix_rank(h-np.eye(2))==1
    return {'eigenvalues':values,'repeated_rank1_two_frobenius_errors':[2.,2.],
            'distinct_values_spectral_errors':spectral_errors,'shear_eigenspace_dimension':1}


def weighted_counterexample():
    a=np.diag([3.,1.]); ordinary=np.diag([3.,0.]); alternative=np.diag([0.,1.])
    weights=np.array([[1.,1.],[1.,100.]])
    ordinary_loss=float(np.sum(weights*(a-ordinary)**2))
    alternative_loss=float(np.sum(weights*(a-alternative)**2))
    assert ordinary_loss==100. and alternative_loss==9.
    return {'ordinary_rank1_weighted_loss':ordinary_loss,'alternative_rank1_weighted_loss':alternative_loss,
            'ordinary_rank1_unweighted_loss':1.,'alternative_rank1_unweighted_loss':9.}


def boundary_checks():
    passed=[]
    invalid=[('vector',[1,2]),('empty',np.zeros((0,2))),('nan',[[np.nan]]),
             ('infinity',[[np.inf]]),('complex',[[1+0j]]),
             ('bool_array',np.array([[True]])),('bool_list',[[True,False]]),
             ('mixed_bool',[[1,False]]),('string_array',np.array([["1"]])),
             ('mixed_string',[[1,"2"]]),('object_array',np.array([[1.]],dtype=object)),
             ('object_leaf',[[object()]])]
    for name, a in invalid:
        try: real_matrix(a)
        except ValueError: passed.append(name)
        else: raise AssertionError(name)
    for name, a in [('not_square',[[1,2,3],[4,5,6]]),('not_symmetric',[[1,2],[0,1]])]:
        try: symmetric_eigh(a)
        except ValueError: passed.append(name)
        else: raise AssertionError(name)
    for k in [-1,3,0.5,True]:
        try: truncated_svd(np.eye(2),k)
        except ValueError: passed.append('invalid_k_'+str(k))
        else: raise AssertionError('invalid k')
    fixtures={'square_full':np.diag([3.,1.]), 'tall_full':main_matrices()[1],
              'wide_full_row':np.array([[1.,0.,0.],[0.,1.,0.]]),
              'rank_deficient':np.array([[1.,2.],[2.,4.],[0.,0.]]),
              'zero':np.zeros((3,2)), 'singleton':np.array([[-5.]]),
              'near_singular':np.diag([1.,2.**-40])}
    shape_report=[]
    for name,a in fixtures.items():
        m,n=a.shape;p=min(m,n)
        uf,s,vf=np.linalg.svd(a,full_matrices=True)
        ur,sr,vr=np.linalg.svd(a,full_matrices=False)
        assert uf.shape==(m,m) and vf.shape==(n,n)
        assert ur.shape==(m,p) and vr.shape==(p,n)
        sm=np.zeros((m,n));sm[:p,:p]=np.diag(s)
        assert np.allclose(a,uf@sm@vf,atol=ATOL,rtol=RTOL)
        assert np.allclose(uf.T@uf,np.eye(m),atol=ATOL,rtol=RTOL)
        assert np.allclose(vf@vf.T,np.eye(n),atol=ATOL,rtol=RTOL)
        assert np.allclose(a,(ur*sr)@vr,atol=ATOL,rtol=RTOL)
        rank,tol=rank_from_s(sr,a.shape)
        expected_rank={'square_full':2,'tall_full':2,'wide_full_row':2,'rank_deficient':1,'zero':0,'singleton':1,'near_singular':2}[name]
        assert rank==expected_rank
        for k in range(p+1): truncated_svd(a,k)
        shape_report.append({'name':name,'shape':[m,n],'full_U':list(uf.shape),'full_Vt':list(vf.shape),
                             'reduced_U':list(ur.shape),'reduced_Vt':list(vr.shape),
                             'numerical_rank':rank,'rank_absolute_tolerance':tol})
        passed.append(name)
    # 零矩阵总能量为零，0/0 没有定义。选择 None 而非假称保留率 100%。
    _, zero_info=truncated_svd(np.zeros((3,2)),0)
    assert zero_info['energy_fraction'] is None
    # 实非对称旋转没有实特征值，但实 SVD 正常存在。
    rotation=np.array([[0.,-1.],[1.,0.]])
    w,_=np.linalg.eig(rotation)
    assert np.allclose(np.sort(np.abs(w.imag)),[1,1])
    assert np.allclose(np.linalg.svd(rotation,compute_uv=False),[1,1])
    passed += ['zero_energy_undefined','nonreal_eigenvalues_real_svd']
    return {'passed_count':len(passed),'names':passed,'shape_report':shape_report}


def run_all(write_outputs=True):
    field,known_s=synthetic_field()
    from_csv=np.loadtxt(BASE/'data'/'synthetic_field.csv',delimiter=',')
    assert np.array_equal(field,from_csv)
    a_csv=np.loadtxt(BASE/'data'/'main_matrix.csv',delimiter=',',skiprows=1)
    assert np.array_equal(a_csv,main_matrices()[1])
    measured=np.linalg.svd(field,compute_uv=False)
    assert np.allclose(measured[:6],known_s,atol=ATOL,rtol=RTOL)
    assert np.max(measured[6:])<ATOL
    curves=[truncated_svd(field,k)[1] for k in range(7)]
    report={'python':sys.version.split()[0],'numpy':np.__version__,'dtype':'float64',
            'tolerances':{'atol':ATOL,'rtol':RTOL},'exact_hand':exact_hand_check(),
            'decompositions':decomposition_checks(),'conditions':condition_checks(),
            'definiteness_and_uniqueness':definiteness_and_uniqueness_checks(),
            'weighted_counterexample':weighted_counterexample(),'boundary':boundary_checks(),
            'field':{'shape':list(field.shape),'known_singular_values':known_s.tolist(),
                     'min':float(field.min()),'max':float(field.max()),'curves':curves}}
    if write_outputs:
        out=BASE/'outputs';out.mkdir(exist_ok=True)
        (out/'experiment_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        with (out/'truncation_errors.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(curves[0]));writer.writeheader();writer.writerows(curves)
    return report


def print_summary(report):
    print('NumPy:',report['numpy'])
    print('手算核对：B 特征值 [1, 3]；A 奇异值 [3, sqrt(3)]')
    print('A 的秩1误差平方：3；保留能量：3/4')
    print('重构 Frobenius 误差:',format(report['decompositions']['A_reconstruction_frobenius'],'.3e'))
    print('合成图奇异值:',report['field']['known_singular_values'])
    print('边界检查通过组数:',report['boundary']['passed_count'])
    print('宽矩阵 cond=1，输入零空间维数仍为1')
    print('加权反例：普通截断损失100，另一个秩1矩阵损失9')
    print('输出文件存在:',(BASE/'outputs'/'experiment_report.json').exists())


if __name__=='__main__':
    print_summary(run_all())
