"""第011讲：线性方程与最小二乘几何。

离线、小CPU、确定性合成数据。NumPy 2.3.5 为制作核验版本。
所有方法均记录实际误差或失败，不预设一个固定的数值排名。
Fraction 提供主例精确参照；浮点路线均使用 float64。
"""
from pathlib import Path
from fractions import Fraction as F
import csv
import json
import math
import numpy as np

BASE = Path(__file__).resolve().parent
CHECK_ATOL = 1e-11
CHECK_RTOL = 1e-11


def validate(A, y):
    """本教学接口只接收单个二维设计与一维右端，避免广播隐藏形状错位。"""
    A = np.asarray(A, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if A.ndim != 2 or y.ndim != 1:
        raise ValueError('要求 A 为二维，y 为一维')
    if A.shape[0] != y.shape[0]:
        raise ValueError('A 的行数必须等于 y 的长度')
    if min(A.shape) == 0:
        raise ValueError('本实验不接收空行或空列设计')
    if not np.isfinite(A).all() or not np.isfinite(y).all():
        raise ValueError('只接收有限数值')
    return A, y


def read_main():
    with (BASE/'data/main.csv').open(encoding='utf-8', newline='') as stream:
        rows=list(csv.DictReader(stream))
    xs=np.array([float(r['x']) for r in rows])
    ys=np.array([float(r['y']) for r in rows])
    return np.column_stack([np.ones(len(xs)),xs]),ys


def exact_main():
    """逐项做 Fraction 正规方程，独立于 NumPy 浮点解。"""
    A=[[F(1),F(0)],[F(1),F(1)],[F(1),F(2)]]
    y=[F(1),F(2),F(2)]
    G=[[sum((row[j]*row[k] for row in A),F(0)) for k in range(2)] for j in range(2)]
    h=[sum((A[i][j]*y[i] for i in range(3)),F(0)) for j in range(2)]
    # 第二条正规方程减第一条：2*beta1=1。
    slope=(h[1]-h[0])/(G[1][1]-G[0][1])
    intercept=(h[0]-G[0][1]*slope)/G[0][0]
    beta=[intercept,slope]
    pred=[sum((row[j]*beta[j] for j in range(2)),F(0)) for row in A]
    residual=[y[i]-pred[i] for i in range(3)]
    sse=sum((r*r for r in residual),F(0))
    orth=[sum((A[i][j]*residual[i] for i in range(3)),F(0)) for j in range(2)]
    assert beta==[F(7,6),F(1,2)]
    assert residual==[F(-1,6),F(1,3),F(-1,6)] and sse==F(1,6)
    assert orth==[0,0]
    return {'gram':[[str(x) for x in row] for row in G], 'rhs':[str(x) for x in h],
            'beta':[str(x) for x in beta], 'prediction':[str(x) for x in pred],
            'residual':[str(x) for x in residual], 'sse':str(sse),
            'orthogonality':[str(x) for x in orth]}


def rank_fraction(rows):
    """小矩阵精确行消元用于核对秩，不是通用高效数值求解器。"""
    M=[[F(x) for x in row] for row in rows]
    m,n=len(M),len(M[0]);rank=0
    for j in range(n):
        pivot=next((i for i in range(rank,m) if M[i][j]!=0),None)
        if pivot is None:continue
        M[rank],M[pivot]=M[pivot],M[rank]
        scale=M[rank][j];M[rank]=[x/scale for x in M[rank]]
        for i in range(m):
            if i==rank:continue
            factor=M[i][j]
            M[i]=[a-factor*b for a,b in zip(M[i],M[rank])]
        rank+=1
        if rank==m:break
    return rank


def mgs(A):
    """一次修改 Gram-Schmidt。教学用，近共线时可能失去正交性。

    只支持 m>=n 的数值列满秩路线；不尝试自动返回秩亏最小范数解。
    """
    A=np.asarray(A,dtype=np.float64)
    m,n=A.shape
    if m<n:raise ValueError('教学 QR 路线要求行数不少于列数')
    Q=np.zeros((m,n));R=np.zeros((n,n))
    threshold=np.finfo(np.float64).eps*max(A.shape)*np.linalg.norm(A,ord=2)
    for j in range(n):
        v=A[:,j].copy()
        for k in range(j):
            R[k,j]=Q[:,k]@v
            v=v-Q[:,k]*R[k,j]
        R[j,j]=np.linalg.norm(v)
        if R[j,j]<=threshold:raise ValueError('教学 QR 检出零或过小剩余方向')
        Q[:,j]=v/R[j,j]
    return Q,R


def qr_solution(A,y,teaching=False):
    if A.shape[0]<A.shape[1]:raise ValueError('此 QR 回代路线不处理宽矩阵')
    Q,R=mgs(A) if teaching else np.linalg.qr(A,mode='reduced')
    threshold=np.finfo(np.float64).eps*max(A.shape)*np.linalg.norm(A,ord=2)
    if np.any(np.abs(np.diag(R))<=threshold):
        raise ValueError('R 的对角元低于此教学回代路线阈值')
    beta=np.linalg.solve(R,Q.T@y)
    return beta,{'reconstruction_norm':float(np.linalg.norm(Q@R-A)),
                 'orthogonality_norm':float(np.linalg.norm(Q.T@Q-np.eye(A.shape[1]))),
                 'diagonal_threshold':float(threshold)}


def finite_or_string(value):
    value=float(value)
    return value if math.isfinite(value) else ('inf' if value>0 else '-inf' if value<0 else 'nan')


def metric_record(A,y,beta,reference=None,extra=None):
    residual=y-A@beta
    result={'status':'ok','beta':beta.tolist(),'residual_norm':float(np.linalg.norm(residual)),
            'sse':float(residual@residual),'orthogonality_defect':float(np.linalg.norm(A.T@residual)),
            'coefficient_error':None if reference is None else float(np.linalg.norm(beta-reference)),
            'prediction_error':None if reference is None else float(np.linalg.norm(A@beta-A@reference))}
    if extra:result.update(extra)
    return result


def compare_methods(A,y,reference=None):
    A,y=validate(A,y)
    G=A.T@A;h=A.T@y
    names=['lstsq','qr_library','qr_teaching_mgs','normal_solve','normal_inverse','solve_A','inverse_A']
    records={}
    for name in names:
        try:
            extra={}
            if name=='lstsq':
                beta,returned_residuals,rank,singular_values=np.linalg.lstsq(A,y,rcond=None)
                extra={'reported_rank':int(rank),'returned_residuals':returned_residuals.tolist(),
                       'singular_values':singular_values.tolist(),'rcond':'None (NumPy default)'}
            elif name=='qr_library':beta,extra=qr_solution(A,y)
            elif name=='qr_teaching_mgs':beta,extra=qr_solution(A,y,teaching=True)
            elif name=='normal_solve':beta=np.linalg.solve(G,h)
            elif name=='normal_inverse':beta=np.linalg.inv(G)@h
            elif name=='solve_A':beta=np.linalg.solve(A,y)
            else:beta=np.linalg.inv(A)@y
            records[name]=metric_record(A,y,beta,reference,extra)
        except (np.linalg.LinAlgError,ValueError) as exc:
            records[name]={'status':'failed','error_type':type(exc).__name__,'message':str(exc)}
    return {'shape':list(A.shape),'dtype':str(A.dtype),'default_numeric_rank':int(np.linalg.matrix_rank(A)),
            'condition_A':finite_or_string(np.linalg.cond(A)),
            'condition_gram_computed':finite_or_string(np.linalg.cond(G)),
            'methods':records}


def near_design(k):
    """精确设定 epsilon=2^-k，t固定；未使用随机样本。"""
    epsilon=2.0**(-k);t=np.arange(-2.,3.)
    A=np.column_stack([np.ones(5),1+epsilon*t])
    beta=np.array([1.,2.]);y=A@beta
    return A,y,beta,epsilon,t


def perturbation_demo():
    A,y,beta,epsilon,t=near_design(20)
    eta=2.0**(-30)
    dy_parallel=eta*t
    dy_orthogonal=eta*np.array([1.,-2.,1.,0.,0.])
    base=np.linalg.lstsq(A,y,rcond=None)[0]
    cases={}
    for name,dy in [('column_space',dy_parallel),('orthogonal',dy_orthogonal)]:
        changed=np.linalg.lstsq(A,y+dy,rcond=None)[0]
        cases[name]={'data_change_norm':float(np.linalg.norm(dy)),
            'coefficient_change_norm':float(np.linalg.norm(changed-base)),
            'prediction_change_norm':float(np.linalg.norm(A@changed-A@base)),
            'coefficients':changed.tolist(),'new_residual_norm':float(np.linalg.norm(y+dy-A@changed))}
    exact_change=np.array([-eta/epsilon,eta/epsilon])
    assert np.allclose(np.array(cases['column_space']['coefficients']),beta+exact_change,atol=2e-9,rtol=2e-9)
    assert cases['column_space']['coefficient_change_norm']>1e5*cases['column_space']['data_change_norm']
    assert cases['orthogonal']['prediction_change_norm']<1e-12
    assert np.linalg.norm(A.T@dy_orthogonal)<1e-22
    return {'epsilon':epsilon,'eta':eta,'exact_parallel_coefficient_change':exact_change.tolist(),
            'orthogonal_dot_columns':(A.T@dy_orthogonal).tolist(),'cases':cases}


def threshold_demo():
    A,y,beta,epsilon,t=near_design(40)
    rows=[]
    for rcond in [None,1e-14,1e-12,1e-10]:
        sol,res,rank,s=np.linalg.lstsq(A,y,rcond=rcond)
        rows.append({'rcond':rcond,'reported_rank':int(rank),'beta':sol.tolist(),
                     'residual_norm':float(np.linalg.norm(y-A@sol)),
                     'coefficient_error':float(np.linalg.norm(sol-beta))})
    assert rows[0]['reported_rank']==2 and rows[-1]['reported_rank']==1
    return {'epsilon':epsilon,'construction_exact_rank':2,'results':rows}


def boundary_tests():
    passed=[]
    invalid=[('one_dimensional_A',[1,2],[1,2]),('column_y',[[1],[2]],[[1],[2]]),
             ('row_mismatch',[[1],[2]],[1]),('empty_rows',np.empty((0,2)),[]),
             ('empty_columns',np.empty((2,0)),[1,2]),('nan_input',[[1],[np.nan]],[1,2])]
    for name,A,y in invalid:
        try:validate(A,y)
        except ValueError:passed.append(name+'_rejected')
        else:raise AssertionError(name)
    singleton=compare_methods([[2]],[6],np.array([3.]))
    assert np.allclose(singleton['methods']['solve_A']['beta'],[3])
    passed.append('single_equation_single_parameter')
    single_col=compare_methods([[1],[1],[1]],[1,2,3],np.array([2.]))
    assert np.allclose(single_col['methods']['lstsq']['beta'],[2])
    passed.append('single_column_mean')
    wide=compare_methods([[1,1]],[2],np.array([1.,1.]))
    assert np.allclose(wide['methods']['lstsq']['beta'],[1,1])
    assert wide['methods']['lstsq']['returned_residuals']==[]
    passed.append('wide_minimum_norm')
    zero=compare_methods(np.zeros((3,2)),[1,2,3],np.zeros(2))
    assert zero['methods']['lstsq']['sse']==14
    assert zero['methods']['lstsq']['returned_residuals']==[]
    passed.append('zero_design_nonzero_residual_despite_empty_return')
    A,y=read_main();Q,R=np.linalg.qr(A,mode='reduced')
    assert Q.shape==(3,2) and R.shape==(2,2)
    assert np.allclose(Q.T@Q,np.eye(2),atol=CHECK_ATOL,rtol=CHECK_RTOL)
    assert np.allclose(Q@R,A,atol=CHECK_ATOL,rtol=CHECK_RTOL)
    P=Q@Q.T
    assert np.allclose(P.T,P) and np.allclose(P@P,P)
    passed.append('qr_shapes_orthogonality_reconstruction_projection')
    assert rank_fraction([[1,0],[1,1],[1,2]])==2
    assert rank_fraction([[1,0,1],[1,1,2],[1,2,2]])==3
    passed.append('exact_augmented_rank_detects_inconsistency')
    # 行缩放会改变不相容最小二乘的权重。
    plain=np.linalg.lstsq(np.ones((2,1)),[0.,2.],rcond=None)[0][0]
    scaled=np.linalg.lstsq(np.array([[1.],[10.]]),[0.,20.],rcond=None)[0][0]
    assert np.isclose(plain,1) and np.isclose(scaled,200/101)
    passed.append('row_scaling_changes_least_squares')
    # 对角条件数可直接从坐标伸缩核对，不依赖读者已学SVD。
    T=np.diag([1.,.01]);assert np.isclose(np.linalg.cond(T),100)
    assert np.isclose(np.linalg.cond(T.T@T),10000)
    passed.append('diagonal_condition_number_squared')
    return passed,{'singleton':singleton,'single_column':single_col,'wide':wide,'zero':zero,
                   'row_scaling':{'original':float(plain),'scaled':float(scaled)}}


def run_experiment(write_outputs=True):
    exact=exact_main();A,y=read_main();ref=np.array([7/6,.5])
    main=compare_methods(A,y,ref)
    for name in ['lstsq','qr_library','qr_teaching_mgs','normal_solve','normal_inverse']:
        out=main['methods'][name];assert out['status']=='ok'
        assert np.allclose(out['beta'],ref,atol=CHECK_ATOL,rtol=CHECK_RTOL)
        assert np.isclose(out['sse'],1/6,atol=CHECK_ATOL,rtol=CHECK_RTOL)
    D=np.column_stack([A,A[:,1]])
    deficient_ref=np.array([7/6,.25,.25])
    deficient=compare_methods(D,y,deficient_ref)
    assert deficient['default_numeric_rank']==2
    assert np.allclose(deficient['methods']['lstsq']['beta'],deficient_ref,atol=CHECK_ATOL,rtol=CHECK_RTOL)
    family=[]
    for t in [-2.,0.,3.]:
        beta=deficient_ref+t*np.array([0.,1.,-1.])
        assert np.allclose(D@beta,A@ref)
        family.append({'t':t,'beta':beta.tolist(),'norm':float(np.linalg.norm(beta)),
                       'prediction':(D@beta).tolist()})
    assert family[1]['norm']<family[0]['norm'] and family[1]['norm']<family[2]['norm']
    near=[]
    for k in [10,20,26,30,40,50]:
        N,z,true_beta,e,t=near_design(k)
        rec=compare_methods(N,z,true_beta);rec.update({'k':k,'epsilon':e,'declared_exact_rank':2,
            'exact_rank_of_stored_float_entries':rank_fraction(N.tolist())})
        assert rec['exact_rank_of_stored_float_entries']==2
        near.append(rec)
    square_eps=F(1,1000);delta=F(1,1000000)
    square_true=np.array([1.,1.]);C=np.array([[1.,1.],[1.,1.+float(square_eps)]])
    square=compare_methods(C,C@square_true,square_true)
    exact_changed=[F(1)-delta/square_eps,F(1)+delta/square_eps]
    assert exact_changed==[F(999,1000),F(1001,1000)]
    checks,boundaries=boundary_tests()
    report={'unit':'011','numpy_version':np.__version__,'dtype':'float64','randomness':'none',
            'exact_main':exact,'main':main,'rank_deficient':deficient,'rank_deficient_family':family,
            'near_collinear':near,'square_case':square,
            'square_exact_perturbed_solution':[str(x) for x in exact_changed],
            'threshold_demo':threshold_demo(),'perturbations':perturbation_demo(),
            'boundary_tests':checks,'boundary_cases':boundaries,
            'tolerances':{'well_conditioned_atol':CHECK_ATOL,'well_conditioned_rtol':CHECK_RTOL,
                           'parallel_perturbation_coefficient_atol':2e-9,
                           'qr_diagonal_guard':'eps * max(shape) * norm(A,2)'},
            'limitations':['Synthetic deterministic cases; no empirical generalization claim.',
                'Teaching MGS is not production QR and may lose orthogonality.',
                'NumPy diagnostic/lstsq internals use decompositions including SVD; its theory is deferred to unit 012.',
                'Last digits and precise failures can depend on BLAS/LAPACK.',
                'Finite checks do not replace geometric proofs.']}
    if write_outputs:
        out=BASE/'outputs';out.mkdir(exist_ok=True)
        (out/'experiment_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        with (out/'method_comparison.csv').open('w',encoding='utf-8',newline='') as stream:
            fields=['case','method','status','coefficient_error','prediction_error','residual_norm','sse','orthogonality_defect','error_type']
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            cases=[('main',main),('rank_deficient',deficient),('square',square)]+[(f'near_2neg{r["k"]}',r) for r in near]
            for case,record in cases:
                for name,r in record['methods'].items():
                    writer.writerow({k:({'case':case,'method':name}.get(k,r.get(k,''))) for k in fields})
    return report


def print_report(report):
    print('NumPy',report['numpy_version'],'；所有数据为确定性合成数据')
    print('精确主例系数：',report['exact_main']['beta'])
    print('精确残差：',report['exact_main']['residual'],'SSE =',report['exact_main']['sse'])
    print('秩亏最小范数系数：',report['rank_deficient']['methods']['lstsq']['beta'])
    print('近共线实际结果：k / 数值秩 / lstsq参数误差 / normal_solve状态 / normal_inverse状态')
    for row in report['near_collinear']:
        ms=row['methods'];print(row['k'],row['default_numeric_rank'],f'{ms["lstsq"]["coefficient_error"]:.3e}',ms['normal_solve']['status'],ms['normal_inverse']['status'])
    print('阈值实验返回秩：',[r['reported_rank'] for r in report['threshold_demo']['results']])
    print('列空间内扰动的参数变化：',report['perturbations']['cases']['column_space']['coefficient_change_norm'])
    print('同次预测变化：',report['perturbations']['cases']['column_space']['prediction_change_norm'])
    print('形状与边界检查通过：',len(report['boundary_tests']))
    print('核验通过；未预设每个例子的求解器误差排名。')


if __name__ == '__main__':
    print_report(run_experiment())
