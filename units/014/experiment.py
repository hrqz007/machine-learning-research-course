"""第014讲：多元导数、Jacobian 与链式法则。离线小CPU实验。

运行 python experiment.py。只有 NumPy 是非标准库依赖。
手推公式、展开式、差分分别实现；没有自动微分框架或训练循环。
"""
from pathlib import Path
from numbers import Real
from fractions import Fraction
import csv
import json
import sys
import numpy as np

BASE=Path(__file__).resolve().parent
ATOL=1e-7
RTOL=1e-7


def finite_scalar(value,name='value'):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,Real):
        raise ValueError(name+' 必须是实数标量，不能为布尔、字符串或复数')
    try: result=float(value)
    except (OverflowError,ValueError) as error: raise ValueError(name+' 超出有限浮点范围') from error
    if not np.isfinite(result): raise ValueError(name+' 必须有限')
    return result


def real_array(values,name='array'):
    """先检查原始叶子，避免混合列表把 bool/string 静默转换为数值。"""
    def check(v):
        if isinstance(v,np.ndarray):
            if v.dtype.kind not in 'iuf': raise ValueError(name+' 拒绝布尔、字符串、复数和object数组')
        elif isinstance(v,(list,tuple)):
            for item in v: check(item)
        else: finite_scalar(v,name)
    check(values)
    try: a=np.asarray(values,dtype=float)
    except (ValueError,OverflowError) as error: raise ValueError(name+' 不能构成有限实数数组') from error
    if not np.all(np.isfinite(a)): raise ValueError(name+' 必须全部有限')
    return a


def theta_vector(theta):
    a=real_array(theta,'theta')
    if a.shape!=(2,): raise ValueError('theta 必须是形状(2,)的两个参数；不接收行/列矩阵')
    return a


def data_vectors(x,y=None):
    x=real_array(x,'x')
    if x.ndim!=1 or x.size==0: raise ValueError('x 必须是非空一维数组')
    if y is None: return x
    y=real_array(y,'y')
    if y.shape!=x.shape: raise ValueError('y 必须是一维且与 x 同形，防止广播改变损失')
    return x,y


def load_data():
    with (BASE/'data'/'observations.csv').open(encoding='utf-8',newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=['record_id','x','y']: raise ValueError('CSV 表头不符合契约')
        rows=list(reader)
    ids=[r['record_id'] for r in rows]
    if any(not v for v in ids) or len(ids)!=len(set(ids)):raise ValueError('记录ID需非空且唯一')
    # CSV 是显式文本解析边界；核心数学函数不接受数字字符串。
    x=[float(r['x']) for r in rows];y=[float(r['y']) for r in rows]
    return data_vectors(x,y)


def prediction(theta,x):
    a,b=theta_vector(theta);x=data_vectors(x)
    with np.errstate(over='raise',invalid='raise'):
        try: result=a*x+b*b
        except FloatingPointError as error:raise ValueError('预测超出有限浮点范围') from error
    return real_array(result,'prediction')


def prediction_jacobian(theta,x):
    _,b=theta_vector(theta);x=data_vectors(x)
    # 行对应输出，列对应参数 a,b。这里返回 n×2，不转置。
    j=np.column_stack((x,np.full_like(x,2*b)))
    return real_array(j,'Jacobian')


def loss(theta,x,y):
    x,y=data_vectors(x,y);p=prediction(theta,x)
    with np.errstate(over='raise',invalid='raise'):
        try: result=np.mean((p-y)**2)
        except FloatingPointError as error:raise ValueError('损失超出有限浮点范围') from error
    return finite_scalar(result,'loss')


def gradient_chain(theta,x,y):
    x,y=data_vectors(x,y)
    output_gradient=2*(prediction(theta,x)-y)/x.size
    return real_array(prediction_jacobian(theta,x).T@output_gradient,'gradient')


def gradient_sum(theta,x,y):
    a,b=theta_vector(theta);x,y=data_vectors(x,y)
    r=prediction([a,b],x)-y
    return real_array([2*np.sum(r*x)/x.size,4*b*np.sum(r)/x.size],'gradient')


def expanded_loss(theta):
    """只对应 CSV 中固定三条记录的独立代数展开式。"""
    a,b=theta_vector(theta)
    return finite_scalar((5/3)*a*a+2*a*b*b-4*a+b**4-(10/3)*b*b+3,'expanded loss')


def expanded_gradient(theta):
    a,b=theta_vector(theta)
    return real_array([(10/3)*a+2*b*b-4,4*a*b+4*b**3-(20/3)*b],'expanded gradient')


def output_vector(value):
    """差分支持实标量或非空一维向量。标量提升为长度1数组。"""
    if np.isscalar(value): return np.array([finite_scalar(value,'function output')])
    a=real_array(value,'function output')
    if a.ndim!=1 or a.size==0:raise ValueError('函数输出必须为实标量或非空一维数组')
    return a


def finite_jacobian(function,theta,h,method='central'):
    """有限差分审计器，不声称证明可微；输出行、输入列。

    分母用名义步长，另记真实浮点位移。无法移动采样点时明确拒绝。
    """
    theta=theta_vector(theta);h=finite_scalar(h,'h')
    if h<=0:raise ValueError('h 必须严格为正')
    if method not in ['central','forward']:raise ValueError('method 仅支持 central 或 forward')
    denominator=finite_scalar(2*h if method=='central' else h,'difference denominator')
    center=output_vector(function(theta.copy()))
    j=np.empty((center.size,2));actual_steps=[]
    for k in range(2):
        plus=theta.copy();minus=theta.copy()
        plus[k]+=h;minus[k]-=h
        if not np.all(np.isfinite(plus)) or not np.all(np.isfinite(minus)):
            raise ValueError('采样点不是有限数')
        if plus[k]==theta[k] or (method=='central' and minus[k]==theta[k]):
            raise ValueError('步长在当前参数尺度下不可分辨')
        fp=output_vector(function(plus))
        if fp.shape!=center.shape:raise ValueError('扰动后函数输出形状改变')
        if method=='central':
            fm=output_vector(function(minus))
            if fm.shape!=center.shape:raise ValueError('扰动后函数输出形状改变')
            j[:,k]=(fp-fm)/denominator
        else:j[:,k]=(fp-center)/denominator
        actual_steps.append({'coordinate':k,'plus_delta':float(plus[k]-theta[k]),
                             'minus_delta':float(theta[k]-minus[k]) if method=='central' else None})
    if not np.all(np.isfinite(j)):raise ValueError('差分结果不是有限数')
    return j,actual_steps


def unit_direction(direction):
    d=theta_vector(direction);n=float(np.linalg.norm(d))
    if not np.isfinite(n) or n==0:raise ValueError('方向长度必须有限且非零')
    if not np.isclose(n,1.,atol=1e-12,rtol=1e-12):raise ValueError('本接口要求单位方向；请先显式归一化')
    return d


def directional_difference(function,theta,d,h):
    theta=theta_vector(theta);d=unit_direction(d);h=finite_scalar(h,'h')
    if h<=0:raise ValueError('h 必须为正')
    denominator=finite_scalar(2*h,'directional denominator')
    with np.errstate(over='ignore',invalid='ignore'):
        plus=theta+h*d;minus=theta-h*d
    if not np.all(np.isfinite(plus)) or not np.all(np.isfinite(minus)):
        raise ValueError('方向采样点不是有限数')
    if np.array_equal(plus,theta) or np.array_equal(minus,theta):raise ValueError('方向步长不可分辨')
    fp=finite_scalar(function(plus),'f plus');fm=finite_scalar(function(minus),'f minus')
    return finite_scalar((fp-fm)/denominator,'directional estimate')


def branched(theta):
    a,b=theta_vector(theta);s=a+b;t=a-b
    return finite_scalar(s*s+s*t,'branched output')


def branched_gradient(theta):
    a,b=theta_vector(theta)
    # s 同时通向平方与乘积，因此上游贡献先相加成 2s+t。
    with np.errstate(over='raise',invalid='raise'):
        try:
            s=a+b;t=a-b
            result=np.array([2*s+t,s])@np.array([[1.,1.],[1.,-1.]])
        except FloatingPointError as error:
            raise ValueError('分支梯度超出有限浮点范围') from error
    return real_array(result,'branched gradient')


def log_gap(theta):
    a,b=theta_vector(theta);z=a-b*b
    if z<=0:raise ValueError('log_gap 定义域要求 a-b²>0')
    return finite_scalar(np.log(z),'log value')


def partial_counterexample(theta):
    """连续且每条方向导数存在，原点仍不全微分：x³/(x²+y²)。"""
    x,y=theta_vector(theta)
    if x==0 and y==0:return 0.
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        try:result=x**3/(x*x+y*y)
        except FloatingPointError as error:
            raise ValueError('反例函数计算超出本实现的有效浮点范围') from error
    return finite_scalar(result,'counterexample output')


def exact_hand():
    f=Fraction;a=f(1);b=f(1,2);x=[f(0),f(1),f(2)];y=[f(1),f(2),f(2)]
    p=[a*t+b*b for t in x];r=[v-w for v,w in zip(p,y)]
    l=sum(t*t for t in r)/3
    ga=2*sum(v*t for v,t in zip(r,x))/3;gb=4*b*sum(r)/3
    d=[f(3,5),f(4,5)];directional=ga*d[0]+gb*d[1]
    assert l==f(19,48) and ga==f(-1,6) and gb==f(-5,6) and directional==f(-23,30)
    # 沿单位方向的中心差分误差，直接展开该四次多项式，精确为224/125*h²。
    def exact_loss(aa,bb):return sum((aa*t+bb*bb-v)**2 for t,v in zip(x,y))/3
    h=f(1,100)
    fd=(exact_loss(a+h*d[0],b+h*d[1])-exact_loss(a-h*d[0],b-h*d[1]))/(2*h)
    assert fd-directional==f(224,125)*h*h
    return {'theta':['1','1/2'],'prediction':[str(v) for v in p],'residual_prediction_minus_target':[str(v) for v in r],
            'loss':str(l),'gradient':[str(ga),str(gb)],'unit_direction':['3/5','4/5'],
            'directional_derivative':str(directional),'directional_central_error':'(224/125)*h^2',
            'branched_value':'3','branched_gradient':['5','2']}


def audit_boundaries():
    x,y=load_data();passed=[]
    def reject(name,action):
        try:action()
        except ValueError:passed.append(name)
        else:raise AssertionError('应拒绝: '+name)
    for name,value in [('theta_column',[[1],[.5]]),('theta_row',[[1,.5]]),('theta_length',[1]),
                       ('theta_bool',[1,True]),('theta_string',[1,'0.5']),('theta_complex',[1,1j]),
                       ('theta_object',np.array([1,.5],dtype=object)),('theta_nan',[1,np.nan]),
                       ('theta_inf',[1,np.inf])]:
        reject(name,lambda v=value:prediction(v,x))
    reject('x_empty',lambda:loss([1,.5],[],[]))
    reject('y_column',lambda:loss([1,.5],x,y[:,None]))
    reject('y_length',lambda:loss([1,.5],x,[1,2]))
    reject('x_mixed_bool',lambda:prediction([1,.5],[0,True,2]))
    reject('object_leaf',lambda:prediction([1,.5],[object()]))
    for name,h in [('zero_step',0),('negative_step',-.1),('bool_step',True),('string_step','0.01'),('infinite_step',np.inf)]:
        reject(name,lambda hh=h:finite_jacobian(lambda t:loss(t,x,y),[1,.5],hh))
    reject('unresolved_step',lambda:finite_jacobian(lambda t:loss(t,x,y),[1,.5],1e-20))
    reject('overflow_denominator',lambda:finite_jacobian(lambda t:1.,[1,.5],1e308))
    reject('directional_denominator_overflow',lambda:directional_difference(lambda t:1.,[1,.5],[.6,.8],1e308))
    callback_calls=[]
    def finite_only_probe(t):
        callback_calls.append(t.copy())
        return 1.
    reject('directional_sample_overflow',lambda:directional_difference(finite_only_probe,[1.7e308,1.7e308],[.6,.8],8e307))
    if callback_calls:raise AssertionError('非有限方向采样点不得传入被测函数')
    reject('branched_gradient_sum_overflow',lambda:branched_gradient([1e308,1e308]))
    reject('branched_gradient_difference_overflow',lambda:branched_gradient([1e308,-1e308]))
    reject('counterexample_overflow',lambda:partial_counterexample([1e308,1e308]))
    reject('counterexample_underflow_division',lambda:partial_counterexample([1e-200,1e-200]))
    reject('bad_method',lambda:finite_jacobian(lambda t:loss(t,x,y),[1,.5],1e-4,'backward'))
    reject('zero_direction',lambda:unit_direction([0,0]))
    reject('nonunit_direction',lambda:unit_direction([3,4]))
    reject('domain_point',lambda:log_gap([.25,.5]))
    reject('domain_crossing',lambda:finite_jacobian(log_gap,[.26,.5],.1))
    reject('changing_output_shape',lambda:finite_jacobian(lambda t:np.ones(1 if t[0]<=1 else 2),[1,.5],.01))
    # 合法但很容易被 squeeze 或广播处理错的单样本形状。
    assert prediction([1,.5],[2]).shape==(1,)
    assert prediction_jacobian([1,.5],[2]).shape==(1,2)
    assert gradient_chain([1,.5],[2],[2]).shape==(2,)
    passed.append('single_sample_shapes')
    return {'passed_count':len(passed),'names':passed}


def run_all(write_outputs=True):
    x,y=load_data();theta=np.array([1.,.5]);g=gradient_chain(theta,x,y);j=prediction_jacobian(theta,x)
    assert np.array_equal(x,[0,1,2]) and np.array_equal(y,[1,2,2])
    assert np.allclose(g,[-1/6,-5/6],atol=1e-14,rtol=1e-14)
    assert np.array_equal(j,[[0,1],[1,1],[2,1]])
    checks=[]
    for a in [-1.,0.,1.,1.2,2.]:
        for b in [-1.,-.5,0.,.5,1.]:
            t=np.array([a,b]);g1=gradient_chain(t,x,y);g2=gradient_sum(t,x,y);g3=expanded_gradient(t)
            fd,_=finite_jacobian(lambda q:loss(q,x,y),t,1e-5)
            jp,_=finite_jacobian(lambda q:prediction(q,x),t,1e-5)
            assert g1.shape==g2.shape==g3.shape==(2,) and fd.shape==(1,2) and jp.shape==(3,2)
            assert np.isclose(loss(t,x,y),expanded_loss(t),atol=1e-12,rtol=1e-12)
            assert np.allclose(g1,g2,atol=1e-12,rtol=1e-12) and np.allclose(g1,g3,atol=1e-12,rtol=1e-12)
            assert np.allclose(fd[0],g1,atol=ATOL,rtol=RTOL)
            assert np.allclose(jp,prediction_jacobian(t,x),atol=ATOL,rtol=RTOL)
            checks.append({'theta':t.tolist(),'gradient_fd_max_abs_error':float(np.max(np.abs(fd[0]-g1))),
                           'prediction_jacobian_fd_max_abs_error':float(np.max(np.abs(jp-prediction_jacobian(t,x))))})
    sweep=[]
    for h in [10.**(-k) for k in range(1,17)]+[1e-20]:
        row={'h':h}
        for method in ['forward','central']:
            try:
                fd,steps=finite_jacobian(lambda t:loss(t,x,y),theta,h,method)
                row[method]={'status':'ok','gradient':fd[0].tolist(),'max_abs_error':float(np.max(np.abs(fd[0]-g))),'actual_steps':steps}
            except ValueError as error:row[method]={'status':'rejected','reason':str(error)}
        sweep.append(row)
    d=np.array([.6,.8]);analytic=float(g@d);fd=directional_difference(lambda t:loss(t,x,y),theta,d,1e-5)
    assert np.isclose(fd,analytic,atol=ATOL,rtol=RTOL)
    # 局部线性余量除以位移范数应缩小；不要求浮点最小步长严格单调。
    local=[]
    for h in [.1,.05,.01,.005,.001]:
        actual=loss(theta+h*d,x,y)-loss(theta,x,y);linear=float(h*g@d)
        local.append({'h':h,'actual_change':actual,'linear_change':linear,'abs_remainder_over_step':abs(actual-linear)/h})
    assert local[-1]['abs_remainder_over_step']<local[0]['abs_remainder_over_step']
    bg=branched_gradient(theta);bfd,_=finite_jacobian(branched,theta,1e-5)
    assert np.allclose(bg,[5,2],atol=1e-12,rtol=1e-12) and np.allclose(bfd[0],bg,atol=ATOL,rtol=RTOL)
    # 反例：所有方向导数存在，却不是梯度的一次线性组合。
    dc=np.array([1.,1.])/np.sqrt(2)
    counter_j,_=finite_jacobian(partial_counterexample,[0.,0.],.01)
    assert counter_j.shape==(1,2) and np.allclose(counter_j[0],[1.,0.],atol=1e-14,rtol=1e-14)
    counter_fd=directional_difference(partial_counterexample,[0.,0.],dc,.01)
    true_direction=1/(2*np.sqrt(2));wrong_dot=1/np.sqrt(2)
    assert np.isclose(counter_fd,true_direction,atol=1e-14,rtol=1e-14)
    # b=0 的驻点没有一阶下降信号，但沿 b 的小变化会降低损失。
    stationary=np.array([1.2,0.]);base=loss(stationary,x,y)
    assert np.allclose(gradient_chain(stationary,x,y),[0,0],atol=1e-14,rtol=1e-14)
    assert loss(stationary+[0,.1],x,y)<base<loss(stationary+[.1,0],x,y)
    log_theta=np.array([2.,.5]);log_fd,_=finite_jacobian(log_gap,log_theta,1e-5)
    gap=log_theta[0]-log_theta[1]**2;log_g=np.array([1.,-2*log_theta[1]])/gap
    assert np.allclose(log_fd[0],log_g,atol=ATOL,rtol=RTOL)
    # 列约定显式测试：2×3 乘 3×1 等于 2×1。
    outg=2*(prediction(theta,x)-y)/x.size
    column_result=j.T@outg[:,None]
    assert column_result.shape==(2,1) and np.allclose(column_result[:,0],g)
    report={'python':sys.version.split()[0],'numpy':np.__version__,'dtype':'float64','exact_hand':exact_hand(),
            'main':{'theta':theta.tolist(),'loss':loss(theta,x,y),'prediction':prediction(theta,x).tolist(),
                    'jacobian_shape':list(j.shape),'jacobian':j.tolist(),'gradient_array_shape':list(g.shape),
                    'column_gradient_shape':list(column_result.shape),'gradient':g.tolist(),
                    'directional_analytic':analytic,'directional_fd':fd},
            'deterministic_grid':{'points':len(checks),'checks':checks},'step_sweep':sweep,'local_linear':local,
            'branch':{'value':branched(theta),'gradient':bg.tolist(),'finite_difference':bfd[0].tolist(),
                      'paths_a':[3.,.5,1.5],'paths_b':[3.,.5,-1.5]},
            'counterexample':{'coordinate_partials':[1.,0.],'diagonal_direction_fd':counter_fd,
                              'true_diagonal_direction':true_direction,'invalid_gradient_dot_prediction':wrong_dot},
            'stationary':{'theta':stationary.tolist(),'loss':base,'loss_after_b_0_1':loss(stationary+[0,.1],x,y),
                          'loss_after_a_0_1':loss(stationary+[.1,0],x,y)},
            'log_domain':{'theta':log_theta.tolist(),'gradient':log_g.tolist(),'finite_difference':log_fd[0].tolist()},
            'boundary':audit_boundaries(),'tolerances':{'fd_atol':ATOL,'fd_rtol':RTOL,'algebra_atol':1e-12,'algebra_rtol':1e-12}}
    if write_outputs:
        out=BASE/'outputs';out.mkdir(exist_ok=True)
        (out/'experiment_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        with (out/'finite_difference_sweep.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.writer(f);writer.writerow(['h','method','status','max_abs_error'])
            for row in sweep:
                for method in ['forward','central']:
                    z=row[method];writer.writerow([row['h'],method,z['status'],z.get('max_abs_error','')])
    return report


def print_summary(report):
    print('NumPy:',report['numpy'])
    print('主点损失 19/48 =',report['main']['loss'])
    print('梯度 [-1/6, -5/6] =',report['main']['gradient'])
    print('预测 Jacobian 形状:',report['main']['jacobian_shape'])
    print('单位方向变化率 -23/30 =',report['main']['directional_analytic'])
    print('分支例梯度 [5,2] =',report['branch']['gradient'])
    print('确定性网格点数:',report['deterministic_grid']['points'])
    print('边界通过组数:',report['boundary']['passed_count'])
    print('输出文件存在:',(BASE/'outputs'/'experiment_report.json').exists())

if __name__=='__main__':print_summary(run_all())
