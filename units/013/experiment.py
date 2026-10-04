"""第十三讲：一元导数、确定性差分与失败条件。核心仅用 Python 标准库。"""
from pathlib import Path
from fractions import Fraction
from numbers import Real
import argparse
import csv
import hashlib
import json
import math
import sys

BASE = Path(__file__).resolve().parent


def finite_real(value, name='value'):
    """接收有限实数标量；拒绝布尔值、字符串、复数和非有限数。"""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f'{name} 必须是实数标量，不能用布尔值或字符串代替')
    try:
        result = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError(f'{name} 无法表示为有限浮点数') from error
    if not math.isfinite(result):
        raise ValueError(f'{name} 必须有限')
    return result


def validate_data(rows):
    data = tuple(tuple(row) for row in rows)
    if not data or any(len(row) != 2 for row in data):
        raise ValueError('数据需要至少一对 (x, y)，每行恰好两个值')
    return tuple((finite_real(x, 'x'), finite_real(y, 'y')) for x, y in data)


def load_data(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['record_id', 'x', 'y']:
            raise ValueError('CSV 表头必须是 record_id,x,y')
        records = list(reader)
    ids = [r['record_id'] for r in records]
    if any(not i for i in ids) or len(ids) != len(set(ids)):
        raise ValueError('记录 ID 必须非空且唯一')
    # CSV 文本是显式解析边界；核心数学函数不做静默字符串转换。
    try:
        rows = [(float(r['x']), float(r['y'])) for r in records]
    except (TypeError, ValueError) as error:
        raise ValueError('CSV 的 x,y 必须是可解析数值') from error
    return validate_data(rows)


def mean_square_loss(w, rows):
    w = finite_real(w, 'w')
    data = validate_data(rows)
    try:
        loss = math.fsum((w*x-y)**2 for x, y in data) / len(data)
    except OverflowError as error:
        raise ValueError('平方误差超出有限浮点范围') from error
    return finite_real(loss, 'loss')


def loss_derivative(w, rows):
    w = finite_real(w, 'w')
    data = validate_data(rows)
    try:
        result = 2 * math.fsum(x*(w*x-y) for x, y in data) / len(data)
    except OverflowError as error:
        raise ValueError('导数计算超出有限浮点范围') from error
    return finite_real(result, 'derivative')


def finite_difference(function, x, h, method='central'):
    """基本固定步长公式，不是可导性证明或自适应生产求导器。

    分母使用名义 h；记录真实浮点采样位移，拒绝未移动的采样点。
    """
    x, h = finite_real(x, 'x'), finite_real(h, 'h')
    if h <= 0:
        raise ValueError('h 必须严格为正')
    if method not in ('forward', 'backward', 'central'):
        raise ValueError('method 只能为 forward/backward/central')
    center = finite_real(function(x), 'f(x)')
    points = {'center': x}
    values = {'center': center}
    if method in ('forward', 'central'):
        plus = finite_real(x+h, 'x+h')
        if plus == x:
            raise ValueError('正向步长在当前尺度下不可分辨')
        points['plus'] = plus
        values['plus'] = finite_real(function(plus), 'f(x+h)')
    if method in ('backward', 'central'):
        minus = finite_real(x-h, 'x-h')
        if minus == x:
            raise ValueError('负向步长在当前尺度下不可分辨')
        points['minus'] = minus
        values['minus'] = finite_real(function(minus), 'f(x-h)')
    if method == 'forward':
        estimate = (values['plus']-center)/h
    elif method == 'backward':
        estimate = (center-values['minus'])/h
    else:
        denominator = finite_real(2*h, '2h')
        estimate = (values['plus']-values['minus'])/denominator
    return {'estimate': finite_real(estimate, 'difference'), 'method': method,
            'h': h, 'points': points, 'values': values,
            'actual_positive_step': points['plus']-x if 'plus' in points else None,
            'actual_negative_step': x-points['minus'] if 'minus' in points else None}


def scan_steps(function, x, truth, steps):
    truth = finite_real(truth, 'reference derivative')
    result = []
    for h in steps:
        for method in ('forward', 'central'):
            try:
                row = finite_difference(function, x, h, method)
                row.update(status='ok', absolute_error=abs(row['estimate']-truth),
                           relative_error=abs(row['estimate']-truth)/abs(truth) if truth != 0 else None)
            except (ValueError, OverflowError, ZeroDivisionError) as error:
                row = {'h': float(h), 'method': method, 'status': 'rejected',
                       'reason': str(error), 'absolute_error': None, 'relative_error': None}
            result.append(row)
    return result


def exact_checks():
    f = Fraction
    rows = [(f(1), f(2)), (f(2), f(3)), (f(3), f(5))]
    loss = lambda w: sum((w*x-y)**2 for x,y in rows)/3
    deriv = lambda w: 2*sum(x*(w*x-y) for x,y in rows)/3
    checks = 0
    def require(condition, label):
        nonlocal checks
        if not condition:
            raise AssertionError(label)
        checks += 1
    require(loss(f(3,2)) == f(1,6), 'main loss')
    require(deriv(f(3,2)) == -f(4,3), 'main derivative')
    require(loss(f(23,14)) == f(1,14), 'minimum loss')
    require(deriv(f(23,14)) == 0, 'stationary derivative')
    require(loss(f(151,100)) == f(769,5000), 'finite change')
    for k in range(1,51):
        a, h = f(k-26,7), f(1,k+3)
        forward = ((a+h)**3-a**3)/h
        central = ((a+h)**3-(a-h)**3)/(2*h)
        require(forward == 3*a*a+3*a*h+h*h, 'cubic forward expansion')
        require(central == 3*a*a+h*h, 'cubic center expansion')
        require(loss(a+h)-loss(a) == deriv(a)*h+f(14,3)*h*h, 'loss remainder')
        require(loss(a) == f(14,3)*(a-f(23,14))**2+f(1,14), 'global square')
    h=f(1,20)
    return {'exact_checks_passed': checks, 'loss_at_3_over_2': '1/6',
            'derivative_at_3_over_2': '-4/3', 'optimum': '23/14', 'minimum': '1/14',
            'loss_at_151_over_100': '769/5000',
            'cubic_forward_error_at_h_1_over_20': str(3*h+h*h),
            'cubic_central_error_at_h_1_over_20': str(h*h)}


def boundary_checks():
    checks=[]
    def require(condition, label):
        if not condition:
            raise AssertionError(label)
        checks.append(label)
    def rejects(call, label):
        try:
            call()
        except (ValueError, OverflowError, ZeroDivisionError, TypeError):
            checks.append(label)
        else:
            raise AssertionError('未拒绝非法输入: '+label)
    rows=((1,2),(2,3),(3,5))
    require(math.isclose(mean_square_loss(1.5,rows),1/6,rel_tol=1e-14), 'loss agrees with hand')
    require(math.isclose(loss_derivative(1.5,rows),-4/3,rel_tol=1e-14), 'derivative agrees with hand')
    require(math.isclose(loss_derivative(23/14,rows),0,abs_tol=1e-14), 'zero derivative absolute tolerance')
    require(math.isclose(finite_difference(lambda z:z**3,1,.01,'central')['estimate'],3.0001,abs_tol=1e-12), 'cubic center')
    require(finite_difference(abs,0,.1,'forward')['estimate']==1, 'abs right')
    require(finite_difference(abs,0,.1,'backward')['estimate']==-1, 'abs left')
    require(finite_difference(abs,0,.1,'central')['estimate']==0, 'abs symmetric false positive')
    relu=lambda z:max(0,z)
    require(finite_difference(relu,0,.1,'central')['estimate']==.5, 'relu symmetric false positive')
    require(finite_difference(lambda z:7,2,.1)['estimate']==0, 'constant')
    require(math.isclose(finite_difference(math.log,.05,.01)['estimate'],(math.log(.06)-math.log(.04))/.02,abs_tol=1e-12), 'valid log domain')
    for bad in [True,'1',1+0j,float('inf'),float('-inf'),float('nan')]:
        rejects(lambda bad=bad:finite_real(bad),'invalid scalar '+repr(bad))
    for bad in [0,-.1,True,float('nan'),float('inf')]:
        rejects(lambda bad=bad:finite_difference(lambda z:z*z,1,bad),'invalid step '+repr(bad))
    rejects(lambda:finite_difference(lambda z:z,1e20,.5), 'unresolved scale')
    rejects(lambda:finite_difference(lambda z:z,1,1e-20,'forward'), 'unresolved tiny step')
    rejects(lambda:finite_difference(math.log,.05,.1), 'central log outside domain')
    rejects(lambda:finite_difference(lambda z:float('nan'),1,.1), 'nonfinite function output')
    rejects(lambda:finite_difference(lambda z:True,1,.1), 'boolean function output')
    rejects(lambda:finite_difference(lambda z:z,1,.1,'mystery'), 'unknown method')
    rejects(lambda:finite_difference(lambda z:1,0,1e308), 'overflowed central denominator')
    rejects(lambda:validate_data([]), 'empty data')
    rejects(lambda:validate_data([(1,2,3)]), 'wrong pair length')
    rejects(lambda:validate_data([(True,2)]), 'boolean data')
    rejects(lambda:validate_data([('1',2)]), 'string in core data')
    rejects(lambda:mean_square_loss(1e308,[(1e308,1)]), 'overflowed loss')
    rejects(lambda:loss_derivative(1e308,[(1e308,1)]), 'overflowed derivative')
    zero_scan=scan_steps(lambda z:z*z,0,0,[.01])
    require(all(v['relative_error'] is None for v in zero_scan), 'zero truth relative error omitted')
    return {'checks_passed':len(checks), 'checks':checks,
            'active_with_python_O':True, 'note':'Checks explicitly raise; no checks rely on the assert statement.'}


def run(data_path=BASE/'data/calibration.csv', output=BASE/'outputs'):
    data_path,output=Path(data_path).resolve(),Path(output).resolve()
    rows=load_data(data_path)
    if rows != ((1.,2.),(2.,3.),(3.,5.)):
        raise ValueError('本验收脚本要求随附的三条基准数据；修改数据后应另建实验及手算基准')
    if data_path in [output/'report.json',output/'difference_scan.csv']:
        raise ValueError('输出不得覆盖输入文件')
    steps=[10.**(-k) for k in range(1,18)]
    scans={'cubic':scan_steps(lambda z:z**3,1,3,steps),
           'loss':scan_steps(lambda w:mean_square_loss(w,rows),1.5,-4/3,steps)}
    report={'unit':'013','python':sys.version.split()[0],
            'data_sha256':hashlib.sha256(data_path.read_bytes()).hexdigest(),
            'data_kind':'original dimensionless synthetic teaching data',
            'exact':exact_checks(),'boundary':boundary_checks(),
            'loss_at_1_5':mean_square_loss(1.5,rows),
            'derivative_at_1_5':loss_derivative(1.5,rows),
            'scans':scans,
            'limitations':['Finite checks do not prove differentiability.','Basic fixed-step formulas are not an adaptive production solver.',
              'Nominal h is used in formulas; actual floating-point offsets are separately recorded.',
              'Numeric curve details depend on floating-point implementation; no universal optimal step is claimed.']}
    output.mkdir(parents=True,exist_ok=True)
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    with (output/'difference_scan.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['function','h','method','status','estimate','absolute_error','actual_positive_step','actual_negative_step','reason'])
        writer.writeheader()
        for name,entries in scans.items():
            for row in entries:
                writer.writerow({'function':name,**{k:row.get(k,'') for k in writer.fieldnames if k!='function'}})
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=BASE/'data/calibration.csv')
    parser.add_argument('--output',type=Path,default=BASE/'outputs')
    args=parser.parse_args();report=run(args.data,args.output)
    print(json.dumps({'unit':'013','exact_checks':report['exact']['exact_checks_passed'],
                      'boundary_checks':report['boundary']['checks_passed'],
                      'loss':report['loss_at_1_5'],'derivative':report['derivative_at_1_5']},ensure_ascii=False))
