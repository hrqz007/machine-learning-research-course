"""030 loss geometry: bounded original teaching code; no external services.

The objective is J = sum((a+b*x-y)**2)/(2*n). Finite-budget paths are
illustrations, not a general optimization library. Read README.md for contracts.
"""
from pathlib import Path
from fractions import Fraction
from decimal import Decimal, InvalidOperation
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
CASES = ('hand', 'constant', 'changed_sample')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def real(value, name='value', low=-1e6, high=1e6):
    permitted = type(value) in (int, float, np.float16, np.float32, np.float64) or isinstance(value, np.integer)
    require(permitted and not isinstance(value, (bool, np.bool_)), name+' requires an ordinary real scalar')
    require(low <= value <= high, name+' is nonfinite or outside supported range')
    require(value == 0 or abs(value) >= 1e-100, name+' nonzero magnitude below 1e-100')
    result = float(value)
    require(math.isfinite(result) and (value == 0 or result != 0), name+' cast loses a nonzero value')
    if isinstance(value, (int, np.integer)):
        require(int(result) == int(value), name+' integer cast is not exact')
    return result


def integer(value, name, low, high):
    require(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)), name+' requires an integer')
    require(low <= value <= high, name+' outside supported range')
    return int(value)


def vector(values, name='values', minimum=1, maximum=2000):
    require(isinstance(values, (list, tuple, np.ndarray)), name+' requires a list, tuple, or numeric ndarray')
    if isinstance(values, np.ndarray):
        require(values.ndim == 1, name+' requires shape (n,)')
        require(values.dtype.kind in 'iuf' and values.dtype.itemsize <= 8, name+' rejects bool/object/complex/extended arrays')
    require(minimum <= len(values) <= maximum, name+' wrong length')
    return [real(v, name+'['+str(i)+']') for i, v in enumerate(values)]


def pair(x, y):
    x, y = vector(x, 'x'), vector(y, 'y')
    require(len(x) == len(y), 'x and y must have equal length')
    return x, y


def finite(value, name):
    require(math.isfinite(value), name+' nonfinite result')
    return value


def product(a, b, name='product'):
    q = finite(a*b, name)
    require(a == 0 or b == 0 or abs(q) >= MIN_NORMAL, name+' nonzero product became zero/subnormal')
    return q


def divide(a, b, name='division'):
    require(b != 0, name+' zero denominator')
    q = finite(a/b, name)
    require(a == 0 or abs(q) >= MIN_NORMAL, name+' nonzero result became zero/subnormal')
    return q


def sumsq(values, name='squared norm'):
    return finite(math.fsum(product(v, v, name) for v in values), name)


def _state(x, y, theta):
    a, b = theta
    residuals = [math.fsum([a, product(b, t, 'b*x'), -v]) for t, v in zip(x, y)]
    value = divide(sumsq(residuals, 'residual square'), 2*len(x), 'loss')
    if value == 0:
        exact_residuals = [Fraction(a)+Fraction(b)*Fraction(t)-Fraction(v) for t,v in zip(x,y)]
        if any(exact_residuals):
            residuals = [float(q) for q in exact_residuals]
            require(all(q == 0 or abs(v) >= MIN_NORMAL for q,v in zip(exact_residuals,residuals)), 'exact residual underflow')
            value = divide(sumsq(residuals, 'recovered residual square'), 2*len(x), 'recovered loss')
    ga = divide(math.fsum(residuals), len(x), 'gradient a')
    gb = divide(math.fsum(product(t, r, 'gradient product') for t, r in zip(x, residuals)), len(x), 'gradient b')
    return {'theta':[a,b], 'loss':value, 'gradient':[ga,gb], 'gradient_norm':math.hypot(ga,gb)}


def loss_gradient(x, y, theta):
    """Validated scalar objective and length-2 gradient; residual sign is pred-y."""
    x, y = pair(x,y)
    theta = vector(theta, 'theta', 2, 2)
    return _state(x,y,theta)


def exact_reference(x, y):
    """Closed form over the exact rational values of stored binary64 inputs.

    This is independent of the path optimizer. It does not restore measurement
    precision lost before entry; bounded input size keeps the teaching cost finite.
    """
    x,y = pair(x,y)
    xq,yq = [Fraction(v) for v in x],[Fraction(v) for v in y]
    n=len(xq);mx=sum(xq)/n;my=sum(yq)/n
    sxx=sum((t-mx)**2 for t in xq)
    sxy=sum((t-mx)*(v-my) for t,v in zip(xq,yq))
    b=Fraction(0) if sxx==0 else sxy/sxx;a=my-b*mx
    j=sum((a+b*t-v)**2 for t,v in zip(xq,yq))/(2*n)
    def convert(q,name):
        v=float(q)
        require(math.isfinite(v) and abs(v)<=1e12, name+' exceeds finite teaching range')
        require(q==0 or abs(v)>=MIN_NORMAL,name+' exact nonzero result became zero/subnormal')
        return v
    theta=[convert(a,'reference a'),convert(b,'reference b')]
    return {'theta':theta,'minimum_loss':convert(j,'reference loss'),
            'exact_theta':[str(a),str(b)],'exact_minimum_loss':str(j),
            'x_mean':convert(mx,'x mean'),'y_mean':convert(my,'y mean'),
            'Sxx':convert(sxx,'Sxx'),'unique_parameters':sxx>0,
            'meaning':'exact rational reference for stored binary64 data; decimal output rounded'}


def geometry(x, y):
    x,y=pair(x,y)
    reference=exact_reference(x,y)
    mean_x=divide(math.fsum(x),len(x),'mean x')
    mean_x2=divide(sumsq(x,'x square'),len(x),'mean x squared')
    h=np.array([[1.,mean_x],[mean_x,mean_x2]])
    ev=np.linalg.eigvalsh(h)
    require(np.isfinite(ev).all(),'Hessian eigenvalues nonfinite')
    # Mathematical uniqueness comes from the exact spread, not eigenvalue rounding.
    conditioned=reference['unique_parameters'] and ev[0]>0
    return {'reference':reference,'hessian':h.tolist(),'eigenvalues':ev.tolist(),
            'condition_number_if_positive':float(ev[-1]/ev[0]) if conditioned else None,
            'positive_mu_for_numeric_bound':float(ev[0]) if conditioned else None}


def gap_from_reference(theta, reference, n):
    """Nonnegative quadratic excess relative to the rounded closed-form anchor.

    Avoid subtracting nearly equal total losses. At floating resolution this is
    a diagnostic, not an exact certificate for arbitrary ill-conditioned inputs.
    """
    da=theta[0]-reference['theta'][0];db=theta[1]-reference['theta'][1]
    c=math.fsum([da,product(reference['x_mean'],db)])
    return finite(.5*product(c,c)+product(reference['Sxx']/(2*n),product(db,db)),'quadratic gap')


def validate_path(spec):
    require(type(spec) is dict and set(spec)=={'name','initial','step','max_steps','gradient_tolerance'}, 'path fields incorrect')
    name=spec['name'];require(type(name) is str and re.fullmatch(r'[a-z][a-z0-9_]{0,31}',name) is not None,'invalid path name')
    return {'name':name,'initial':vector(spec['initial'],'initial',2,2),
            'step':real(spec['step'],'step',1e-8,1.),
            'max_steps':integer(spec['max_steps'],'max_steps',0,10000),
            'gradient_tolerance':real(spec['gradient_tolerance'],'gradient_tolerance',1e-12,1.)}


def _path(x,y,spec,geo):
    theta=spec['initial'].copy();ref=geo['reference'];history=[];status='budget_exhausted'
    for k in range(spec['max_steps']+1):
        s=_state(x,y,theta);s['iteration']=k
        s['quadratic_gap']=gap_from_reference(theta,ref,len(x));history.append(s)
        if s['gradient_norm']<=spec['gradient_tolerance']:
            status='gradient_tolerance_met';break
        if k==spec['max_steps']:
            break
        candidate=[math.fsum([v,-product(spec['step'],g,'step*gradient')]) for v,g in zip(theta,s['gradient'])]
        if any(not math.isfinite(v) or abs(v)>1e6 for v in candidate):
            status='growth_limit_reached';break
        if candidate==theta:
            status='stagnated_without_tolerance';break
        theta=candidate
    return {'name':spec['name'],'status':status,'success':status=='gradient_tolerance_met',
            'updates_completed':history[-1]['iteration'],'max_steps':spec['max_steps'],
            'gradient_tolerance':spec['gradient_tolerance'],'step':spec['step'],
            'history':history,'final':history[-1]}


def trace_path(x,y,spec):
    x,y=pair(x,y);spec=validate_path(spec)  # All arguments, including unused budget, first.
    geo=geometry(x,y)
    return _path(x,y,spec,geo)


def constrained_reference(x,y,slope_upper):
    x,y=pair(x,y);upper=real(slope_upper,'slope_upper')
    ref=exact_reference(x,y)
    require(ref['unique_parameters'],'this constrained illustration requires nonconstant x')
    xq,yq=[Fraction(v) for v in x],[Fraction(v) for v in y]
    b=min(Fraction(ref['exact_theta'][1]),Fraction(upper))
    a=sum(yq)/len(yq)-b*sum(xq)/len(xq)
    require(abs(a)<=1e6 and abs(b)<=1e6,'constrained parameters outside state range')
    state=_state(x,y,[float(a),float(b)])
    state['exact_theta']=[str(a),str(b)]
    state['exact_loss']=str(sum((a+b*t-v)**2 for t,v in zip(xq,yq))/(2*len(xq)))
    state['constraint']='slope <= upper';state['slope_upper']=upper
    return state


def number_token(token):
    require(type(token) is str and len(token)<=64 and TOKEN.fullmatch(token) is not None,'invalid numeric token')
    try:d=Decimal(token)
    except InvalidOperation as exc:raise ValueError('invalid decimal token') from exc
    require(d.is_finite() and abs(d)<=Decimal('1e6'),'numeric token outside finite range')
    require(d==0 or abs(d)>=Decimal('1e-100'),'nonzero numeric token below 1e-100')
    return real(float(d),'numeric token')


def reject_constant(token):
    raise ValueError('nonstandard JSON constant: '+token)


def unique_object(pairs):
    result={}
    for k,v in pairs:
        require(k not in result,'duplicate JSON key: '+k);result[k]=v
    return result


def validate_config(cfg):
    require(type(cfg) is dict and set(cfg)=={'focus_case','paths','candidate_parameters','slope_upper','grid'},'config fields incorrect')
    require(type(cfg['focus_case']) is str and cfg['focus_case'] in CASES,'unknown focus_case')
    require(type(cfg['paths']) is list and 1<=len(cfg['paths'])<=12,'paths needs 1..12 entries')
    paths=[validate_path(s) for s in cfg['paths']]
    require(len({s['name'] for s in paths})==len(paths),'duplicate path name')
    candidates=cfg['candidate_parameters']
    require(type(candidates) is list and 1<=len(candidates)<=30,'candidate_parameters needs 1..30 entries')
    candidates=[vector(t,'candidate',2,2) for t in candidates]
    upper=real(cfg['slope_upper'],'slope_upper')
    grid=cfg['grid'];require(type(grid) is dict and set(grid)=={'a_bounds','b_bounds','points'},'grid fields incorrect')
    ab=vector(grid['a_bounds'],'a_bounds',2,2);bb=vector(grid['b_bounds'],'b_bounds',2,2)
    require(ab[0]<ab[1] and bb[0]<bb[1],'grid bounds must increase')
    points=integer(grid['points'],'grid points',5,201)
    return {'focus_case':cfg['focus_case'],'paths':paths,'candidate_parameters':candidates,'slope_upper':upper,
            'grid':{'a_bounds':ab,'b_bounds':bb,'points':points}}


def read_config(path):
    cfg=json.loads(Path(path).read_text(encoding='utf-8'),parse_float=number_token,parse_int=int,
                   parse_constant=reject_constant,object_pairs_hook=unique_object)
    return validate_config(cfg)


def read_cases(path):
    groups={c:{'ids':[],'x':[],'y':[]} for c in CASES};seen=set();count=0
    with Path(path).open(newline='',encoding='utf-8') as f:
        r=csv.DictReader(f)
        require(r.fieldnames==['case','id','x','y'],'CSV header must be case,id,x,y')
        for row in r:
            count+=1;require(count<=6000,'CSV too large')
            require(set(row)=={'case','id','x','y'} and all(v is not None for v in row.values()),'malformed CSV row')
            case,key=row['case'],row['id']
            require(case in groups and re.fullmatch(r'[A-Z][0-9]{2,4}',key) is not None,'invalid CSV case/id')
            require(key not in seen,'duplicate CSV id');seen.add(key)
            x,y=number_token(row['x']),number_token(row['y'])
            groups[case]['ids'].append(key);groups[case]['x'].append(x);groups[case]['y'].append(y)
    for case,g in groups.items():
        require(2<=len(g['x'])<=2000,'each case needs 2..2000 rows')
    return groups


def validate_groups(groups):
    require(type(groups) is dict and set(groups)==set(CASES),'case mapping incorrect')
    clean={}
    for name,g in groups.items():
        require(type(g) is dict and set(g)=={'ids','x','y'},'case fields incorrect')
        x,y=pair(g['x'],g['y']);require(2<=len(x)<=2000,'case needs 2..2000 rows')
        require(type(g['ids']) is list and len(g['ids'])==len(x),'ids length mismatch')
        require(all(type(k) is str and re.fullmatch(r'[A-Z][0-9]{2,4}',k) for k in g['ids']),'invalid ids')
        clean[name]={'ids':g['ids'].copy(),'x':x,'y':y}
    ids=[k for g in clean.values() for k in g['ids']];require(len(set(ids))==len(ids),'duplicate ids')
    return clean


def surface(x,y,grid):
    x,y=pair(x,y)
    require(type(grid) is dict and set(grid)=={'a_bounds','b_bounds','points'},'grid fields incorrect')
    ab=vector(grid['a_bounds'],'a_bounds',2,2);bb=vector(grid['b_bounds'],'b_bounds',2,2)
    require(ab[0]<ab[1] and bb[0]<bb[1],'grid bounds must increase')
    n=integer(grid['points'],'points',5,201)
    av=np.linspace(*ab,n);bv=np.linspace(*bb,n);aa,bbm=np.meshgrid(av,bv,indexing='xy')
    # Checked scalar loss avoids silent underflow in a vectorized power operation.
    zz=np.array([_state(x,y,[float(a),float(b)])['loss'] for a,b in zip(aa.ravel(),bbm.ravel())]).reshape(aa.shape)
    return aa,bbm,zz


def compile_report(groups,cfg):
    groups=validate_groups(groups);cfg=validate_config(cfg)  # validate every last/unselected field first
    geos={name:geometry(g['x'],g['y']) for name,g in groups.items()}
    focus=groups[cfg['focus_case']];x,y=focus['x'],focus['y']
    candidates=[loss_gradient(x,y,t) for t in cfg['candidate_parameters']]
    # Full surface construction is part of validation, even when only summary is saved.
    aa,bb,zz=surface(x,y,cfg['grid']);index=np.unravel_index(int(np.argmin(zz)),zz.shape)
    paths=[_path(x,y,s,geos[cfg['focus_case']]) for s in cfg['paths']]
    constrained=constrained_reference(x,y,cfg['slope_upper']) if geos[cfg['focus_case']]['reference']['unique_parameters'] else {'status':'not_applied_nonunique_parameters'}
    return {'unit':'030','objective':'SSE/(2*n)','data_kind':'deterministic synthetic teaching data',
            'focus_case':cfg['focus_case'],'geometry':geos,'candidates':candidates,'paths':paths,'constrained':constrained,
            'grid':{'shape':list(zz.shape),'minimum_sampled_loss':float(zz[index]),'minimum_sampled_theta':[float(aa[index]),float(bb[index])],
                    'note':'finite grid minimum is not a proof of continuous global optimality'},
            'limitations':['finite binary64 teaching domain','optimizer status is computational, not statistical validity',
                           'no RNG or real-world performance evaluation','gradient tolerance does not certify an arbitrary nonconvex global minimum']}


def run(data=None,config=None,output=None):
    data=ROOT/'data/cases.csv' if data is None else Path(data)
    config=ROOT/'data/model_spec.json' if config is None else Path(config)
    output=ROOT/'outputs/report.json' if output is None else Path(output)
    cfg=read_config(config);groups=read_cases(data);report=compile_report(groups,cfg)
    payload=json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile('w',encoding='utf-8',dir=output.parent,prefix='.030-',suffix='.tmp',delete=False) as f:
            temporary=Path(f.name);f.write(payload)
        os.replace(temporary,output)
    finally:
        if temporary is not None and temporary.exists():temporary.unlink()
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data');p.add_argument('--config');p.add_argument('--output')
    a=p.parse_args();report=run(a.data,a.config,a.output)
    print(json.dumps({'unit':'030','focus':report['focus_case'],'reference':report['geometry'][report['focus_case']]['reference'],
                      'paths':[{'name':r['name'],'status':r['status'],'updates':r['updates_completed'],'final_loss':r['final']['loss']} for r in report['paths']]},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
