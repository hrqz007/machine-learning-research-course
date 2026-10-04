"""Deterministic scalar GD teaching experiment; no plotting dependency for CLI.

The solver performs actual binary64 updates. Fraction functions are explicitly
separate exact references, never substituted for iterative solver updates.
"""
from pathlib import Path
from fractions import Fraction as F
from decimal import Decimal, InvalidOperation
import argparse, csv, hashlib, json, math, os, re, sys, tempfile
import numbers

ROOT=Path(__file__).resolve().parent
MIN_INPUT=1e-100
MIN_NORMAL=sys.float_info.min

def real(value,name='value',limit=1e8,positive=False,nonnegative=False):
    # Inspect the original type. No silent coercion of text, booleans or complex.
    typ=type(value)
    allowed=typ in (int,float) or (typ.__module__=='numpy' and typ.__name__ in
        ('int8','int16','int32','int64','uint8','uint16','uint32','uint64','float16','float32','float64'))
    if not allowed:raise ValueError(name+' must be an ordinary real scalar')
    try:v=float(value)
    except (OverflowError,ValueError) as exc:raise ValueError(name+' cannot convert safely') from exc
    if not math.isfinite(v) or abs(v)>limit:raise ValueError(name+' outside finite range')
    if isinstance(value,numbers.Integral) and int(v)!=int(value):raise ValueError(name+' integer cast loses information')
    if v and abs(v)<MIN_INPUT:raise ValueError(name+' below teaching input floor')
    if positive and v<=0 or nonnegative and v<0:raise ValueError(name+' has invalid sign')
    return v

def integer(value,name,maximum=10000):
    if type(value) is not int or not 0<=value<=maximum:raise ValueError(name+' must be bounded integer')
    return value

def computed(v,name,positive_expected=False):
    if not math.isfinite(v) or abs(v)>1e150:raise ValueError(name+' exceeds computation range')
    if v and abs(v)<MIN_NORMAL or positive_expected and v==0:raise ValueError(name+' underflows supported normal range')
    return v

def product(a,b,name):return computed(a*b,name,positive_expected=(a!=0 and b!=0))

def token(text):
    if type(text) is not str or not text.strip():raise ValueError("Expected nonempty numeric token")
    try:d=Decimal(text)
    except InvalidOperation as exc:raise ValueError('Invalid numeric token') from exc
    if not d.is_finite():raise ValueError('Nonfinite token')
    v=float(d)
    if d!=0 and v==0:raise ValueError('Nonzero token underflow')
    return real(v,'numeric token')

def pairs(items):
    out={}
    for k,v in items:
        if k in out:raise ValueError('Duplicate JSON key: '+k)
        out[k]=v
    return out

def read_spec(path=ROOT/'data/model_spec.json'):
    x=json.loads(Path(path).read_text(),parse_float=token,
                 parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON')),
                 object_pairs_hook=pairs)
    keys={'kind','initial_slope','trajectory_steps','max_steps','gradient_tolerance',
          'parameter_error_tolerance','objective_multiplier','parameter_scale',
          'counterexample_initial','counterexample_rate'}
    if type(x) is not dict or set(x)!=keys or x['kind']!='synthetic_scalar_gradient_descent_demo':raise ValueError('Bad configuration schema')
    integer(x['trajectory_steps'],'trajectory_steps',1000);integer(x['max_steps'],'max_steps')
    for k in keys-{'kind','trajectory_steps','max_steps'}:
        x[k]=real(x[k],k,positive=k in {'gradient_tolerance','parameter_error_tolerance','objective_multiplier','parameter_scale','counterexample_rate'})
    return x

def read_data(path=ROOT/'data/regression.csv'):
    with Path(path).open(newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=['id','x','y']:raise ValueError('Bad regression header')
        rows=list(reader)
    if not 2<=len(rows)<=1000:raise ValueError('Need 2..1000 observations')
    ids=set();x=[];y=[]
    for r in rows:
        if set(r)!= {'id','x','y'} or not r['id'] or r['id'] in ids:raise ValueError('Bad or duplicate row ID')
        ids.add(r['id']);x.append(token(r['x']));y.append(token(r['y']))
    return x,y

def read_rates(path=ROOT/'data/learning_rates.csv'):
    with Path(path).open(newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=['label','numerator','denominator']:raise ValueError('Bad rate header')
        rows=list(reader)
    if not 1<=len(rows)<=30:raise ValueError('Need 1..30 rates')
    rates=[];seen=set()
    for r in rows:
        if set(r)!= {'label','numerator','denominator'} or not re.fullmatch('[a-z][a-z0-9_]{0,39}',r['label'] or '') or r['label'] in seen:raise ValueError('Bad rate label')
        if any(not re.fullmatch(r'[0-9]{1,9}',r[k] or '') for k in ['numerator','denominator']):raise ValueError('Rate must use bounded integer tokens')
        n,d=int(r['numerator']),int(r['denominator'])
        if d==0:raise ValueError('Zero denominator')
        ratio=F(n,d);eta=real(float(ratio),'rate',nonnegative=True)
        rates.append({'label':r['label'],'exact':ratio,'eta':eta});seen.add(r['label'])
    return rates

def profile(xs,ys):
    if type(xs) not in (list,tuple) or type(ys) not in (list,tuple) or len(xs)!=len(ys) or not 2<=len(xs)<=1000:raise ValueError('Expected equal nonempty 1D lists')
    x=[F.from_float(real(v,'x')) for v in xs];y=[F.from_float(real(v,'y')) for v in ys];n=len(x)
    sx=sum(x);sy=sum(y);sxx=sum(v*v for v in x)-sx*sx/n;sxy=sum(a*b for a,b in zip(x,y))-sx*sy/n
    if sxx<=0:raise ValueError('Slope is not identifiable')
    w=sxy/sxx;a=sy/n-w*sx/n;c=sum((a+w*u-v)**2 for u,v in zip(x,y))/(2*n);q=sxx/n
    model=validate_model({'q':float(q),'center':float(w),'constant':float(c)})
    if c and not model['constant']:raise ValueError('Profile minimum underflows')
    return {'exact':{'q':str(q),'center':str(w),'constant':str(c),'intercept':str(a),'x_mean':str(sx/n),'y_mean':str(sy/n)},'model':model,'n':n}

def validate_model(model):
    if type(model) is not dict or set(model)!= {'q','center','constant'}:raise ValueError('Bad quadratic model schema')
    return {'q':real(model['q'],'q',positive=True),'center':real(model['center'],'center'),
            'constant':real(model['constant'],'constant',nonnegative=True)}

def state(model,w,t=0,step=0.):
    # Called after validation by trace; usable on its own with validation.
    m=validate_model(model);w=real(w,'w',limit=1e8);integer(t,'t');step=real(step,'step',limit=1e9)
    error=computed(w-m['center'],'error');g=product(m['q'],error,'gradient')
    square=product(error,error,'squared error');excess=product(m['q']/2,square,'excess loss')
    value=computed(m['constant']+excess,'objective')
    return {'t':t,'w':w,'error':error,'gradient':g,'objective':value,'excess_direct':excess,
            'excess_subtracted':computed(value-m['constant'],'subtracted excess'),'step_from_previous':step}

def trace(model,initial,eta,max_steps,gradient_tolerance=1e-10,mode='stopping',growth_limit=1e7):
    m=validate_model(model);w=real(initial,'initial');rate=real(eta,'eta',nonnegative=True)
    budget=integer(max_steps,'max_steps');tol=real(gradient_tolerance,'gradient_tolerance',positive=True)
    growth=real(growth_limit,'growth_limit',positive=True)
    if mode not in ('stopping','fixed'):raise ValueError('mode must be stopping or fixed')
    if abs(w)>growth:raise ValueError('Initial state exceeds growth bound')
    rows=[];step=0.;status=None
    for t in range(budget+1):
        row=state(m,w,t,step);rows.append(row)
        if mode=='stopping':
            if row['error']==0:status='exact_stored_optimum';break
            if abs(row['gradient'])<=tol:status='gradient_tolerance';break
        if t==budget:status='fixed_budget_completed' if mode=='fixed' else 'max_steps';break
        delta=product(rate,row['gradient'],'proposed step');nxt=computed(w-delta,'candidate parameter')
        if abs(nxt)>growth:status='growth_guard';break
        if mode=='stopping' and nxt==w:status='stagnation';break
        # Record both endpoints before reporting a two-cycle.
        if mode=='stopping' and len(rows)>=2 and nxt==rows[-2]['w']:
            rows.append(state(m,nxt,t+1,computed(nxt-w,'actual step')));status='two_cycle';break
        step=computed(nxt-w,'actual step');w=nxt
    return {'eta':rate,'mode':mode,'requested_updates':budget,'updates_completed':len(rows)-1,
            'status':status,'history':rows,'final':rows[-1]}

def exact_trace(q,center,constant,initial,eta,steps):
    if any(type(v) is not F for v in [q,center,constant,initial,eta]):raise ValueError('Exact interface requires Fraction arguments')
    integer(steps,'steps',1000)
    if q<=0 or constant<0 or eta<0:raise ValueError('Invalid exact quadratic')
    rho=1-eta*q
    return [{'t':t,'w':str(center+rho**t*(initial-center)),
             'error':str(rho**t*(initial-center)),
             'excess':str(q/2*rho**(2*t)*(initial-center)**2)} for t in range(steps+1)]

def first_threshold(q,error,eta,tolerance,kind='parameter',budget=10000):
    if any(type(v) is not F for v in [q,error,eta,tolerance]) or q<=0 or eta<0 or tolerance<=0:raise ValueError('Invalid exact threshold request')
    integer(budget,'budget');rho=1-eta*q
    if kind not in ('parameter','gradient'):raise ValueError('Unknown threshold kind')
    for t in range(budget+1):
        amount=abs(error)*(q if kind=='gradient' else 1)
        if amount<=tolerance:return t
        error*=rho
    return None

def compile_report(xs,ys,rates,spec):
    # Validate every source before the first trajectory, including unused rates.
    p=profile(xs,ys);m=p['model']
    if type(spec) is not dict:raise ValueError('Bad spec')
    # Round-trip through schema parser also catches manually constructed dictionaries.
    expected={'kind','initial_slope','trajectory_steps','max_steps','gradient_tolerance','parameter_error_tolerance','objective_multiplier','parameter_scale','counterexample_initial','counterexample_rate'}
    if set(spec)!=expected or spec['kind']!='synthetic_scalar_gradient_descent_demo':raise ValueError('Bad spec schema')
    s=dict(spec)
    for k in ['trajectory_steps','max_steps']:integer(s[k],k,1000 if k=='trajectory_steps' else 10000)
    for k in expected-{'kind','trajectory_steps','max_steps'}:s[k]=real(s[k],k,positive=k not in {'initial_slope','counterexample_initial'})
    if type(rates) is not list or not 1<=len(rates)<=30:raise ValueError('Bad rates')
    seen=set()
    for r in rates:
        if type(r) is not dict or set(r)!= {'label','exact','eta'} or not isinstance(r['label'],str) or not re.fullmatch('[a-z][a-z0-9_]{0,39}',r['label']) or r['label'] in seen:raise ValueError('Bad rate record')
        if type(r['exact']) is not F or r['exact']<0:raise ValueError('Bad rational rate')
        if real(r['eta'],'eta',nonnegative=True)!=float(r['exact']):raise ValueError('Rate mismatch')
        seen.add(r['label'])
    q,center,c=(F(p['exact'][k]) for k in ['q','center','constant']);initial=F.from_float(s['initial_slope'])
    qm,cm,km=(F.from_float(m[k]) for k in ['q','center','constant'])
    eta=.25;k=s['objective_multiplier'];scale=s['parameter_scale']
    objective=validate_model({'q':product(k,m['q'],'scaled q'),'center':m['center'],'constant':product(k,m['constant'],'scaled constant')})
    coordinate=validate_model({'q':m['q']/product(scale,scale,'scale square'),'center':product(scale,m['center'],'scaled center'),'constant':m['constant']})
    matched_rate=product(eta,product(scale,scale,'scale square'),'matched rate')
    real(matched_rate,'matched coordinate rate',nonnegative=True);real(eta/k,'matched objective rate',nonnegative=True)
    real(scale*s['initial_slope'],'scaled initial')
    z=s['counterexample_initial'];rate=s['counterexample_rate'];z2=product(z,z,'quartic square');g=product(4,product(z2,z,'quartic cube'),'quartic gradient');nxt=computed(z-product(rate,g,'quartic step'),'quartic next');n2=product(nxt,nxt,'quartic new square')
    counter={'initial':z,'gradient':g,'next':nxt,'loss_before':product(z2,z2,'quartic value'),'loss_after':product(n2,n2,'quartic new value')}
    results=[]
    for r in rates:
        fixed=trace(m,s['initial_slope'],r['eta'],s['trajectory_steps'],s['gradient_tolerance'],mode='fixed')
        stop=trace(m,s['initial_slope'],r['eta'],s['max_steps'],s['gradient_tolerance'])
        steps=fixed['updates_completed']
        theory=exact_trace(q,center,c,initial,r['exact'],steps)
        stored=exact_trace(qm,cm,km,initial,F.from_float(r['eta']),steps)
        results.append({'label':r['label'],'exact_eta':str(r['exact']),'rho_exact':str(1-r['exact']*q),
                        'fixed':fixed,'stopping':stop,'rational_theory':theory,'stored_model_exact_arithmetic':stored})
    scaling={'base':trace(m,s['initial_slope'],eta,12,mode='fixed'),
             'objective_unmatched':trace(objective,s['initial_slope'],eta,12,mode='fixed'),
             'objective_matched':trace(objective,s['initial_slope'],eta/k,12,mode='fixed'),
             'coordinate_unmatched':trace(coordinate,scale*s['initial_slope'],eta,12,mode='fixed'),
             'coordinate_matched':trace(coordinate,scale*s['initial_slope'],matched_rate,12,mode='fixed'),
             'objective_multiplier':k,'coordinate_scale':scale,'matched_coordinate_rate':matched_rate}
    near=math.nextafter(m['center'],math.inf);cancellation=state(m,near)
    stagnant=trace(m,near,1e-10,10,1e-30)
    return {'unit':'031','profile':p,'initial_slope':s['initial_slope'],'rates':results,'scaling':scaling,
            'exact_thresholds':{'parameter':first_threshold(q,initial-center,F(1,4),F.from_float(s['parameter_error_tolerance'])),
                                'gradient':first_threshold(q,initial-center,F(1,4),F.from_float(s['gradient_tolerance']),'gradient')},
            'cancellation_example':cancellation,'stagnation_example':stagnant,'quartic_counterexample':counter,
            'interpretation':'Binary64 iterations, exact rational theory and exact arithmetic on stored binary64 coefficients are distinct references. Fixed traces may end early only on an explicit growth guard; every final status and completed count is retained.'}

def run(data=ROOT/'data/regression.csv',rates=ROOT/'data/learning_rates.csv',spec=ROOT/'data/model_spec.json',output=ROOT/'outputs/result.json'):
    xs,ys=read_data(data);rs=read_rates(rates);cfg=read_spec(spec);report=compile_report(xs,ys,rs,cfg)
    text=json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    dest=Path(output);inputs=[Path(p).resolve() for p in [data,rates,spec]]
    if dest.resolve() in inputs:raise ValueError('Output cannot overwrite an input')
    dest.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile('w',encoding='utf-8',dir=dest.parent,delete=False) as f:
        f.write(text);tmp=Path(f.name)
    os.replace(tmp,dest)
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'data/regression.csv');parser.add_argument('--rates',type=Path,default=ROOT/'data/learning_rates.csv');parser.add_argument('--spec',type=Path,default=ROOT/'data/model_spec.json');parser.add_argument('--output',type=Path,default=ROOT/'outputs/result.json')
    args=parser.parse_args()
    try:result=run(args.data,args.rates,args.spec,args.output)
    except (ValueError,OverflowError,OSError) as exc:parser.exit(2,'Input or numerical range error: '+str(exc)+'\n')
    print(json.dumps({'unit':'031','status':'executed','rates':len(result['rates']),'threshold_updates':result['exact_thresholds']},ensure_ascii=False))
