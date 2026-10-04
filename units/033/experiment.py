"""033: deterministic descent and convergence experiments, no optimizer library.

All public inputs are checked in their original types before any trajectory.
The JSON report separates mathematical hypotheses from finite numerical evidence.
Python -O is supported: validation never depends on assert statements.
"""
from pathlib import Path
from decimal import Decimal, localcontext, ROUND_CEILING
from fractions import Fraction
import argparse, csv, hashlib, io, json, math, os, re, sys, tempfile
ROOT = Path(__file__).resolve().parent
MIN_NORMAL = sys.float_info.min
TOKEN = re.compile(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z')
BASELINE = {'data/quadratic.csv': 'a4b039de0bb648a052077f88b8595d4fd132908598c21964c043e30bd2894fff', 'data/model_spec.json': '05345827fc68bd7b6621b502c49579fd3f02ed3b5f00571e724cd6e1f432ae9f'}
CONFIG_KEYS = {'kind','steps','claimed_L_values','claimed_mu_values','quadratic_rate',
               'convex_initial','convex_rate','nonconvex_initials','nonconvex_rate',
               'boundary_rates','target_gap'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def real(value, name='value', low=-100, high=100):
    # No cast may hide bool, string, complex, Decimal, object, or extended dtype.
    require(type(value) in (int, float), name+' requires a built-in int or float, not bool/complex/extended scalar')
    require(low <= value <= high, name+' outside supported finite range')
    result = float(value)
    require(math.isfinite(result), name+' must be finite')
    require(value == 0 or abs(result) >= 1e-150, name+' nonzero magnitude below 1e-150')
    if type(value) is int:
        require(int(result) == value, name+' integer conversion not exact')
    return result


def integer(value, name='steps', low=0, high=200):
    require(type(value) is int, name+' requires a built-in integer, not bool or float')
    require(low <= value <= high, name+' outside supported range')
    return value


def vector(value, name, length=2, low=-100, high=100):
    require(type(value) in (list, tuple), name+' requires a list or tuple')
    require(len(value) == length, name+' wrong length')
    return [real(x, name+'['+str(i)+']', low, high) for i,x in enumerate(value)]


def numeric_token(token):
    require(type(token) is str and len(token) <= 400 and TOKEN.fullmatch(token) is not None, 'invalid numeric token')
    d = Decimal(token)
    f = float(d)
    require(math.isfinite(f), 'numeric token overflows binary64')
    require(d == 0 or f != 0, 'nonzero numeric token underflows binary64')
    return f


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key: '+key)
        result[key] = value
    return result


def reject_constant(token):
    raise ValueError('nonstandard JSON constant: '+token)


def validate_inputs(data, config):
    require(type(data) is dict and set(data) == {'coordinates','curvature','initial'}, 'data fields must match schema')
    require(type(data['coordinates']) is list and data['coordinates'] == ['u','v'], 'coordinates must be ordered u,v')
    clean_data = {'coordinates':['u','v'], 'curvature':vector(data['curvature'],'curvature',low=0.01,high=100),
                  'initial':vector(data['initial'],'initial')}
    require(type(config) is dict and set(config) == CONFIG_KEYS, 'configuration fields must match schema exactly')
    require(type(config['kind']) is str and config['kind'] == 'synthetic_smoothness_certificate_demo', 'unsupported kind')
    clean = {'kind':config['kind'], 'steps':integer(config['steps'])}
    for name in ('claimed_L_values','claimed_mu_values'):
        clean[name] = vector(config[name], name, 3, 1e-8, 1000)
    for name in ('quadratic_rate','convex_rate','nonconvex_rate'):
        clean[name] = real(config[name],name,1e-8,2)
    clean['convex_initial'] = real(config['convex_initial'],'convex_initial')
    clean['nonconvex_initials'] = vector(config['nonconvex_initials'],'nonconvex_initials')
    clean['boundary_rates'] = vector(config['boundary_rates'],'boundary_rates',2,1e-8,2)
    clean['target_gap'] = real(config['target_gap'],'target_gap',1e-150,1000)
    # Deliberately false positive L/mu claims are legal experimental inputs.
    # They are diagnosed separately, never silently promoted to certificates.
    return clean_data, clean


def load_inputs(data_path=None, config_path=None):
    dp = Path(data_path) if data_path is not None else ROOT/'data/quadratic.csv'
    cp = Path(config_path) if config_path is not None else ROOT/'data/model_spec.json'
    raw_data = dp.read_text(encoding='utf-8')
    rows = list(csv.reader(io.StringIO(raw_data),strict=True))
    require(len(rows) == 3 and rows[0] == ['coordinate','curvature','initial'], 'CSV exact header and two rows required')
    require(all(len(row) == 3 for row in rows[1:]), 'CSV rows must each have three fields')
    data = {'coordinates':[row[0] for row in rows[1:]],
            'curvature':[numeric_token(row[1]) for row in rows[1:]],
            'initial':[numeric_token(row[2]) for row in rows[1:]]}
    config = json.loads(cp.read_text(encoding='utf-8'), parse_float=numeric_token,
                        parse_constant=reject_constant, object_pairs_hook=unique_object)
    return validate_inputs(data,config)


def checked(value, name):
    require(math.isfinite(value), name+' computed nonfinite value')
    require(value == 0 or abs(value) >= MIN_NORMAL, name+' computed subnormal value')
    return value


def multiply(a, b, name):
    value = a*b
    require(math.isfinite(value) and (a == 0 or b == 0 or abs(value) >= MIN_NORMAL), name+' overflow or nonzero underflow')
    return value


def square(a, name):
    return multiply(a,a,name)


def positive_fraction_float(value,name):
    result=float(value)
    require(math.isfinite(result) and (value==0 or abs(result)>=MIN_NORMAL),name+' overflow or nonzero underflow')
    return result


def _quadratic_state(q,w,t):
    squares = [square(z,'coordinate square') for z in w]
    contributions = [multiply(a,b,'loss contribution')/2 for a,b in zip(q,squares)]
    for z,c in zip(squares,contributions):
        require(z == 0 or c >= MIN_NORMAL, 'loss contribution division underflow')
    gradient = [multiply(a,b,'gradient component') for a,b in zip(q,w)]
    g2 = checked(math.fsum(square(z,'gradient square') for z in gradient),'gradient norm square')
    value = checked(math.fsum(contributions),'objective')
    d2 = checked(math.fsum(squares),'distance square')
    return {'iteration':t,'w':w[:],'squares':squares,'loss_contributions':contributions,
            'local_derivatives':{'square_wrt_coordinate':[2*z for z in w],
                                 'contribution_wrt_square':[a/2 for a in q],
                                 'sum_wrt_contribution':[1.0,1.0]},
            'value':value,'gradient':gradient,'gradient_squared':g2,'gradient_norm':math.sqrt(g2),
            'distance_squared':d2,'distance':math.sqrt(d2)}


def quadratic_state(q,w,t=0):
    q = vector(q,'curvature',low=.01,high=100)
    w = vector(w,'w')
    return _quadratic_state(q,w,integer(t))


def _quadratic_trace(q,initial,eta,steps):
    history = [_quadratic_state(q,initial,0)]
    L,mu = max(q),min(q)
    safe = Fraction.from_float(eta)*Fraction.from_float(L) <= 1
    for t in range(steps):
        old = history[-1]
        delta = [-multiply(eta,g,'update product') for g in old['gradient']]
        new_w = [checked(w+h,'updated coordinate') for w,h in zip(old['w'],delta)]
        new = _quadratic_state(q,new_w,t+1)
        rhs = checked(old['value']-eta*(1-L*eta/2)*old['gradient_squared'],'descent right side')
        old.update(update=delta,descent_upper=rhs,descent_slack=rhs-new['value'],
                   potential_upper=(old['distance_squared']-new['distance_squared'])/(2*eta) if safe else None)
        history.append(new)
    f0,d0=history[0]['value'],history[0]['distance_squared']
    for state in history:
        t=state['iteration']
        state['convex_bound'] = d0/(2*eta*t) if safe and t>=1 else None
        state['strong_bound'] = positive_fraction_float(Fraction.from_float(f0)*(1-Fraction.from_float(eta)*Fraction.from_float(mu))**t, 'strong bound') if safe else None
        state['minimum_gradient_squared_before_T'] = min(x['gradient_squared'] for x in history[:t]) if t>=1 else None
        state['nonconvex_gradient_bound'] = 2*f0/(eta*t) if safe and t>=1 else None
    return {'status':'budget_completed','requested_updates':steps,'updates_completed':steps,
            'L':L,'mu':mu,'eta':eta,'safe_rate_for_stated_bounds':safe,
            'descent_strict_when_gradient_nonzero':Fraction.from_float(eta)*Fraction.from_float(L)<2,
            'history':history}


def quadratic_trace(q,initial,eta,steps):
    q=vector(q,'curvature',low=.01,high=100);initial=vector(initial,'initial')
    eta=real(eta,'eta',1e-8,2);steps=integer(steps)
    return _quadratic_trace(q,initial,eta,steps)


def _convex_state(w,t):
    w2=square(w,'convex coordinate square')
    root=math.hypot(1,w)
    value=w2/(root+1)
    require(w == 0 or value>=MIN_NORMAL,'convex stable objective underflow')
    gradient=w/root
    g2=square(gradient,'convex gradient square')
    return {'iteration':t,'w':w,'square':w2,'root':root,'value':value,'gradient':gradient,
            'gradient_squared':g2,'gradient_norm':abs(gradient),'distance':abs(w),'distance_squared':w2}


def _nonconvex_state(w,t):
    half_cos=math.cos(w/2)
    # This identity avoids 1 + cos(w) cancellation near odd multiples of pi.
    value=2*square(half_cos,'nonconvex half cosine square')
    gradient=-math.sin(w)
    g2=square(gradient,'nonconvex gradient square')
    return {'iteration':t,'w':w,'half_cosine':half_cos,'value':value,'gradient':gradient,
            'gradient_squared':g2,'gradient_norm':abs(gradient),
            'distance_to_nearest_stored_odd_pi':abs(w-(2*round((w/math.pi-1)/2)+1)*math.pi),
            'distance_note':'distance to a binary64 approximation, not exact mathematical pi'}


def _auxiliary_trace(kind,initial,eta,steps):
    state_function=_convex_state if kind=='convex' else _nonconvex_state
    history=[state_function(initial,0)]
    status='budget_completed';detail='Requested finite budget completed; no convergence certificate inferred from the curve.'
    for t in range(steps):
        old=history[-1]
        if old['gradient']==0:
            status='exact_stationary_in_implemented_model'
            detail='Zero gradient; for cosine at zero this is the maximum, not an optimum.' if kind=='nonconvex' else 'Initial zero is the proven minimizer.'
            break
        try:
            if kind=='convex' and eta==1:
                # w - w/root = w * w^2 / [root*(root+1)]
                # Avoid losing a small nonzero iterate when root rounds to 1.
                factor=old['square']/(old['root']*(old['root']+1))
                require(factor>=MIN_NORMAL,'stable update factor underflow')
                new_w=multiply(old['w'],factor,'stable convex update')
            else:
                new_w=checked(old['w']-multiply(eta,old['gradient'],'auxiliary update'),'new auxiliary coordinate')
            if new_w==old['w']:
                status='floating_stagnation';detail='Rounded update equals old value although gradient is nonzero; not exact stationarity.'
                break
            new=state_function(new_w,t+1)
        except ValueError as error:
            status='numeric_range_limit';detail=str(error)+'; proposed state not appended and no zero value invented.'
            break
        naive_upper=old['value']-eta*(1-eta/2)*old['gradient_squared']
        if kind=='convex':
            # h - g^2/2 = w^4(2r+1)/[2r^2(r+1)^2], avoiding cancellation.
            r=old['root']
            w4=multiply(old['square'],old['square'],'stable descent upper fourth power')
            base=w4*((2*r+1)/(2*r*r*(r+1)**2))
            require(w4==0 or base>=MIN_NORMAL,'stable descent upper division underflow')
            stable_upper=base+0.5*(1-eta)**2*old['gradient_squared']
        else:
            stable_upper=naive_upper
        old.update(update=new_w-old['w'],descent_upper=stable_upper,descent_upper_naive_diagnostic=naive_upper)
        old['descent_slack']=old['descent_upper']-new['value']
        history.append(new)
    for s in history:
        t=s['iteration']
        s['convex_bound']=history[0].get('distance_squared',0)/(2*eta*t) if kind=='convex' and eta<=1 and t>=1 else None
        s['minimum_gradient_squared_before_T']=min(z['gradient_squared'] for z in history[:t]) if t>=1 else None
        s['nonconvex_gradient_bound']=2*history[0]['value']/(eta*t) if eta<=1 and t>=1 else None
    return {'kind':kind,'status':status,'status_detail':detail,'requested_updates':steps,
            'updates_completed':len(history)-1,'L':1,'mu':None,'eta':eta,
            'global_convex':kind=='convex','global_strong_convex':False,
            'analytic_stationary_extension': {'T':steps,'minimum_gradient_squared':0.0,'bound':2*history[0]['value']/(eta*steps),'note':'Analytic repeated zero-gradient path, not additional executed updates'} if kind=='nonconvex' and initial==0 and eta<=1 and steps>=1 else None,
            'history':history}


def auxiliary_trace(kind,initial,eta,steps):
    require(type(kind) is str and kind in ('convex','nonconvex'),'unknown auxiliary kind')
    initial=real(initial,'initial');eta=real(eta,'eta',1e-8,2);steps=integer(steps)
    return _auxiliary_trace(kind,initial,eta,steps)


def strong_budget(delta0,epsilon,eta,mu,L):
    delta0=real(delta0,'delta0',0,1e6);epsilon=real(epsilon,'epsilon',1e-150,1e6)
    eta=real(eta,'eta',1e-8,2);mu=real(mu,'mu',1e-8,1000);L=real(L,'L',1e-8,1000)
    require(mu<=L and Fraction.from_float(eta)*Fraction.from_float(L)<=1,'budget assumptions false')
    exact_product=Fraction.from_float(eta)*Fraction.from_float(mu)
    exact_factor=1-exact_product
    factor=float(exact_factor)
    if delta0<=epsilon:
        return {'geometric_sufficient':0,'exponential_sufficient':0,'factor':factor}
    # Decimal logs avoid turning a nonzero exact stored-input factor into 0
    # when binary64 eta*mu rounds to 1 (for example eta=1/3, mu=3).
    with localcontext() as context:
        context.prec=100
        product=Decimal(exact_product.numerator)/Decimal(exact_product.denominator)
        ratio=Decimal.from_float(delta0)/Decimal.from_float(epsilon)
        log_ratio=ratio.ln()
        exponential=int((log_ratio/product).to_integral_value(rounding=ROUND_CEILING))
        if exact_factor==0:
            count=1
        else:
            rho=Decimal(exact_factor.numerator)/Decimal(exact_factor.denominator)
            denominator=-rho.ln()
            count=max(1,int((log_ratio/denominator).to_integral_value(rounding=ROUND_CEILING)))
            while count*denominator<log_ratio:
                count+=1
    return {'geometric_sufficient':count,'exponential_sufficient':exponential,'factor':factor}



def certificate_diagnostics(q,claims_L,claims_mu):
    q=vector(q,'curvature',low=.01,high=100)
    claims_L=vector(claims_L,'claimed L',3,1e-8,1000)
    claims_mu=vector(claims_mu,'claimed mu',3,1e-8,1000)
    imax=q.index(max(q));imin=q.index(min(q));origin=[0.0,0.0]
    result={'L':[],'mu':[]}
    for claim in claims_L:
        valid=claim>=max(q);y=origin[:];y[imax]=1.0
        result['L'].append({'claim':claim,'assumptions_satisfied':valid,
            'label':'valid_global_hessian_certificate' if valid else 'assumptions_false_counterexample',
            'x':origin,'y':y,'actual_value':max(q)/2,'proposed_upper':claim/2})
    for claim in claims_mu:
        valid=claim<=min(q);y=origin[:];y[imin]=1.0
        result['mu'].append({'claim':claim,'assumptions_satisfied':valid,
            'label':'valid_global_hessian_certificate' if valid else 'assumptions_false_counterexample',
            'x':origin,'y':y,'actual_value':min(q)/2,'proposed_lower':claim/2})
    return result


def compile_report(data,config):
    data,config=validate_inputs(data,config)  # No trace until ALL fields validate.
    q,initial=data['curvature'],data['initial'];n=config['steps'];eta=config['quadratic_rate']
    main=_quadratic_trace(q,initial,eta,n)
    high=[0.0,0.0];high[q.index(max(q))]=1.0
    boundary=[_quadratic_trace(q,high,rate,n) for rate in config['boundary_rates']]
    convex=_auxiliary_trace('convex',config['convex_initial'],config['convex_rate'],n)
    nonconvex=[_auxiliary_trace('nonconvex',w,config['nonconvex_rate'],n) for w in config['nonconvex_initials']]
    budget=strong_budget(main['history'][0]['value'],config['target_gap'],eta,min(q),max(q)) if main['safe_rate_for_stated_bounds'] else None
    first=next((s['iteration'] for s in main['history'] if s['value']<=config['target_gap']),None)
    return {'unit':'033','scope':'deterministic synthetic finite mathematical experiments',
            'configuration':config,'data':data,'quadratic':main,'convex':convex,'nonconvex':nonconvex,
            'boundary':boundary,'certificates':certificate_diagnostics(q,config['claimed_L_values'],config['claimed_mu_values']),
            'budget':budget,'first_observed_target_update':first,
            'quartic_counterexample':{'initial':1,'gradient':4,'eta':1,'new_w':-3,'old_value':1,'new_value':81,
                'label':'assumptions_false_no_global_L'},
            'nondifferentiable_counterexample':{'function':'abs(w)','point':0,'label':'assumptions_false_not_differentiable'}}


def check_notebook_baseline():
    require(set(BASELINE)=={'data/quadratic.csv','data/model_spec.json'},'baseline hash set incomplete')
    for relative,digest in BASELINE.items():
        require(hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==digest,'Notebook fixed teaching baseline changed: '+relative)
    return load_inputs()


def report_bytes(report):
    return (json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8')


def run(data_path=None,config_path=None,output=None):
    path=Path(output) if output is not None else ROOT/'outputs/report.json'
    sources=[Path(data_path or ROOT/'data/quadratic.csv'),Path(config_path or ROOT/'data/model_spec.json'),Path(__file__)]
    resolved=path.resolve()
    require(resolved not in [p.resolve() for p in sources],'output may not overwrite any input/source')
    require(not resolved.is_relative_to(ROOT) or resolved.is_relative_to((ROOT/'outputs').resolve()),'package files protected: use outputs/ or an external directory')
    if path.exists():
        require(path.is_file(),'output must be a regular file path')
        for source in sources:
            require(not os.path.samefile(path,source),'output is a hard-link alias of protected input/source')
    report=compile_report(*load_inputs(data_path,config_path))
    payload=report_bytes(report)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile('wb',dir=path.parent,prefix='.033-report-',delete=False) as stream:
            temporary=Path(stream.name);stream.write(payload);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        if temporary is not None and temporary.exists():temporary.unlink()
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data');parser.add_argument('--config');parser.add_argument('--output')
    args=parser.parse_args()
    report=run(args.data,args.config,args.output)
    print(json.dumps({'quadratic':report['quadratic']['status'],'convex':report['convex']['status'],
                      'nonconvex':[p['status'] for p in report['nonconvex']],
                      'first_observed_target_update':report['first_observed_target_update']},ensure_ascii=False))


if __name__=='__main__':
    main()
