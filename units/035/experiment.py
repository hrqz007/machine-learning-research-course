"""035: validated full-batch heavy-ball and lookahead Nesterov, from scratch.

Core uses only Python's standard library. No assertion is a validation gate.
The documented CSV family is four corners (±1, ±s), y=x1, 1<=s<=10.
"""
from pathlib import Path
from fractions import Fraction
from decimal import Decimal, InvalidOperation
import argparse, cmath, csv, hashlib, json, math, os, tempfile
ROOT = Path(__file__).resolve().parent
KEYS = {'kind','initial','initial_velocity','steps','target_gap','hand_rate',
        'hand_momentum','stress_rate','stress_momentum','library_steps',
        'schedule_rates','dampening'}

def real(v, name, low=None, high=None):
    if type(v) not in (int, float): raise ValueError(name+': built-in int/float required, not bool')
    if abs(v)>1e100: raise ValueError(name+': input too large')
    if not math.isfinite(v): raise ValueError(name+': finite number required')
    if v != 0 and abs(v) < 1e-100: raise ValueError(name+': nonzero input too small')
    if abs(v)>1e100: raise ValueError(name+': input too large')
    if low is not None and v < low or high is not None and v > high: raise ValueError(name+': outside range')
    return float(v)

def integer(v, name, low, high):
    if type(v) is not int or not low <= v <= high: raise ValueError(name+': bounded built-in int required')
    return v

def vector(v, name, bound=100):
    if type(v) not in (list,tuple) or len(v)!=2: raise ValueError(name+': length-two list/tuple required')
    return [real(x,name+str(i),-bound,bound) for i,x in enumerate(v)]

def validate_data(data):
    if type(data) not in (list,tuple) or len(data)!=4: raise ValueError('data: exactly four rows required')
    parsed=[]
    # Validate ALL original scalar types first; never astype/float before checking.
    for i,row in enumerate(data):
        if type(row) not in (list,tuple) or len(row)!=3: raise ValueError('data row: x1,x2,y required')
        parsed.append([real(v,f'data[{i},{j}]',-10,10) for j,v in enumerate(row)])
    scale=parsed[0][1]
    if not 1 <= scale <= 10: raise ValueError('positive scale in [1,10] required')
    expected=[[1,scale,1],[1,-scale,1],[-1,scale,-1],[-1,-scale,-1]]
    if parsed != expected: raise ValueError('only ordered four-corner family with y=x1 is supported')
    return parsed

def validate_spec(spec):
    if type(spec) is not dict or set(spec)!=KEYS: raise ValueError('config: exact fields required')
    if spec['kind']!='four_corner_momentum_v1': raise ValueError('config kind mismatch')
    s={'kind':spec['kind']}
    for name in ['initial','initial_velocity']: s[name]=vector(spec[name],name)
    for name,lo,hi in [('steps',0,400),('library_steps',1,40)]:s[name]=integer(spec[name],name,lo,hi)
    s['target_gap']=real(spec['target_gap'],'target_gap',1e-30,100)
    for name in ['hand_rate','stress_rate']:s[name]=real(spec[name],name,1e-8,.3)
    for name in ['hand_momentum','stress_momentum']:s[name]=real(spec[name],name,0,.99)
    rates=spec['schedule_rates']
    if type(rates) not in (list,tuple) or not 2<=len(rates)<=20:raise ValueError('schedule_rates: 2..20 items')
    s['schedule_rates']=[real(v,'schedule_rates item',1e-8,.3) for v in rates]
    s['dampening']=real(spec['dampening'],'dampening',0,.99)
    return s

def decimal_token(token):
    try:d=Decimal(token)
    except InvalidOperation as ex:raise ValueError('invalid decimal input') from ex
    if not d.is_finite() or d!=0 and not Decimal('1e-100')<=abs(d)<=Decimal('1e100'):
        raise ValueError('decimal token out of input range before float conversion')
    return float(d)

def unique_pairs(pairs):
    d={}
    for k,v in pairs:
        if k in d:raise ValueError('duplicate JSON key: '+k)
        d[k]=v
    return d

def load_inputs(data_path=None, config_path=None):
    dp=Path(data_path) if data_path is not None else ROOT/'data/regression.csv'
    cp=Path(config_path) if config_path is not None else ROOT/'data/model_spec.json'
    with dp.open(newline='',encoding='utf-8') as f:rows=list(csv.reader(f))
    if len(rows)!=5 or rows[0]!=['id','x1','x2','y']:raise ValueError('CSV header or row count wrong')
    data=[]
    for i,row in enumerate(rows[1:]):
        if len(row)!=4 or row[0]!=f'S{i+1}':raise ValueError('CSV row shape/order wrong')
        data.append([decimal_token(v) for v in row[1:]])
    s=json.loads(cp.read_text(encoding='utf-8'),parse_float=decimal_token,
                 parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonstandard JSON number')),
                 object_pairs_hook=unique_pairs)
    return validate_data(data),validate_spec(s)

def _forward(data, theta):
    rows=[];gradient=[0.0,0.0];loss=0.0
    for i,(x1,x2,y) in enumerate(data):
        pred=x1*theta[0]+x2*theta[1];r=pred-y;part=r*r/8
        c=[r*x1/4,r*x2/4];loss+=part
        gradient=[gradient[j]+c[j] for j in range(2)]
        rows.append({'id':f'S{i+1}','prediction':pred,'residual':r,'loss_contribution':part,
                     'local_dloss_dresidual':r/4,'dresidual_dprediction':1,
                     'dprediction_dtheta':[x1,x2],'gradient_contribution':c})
    if not all(math.isfinite(v) for v in [loss,*gradient]):raise ArithmeticError('nonfinite forward')
    gap=((theta[0]-1)**2+(data[0][1]*theta[1])**2)/2
    return {'theta':list(theta),'rows':rows,'loss':loss,'gradient':gradient,
            'geometric_gap':gap,'geometric_gradient':[theta[0]-1,data[0][1]**2*theta[1]]}

def forward(data,theta):
    dd=validate_data(data);tt=vector(theta,'theta')
    return _forward(dd,tt)

def _spectral(method,eta,beta,lam):
    a=eta*lam
    if method=='gd':A=1-a;D=0.0
    elif method=='hb':A=1+beta-a;D=beta
    else:A=(1+beta)*(1-a);D=beta*(1-a)
    delta=A*A-4*D;root=cmath.sqrt(delta);roots=[(A+root)/2,(A-root)/2]
    rho=max(abs(z) for z in roots)
    return {'lambda':lam,'eta_lambda':a,'A':A,'D':D,'discriminant':delta,
            'matrix':[[A,-D],[1,0]],'roots':[[z.real,z.imag] for z in roots],
            'spectral_radius':rho,'strict_jury':abs(D)<1 and 1-A+D>0 and 1+A+D>0}

def spectral(method,eta,beta,lam):
    if method not in ('gd','hb','nag'):raise ValueError('unknown method')
    e=real(eta,'eta',0,1);b=real(beta,'beta',0,.9999);l=real(lam,'lambda',.001,100)
    if method=='gd' and b!=0:raise ValueError('GD beta must be zero')
    return _spectral(method,e,b,l)

def _run(data,initial,velocity,method,eta,beta,steps,target):
    theta=list(initial);d=list(velocity);old=_forward(data,theta)
    states=[{'t':0,**old,'velocity':list(d)}];trace=[];status='budget_exhausted'
    training_forwards=0;diagnostic_forwards=1
    for t in range(steps):
        if theta==[1.0,0.0] and d==[0.0,0.0]:status='represented_optimum_zero_velocity';break
        point=[theta[j]+beta*d[j] for j in range(2)] if method=='nag' else list(theta)
        if max(abs(v) for v in point)>1e100:status='numeric_range_limit';break
        training_forwards+=1
        evaluation=_forward(data,point)
        if evaluation['gradient']==[0.0,0.0] and point!=[1.0,0.0]:
            status='forward_cancellation_limit';break
        nd=[beta*d[j]-eta*evaluation['gradient'][j] for j in range(2)]
        nt=[theta[j]+nd[j] for j in range(2)]
        if max(abs(v) for v in nt+nd)>1e100 or not all(math.isfinite(v) for v in nt+nd):
            status='numeric_range_limit';break
        diagnostic_forwards+=1
        new=_forward(data,nt)
        item={'t':t,'old':old,'old_velocity':list(d),'evaluation':evaluation,
              'gradient_point':'lookahead' if method=='nag' else 'old',
              'new_velocity':nd,'new':new,'parameter_unchanged':nt==theta,
              'whole_state_unchanged':nt==theta and nd==d}
        trace.append(item);states.append({'t':t+1,**new,'velocity':list(nd)})
        stopped=item['whole_state_unchanged'];theta,d,old=nt,nd,new
        if stopped:status='floating_state_stagnation';break
    if states[-1]['theta']==[1.0,0.0] and states[-1]['velocity']==[0.0,0.0]:status='represented_optimum_zero_velocity'
    return {'method':method,'rate':eta,'momentum':beta,'requested_updates':steps,
            'updates':len(trace),'status':status,'states':states,'trace':trace,
            'first_target_t':next((r['t'] for r in states if r['geometric_gap']<=target),None),
            'target_gap':target,'loss_increase_steps':[i for i in range(1,len(states)) if states[i]['loss']>states[i-1]['loss']],
            'training_sample_gradients':4*training_forwards,'diagnostic_full_forwards':diagnostic_forwards,
            'training_full_forwards':training_forwards,'all_forward_calls':training_forwards+diagnostic_forwards,
            'spectrum':[_spectral(method,eta,beta,l) for l in [1,data[0][1]**2]]}

def run_method(data,initial,velocity,method,eta,beta,steps,target=1e-6):
    dd=validate_data(data);ii=vector(initial,'initial');vv=vector(velocity,'velocity')
    if method not in ('gd','hb','nag'):raise ValueError('unknown method')
    ee=real(eta,'rate',1e-8,.3);bb=real(beta,'momentum',0,.99)
    tt=integer(steps,'steps',0,400);target=real(target,'target',1e-30,100)
    if method=='gd' and (bb!=0 or vv!=[0.0,0.0]):raise ValueError('GD requires zero momentum and velocity')
    return _run(dd,ii,vv,method,ee,bb,tt,target)

def hand_chain(data,spec,method):
    dd=validate_data(data);s=validate_spec(spec)
    if method not in ('hb','nag'):raise ValueError('hand method must be hb/nag')
    f=lambda x:Fraction(str(x));dataq=[[f(x) for x in row] for row in dd]
    theta=list(map(f,s['initial']));d=list(map(f,s['initial_velocity']));eta=f(s['hand_rate']);beta=f(s['hand_momentum'])
    def fw(p):
        rows=[];g=[Fraction(0),Fraction(0)];loss=Fraction(0)
        for x1,x2,y in dataq:
            pred=x1*p[0]+x2*p[1];r=pred-y;c=[r*x1/4,r*x2/4];part=r*r/8
            rows.append({'prediction':str(pred),'residual':str(r),'loss_contribution':str(part),
                         'local_dloss_dresidual':str(r/4),'gradient_contribution':list(map(str,c))})
            loss+=part;g=[g[j]+c[j] for j in range(2)]
        return {'theta':list(map(str,p)),'rows':rows,'gradient':list(map(str,g)),'loss':str(loss)},g
    out=[]
    for t in range(2):
        old,_=fw(theta);p=[theta[j]+beta*d[j] for j in range(2)] if method=='nag' else list(theta)
        ev,g=fw(p);nd=[beta*d[j]-eta*g[j] for j in range(2)];nt=[theta[j]+nd[j] for j in range(2)];new,_=fw(nt)
        out.append({'t':t,'old':old,'old_velocity':list(map(str,d)),'evaluation':ev,
                    'new_velocity':list(map(str,nd)),'new':new});theta,d=nt,nd
    return out

def run_experiment(data,spec):
    dd=validate_data(data);s=validate_spec(spec) # complete last-field validation before _run
    L=dd[0][1]**2;root=math.sqrt(L);b=((root-1)/(root+1))**2
    settings={'gd_safe':('gd',1/L,0),'gd_balanced':('gd',2/(1+L),0),
              'hb_hand':('hb',s['hand_rate'],s['hand_momentum']),
              'nag_hand':('nag',s['hand_rate'],s['hand_momentum']),
              'hb_quadratic_tuned':('hb',4/(root+1)**2,b),
              'nag_strongconvex':('nag',1/L,(root-1)/(root+1)),
              'hb_stress':('hb',s['stress_rate'],s['stress_momentum']),
              'nag_stress':('nag',s['stress_rate'],s['stress_momentum'])}
    methods={}
    for name,(m,eta,beta) in settings.items():
        velocity=[0.0,0.0] if m=='gd' else s['initial_velocity']
        methods[name]=_run(dd,s['initial'],velocity,m,eta,beta,s['steps'],s['target_gap'])
    probes={
      'same_point_with_velocity':_run(dd,[1.0,0.0],[.25,0.0],'hb',1/L,.5,8,s['target_gap']),
      'nearby_floating_stagnation':_run(dd,[math.nextafter(1.0,2.0),0.0],[0.0,0.0],'hb',1e-8,0.0,8,s['target_gap']),
      'parameter_pause_not_state_stop':_run(dd,[math.nextafter(1.0,2.0),0.0],[1e-30,0.0],'hb',1e-8,.5,4,s['target_gap']),
      'hb_boundary':_run(dd,[1.0,1.0],[0.0,0.0],'hb',3/L,.5,min(20,s['steps']),s['target_gap']),
      'nag_boundary':_run(dd,[1.0,1.0],[0.0,0.0],'nag',1.5/L,.5,min(20,s['steps']),s['target_gap'])}
    return {'unit':'035','data':dd,'spec':s,'objective':'sum((x1*a+x2*b-y)^2)/(2*4)',
            'optimum':[1.0,0.0],'minimum_loss':0.0,'curvatures':[1.0,L],
            'methods':methods,'hand':{m:hand_chain(dd,s,m) for m in ['hb','nag']},'probes':probes,
            'limits':'finite synthetic full-gradient experiment; no stochastic or global nonquadratic speed claim'}

def report_bytes(report):return (json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()

def protect_output(output,inputs=()):
    p=Path(output).absolute();resolved=p.resolve();rr=ROOT.resolve()
    protected=[ROOT/'experiment.py',*(ROOT/'data').glob('*'),*map(Path,inputs)]
    if resolved==rr or rr in resolved.parents and not (rr/'results'==resolved.parent or rr/'results' in resolved.parents):
        raise ValueError('inside unit only results/ is writable by CLI')
    for q in protected:
        if q.exists() and (resolved==q.resolve() or p.exists() and os.path.samefile(p,q)):raise ValueError('output aliases protected input/source')
    if p.exists() and not p.is_file():raise ValueError('output must be a file')
    return p

def atomic_write(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.report-',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path);p.add_argument('--config',type=Path)
    p.add_argument('--output',type=Path,default=ROOT/'results/report.json');a=p.parse_args()
    try:
        data,s=load_inputs(a.data,a.config);out=protect_output(a.output,[v for v in [a.data,a.config] if v is not None])
        report=run_experiment(data,s);payload=report_bytes(report);atomic_write(out,payload)
    except (ValueError,ArithmeticError,OSError) as err:p.exit(2,'error: '+str(err)+'\n')
    print('report',out,'SHA256',hashlib.sha256(payload).hexdigest())
    for name,m in report['methods'].items():print(name,m['updates'],m['status'],'target',m['first_target_t'],'final loss',m['states'][-1]['loss'])
if __name__=='__main__':main()
