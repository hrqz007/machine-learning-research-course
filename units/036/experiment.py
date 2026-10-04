"""Full-batch adaptive linear regression, explicit optimizer state and input guards.

Core dependency: NumPy. The optional library_check.py adds pinned PyTorch checks.
All updates use row-derived gradients. No theoretical solution is used to update.
"""
from pathlib import Path
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import argparse, copy, csv, hashlib, json, math, os, tempfile
import numpy as np

ROOT=Path(__file__).resolve().parent
FIELDS={'kind','initial_parameters','steps','learning_rate','beta1','beta2','rmsprop_alpha','epsilon','weight_decay','decay_mask','decay_initial_parameters','rate_grid','epsilon_grid','objective_scale','resume_split'}
METHODS=('GD','AdaGrad','RMSProp','Adam','AdamL2','AdamW')

def real(x,name='number',limit=1e100,floor=1e-100):
    t=type(x)
    if t not in (int,float) and not (t.__module__=='numpy' and t.__name__ in ('int8','int16','int32','int64','uint8','uint16','uint32','uint64','float16','float32','float64')):raise ValueError(name+' must be a real scalar, not bool/string/extended precision')
    try:v=float(x)
    except (ValueError,OverflowError) as ex:raise ValueError(name+' conversion failed') from ex
    if not math.isfinite(v) or abs(v)>limit or (v!=0 and abs(v)<floor):raise ValueError(name+' outside supported finite range')
    if isinstance(x,(int,np.integer)) and int(v)!=int(x):raise ValueError(name+' integer precision loss')
    return v

def integer(x,name,lo,hi):
    if type(x) is not int or not lo<=x<=hi:raise ValueError(name+' must be a bounded Python integer')
    return x

def vector(x,name,length=2,limit=1e100,nonnegative=False,floor=1e-100):
    if not isinstance(x,(list,tuple,np.ndarray)) or isinstance(x,np.ndarray) and x.ndim!=1 or len(x)!=length:raise ValueError(name+' shape')
    out=[real(v,name,limit,floor) for v in x]
    if nonnegative and any(v<0 for v in out):raise ValueError(name+' must be nonnegative')
    return np.array(out,dtype=float)

def data_checked(data):
    if not isinstance(data,(list,tuple,np.ndarray)) or not 2<=len(data)<=64:raise ValueError('need 2..64 rows')
    a=np.array([vector(row,'data row',2,100) for row in data])
    return a

def validate(data,spec):
    data=data_checked(data)
    if type(spec) is not dict or set(spec)!=FIELDS:raise ValueError('incorrect config fields')
    s=copy.deepcopy(spec)
    if type(s['kind']) is not str or s['kind']!='synthetic_adaptive_linear_regression':raise ValueError('incorrect problem kind')
    for k in ('initial_parameters','decay_initial_parameters'):s[k]=vector(s[k],k,2,100).tolist()
    s['steps']=integer(s['steps'],'steps',8,300)
    for k in ('learning_rate','beta1','beta2','rmsprop_alpha','epsilon','weight_decay','objective_scale'):s[k]=real(s[k],k,10000)
    if not 1e-12<=s['learning_rate']<=1 or not 1e-12<=s['epsilon']<=100 or not 0<=s['weight_decay']<=1 or not 1e-4<=s['objective_scale']<=10000:raise ValueError('rate/epsilon/decay/scale outside teaching range')
    if any(not 0<=s[k]<=.999 for k in ('beta1','beta2','rmsprop_alpha')):raise ValueError('EMA coefficients must be in [0,.999]')
    mask=s['decay_mask']
    if type(mask) is not list or len(mask)!=2 or any(type(v) is not int or v not in (0,1) for v in mask):raise ValueError('mask must be two integers in {0,1}')
    for k,lo,hi in [('rate_grid',1e-12,1),('epsilon_grid',1e-12,100)]:
        seq=s[k]
        if type(seq) is not list or not 1<=len(seq)<=8:raise ValueError(k+' must be a bounded nonempty list')
        seq=[real(v,k,hi) for v in seq]
        if any(not lo<=v<=hi for v in seq) or len(set(seq))!=len(seq):raise ValueError(k+' invalid or duplicated entry')
        s[k]=seq
    s['resume_split']=integer(s['resume_split'],'resume split',1,s['steps']-1)
    if len(s['rate_grid'])*len(s['epsilon_grid'])*s['steps']*len(data)>150000:raise ValueError('excess teaching work budget')
    return data,s

def token(x):
    try:d=Decimal(x)
    except (InvalidOperation,TypeError) as ex:raise ValueError('invalid numeric token') from ex
    if not d.is_finite():raise ValueError('nonfinite token')
    v=float(d)
    if d!=0 and v==0:raise ValueError('numeric token underflow')
    return real(v)

def pairs(items):
    out={}
    for k,v in items:
        if k in out:raise ValueError('duplicate JSON key')
        out[k]=v
    return out

def load_inputs(data=ROOT/'data/regression.csv',config=ROOT/'data/model_spec.json'):
    with Path(data).open(newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=['id','x','y']:raise ValueError('CSV header must be id,x,y')
        rows=list(reader)
    if any(set(r)!= {'id','x','y'} or None in r.values() or not r['id'] or len(r['id'])>40 for r in rows):raise ValueError('invalid CSV row')
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('duplicate sample id')
    a=[[token(r['x']),token(r['y'])] for r in rows]
    s=json.loads(Path(config).read_text(),parse_float=token,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite JSON')),object_pairs_hook=pairs)
    return validate(a,s)

def _finite(a,name):
    a=np.asarray(a,dtype=float)
    if not np.all(np.isfinite(a)) or np.max(np.abs(a),initial=0)>1e140:raise FloatingPointError(name+' exceeds finite teaching arithmetic range')
    return a

def forward(data,theta,scale=1.):
    """All n rows; local half-square losses and contributions before their sum."""
    x,y=data[:,0],data[:,1];a,b=theta
    pred=_finite(a+b*x,'prediction');r=_finite(pred-y,'residual');sq=_finite(r*r,'squared residual')
    if np.any((r!=0)&(sq==0)):raise FloatingPointError('nonzero residual square underflow')
    losses=sq/2;contrib=np.column_stack((r,r*x))/len(data)*scale
    _finite(contrib,'gradient contributions');g=_finite(np.sum(contrib,axis=0),'full gradient')
    loss=float(np.sum(losses)/len(data));_finite([loss,loss*scale],'objective')
    rows=[{'row':i+1,'x':float(x[i]),'y':float(y[i]),'prediction':float(pred[i]),'residual':float(r[i]),'half_square':float(losses[i]),'dhalf_square_dresidual':float(r[i]),'dresidual_dprediction':1.,'dprediction_da':1.,'dprediction_db':float(x[i]),'scaled_mean_gradient_contribution':contrib[i].tolist()} for i in range(len(data))]
    default_data=np.array_equal(data,np.array([[-2.,-1.5],[-2.,-.5],[2.,2.5],[2.,3.5]]))
    gap=float((a-1)**2/2+2*(b-1)**2) if default_data else None
    return {'theta':theta.tolist(),'rows':rows,'data_loss':loss,'scaled_objective':loss*scale,'gradient':g.tolist(),'default_closed_form_gap':gap}

def options(s,method,eta=None,epsilon=None,scale=1.,inside=False):
    if type(method) is not str or method not in METHODS:raise ValueError('unknown method')
    eta=s['learning_rate'] if eta is None else real(eta,'rate',1)
    eps=s['epsilon'] if epsilon is None else real(epsilon,'epsilon',1e6)
    scale=real(scale,'objective scale',10000)
    if not 1e-12<=eta<=1 or not 1e-12<=eps<=1e6 or not 1e-4<=scale<=10000 or type(inside) is not bool:raise ValueError('invalid optimizer options')
    return {'method':method,'eta':eta,'epsilon':eps,'scale':scale,'beta1':s['beta1'],'beta2':s['beta2'],'alpha':s['rmsprop_alpha'],'weight_decay':s['weight_decay'] if method in ('AdamL2','AdamW') else 0.,'mask':s['decay_mask'],'epsilon_inside_root':inside}

def initial_state(theta,opt):return {'theta':theta.tolist(),'m':[0.,0.],'v':[0.,0.],'s':[0.,0.],'t':0,'options':copy.deepcopy(opt)}

def validate_state(state,opt,max_step):
    if type(state) is not dict or set(state)!= {'theta','m','v','s','t','options'}:raise ValueError('invalid state fields')
    # Check raw scalar types BEFORE comparing values (True == 1 is not validation).
    t=integer(state['t'],'state step',0,max_step)
    vectors={k:vector(state[k],'state '+k,nonnegative=k in ('v','s'),limit=1e140,floor=0.) for k in ('theta','m','v','s')}
    saved=state['options']
    if type(saved) is not dict or set(saved)!=set(opt):raise ValueError('state option fields')
    for k,v in saved.items():
        if k=='method':
            if type(v) is not str:raise ValueError('state method type')
        elif k=='epsilon_inside_root':
            if type(v) is not bool:raise ValueError('state root flag type')
        elif k=='mask':
            if type(v) is not list or len(v)!=2 or any(type(z) is not int or z not in (0,1) for z in v):raise ValueError('state mask type')
        else:real(v,'state '+k)
    if saved!=opt:raise ValueError('state hyperparameters do not match requested run')
    method=opt['method']
    if (method not in ('Adam','AdamL2','AdamW') and np.any(vectors['m'])) or (method not in ('Adam','AdamL2','AdamW','RMSProp') and np.any(vectors['v'])) or (method!='AdaGrad' and np.any(vectors['s'])):raise ValueError('unexpected state for method')
    if t==0 and any(np.any(vectors[k]) for k in ('m','v','s')):raise ValueError('nonzero history at step zero')
    return {**{k:v.tolist() for k,v in vectors.items()},'t':t,'options':copy.deepcopy(opt)}

def step(data,state,opt):
    """One simultaneous update; return row-level old/new forwards and all states."""
    theta=np.array(state['theta']);m=np.array(state['m']);v=np.array(state['v']);s=np.array(state['s']);t=state['t']+1
    old=forward(data,theta,opt['scale']);g=np.array(old['gradient']);mask=np.array(opt['mask']);wd=opt['weight_decay'];method=opt['method']
    regularizer_gradient=wd*mask*theta if method=='AdamL2' else np.zeros(2)
    used=_finite(g+regularizer_gradient,'used gradient');eta=opt['eta'];eps=opt['epsilon']
    mh=used.copy();vh=np.zeros(2);den=np.ones(2);correction=[1.,1.]
    if method!='GD':
        squared=_finite(used*used,'squared used gradient')
        if np.any((used!=0)&(squared==0)):raise FloatingPointError('nonzero gradient square underflow')
    if method=='AdaGrad':s=_finite(s+squared,'accumulated squares');vh=s
    elif method=='RMSProp':v=_finite(opt['alpha']*v+(1-opt['alpha'])*squared,'RMS squares');vh=v
    elif method in ('Adam','AdamL2','AdamW'):
        m=_finite(opt['beta1']*m+(1-opt['beta1'])*used,'first moment');v=_finite(opt['beta2']*v+(1-opt['beta2'])*squared,'second moment')
        correction=[1-opt['beta1']**t,1-opt['beta2']**t];mh=m/correction[0];vh=v/correction[1]
    if method!='GD':den=np.sqrt(vh+eps) if opt['epsilon_inside_root'] else np.sqrt(vh)+eps
    coeff=eta/den;adaptive=-coeff*mh;decay=-eta*wd*mask*theta if method=='AdamW' else np.zeros(2)
    new_theta=_finite(theta+decay+adaptive,'new parameter');new=forward(data,new_theta,opt['scale'])
    new_state={'theta':new_theta.tolist(),'m':m.tolist(),'v':v.tolist(),'s':s.tolist(),'t':t,'options':copy.deepcopy(opt)}
    penalty_old=wd*float(np.sum(mask*theta*theta))/2;penalty_new=wd*float(np.sum(mask*new_theta*new_theta))/2
    trace={'update':t,'old_forward':old,'data_gradient':g.tolist(),'regularizer_gradient':regularizer_gradient.tolist(),'used_gradient':used.tolist(),'old_state':copy.deepcopy(state),'new_state':new_state,'correction':correction,'direction':mh.tolist(),'raw_scale_statistic':vh.tolist(),'denominator':den.tolist(),'effective_coefficient':coeff.tolist(),'adaptive_delta':adaptive.tolist(),'decay_delta':decay.tolist(),'actual_delta':(new_theta-theta).tolist(),'new_forward':new,'old_F_plus_L2':old['data_loss']+penalty_old,'new_F_plus_L2':new['data_loss']+penalty_new}
    return new_state,trace

def run_path(data,s,method,*,eta=None,epsilon=None,scale=1.,inside=False,initial=None,resume=None,stop=None):
    data,s=validate(data,s);opt=options(s,method,eta,epsilon,scale,inside)
    theta=vector(s['decay_initial_parameters'] if method in ('AdamL2','AdamW') else s['initial_parameters'],'initial') if initial is None else vector(initial,'initial',limit=100)
    end=s['steps'] if stop is None else integer(stop,'stop',0,s['steps'])
    state=initial_state(theta,opt) if resume is None else validate_state(resume,opt,end)
    # Even a zero-update path validates the whole checkpoint and all options first.
    first=forward(data,np.array(state['theta']),scale);trace=[];status='fixed_budget_completed';attempts=0
    for _ in range(state['t'],end):
        attempts+=1
        try:new,tr=step(data,state,opt)
        except FloatingPointError as ex:status='arithmetic_range_stop: '+str(ex);break
        state=new;trace.append(tr)
    return {'method':method,'options':opt,'start_state':initial_state(np.array(first['theta']),opt) if resume is None else copy.deepcopy(resume),'initial_forward':first,'trace':trace,'final_state':state,'status':status,'completed_updates':len(trace),'attempted_updates':attempts,'training_sample_gradients':attempts*len(data),'diagnostic_full_forward_rows':len(data)*(1+len(trace))}

def exact_hand(data,s):
    """Exact rational first update, then a separate 100-digit root reference."""
    from decimal import localcontext
    Q=Fraction;rows=[list(map(lambda v:Q(float(v)),r)) for r in data];n=len(rows);theta=list(map(Q,s['initial_parameters']));eta=Q(s['learning_rate']);b1=Q(s['beta1']);b2=Q(s['beta2']);eps=Q(s['epsilon'])
    def fwd(th):
        rs=[th[0]+th[1]*x-y for x,y in rows];return {'theta':list(map(str,th)),'predictions':[str(th[0]+th[1]*x) for x,y in rows],'residuals':list(map(str,rs)),'half_squares':[str(r*r/2) for r in rs],'gradient_contributions':[[str(r/n),str(r*x/n)] for r,(x,y) in zip(rs,rows)],'gradient':[sum(rs)/n,sum(r*x for r,(x,y) in zip(rs,rows))/n],'loss':str(sum(r*r for r in rs)/(2*n))}
    f0=fwd(theta);g=f0['gradient'];m=[(1-b1)*v for v in g];v=[(1-b2)*z*z for z in g];mh=[z/(1-b1) for z in m];vh=[z/(1-b2) for z in v]
    # First vhat is g^2, so the square root is exact abs(g), for all valid data.
    th1=[a-eta*d/(abs(gg)+eps) for a,d,gg in zip(theta,mh,g)];f1=fwd(th1);g2=f1['gradient'];m2=[b1*a+(1-b1)*b for a,b in zip(m,g2)];v2=[b2*a+(1-b2)*b*b for a,b in zip(v,g2)];mh2=[a/(1-b1*b1) for a in m2];vh2=[a/(1-b2*b2) for a in v2]
    for f in (f0,f1):f['gradient']=list(map(str,f['gradient']))
    with localcontext() as ctx:
        ctx.prec=100
        def D(q):return Decimal(q.numerator)/Decimal(q.denominator)
        th2=[D(a)-D(eta)*D(mh)/(D(vh).sqrt()+D(eps)) for a,mh,vh in zip(th1,mh2,vh2)];pr=[th2[0]+th2[1]*D(x) for x,y in rows];rs=[p-D(y) for p,(x,y) in zip(pr,rows)];ls=[r*r/2 for r in rs]
        f2={'theta':list(map(str,th2)),'predictions':list(map(str,pr)),'residuals':list(map(str,rs)),'half_squares':list(map(str,ls)),'loss':str(sum(ls)/n)}
    return {'reference':'Fractions of stored binary64 inputs; independent Decimal100 roots, never fed into solver','old':f0,'step1':{'m':list(map(str,m)),'v':list(map(str,v)),'mhat':list(map(str,mh)),'vhat':list(map(str,vh))},'after1':f1,'step2':{'m':list(map(str,m2)),'v':list(map(str,v2)),'mhat':list(map(str,mh2)),'vhat':list(map(str,vh2))},'after2_decimal100':f2}

def compact(path):
    first=path['initial_forward'];curve=[{'t':path['start_state']['t'],'theta':first['theta'],'F':first['data_loss'],'default_gap':first['default_closed_form_gap']}]+[{'t':tr['update'],'theta':tr['new_state']['theta'],'F':tr['new_forward']['data_loss'],'default_gap':tr['new_forward']['default_closed_form_gap']} for tr in path['trace']]
    return {'status':path['status'],'sample_gradients':path['training_sample_gradients'],'curve':curve}

def run_experiment(data,s):
    data,s=validate(data,s)
    methods={name:run_path(data,s,name) for name in METHODS}
    grid={f'{m}:eta={eta}:eps={ep}':compact(run_path(data,s,m,eta=eta,epsilon=ep)) for m in METHODS[:4] for eta in s['rate_grid'] for ep in s['epsilon_grid']}
    base=methods['Adam'];scaled=run_path(data,s,'Adam',scale=s['objective_scale']);matched=run_path(data,s,'Adam',scale=s['objective_scale'],epsilon=s['epsilon']*s['objective_scale']);inside=run_path(data,s,'Adam',inside=True)
    split=s['resume_split'];whole={};restart={}
    for name in METHODS:
        path=methods[name];prefix=run_path(data,s,name,stop=split)
        reached=prefix['final_state']['t']
        if reached!=split:
            reason='not_run: arithmetic range prevented reaching the requested split'
            whole[name]={'split':split,'actual_prefix_updates':reached,'status':reason,'identical_suffix':None,'resumed':{'status':reason,'sample_gradients':0,'curve':[]}}
            restart[name]={'status':reason,'requested_split':split,'actual_prefix_updates':reached,'sample_gradients':0,'curve':[]}
            continue
        resumed=run_path(data,s,name,resume=prefix['final_state']);whole[name]={'split':split,'actual_prefix_updates':reached,'status':'comparison_executed','identical_suffix':resumed['trace']==path['trace'][split:],'resumed':compact(resumed)}
        # A reached internal iterate is NOT a new user-supplied initial value.
        # Reset only optimizer history/time and validate it as a complete state.
        reset=initial_state(np.array(prefix['final_state']['theta']),path['options'])
        restarted=run_path(data,s,name,resume=reset,stop=s['steps']-split);restart[name]=compact(restarted)
    # Same data and the declared teaching point. This probe is explicitly not a
    # claim that (1,1) is the optimum for arbitrary custom data.
    near=run_path(data,s,'Adam',initial=[1+1e-6,1.],eta=.25,epsilon=1e-8,stop=2)
    return {'unit':'036','numpy_version':np.__version__,'config':s,'data':data.tolist(),'hand':exact_hand(data,s),'methods':methods,'grid':grid,'scale_probe':{'factor':s['objective_scale'],'same_epsilon':compact(scaled),'matched_epsilon':compact(matched),'inside_root':compact(inside)},'resume':whole,'parameter_only_restart':restart,'near_declared_point_probe':near,'limitations':['All updates use finite stored binary64 row computations. Display rounding is not fed back.','No early convergence declaration; valid paths use a fixed update budget. Range stops retain only complete finite steps and count attempted training gradients.','Grid candidates are predeclared; no generalization, hardware-speed or universal winner claim.','Full row traces are retained for main methods; grid stores all parameters/objectives only.','Near-point probe is near the optimum only for the shipped four-row dataset.','Diagnostic forwards are additional work, reported separately from training sample gradients.']}

def report_bytes(report):return (json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()

def safe_target(out,inputs):
    """Do not overwrite input/package assets, even through symlink/hardlink aliases."""
    out=Path(out)
    if out.is_symlink() or out.exists() and not out.is_dir():raise ValueError('unsafe output directory')
    target=out/'report.json'
    if target.is_symlink():raise ValueError('output target symlink rejected')
    protected=[Path(p).resolve() for p in inputs]+[p.resolve() for p in ROOT.rglob('*') if p.is_file() and p.parts[len(ROOT.parts)] not in ('outputs','notebook_results','qa','__pycache__')]
    resolved=target.resolve()
    if resolved in protected or target.exists() and any(p.exists() and os.path.samefile(target,p) for p in protected):raise ValueError('output aliases a protected input or package file')
    if out.resolve() in protected:raise ValueError('output directory aliases a protected file')
    return target

def write_report(report,out,inputs):
    target=safe_target(out,inputs);payload=report_bytes(report)
    target=safe_target(out,inputs);target.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.report-',suffix='.tmp',dir=target.parent)
    try:
        with os.fdopen(fd,'wb') as f:f.write(payload)
        os.replace(tmp,target)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    return payload

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,default=ROOT/'data/regression.csv');p.add_argument('--config',type=Path,default=ROOT/'data/model_spec.json');p.add_argument('--output-dir',type=Path,default=ROOT/'outputs');a=p.parse_args()
    data,s=load_inputs(a.data,a.config);safe_target(a.output_dir,[a.data,a.config]);payload=write_report(run_experiment(data,s),a.output_dir,[a.data,a.config])
    print(json.dumps({'unit':'036','bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest(),'main_methods':6}))
if __name__=='__main__':main()
