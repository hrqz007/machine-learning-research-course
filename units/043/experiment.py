"""ML043: original multiclass row calculus, stable probabilities and information.

Theta has shape (d+1,K), with first row intercepts; X never includes a constant
column. Public numerical scope and failure behavior are documented in README.
"""
from pathlib import Path
from decimal import Decimal
import argparse, csv, hashlib, io, json, math, os, tempfile
import numpy as np
ROOT = Path(__file__).resolve().parent
CONFIG_FIELDS = {'kind','class_names','initial_theta','learning_rate','max_updates',
                 'gradient_tolerance','fd_steps','shift_constants','extreme_logits',
                 'information_q','information_p'}


def scalar(value, name, low=-10000, high=10000, integer=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int,float,np.integer,np.floating)):
        raise ValueError(name + ' must be a real numeric scalar, not bool/string/complex')
    if integer and not isinstance(value, (int,np.integer)):
        raise ValueError(name + ' must have integer type')
    try:
        result = float(value)
    except (OverflowError, ValueError) as ex:
        raise ValueError(name + ' not representable') from ex
    if not math.isfinite(result) or not low <= result <= high:
        raise ValueError(name + ' outside supported finite range')
    if value != 0 and (result == 0 or abs(result) < 1e-100):
        raise ValueError(name + ' nonzero raw magnitude below 1e-100')
    return int(value) if integer else result


def array(value, name, ndim, shape=None, low=-100, high=100):
    # Object conversion keeps each original scalar type until validation.
    if not isinstance(value,(list,tuple,np.ndarray)):
        raise ValueError(name + ' must be a list/tuple/array')
    try:
        raw = np.asarray(value, dtype=object)
    except ValueError as ex:
        raise ValueError(name + ' must be rectangular') from ex
    if raw.ndim != ndim or any(v == 0 for v in raw.shape) or raw.size > 40000:
        raise ValueError(name + ' invalid nonempty shape')
    if shape is not None and raw.shape != shape:
        raise ValueError(name + ' shape mismatch')
    return np.array([scalar(v,name,low,high) for v in raw.flat],dtype=float).reshape(raw.shape)


def validate_problem(X,y,theta):
    X = array(X,'X',2)
    n,d = X.shape
    if not (1 <= n <= 2000 and 1 <= d <= 20):
        raise ValueError('X scope: n=1..2000 and d=1..20')
    theta = array(theta,'theta',2,low=-100,high=100)
    if theta.shape[0] != d+1 or not 2 <= theta.shape[1] <= 20:
        raise ValueError('theta shape must be (d+1,K), K=2..20')
    y = array(y,'y',1,(n,),0,theta.shape[1]-1)
    if np.any(y != np.floor(y)):
        raise ValueError('labels must be integer-valued class indices')
    return X,y.astype(int),theta


def _log_probabilities(z):
    """Internal finite matrix; retain small log-probability for winning class."""
    if not np.isfinite(z).all():
        raise FloatingPointError('nonfinite logits')
    shifted = z - z.max(axis=1,keepdims=True)
    exponents = np.exp(shifted)
    winners = np.argmax(shifted,axis=1)
    tails = exponents.copy()
    tails[np.arange(len(z)),winners] = 0
    # One maximal exponential is exactly 1. log1p avoids rounding 1+small to 1.
    logden = np.log1p(tails.sum(axis=1,keepdims=True))
    logp = shifted - logden
    return np.exp(logp), logp


def stable_softmax(logits):
    z = array(logits,'logits',2,low=-10000,high=10000)
    if not 2 <= z.shape[1] <= 20 or z.shape[0] > 2000:
        raise ValueError('logits scope: n=1..2000, K=2..20')
    return _log_probabilities(z)


def _state(X,y,theta,include_rows=True):
    A = np.column_stack([np.ones(len(X)),X])
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        z = A @ theta
        p, logp = _log_probabilities(z)
        n,K = p.shape
        loss = -logp[np.arange(n),y]
        residual = p.copy()
        # True-class derivative is -sum(other p), not rounded p_y minus one.
        for i in range(n):
            residual[i,y[i]] = -math.fsum(p[i,k] for k in range(K) if k != y[i])
        gradient = A.T @ residual / n
        objective = float(math.fsum(loss)/n)
        if not np.isfinite(gradient).all() or not math.isfinite(objective):
            raise FloatingPointError('nonfinite state')
        state = {'theta':theta.tolist(),'objective':objective,'gradient':gradient.tolist(),
                 'gradient_norm':float(np.linalg.norm(gradient)), 'probabilities':p.tolist()}
        if include_rows:
            rows=[]
            for i in range(n):
                J = -np.outer(p[i],p[i])
                for k in range(K):
                    J[k,k] = p[i,k]*math.fsum(p[i,j] for j in range(K) if j != k)
                local = [0.]*K
                local[y[i]] = -1./p[i,y[i]] if p[i,y[i]] >= 1/np.finfo(float).max else None
                rows.append({'index':i+1,'x':X[i].tolist(),'y':int(y[i]),'logits':z[i].tolist(),
                    'probabilities':p[i].tolist(),'log_probabilities':logp[i].tolist(),
                    'loss':float(loss[i]),'mean_loss_contribution':float(loss[i]/n),
                    'local_dloss_dp':local,'local_softmax_jacobian':J.tolist(),
                    'stable_dloss_dz':residual[i].tolist(),'design_row':A[i].tolist(),
                    'gradient_contribution':(np.outer(A[i],residual[i])/n).tolist(),
                    'probability_underflow':bool(np.any(p[i]==0))})
            state['rows']=rows
        return state


def row_ledger(X,y,theta):
    return _state(*validate_problem(X,y,theta))


def fit(X,y,initial,learning_rate=.6,max_updates=600,tol=1e-10):
    X,y,theta=validate_problem(X,y,initial)
    eta=scalar(learning_rate,'learning_rate',1e-6,100)
    budget=scalar(max_updates,'max_updates',0,5000,True)
    tol=scalar(tol,'tol',1e-13,.01)
    state=_state(X,y,theta);trace=[state];trials=[];attempts=1;successes=1
    status='max_updates'
    for k in range(budget):
        if state['gradient_norm']<=tol:
            status='gradient_tolerance';break
        candidate=theta-eta*np.asarray(state['gradient'])
        if not np.isfinite(candidate).all():
            status='numeric_failure';trials.append({'update':k+1,'accepted':False,'failure':'nonfinite candidate'});break
        attempts+=1
        try:
            new=_state(X,y,candidate)
        except (FloatingPointError,ValueError):
            status='numeric_failure';trials.append({'update':k+1,'accepted':False,'failure':'state evaluation'});break
        successes+=1
        accepted=new['objective']<=state['objective']+8*np.finfo(float).eps*max(1,abs(state['objective']))
        trials.append({'update':k+1,'accepted':bool(accepted),'objective':new['objective']})
        if not accepted:
            status='step_rejected';break
        if np.array_equal(candidate,theta):
            status='parameter_stagnation';break
        theta=candidate;state=new;trace.append(state)
    if state['gradient_norm']<=tol:
        status='gradient_tolerance'
    return {'status':status,'updates':len(trace)-1,'converged_numerically':status=='gradient_tolerance',
            'trace':trace,'trials':trials,'final':state,'state_attempts':attempts,'state_successes':successes,
            'sample_state_attempts':attempts*len(X),
            'cost_scope':'planned full-row state attempts, including rejected/failed candidates; not FLOPs or wall-clock'}


def distribution(v,name):
    a=array(v,name,1,low=0,high=1)
    if not 2<=len(a)<=20:
        raise ValueError('distribution length 2..20')
    total=math.fsum(a)
    if abs(total-1)>1e-12:
        raise ValueError(name+' must sum to one within 1e-12; arbitrary counts are not normalized')
    return a/total  # only correct the explicitly accepted floating summation tolerance


def information(q,p):
    q=distribution(q,'q');p=distribution(p,'p')
    if len(q)!=len(p):
        raise ValueError('q and p must have same ordered class support')
    entropy=-math.fsum(float(a)*math.log(float(a)) for a in q if a>0)
    missing=any(a>0 and b==0 for a,b in zip(q,p))
    if missing:
        return {'q':q.tolist(),'p':p.tolist(),'entropy':entropy,'cross_entropy':None,'kl':None,
                'infinite_cross_entropy_and_kl':True,'log_base':'e'}
    cross=-math.fsum(float(a)*math.log(float(b)) for a,b in zip(q,p) if a>0)
    divergence=math.fsum(float(a)*(math.log(float(a))-math.log(float(b))) for a,b in zip(q,p) if a>0)
    return {'q':q.tolist(),'p':p.tolist(),'entropy':entropy,'cross_entropy':cross,'kl':divergence,
            'infinite_cross_entropy_and_kl':False,'log_base':'e'}


def pairs(items):
    out={}
    for k,v in items:
        if k in out:raise ValueError('duplicate JSON key: '+k)
        out[k]=v
    return out


def number(text):
    dec=Decimal(text);v=float(dec)
    if not math.isfinite(v) or (dec!=0 and (v==0 or abs(v)<1e-100)):
        raise ValueError('numeric token outside finite raw range')
    return v


def read_json(path):
    return json.loads(Path(path).read_text(),parse_float=number,object_pairs_hook=pairs,
       parse_constant=lambda s:(_ for _ in ()).throw(ValueError('nonfinite JSON token')))


def validate_config(c):
    if not isinstance(c,dict) or set(c)!=CONFIG_FIELDS or c['kind']!='multiclass_information_v1':
        raise ValueError('unknown/missing config fields')
    names=c['class_names']
    if not isinstance(names,list) or not 2<=len(names)<=20 or any(not isinstance(s,str) or not s or len(s)>20 for s in names) or len(set(names))!=len(names):
        raise ValueError('class_names must be distinct short strings')
    q=dict(c);theta=array(c['initial_theta'],'initial_theta',2,low=-100,high=100)
    if theta.shape!=(2,len(names)):raise ValueError('CSV teaching config requires (2,K) theta')
    q['initial_theta']=theta.tolist()
    q['learning_rate']=scalar(c['learning_rate'],'learning_rate',1e-6,100)
    q['max_updates']=scalar(c['max_updates'],'max_updates',0,5000,True)
    q['gradient_tolerance']=scalar(c['gradient_tolerance'],'gradient_tolerance',1e-13,.01)
    for key,lo,hi in [('fd_steps',1e-8,.1),('shift_constants',-1000,1000)]:
        v=array(c[key],key,1,low=lo,high=hi)
        if len(v)>20:raise ValueError('too many diagnostics')
        q[key]=v.tolist()
    z=array(c['extreme_logits'],'extreme_logits',2,low=-10000,high=10000)
    if z.shape[1]!=len(names) or len(z)>20:raise ValueError('extreme logits shape')
    q['extreme_logits']=z.tolist()
    for key in ['information_q','information_p']:
        v=distribution(c[key],key)
        if len(v)!=len(names):raise ValueError('information shape')
        q[key]=v.tolist()
    return q


def load_inputs(data=None,config=None):
    c=validate_config(read_json(config or ROOT/'data/model_spec.json'))
    reader=csv.DictReader(io.StringIO(Path(data or ROOT/'data/observations.csv').read_text()))
    if reader.fieldnames!=['id','x','y']:raise ValueError('CSV columns must be id,x,y')
    rows=list(reader);ids=[];xs=[];ys=[]
    for row in rows:
        if set(row)!= {'id','x','y'} or any(v is None for v in row.values()):raise ValueError('malformed CSV row')
        sid=row['id']
        if not sid or len(sid)>40 or not sid.isascii() or not all(c.isalnum() or c in '_-' for c in sid) or sid in ids:raise ValueError('bad/duplicate row id')
        ids.append(sid);xs.append([number(row['x'])]);ys.append(number(row['y']))
    X,y,t=validate_problem(xs,ys,c['initial_theta'])
    return X,y,ids,c


def exact_main_certificate():
    v=math.log(2)
    return {'theta':[[v/6,-v/3,v/6],[-v/2,0,v/2]],
            'probabilities_at_minus_one':[.5,.25,.25],
            'probabilities_at_plus_one':[.25,.25,.5],
            'minimum_objective':1.5*v,
            'scope':'only eight declared rows; unique probabilities at x=-1,+1, sum-zero representative; common-column shifts remain unidentifiable'}


def diagnostics(X,y,c):
    K=len(c['class_names']);base=np.arange(2*K,dtype=float).reshape(2,K)/7-.3
    scan=[]
    for scale in [1.,-.7]:
        theta=scale*base;s=row_ledger(X,y,theta);g=np.asarray(s['gradient'])
        for h in c['fd_steps']:
            numeric=np.empty(theta.shape)
            for index in np.ndindex(theta.shape):
                d=np.zeros_like(theta);d[index]=h
                numeric[index]=(row_ledger(X,y,theta+d)['objective']-row_ledger(X,y,theta-d)['objective'])/(2*h)
            scan.append({'scale':scale,'h':h,'maximum_gradient_error':float(np.max(abs(numeric-g)))})
    z=np.array([[.2,-.4,.7]]) if K==3 else np.arange(K,dtype=float)[None,:]/5
    p,lp=stable_softmax(z);shifts=[]
    for shift in c['shift_constants']:
        pp,ll=stable_softmax(z+shift)
        shifts.append({'shift':shift,'probability_max_error':float(np.max(abs(pp-p))),
                       'log_probability_max_error':float(np.max(abs(ll-lp)))})
    from scipy.special import softmax,log_softmax
    extremes=[]
    for row in c['extreme_logits']:
        zz=np.array([row]);pp,ll=stable_softmax(zz)
        extremes.append({'logits':row,'probabilities':pp[0].tolist(),'log_probabilities':ll[0].tolist(),
              'scipy_probabilities':softmax(zz,axis=1)[0].tolist(),
              'scipy_log_probabilities':log_softmax(zz,axis=1)[0].tolist()})
    return {'finite_difference':scan,'common_shifts':shifts,'extremes':extremes}


def main_report(X,y,ids,c):
    c=validate_config(c);X,y,_=validate_problem(X,y,c['initial_theta'])
    if len(ids)!=len(X) or len(set(ids))!=len(ids) or any(not isinstance(s,str) or not s for s in ids):raise ValueError('report ids invalid')
    fitted=fit(X,y,c['initial_theta'],c['learning_rate'],c['max_updates'],c['gradient_tolerance'])
    is_main=(X.shape==(8,1) and np.array_equal(X[:,0],[-1,-1,-1,-1,1,1,1,1]) and np.array_equal(y,[0,0,1,2,0,1,2,2]) and len(c['class_names'])==3)
    cert=exact_main_certificate() if is_main else None
    # Independent mature optimizer on a reference-class parameterization.
    from scipy.optimize import minimize
    K=len(c['class_names'])
    def decode(v):return np.column_stack([v.reshape(2,K-1),np.zeros(2)])
    def obj(v):
        state=_state(X,y,decode(v),False)
        return state['objective'],np.asarray(state['gradient'])[:,:-1].ravel()
    ref=minimize(obj,np.zeros(2*(K-1)),method='BFGS',jac=True,options={'gtol':1e-11,'maxiter':1000})
    rt=decode(ref.x);rt-=rt.mean(axis=1,keepdims=True);rs=_state(X,y,rt,False)
    pred=np.argmax(fitted['final']['probabilities'],axis=1)
    import scipy
    return {'unit':'043','config':c,'data':{'X':X.tolist(),'y':y.tolist(),'ids':ids},
        'fit':fitted,'hand_states':fitted['trace'][:3],
        'second_update':fitted['trace'][2]['theta'] if len(fitted['trace'])>2 else None,
        'certificate':cert,'reference_optimizer':{'name':'SciPy BFGS, last class fixed at zero','success':bool(ref.success),'message':str(ref.message),'iterations':int(ref.nit),'state':rs},
        'training_predictions':pred.tolist(),'training_accuracy':float(np.mean(pred==y)),
        'information':information(c['information_q'],c['information_p']),
        'reverse_information':information(c['information_p'],c['information_q']),
        'support_failure':information([1,0,0],[0,.5,.5]) if K==3 else None,
        'diagnostics':diagnostics(X,y,c),'versions':{'numpy':np.__version__,'scipy':scipy.__version__},
        'limitations':['synthetic mechanism data, no generalization or calibration claim','float64 supported finite input ranges, representable tails only','finite MLE certificate specific to declared eight rows','SciPy warnings/status retained; no browser/Anaconda/cross-platform test claim']}


def canonical_bytes(value):
    return (json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def safe_write(path,report,extra_protected=()):
    raw=canonical_bytes(report);path=Path(path).absolute()
    if path.suffix!='.json':raise ValueError('JSON suffix required')
    for p in (path,*path.parents):
        if p.is_symlink():raise ValueError('symlink output component')
    if path.exists() and (not path.is_file() or path.stat().st_nlink>1):raise ValueError('unsafe output file')
    protected=[p for p in ROOT.rglob('*') if p.is_file() and 'outputs' not in p.relative_to(ROOT).parts]
    protected.extend(Path(p) for p in extra_protected)
    if any(path.resolve()==p.resolve() or (path.exists() and p.exists() and os.path.samefile(path,p)) for p in protected):
        raise ValueError('cannot overwrite course/input assets')
    path.parent.mkdir(parents=True,exist_ok=True);temp=None
    try:
        fd,temp=tempfile.mkstemp(prefix='.ml043-',dir=path.parent)
        with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        os.replace(temp,path);temp=None
    finally:
        if temp is not None:Path(temp).unlink(missing_ok=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--data');p.add_argument('--config');p.add_argument('--out',default=str(ROOT/'outputs/result.json'));args=p.parse_args()
    if args.data is None and args.config is None:
        contract=read_json(ROOT/'data_integrity.json')
        for rel,h in contract['sha256'].items():
            if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=h:raise ValueError('teaching input changed: '+rel)
    r=main_report(*load_inputs(args.data,args.config));safe_write(args.out,r,[q for q in [args.data,args.config] if q])
    print(json.dumps({'status':r['fit']['status'],'updates':r['fit']['updates'],'objective':r['fit']['final']['objective'],'core_sha256':hashlib.sha256(canonical_bytes(r)).hexdigest()}))
if __name__=='__main__':main()
