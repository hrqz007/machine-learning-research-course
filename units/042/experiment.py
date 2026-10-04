"""ML042: stable binary likelihood, row-level calculus and existence diagnostics.

Original synthetic data; one feature plus unpenalized intercept. Fit requires both
classes. A numerical gradient stop is not a general finite-MLE certificate.
"""
from pathlib import Path
from decimal import Decimal, localcontext
import argparse, csv, hashlib, io, json, math, os, tempfile, warnings
import numpy as np
ROOT=Path(__file__).resolve().parent
FIELDS={'kind','initial_parameters','learning_rate','max_updates','hand_updates','gradient_tolerance','regularization_lambda','separation_ray_scales','thresholds','tie_rule','fd_steps','extreme_logits'}

def scalar(x,name,lo=-1e6,hi=1e6,integer=False):
    if isinstance(x,(bool,np.bool_)) or not isinstance(x,(int,float,np.integer,np.floating)):
        raise ValueError(name+' requires a real numeric scalar')
    if integer and not isinstance(x,(int,np.integer)):raise ValueError(name+' requires integer')
    try:v=float(x)
    except (ValueError,OverflowError) as ex:raise ValueError(name+' unrepresentable') from ex
    if v==0 and x!=0:raise ValueError(name+' nonzero lost during float conversion')
    if not math.isfinite(v) or not lo<=v<=hi or (v!=0 and abs(v)<1e-100):raise ValueError(name+' outside supported raw range')
    return int(x) if integer else v

def vector(x,name,size=None,lo=-1e6,hi=1e6):
    if not isinstance(x,(list,tuple,np.ndarray)) or (isinstance(x,np.ndarray) and x.ndim!=1):raise ValueError(name+' requires vector')
    if not 1<=len(x)<=5000 or (size is not None and len(x)!=size):raise ValueError(name+' length mismatch')
    return np.array([scalar(v,name,lo,hi) for v in x])

def inputs(x,y,both=False):
    x=vector(x,'x');y=vector(y,'y',len(x),0,1)
    if np.any((y!=0)&(y!=1)):raise ValueError('labels must be binary 0/1')
    if both and set(y.tolist())!={0.,1.}:raise ValueError('fit requires both label classes')
    return x,y

def parse_number(s):
    d=Decimal(s);v=float(d)
    if not math.isfinite(v) or (d!=0 and (v==0 or abs(v)<1e-100)):raise ValueError('unrepresentable JSON/CSV number')
    return v

def pairs_object(items):
    d={}
    for k,v in items:
        if k in d:raise ValueError('duplicate JSON key')
        d[k]=v
    return d

def read_json(path):
    return json.loads(Path(path).read_text(),parse_float=parse_number,object_pairs_hook=pairs_object,parse_constant=lambda s:(_ for _ in ()).throw(ValueError('nonfinite JSON')))

def validate_config(c):
    if not isinstance(c,dict) or set(c)!=FIELDS or c['kind']!='binary_logistic_likelihood_v1':raise ValueError('config fields/kind mismatch')
    q=dict(c);q['initial_parameters']=vector(c['initial_parameters'],'initial',2,-100,100).tolist()
    for k,lo,hi in [('learning_rate',1e-8,100),('gradient_tolerance',1e-14,.01),('regularization_lambda',0,100)]:q[k]=scalar(c[k],k,lo,hi)
    for k,hi in [('max_updates',10000),('hand_updates',2)]:q[k]=scalar(c[k],k,1,hi,True)
    for k,lo,hi in [('separation_ray_scales',0,1000),('thresholds',0,1),('fd_steps',1e-10,.1),('extreme_logits',-10000,10000)]:
        q[k]=vector(c[k],k,lo=lo,hi=hi).tolist()
        if len(q[k])>30 or len(set(q[k]))!=len(q[k]):raise ValueError(k+' duplicate/excess values')
    if c['tie_rule']!='positive if probability >= threshold':raise ValueError('tie rule mismatch')
    return q

def load_csv(path):
    r=csv.DictReader(io.StringIO(Path(path).read_text()))
    if r.fieldnames!=['id','x','y']:raise ValueError('CSV header mismatch')
    rows=list(r);seen=set()
    for row in rows:
        if set(row)!= {'id','x','y'} or any(v is None for v in row.values()):raise ValueError('malformed row')
        sid=row['id']
        if not sid or len(sid)>40 or not sid.isascii() or not all(c.isalnum() or c in '_-' for c in sid) or sid in seen:raise ValueError('bad/duplicate ID')
        seen.add(sid)
    x,y=inputs([parse_number(v['x']) for v in rows],[parse_number(v['y']) for v in rows],True)
    return x,y,[v['id'] for v in rows]

def load_inputs(data=None,separable=None,config=None):
    c=validate_config(read_json(config or ROOT/'data/model_spec.json'))
    x,y,ids=load_csv(data or ROOT/'data/observations.csv');sx,sy,sids=load_csv(separable or ROOT/'data/separable.csv')
    return x,y,ids,sx,sy,sids,c

def sigmoid(z):
    z=np.asarray(z,dtype=float)
    if not np.isfinite(z).all():raise ValueError('sigmoid requires finite logits')
    t=np.exp(-np.abs(z));return np.where(z>=0,1/(1+t),t/(1+t))

def softplus(u):
    u=np.asarray(u,dtype=float)
    if not np.isfinite(u).all():raise ValueError('softplus requires finite logits')
    return np.maximum(u,0)+np.log1p(np.exp(-np.abs(u)))

def primitives(z,y):
    """Validated raw public primitive, all labels/logits checked before exponentials."""
    z=vector(z,'z',lo=-10000,hi=10000);y=vector(y,'y',len(z),0,1)
    if np.any((y!=0)&(y!=1)):raise ValueError('labels must be 0/1')
    p=sigmoid(z);other=sigmoid(-z);loss=softplus(-(2*y-1)*z)
    return p,loss,np.where(y==0,p,-other),p*other

def _state(x,y,theta,lam):
    """Internal finite iterates are not rechecked against the RAW initial box."""
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        A=np.column_stack((np.ones(len(x)),x));z=A@theta
        if not np.isfinite(z).all():raise FloatingPointError('nonfinite score')
        p=sigmoid(z);om=sigmoid(-z);loss=softplus(-(2*y-1)*z)
        dz=np.where(y==0,p,-om);weight=p*om;n=len(x)
        gc=dz[:,None]*A/n;hc=weight[:,None,None]*A[:,:,None]*A[:,None,:]/n
        g=gc.sum(axis=0)+np.array([0.,lam*theta[1]]);H=hc.sum(axis=0)+np.diag([0.,lam])
        F=float(loss.mean());penalty=float(lam*theta[1]*theta[1]/2);J=F+penalty
        if not all(np.isfinite(v).all() for v in (theta,g,H,J)):raise FloatingPointError('nonfinite state')
        rows=[]
        for i in range(n):
            denominator=p[i] if y[i] else om[i]
            local=(-1. if y[i] else 1.)/denominator if denominator>=1/np.finfo(float).max else None
            rows.append({'row':i+1,'x':float(x[i]),'y':int(y[i]),'logit':float(z[i]),'probability':float(p[i]),
             'complement_probability':float(om[i]),'loss':float(loss[i]),'mean_loss_contribution':float(loss[i]/n),
             'local_dloss_dp':None if local is None else float(local),'local_dp_dz':float(weight[i]),
             'stable_dloss_dz':float(dz[i]),'jacobian':A[i].tolist(),'gradient_contribution':gc[i].tolist(),
             'hessian_contribution':hc[i].tolist(),'tail_underflow':bool(loss[i]==0)})
        return {'theta':theta.tolist(),'data_loss':F,'penalty':penalty,'objective':J,'gradient':g.tolist(),
         'gradient_norm':float(np.linalg.norm(g)),'hessian':H.tolist(),'rows':rows}

def row_ledger(x,y,theta,lam=0.):
    x,y=inputs(x,y);theta=vector(theta,'theta',2);lam=scalar(lam,'lambda',0,100)
    return _state(x,y,theta,lam)

def fit(x,y,initial=(0.,0.),lam=0.,method='gd',learning_rate=.5,max_updates=400,tol=1e-10):
    x,y=inputs(x,y,True);theta=vector(initial,'initial',2,-100,100);lam=scalar(lam,'lambda',0,100)
    eta=scalar(learning_rate,'learning_rate',1e-8,100);budget=scalar(max_updates,'max_updates',1,10000,True);tol=scalar(tol,'tol',1e-14,.01)
    if method not in ('gd','newton'):raise ValueError('unknown method')
    cost={'state_attempts':0,'state_successes':0,'sample_forward_attempts':0,'sample_gradient_attempts':0,'sample_hessian_attempts':0,'solve_attempts':0,'cholesky_attempts':0}
    def evaluate(t):
        cost['state_attempts']+=1
        for k in ('sample_forward_attempts','sample_gradient_attempts','sample_hessian_attempts'):cost[k]+=len(x)
        s=_state(x,y,t,lam);cost['state_successes']+=1;return s
    state=evaluate(theta);trace=[state];trials=[];status='max_updates'
    for k in range(budget):
        if state['gradient_norm']<=tol:status='gradient_tolerance';break
        g=np.array(state['gradient']);H=np.array(state['hessian']);direction=-g;shift=0.
        if method=='newton':
            ok=False
            for shift in (0.,1e-10,1e-8,1e-6,1e-4,.01,1.,100.):
                cost['cholesky_attempts']+=1
                try:
                    np.linalg.cholesky(H+shift*np.eye(2));cost['solve_attempts']+=1
                    direction=np.linalg.solve(H+shift*np.eye(2),-g)
                    if np.isfinite(direction).all() and g@direction<0:ok=True;break
                except np.linalg.LinAlgError:pass
            if not ok:status='direction_failure';break
        accepted=False
        for j in range(1 if method=='gd' else 30):
            alpha=eta if method=='gd' else 2.**(-j);candidate=theta+alpha*direction
            item={'update':k+1,'alpha':alpha,'shift':shift,'candidate':candidate.tolist(),'accepted':False}
            if not np.isfinite(candidate).all():item['failure']='nonfinite_candidate';trials.append(item);status='numeric_failure';break
            try:new=evaluate(candidate)
            except (FloatingPointError,ValueError):item['failure']='state_failure';trials.append(item);status='numeric_failure';break
            item['objective']=new['objective'];bound=state['objective'] if method=='gd' else state['objective']+1e-4*alpha*float(g@direction)
            # Equality can occur at machine precision; no artificial rounded progress claim.
            item['accepted']=bool(new['objective']<=bound+8*np.finfo(float).eps*max(1.,abs(state['objective'])))
            trials.append(item)
            if item['accepted']:
                if np.array_equal(candidate,theta):status='parameter_stagnation';break
                theta=candidate;state=new;trace.append(state);accepted=True;break
        if not accepted:
            if status=='max_updates':status='step_rejected'
            break
    if state['gradient_norm']<=tol:status='gradient_tolerance'
    return {'method':method,'lambda':lam,'status':status,'converged_numerically':status=='gradient_tolerance','updates':len(trace)-1,
      'trace':trace,'trials':trials,'final':state,'cost':cost,'cost_semantics':'attempts charged before each full row-state evaluation; each attempt plans n forward, gradient and Hessian rows; successes separately recorded; rejected trials included; no claim partially failed rows all completed'}

def cubic_certificate():
    with localcontext() as c:
        c.prec=110;lo=Decimal(1);hi=Decimal(2)
        for _ in range(400):
            mid=(lo+hi)/2
            if mid**3-mid-2>0:hi=mid
            else:lo=mid
        t=(lo+hi)/2;w=t.ln()
        return {'precision':110,'t':str(t),'w':str(w),'cubic_residual':str(t**3-t-2),'interval_width':str(hi-lo),
          'scope':'only the declared symmetric four-row main example; strict convexity plus full two-dimensional stationary point proves unique finite MLE'}

def fit_library(x,y,lam):
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning
    x,y=inputs(x,y,True);lam=scalar(lam,'lambda',0,100);C=np.inf if lam==0 else 1/(len(x)*lam)
    model=LogisticRegression(C=C,l1_ratio=0,solver='lbfgs',fit_intercept=True,tol=1e-12,max_iter=2000)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');model.fit(x[:,None],y)
    theta=[float(model.intercept_[0]),float(model.coef_[0,0])];s=row_ledger(x,y,theta,lam)
    # A controlled exact-zero probe is separate from the fitted model's near-zero intercept.
    probe=LogisticRegression();probe.classes_=np.array([0.,1.]);probe.coef_=np.array([[0.]]);probe.intercept_=np.array([0.]);probe.n_features_in_=1
    return {'lambda':lam,'C':None if lam==0 else C,'C_unregularized_infinity':lam==0,'theta':theta,'classes':model.classes_.tolist(),
     'decision_function':model.decision_function(x[:,None]).tolist(),'positive_probability':model.predict_proba(x[:,None])[:,1].tolist(),
     'predict':model.predict(x[:,None]).astype(int).tolist(),'n_iter':model.n_iter_.tolist(),'gradient_norm':s['gradient_norm'],'objective':s['objective'],
     'warnings':[{'category':type(w.message).__name__,'message':str(w.message)} for w in caught],
     'convergence_warning':any(isinstance(w.message,ConvergenceWarning) for w in caught),
     'controlled_zero_score_probe':{'library_predict':int(probe.predict([[0.]])[0]),'own_probability_ge_half':1,'not_fitted_score':True}}

def diagnostics(x,y,c):
    points=[[.3,-.2],[-.4,.7],[.1,.4]];direction=np.array([.6,-.8]);scan=[]
    for theta in points:
        theta=np.array(theta);s=row_ledger(x,y,theta);g=np.array(s['gradient']);H=np.array(s['hessian'])
        for h in c['fd_steps']:
            numeric=[]
            for j in range(2):
                e=np.eye(2)[j]*h;numeric.append((row_ledger(x,y,theta+e)['objective']-row_ledger(x,y,theta-e)['objective'])/(2*h))
            hv=(np.array(row_ledger(x,y,theta+h*direction)['gradient'])-np.array(row_ledger(x,y,theta-h*direction)['gradient']))/(2*h)
            scan.append({'theta':theta.tolist(),'h':h,'gradient_error':float(np.linalg.norm(numeric-g)),'hvp_error':float(np.linalg.norm(hv-H@direction))})
    tails=[]
    for z in c['extreme_logits']:
        for y0 in (0,1):
            p,l,g,h=primitives([z],[y0]);tails.append({'z':z,'y':y0,'probability':float(p[0]),'loss':float(l[0]),'gradient':float(g[0]),'hessian_weight':float(h[0]),'loss_underflow':bool(l[0]==0),'curvature_underflow':bool(h[0]==0)})
    return {'finite_difference':scan,'tails':tails}

def threshold_report(p,y,thresholds):
    p=vector(p,'p',lo=0,hi=1);y=vector(y,'y',len(p),0,1);thresholds=vector(thresholds,'thresholds',lo=0,hi=1)
    if np.any((y!=0)&(y!=1)):raise ValueError('labels must be binary')
    return [{'threshold':float(t),'predictions':(p>=t).astype(int).tolist(),'accuracy':float(np.mean((p>=t)==y))} for t in thresholds]

def main_report(x,y,ids,sx,sy,sids,c):
    c=validate_config(c);x,y=inputs(x,y,True);sx,sy=inputs(sx,sy,True)
    for names,n in ((ids,len(x)),(sids,len(sx))):
        if not isinstance(names,(list,tuple)) or len(names)!=n or any(not isinstance(v,str) or not v or len(v)>40 or not v.isascii() or not all(k.isalnum() or k in '_-' for k in v) for v in names) or len(set(names))!=n:raise ValueError('invalid report sample IDs')
    gd=fit(x,y,c['initial_parameters'],method='gd',learning_rate=c['learning_rate'],max_updates=c['max_updates'],tol=c['gradient_tolerance'])
    newton=fit(x,y,c['initial_parameters'],method='newton',max_updates=c['max_updates'],tol=c['gradient_tolerance'])
    reg=fit(x,y,c['initial_parameters'],lam=c['regularization_lambda'],method='newton',max_updates=c['max_updates'],tol=c['gradient_tolerance'])
    hand=gd['trace'][:c['hand_updates']+1];ray=[]
    for s in c['separation_ray_scales']:
        st=row_ledger(sx,sy,[0.,s]);margin=(2*sy-1)*sx*s
        ray.append({'scale':s,'loss':st['data_loss'],'gradient_norm':st['gradient_norm'],'theta_norm':s,'minimum_margin':float(margin.min()),'hessian_eigenvalues':np.linalg.eigvalsh(st['hessian']).tolist()})
    library=[fit_library(x,y,lam) for lam in (0.,c['regularization_lambda'])]
    doubled=fit_library(np.tile(x,2),np.tile(y,2),c['regularization_lambda'])
    A=np.column_stack((np.ones(len(x)),x));ols=np.linalg.lstsq(A,y,rcond=None)[0]
    default=np.array_equal(x,[-2,-1,1,2]) and np.array_equal(y,[0,1,0,1])
    probs=[r['probability'] for r in gd['final']['rows']]
    import sklearn,scipy
    return {'unit':'042','config':c,'data':{'x':x.tolist(),'y':y.astype(int).tolist(),'ids':ids,'separable_x':sx.tolist(),'separable_y':sy.astype(int).tolist(),'separable_ids':sids},
      'hand_states':hand,'second_update':hand[2]['theta'] if len(hand)>2 else None,'gd':gd,'newton':newton,'regularized_newton':reg,
      'finite_mle_certificate':cubic_certificate() if default else None,'separation_ray':ray,
      'separation_certificate_applies':bool(np.all((2*sy-1)*sx>0)),
      'library':library,'duplicated_regularized_library':doubled,'decisions':threshold_report(probs,y,c['thresholds']),
      'linear_probability':{'theta':ols.tolist(),'at_x_10':float(ols@[1.,10.])},'diagnostics':diagnostics(x,y,c),
      'versions':{'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__},
      'limitations':['four synthetic rows cannot measure real generalization/calibration','gradient_tolerance is numerical, not finite-MLE existence proof','known one-dimensional separation ray only; no general separation detection','no timing benchmark, browser UI or cross-platform execution claim']}

def canonical_bytes(report):return (json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()

def verify_fixture():
    contract=read_json(ROOT/'data_integrity.json')
    for rel,digest in contract['sha256'].items():
        if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=digest:raise ValueError('teaching fixture changed: '+rel)

def safe_write(path,report,extra_protected=()):
    data=canonical_bytes(report);p=Path(path).absolute()
    for q in (p,*p.parents):
        if q.is_symlink():raise ValueError('symlink output component')
    resolved=p.resolve();protected=[q for q in ROOT.rglob('*') if q.is_file() and 'outputs' not in q.relative_to(ROOT).parts]
    protected += [Path(v).resolve() for v in extra_protected]
    if resolved.suffix!='.json' or (p.exists() and (not p.is_file() or p.stat().st_nlink>1)):raise ValueError('unsafe output type/hardlink')
    if any(resolved==q.resolve() or (p.exists() and q.exists() and os.path.samefile(p,q)) for q in protected):raise ValueError('cannot overwrite inputs or course assets')
    made=[];parent=p.parent
    while not parent.exists():made.append(parent);parent=parent.parent
    temp=None
    try:
        p.parent.mkdir(parents=True,exist_ok=True);fd,temp=tempfile.mkstemp(prefix='.ml042-',dir=p.parent)
        with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        os.replace(temp,p);temp=None
    finally:
        if temp is not None and Path(temp).exists():Path(temp).unlink()
        for d in made:
            try:d.rmdir()
            except OSError:pass

def main():
    a=argparse.ArgumentParser();a.add_argument('--out',default=str(ROOT/'outputs/result.json'));a.add_argument('--data');a.add_argument('--separable');a.add_argument('--config');args=a.parse_args()
    if args.data is None and args.separable is None and args.config is None:verify_fixture()
    v=load_inputs(args.data,args.separable,args.config);r=main_report(*v);safe_write(args.out,r,[p for p in (args.data,args.separable,args.config) if p])
    print(json.dumps({'status':r['gd']['status'],'updates':r['gd']['updates'],'final':r['gd']['final']['theta'],'second_update':r['second_update'],'core_sha256':hashlib.sha256(canonical_bytes(r)).hexdigest()}))
if __name__=='__main__':main()
