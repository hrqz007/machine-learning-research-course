"""ML038: row-derived Newton, Armijo BFGS and finite-history L-BFGS.

High-level entry points validate raw inputs before callbacks/linear algebra.
Timing lives in benchmark.py, separate from this deterministic report.
"""
from pathlib import Path
import argparse, csv, io, json, math, os, tempfile
from decimal import Decimal
import numpy as np

ROOT = Path(__file__).resolve().parent
KIND = 'synthetic_newton_linear_and_square_parameter_regression'
FIELDS = {'kind','linear_initial_parameters','nonlinear_initial_parameters',
'indefinite_initial_parameters','stationary_initial_parameters','max_updates',
'gradient_tolerance','armijo_c1','backtrack_factor','max_line_trials',
'positive_shift_candidates','bfgs_curvature_tolerance','lbfgs_history',
'benchmark_dimensions','benchmark_repetitions','benchmark_seed',
'benchmark_gradient_relative_tolerance','benchmark_max_updates'}
METHODS = ('gd','newton','bfgs','lbfgs')

class ArithmeticRangeError(ValueError):
    pass


def scalar(x, name, lo=-1e100, hi=1e100, integer=False):
    if integer:
        if isinstance(x,(bool,np.bool_)) or not isinstance(x,(int,np.integer)):
            raise ValueError(name+' must be an integer, not bool')
    elif isinstance(x,(bool,np.bool_)) or type(x) not in (int,float,np.int32,np.int64,np.float32,np.float64):
        raise ValueError(name+' must be a real native scalar')
    v=float(x)
    if not math.isfinite(v) or not lo <= v <= hi or (v != 0 and abs(v)<1e-100):
        raise ValueError(name+' outside finite teaching range')
    return int(x) if integer else v


def vector(v, name, d=None, limit=1e100):
    if not isinstance(v,(list,tuple,np.ndarray)):
        raise ValueError(name+' must be a vector')
    if isinstance(v,np.ndarray) and v.ndim != 1:
        raise ValueError(name+' must have one axis')
    if d is not None and len(v)!=d:
        raise ValueError(name+' wrong dimension')
    return np.array([scalar(z,name,-limit,limit) for z in v],dtype=float)


def matrix(v,name,d,limit=1e100):
    if not isinstance(v,(list,tuple,np.ndarray)) or len(v)!=d:
        raise ValueError(name+' wrong matrix shape')
    rows=[vector(row,name,d,limit) for row in v]
    return np.array(rows,dtype=float)


def parse_number(s):
    dec=Decimal(s); v=float(dec)
    if not math.isfinite(v) or (dec != 0 and (v==0 or abs(v)<1e-100)):
        raise ValueError('nonfinite or underflowing numeric literal')
    return v


def pairs_object(items):
    out={}
    for k,v in items:
        if k in out: raise ValueError('duplicate JSON key: '+k)
        out[k]=v
    return out


def read_json(path):
    return json.loads(Path(path).read_text(),parse_float=parse_number,
        parse_constant=lambda s: (_ for _ in ()).throw(ValueError('nonfinite JSON')),
        object_pairs_hook=pairs_object)


def validate_config(c):
    if not isinstance(c,dict) or set(c)!=FIELDS: raise ValueError('configuration fields do not match')
    if c['kind']!=KIND: raise ValueError('unsupported data/model kind')
    q=dict(c)
    for key in ('linear_initial_parameters','nonlinear_initial_parameters','indefinite_initial_parameters','stationary_initial_parameters'):
        q[key]=vector(c[key],key,2,100).tolist()
    for key,lo,hi in [('max_updates',1,300),('max_line_trials',1,60),('lbfgs_history',1,10),
                       ('benchmark_repetitions',1,9),('benchmark_seed',0,2**32-1),('benchmark_max_updates',1,2000)]:
        q[key]=scalar(c[key],key,lo,hi,True)
    for key,lo,hi in [('gradient_tolerance',1e-14,1e-2),('armijo_c1',1e-8,.49),
                     ('backtrack_factor',.05,.95),('bfgs_curvature_tolerance',0,.1),
                     ('benchmark_gradient_relative_tolerance',1e-10,.01)]:
        q[key]=scalar(c[key],key,lo,hi)
    if not isinstance(c['positive_shift_candidates'],list) or not 1<=len(c['positive_shift_candidates'])<=20:
        raise ValueError('invalid finite shift list')
    shifts=[scalar(z,'shift',0,1e6) for z in c['positive_shift_candidates']]
    if shifts!=sorted(set(shifts)): raise ValueError('shifts must be strictly increasing')
    q['positive_shift_candidates']=shifts
    dims=c['benchmark_dimensions']
    if not isinstance(dims,list) or not 1<=len(dims)<=3: raise ValueError('invalid benchmark dimensions')
    q['benchmark_dimensions']=[scalar(z,'dimension',2,64,True) for z in dims]
    if len(set(q['benchmark_dimensions']))!=len(dims): raise ValueError('duplicate benchmark dimension')
    return q


def load_inputs(data_path=None, config_path=None):
    c=validate_config(read_json(config_path or ROOT/'data/model_spec.json'))
    stream=io.StringIO(Path(data_path or ROOT/'data/regression.csv').read_text())
    rd=csv.DictReader(stream)
    if rd.fieldnames!=['id','x','y']: raise ValueError('CSV header must be id,x,y')
    ids=[]; x=[]; y=[]
    for row in rd:
        if set(row)!=set(rd.fieldnames) or any(v is None for v in row.values()):raise ValueError('bad CSV row')
        sid=row['id']
        if not sid or len(sid)>40 or not sid.isascii() or not all(z.isalnum() or z in '_-' for z in sid):raise ValueError('bad sample ID')
        if sid in ids:raise ValueError('duplicate sample ID')
        ids.append(sid)
        x.append(scalar(parse_number(row['x']),'x',-100,100)); y.append(scalar(parse_number(row['y']),'y',-100,100))
    if not 2<=len(x)<=64:raise ValueError('2 to 64 data rows required')
    return vector(x,'x'),vector(y,'y'),c


def finite(a):
    a=np.asarray(a)
    if not np.isfinite(a).all() or (np.abs(a)>1e140).any():
        raise ArithmeticRangeError('arithmetic exceeded finite magnitude limit 1e140')
    return a


def checked_square(a):
    a=finite(a); z=a*a
    finite(z)
    if ((a!=0)&(z==0)).any():raise ArithmeticRangeError('nonzero square underflow')
    return z


class RowObjective:
    """Validated four-row-style data, with arbitrary 2..64 row inputs allowed."""
    def __init__(self,x,y,model):
        self.x=vector(x,'x',limit=100); self.y=vector(y,'y',len(self.x),100)
        if not 2<=len(self.x)<=64:raise ValueError('bad row count')
        if model not in ('linear','square'):raise ValueError('unknown model')
        self.model=model; self.d=2; self.n=len(self.x); self.counts={'f':0,'g':0,'H':0}
    def evaluate(self,theta,gradient=True,hessian=False,rows=False):
        # Internal numerical primitive: optimize validates user state first.
        self.counts['f']+=1
        a,b=finite(theta)
        with np.errstate(over='ignore',invalid='ignore',under='ignore'):
            slope=b if self.model=='linear' else b*b
            pred=finite(a+slope*self.x); r=finite(pred-self.y); sq=checked_square(r)
            f=float(finite(np.mean(sq)/2))
            out={'f':f}
            if gradient or hessian or rows:
                J=finite(np.column_stack((np.ones(self.n),self.x if self.model=='linear' else 2*b*self.x)))
                if gradient or rows:
                    contrib=finite(r[:,None]*J/self.n);g=finite(np.sum(contrib,axis=0));self.counts['g']+=1
                    if gradient:out['g']=g
            if hessian or rows:
                outer=finite(J[:,:,None]*J[:,None,:]/self.n)
                residual_second=np.zeros((self.n,2,2))
                if self.model=='square':residual_second[:,1,1]=2*self.x*r/self.n
                row_h=finite(outer+residual_second); H=finite(np.sum(row_h,axis=0))
                self.counts['H']+=1
                if hessian:out['H']=H
            if rows:
                is_default=np.array_equal(self.x,[-2.,-2.,2.,2.]) and np.array_equal(self.y,[-1.5,-.5,2.5,3.5])
                out['default_gap']=float(.5*(a-1)**2+2*(slope-1)**2) if is_default else None
                out['rows']=[{'id':i+1,'x':float(self.x[i]),'y':float(self.y[i]),
                    'prediction':float(pred[i]),'residual':float(r[i]),'half_square':float(sq[i]/2),
                    'loss_contribution':float(sq[i]/(2*self.n)),'local_dloss_dr':float(r[i]),
                    'jacobian':J[i].tolist(),'gradient_contribution':contrib[i].tolist(),
                    'hessian_J_outer':outer[i].tolist(),'hessian_residual_second':residual_second[i].tolist(),
                    'hessian_contribution':row_h[i].tolist()} for i in range(self.n)]
            return out


def jsonable(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:jsonable(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [jsonable(z) for z in x]
    return x


def validate_state(d,C=None,history=(),need_C=True):
    """Validate every raw item before first dot or Cholesky, including last pair."""
    c=(np.eye(d) if need_C else None) if C is None else matrix(C,'initial inverse approximation',d)
    if not isinstance(history,(tuple,list)) or len(history)>10:raise ValueError('invalid history container')
    pairs=[]
    for pair in history:
        if not isinstance(pair,(tuple,list)) or len(pair)!=2:raise ValueError('history pair must contain s,y')
        pairs.append((vector(pair[0],'history s',d),vector(pair[1],'history y',d)))
    # All raw scalar/type validation has finished; mathematical checks may now run.
    if c is not None:
        if not np.allclose(c,c.T,rtol=0,atol=1e-14):raise ValueError('C must be symmetric')
        try:np.linalg.cholesky(c)
        except np.linalg.LinAlgError as ex:raise ValueError('C must be positive definite') from ex
    for s,y in pairs:
        sy=float(finite(s@y)); yy=float(finite(y@y))
        if sy<=0 or yy<=0:raise ValueError('history requires finite positive curvature')
    return c,pairs


def lbfgs_direction(g,history,validated=False):
    g=vector(g,'gradient') if not validated else g
    if not validated:_,history=validate_state(len(g),history=history,need_C=False)
    q=g.copy(); alpha=[]; rho=[]
    for s,y in reversed(history):
        r=1/float(s@y); a=r*float(s@q); q=finite(q-a*y)
        alpha.append(a);rho.append(r)
    gamma=float(history[-1][0]@history[-1][1])/float(history[-1][1]@history[-1][1]) if history else 1.
    if not math.isfinite(gamma) or gamma<=0:raise ArithmeticRangeError('invalid L-BFGS scale')
    r=finite(gamma*q)
    for (s,y),a,ri in zip(history,reversed(alpha),reversed(rho)):
        beta=ri*float(y@r);r=finite(r+s*(a-beta))
    return -r,gamma


def inverse_update(C,s,y,tol):
    sy=float(finite(s@y)); scale=float(np.linalg.norm(s)*np.linalg.norm(y))
    info={'sTy':sy,'threshold':tol*scale,'updated':False,'cholesky_calls':0}
    if sy<=tol*scale or sy<=0:return C,dict(info,reason='curvature_skip')
    tau=1/sy; V=finite(np.eye(len(s))-tau*np.outer(s,y))
    c=finite(V@C@V.T+tau*np.outer(s,s));c=(c+c.T)/2
    try:np.linalg.cholesky(c)
    except np.linalg.LinAlgError:return C,dict(info,reason='numerical_spd_skip',cholesky_calls=1)
    return c,dict(info,updated=True,cholesky_calls=1,reason='accepted',secant_residual=float(np.linalg.norm(c@y-s)))


def optimize(obj,initial,cfg,method='newton',C0=None,history=(),trace=True,gtol=None,max_updates=None):
    c=validate_config(cfg)
    if method not in METHODS:raise ValueError('unknown method')
    if not isinstance(trace,bool):raise ValueError('trace must be bool')
    theta=vector(initial,'initial parameters',obj.d,100)
    tol=c['gradient_tolerance'] if gtol is None else scalar(gtol,'gtol',1e-14,1e4)
    budget=c['max_updates'] if max_updates is None else scalar(max_updates,'max_updates',1,2000,True)
    # Optional state is validated even for GD/Newton rather than silently ignored.
    C,pairs=validate_state(obj.d,C0,history,need_C=method=='bfgs')
    if len(pairs)>c['lbfgs_history']:raise ValueError('initial history exceeds configured window')
    counts0=dict(obj.counts); counts={'solve':0,'cholesky':int(C is not None),'line_trials':0}
    records=[]; states=[]; accepted=0; skipped=0; status='max_updates'
    with np.errstate(over='ignore',invalid='ignore',under='ignore'):
        current=obj.evaluate(theta,True,method=='newton',trace)
        finite(current['g'])
        if trace:states.append({'theta':theta.tolist(),'f':current['f'],'gradient_norm':float(np.linalg.norm(current['g'])),'default_gap':current.get('default_gap')})
        while accepted<budget:
            g=current['g']; norm=float(np.linalg.norm(g))
            if norm<=tol:status='gradient_stationary';break
            detail={}; direction=None
            try:
                if method=='newton':
                    H=current['H']; finite(H)
                    if H.shape!=(obj.d,obj.d) or not np.allclose(H,H.T,rtol=0,atol=1e-12):raise ArithmeticRangeError('bad Hessian')
                    trials_shift=[]
                    for shift in c['positive_shift_candidates']:
                        B=finite(H+shift*np.eye(obj.d));counts['cholesky']+=1
                        try:np.linalg.cholesky(B)
                        except np.linalg.LinAlgError:trials_shift.append({'shift':shift,'positive_definite':False});continue
                        counts['solve']+=1; p=finite(np.linalg.solve(B,-g));slope=float(finite(g@p))
                        trials_shift.append({'shift':shift,'positive_definite':True,'gTp':slope})
                        if slope<0:
                            direction=p; den=np.linalg.norm(B)*np.linalg.norm(p)+norm
                            detail={'shift':shift,'B':B.tolist(),'shift_trials':trials_shift,
                            'linear_residual_norm':float(np.linalg.norm(B@p+g)),
                            'linear_residual_scaled':float(np.linalg.norm(B@p+g)/den) if den else 0.,
                            'newton_decrement_squared':-slope};break
                    if direction is None:status='positive_shift_exhausted';break
                elif method=='gd':direction=-g
                elif method=='bfgs':direction=finite(-C@g);detail={'C_before':C.tolist()} if trace else {}
                else:
                    direction,gamma=lbfgs_direction(g,pairs,True)
                    detail={'gamma':gamma,'history_before':[[s.tolist(),y.tolist()] for s,y in pairs]} if trace else {'gamma':gamma}
                slope=float(finite(g@direction))
                if slope>=0:status='non_descent_direction';break
                alpha=1.; line=[]; trial_accepted=False; candidate=None
                for j in range(c['max_line_trials']):
                    trial=theta+alpha*direction; counts['line_trials']+=1
                    if np.array_equal(trial,theta):
                        line.append({'alpha':alpha,'theta':trial.tolist(),'f':None,'armijo_rhs':None,'accepted':False,'reason':'floating_stagnation'})
                        status='line_search_stagnation';break
                    rhs=current['f']+c['armijo_c1']*alpha*slope
                    try:
                        finite(trial); probe=obj.evaluate(trial,False,False,trace)
                        f=probe['f']; good=f<=rhs
                        line.append({'alpha':alpha,'theta':trial.tolist(),'f':f,'armijo_rhs':rhs,'accepted':bool(good),
                            'reason':'armijo_accept' if good else 'armijo_reject',**({'rows':probe['rows']} if trace else {})})
                    except (ArithmeticRangeError,FloatingPointError,OverflowError):
                        good=False;line.append({'alpha':alpha,'theta':trial.tolist() if np.isfinite(trial).all() else None,
                            'f':None,'armijo_rhs':rhs,'accepted':False,'reason':'arithmetic_range_reject'})
                    if good:
                        # A finite loss alone is not enough: all needed derivatives must be valid before commit.
                        candidate=obj.evaluate(trial,True,method=='newton',trace)
                        finite(candidate['g']);trial_accepted=True;break
                    alpha*=c['backtrack_factor']
                if not trial_accepted:
                    if status!='line_search_stagnation':status='line_search_exhausted'
                    if trace:records.append({'step':accepted+1,'accepted':False,'old_theta':theta.tolist(),'old':jsonable(current),'direction':direction.tolist(),'gTp':slope,'line_trials':line,'direction_state':detail})
                    break
                old_theta=theta.copy();s=trial-old_theta;y=candidate['g']-g
                update=None
                if method=='bfgs':
                    C,update=inverse_update(C,s,y,c['bfgs_curvature_tolerance'])
                    skipped+=not update['updated'];counts['cholesky']+=update['cholesky_calls']
                    if trace:detail['C_after']=C.tolist()
                if method=='lbfgs':
                    sy=float(finite(s@y)); threshold=c['bfgs_curvature_tolerance']*np.linalg.norm(s)*np.linalg.norm(y)
                    good_pair=sy>threshold and sy>0
                    update={'sTy':sy,'threshold':float(threshold),'updated':bool(good_pair),'reason':'accepted' if good_pair else 'curvature_skip'}
                    if good_pair:pairs=(pairs+[(s.copy(),y.copy())])[-c['lbfgs_history']:]
                    else:skipped+=1
                    if trace:detail['history_after']=[[z.tolist(),v.tolist()] for z,v in pairs]
                theta=trial;accepted+=1
                if trace:records.append({'step':accepted,'accepted':True,'old_theta':old_theta.tolist(),'old':jsonable(current),
                    'direction':direction.tolist(),'gTp':slope,'line_trials':line,'direction_state':detail,'alpha':alpha,
                    's':s.tolist(),'gradient_difference':y.tolist(),'curvature_update':update,'new_theta':theta.tolist(),'new':jsonable(candidate)})
                current=candidate
                if trace:states.append({'theta':theta.tolist(),'f':current['f'],'gradient_norm':float(np.linalg.norm(current['g'])),'default_gap':current.get('default_gap')})
            except (ArithmeticRangeError,FloatingPointError,OverflowError,np.linalg.LinAlgError):
                status='arithmetic_range_stop';break
        if float(np.linalg.norm(current['g']))<=tol:status='gradient_stationary'
        # Separate curvature diagnostic; not a global minimization certificate.
        final_h=obj.evaluate(theta,False,True,False)['H']
        eig=np.linalg.eigvalsh(final_h)
        curvature='positive_definite' if eig.min()>1e-12 else ('indefinite_or_negative' if eig.min()<-1e-12 else 'semidefinite_or_unresolved')
        if status=='gradient_stationary' and curvature=='indefinite_or_negative':status='stationary_indefinite'
    resident=theta.nbytes+current['g'].nbytes
    if method=='newton':resident+=final_h.nbytes
    elif method=='bfgs':resident+=C.nbytes
    elif method=='lbfgs':resident+=sum(s.nbytes+y.nbytes for s,y in pairs)
    return jsonable({'method':method,'status':status,'accepted_updates':accepted,'theta':theta,'f':current['f'],
        'gradient':current['g'],'gradient_norm':float(np.linalg.norm(current['g'])),'gradient_target':tol,
        'curvature_diagnostic':curvature,'hessian_eigenvalues':eig,
        'counts':{**{k:obj.counts[k]-counts0[k] for k in counts0},**counts},
        'skipped_curvature_updates':skipped,'history_size':len(pairs),'algorithm_state_nbytes':resident,
        'memory_note':'parameter, gradient, and retained H/C/history arrays; excludes data, model, trace, temporaries and native solver workspace',
        'states':states if trace else None,'trace':records if trace else None})


def run_experiment(x,y,cfg):
    c=validate_config(cfg);x=vector(x,'x',limit=100);y=vector(y,'y',len(x),100)
    if not 2<=len(x)<=64:raise ValueError('bad row count')
    paths={}
    for model,initial in [('linear',c['linear_initial_parameters']),('square',c['nonlinear_initial_parameters'])]:
        for method in METHODS:
            paths[model+'_'+method]=optimize(RowObjective(x,y,model),initial,c,method)
    paths['indefinite_modified_newton']=optimize(RowObjective(x,y,'square'),c['indefinite_initial_parameters'],c)
    paths['stationary_newton']=optimize(RowObjective(x,y,'square'),c['stationary_initial_parameters'],c)
    obj=RowObjective(x,y,'square'); old=obj.evaluate(c['indefinite_initial_parameters'],True,True,True)
    try:
        raw=np.linalg.solve(old['H'],-old['g']);new=np.array(c['indefinite_initial_parameters'])+raw
        raw_demo={'old':jsonable(old),'direction':raw.tolist(),'gTp':float(old['g']@raw),'new_theta':new.tolist(),'new':jsonable(obj.evaluate(new,True,True,True)),'status':'computed_not_applied_by_optimizer'}
    except np.linalg.LinAlgError:raw_demo={'status':'singular_raw_hessian','old':jsonable(old)}
    histories={}
    for m in (1,3,5):
        cm=dict(c,lbfgs_history=m); histories[str(m)]=optimize(RowObjective(x,y,'square'),c['nonlinear_initial_parameters'],cm,'lbfgs')
    return {'unit':'038','schema':1,'data':{'x':x.tolist(),'y':y.tolist()},'config':c,'paths':paths,
        'raw_indefinite_demo':raw_demo,'lbfgs_windows':histories,
        'limits':['small synthetic examples only','stationarity is not a global minimum certificate',
        'Armijo plus curvature skipping is not SciPy Wolfe BFGS','no timing in deterministic core report']}


def serialized(report):
    return (json.dumps(jsonable(report),ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()


def safe_write(output_dir,filename,body,extra_protected=()):
    """Serialize first, then validate all output aliases before creating directories."""
    if not isinstance(body,bytes):raise ValueError('serialize to bytes first')
    out=Path(output_dir).absolute(); target=out/filename
    if filename not in ('report.json','benchmark-result.json','library-result.json','audit-result.json'):raise ValueError('unsupported report filename')
    for p in (out,*out.parents):
        if p.is_symlink():raise ValueError('output directory cannot traverse symlinks')
    if out.exists() and not out.is_dir():raise ValueError('output directory is not a directory')
    if target.is_symlink() or (target.exists() and not target.is_file()):raise ValueError('unsafe target type')
    protected=[p for p in ROOT.rglob('*') if p.is_file() and not any(z in p.parts for z in ('outputs','notebook_results','__pycache__','qa','results'))]
    protected += [Path(p).resolve() for p in extra_protected]
    for p in protected:
        if target.resolve()==p.resolve() or (target.exists() and p.exists() and os.path.samefile(target,p)):
            raise ValueError('output aliases input or teaching asset')
    out.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.'+filename+'.',dir=out)
    try:
        with os.fdopen(fd,'wb') as f:f.write(body);f.flush();os.fsync(f.fileno())
        os.replace(tmp,target)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    return target


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,default=ROOT/'data/regression.csv');ap.add_argument('--config',type=Path,default=ROOT/'data/model_spec.json');ap.add_argument('--output-dir',type=Path,default=Path('outputs'));args=ap.parse_args()
    try:
        x,y,c=load_inputs(args.data,args.config);report=run_experiment(x,y,c);body=serialized(report)
        safe_write(args.output_dir,'report.json',body,[args.data,args.config])
    except (ValueError,OSError,OverflowError) as ex:ap.exit(2,'error: '+str(ex)+'\n')
    print(json.dumps({'status':'completed','paths':len(report['paths']),'report_bytes':len(body)},ensure_ascii=False))

if __name__=='__main__':main()
