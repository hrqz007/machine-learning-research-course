"""041: deterministic regression evidence; no computation or RNG at import.

Public run_experiment validates every raw field before invoking any numerical
experiment, RNG, callback, or writer. Timings live in a separate benchmark report.
"""
from pathlib import Path
from decimal import Decimal, InvalidOperation
import argparse
import copy
import csv
import hashlib
import io
import json
import math
import os
import tempfile
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
CONFIG_KEYS = {'schema_version','hand_lambda','hand_step','initial','lambdas',
               'max_epochs','batch_size','gradient_tolerance','adam_step',
               'seed','fd_steps','coverage_repetitions'}
METHODS = ('gd', 'diagonal_gd', 'minibatch', 'adam', 'newton')
CASES = ('regular', 'collinear', 'scaled')


def real(v, name, lo=-1e8, hi=1e8):
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)):
        raise TypeError(name + ': finite real scalar required, not bool/string/complex')
    try:
        x = float(v)
    except (OverflowError, ValueError) as e:
        raise ValueError(name + ': unrepresentable') from e
    if not math.isfinite(x) or not lo <= x <= hi or (v != 0 and (x == 0 or abs(x) < 1e-150)):
        raise ValueError(name + ': unsupported raw number')
    return x


def integer(v, name, lo, hi):
    if type(v) is not int or not lo <= v <= hi:
        raise ValueError(name + ': integer in declared domain required')
    return v


def seq(v, name, length=None):
    if not isinstance(v, (list, tuple, np.ndarray)) or isinstance(v,np.ndarray) and v.ndim != 1:
        raise TypeError(name + ': one-dimensional sequence required')
    if length is not None and len(v) != length:
        raise ValueError(name + ': wrong length')
    return v


def validate(hand, bench, selection, spec, observer=None):
    """All input validation is deliberately scalar and precedes np.linalg/RNG."""
    if not isinstance(spec, dict) or set(spec) != CONFIG_KEYS:
        raise ValueError('exact configuration key set required')
    s = dict(spec)
    integer(s['schema_version'],'schema_version',1,1)
    s['hand_lambda'] = real(s['hand_lambda'],'hand_lambda',0,10)
    s['hand_step'] = real(s['hand_step'],'hand_step',1e-8,1)
    s['initial'] = [real(x,'initial',-10,10) for x in seq(s['initial'],'initial',2)]
    la = seq(s['lambdas'],'lambdas')
    if not 2 <= len(la) <= 20:
        raise ValueError('2..20 lambdas required')
    s['lambdas'] = [real(x,'lambda',0,100) for x in la]
    if len(set(s['lambdas'])) != len(la) or 0. not in s['lambdas']:
        raise ValueError('unique lambdas including OLS zero required')
    integer(s['max_epochs'],'max_epochs',1,5000)
    integer(s['batch_size'],'batch_size',1,64)
    s['gradient_tolerance'] = real(s['gradient_tolerance'],'gradient_tolerance',1e-12,1e-2)
    s['adam_step'] = real(s['adam_step'],'adam_step',1e-6,1)
    integer(s['seed'],'seed',0,2**32-1)
    fs = seq(s['fd_steps'],'fd_steps')
    if not 2 <= len(fs) <= 20:
        raise ValueError('2..20 finite difference steps required')
    s['fd_steps'] = [real(x,'fd_step',1e-14,.1) for x in fs]
    if len(set(s['fd_steps'])) != len(fs):
        raise ValueError('duplicate finite difference step')
    integer(s['coverage_repetitions'],'coverage_repetitions',30,5000)
    def rows(data, kind):
        if not isinstance(data,(list,tuple)) or not 4 <= len(data) <= 3000:
            raise ValueError(kind+': row list length outside [4,3000]')
        out=[]; ids=set()
        for row in data:
            r=seq(row,kind+' row',4 if kind=='hand' else 9)
            ident=integer(r[0],'id',1,100000)
            if ident in ids:
                raise ValueError(kind+': duplicate ID')
            ids.add(ident)
            if kind=='hand':
                out.append([ident]+[real(x,'hand numeric',-100,100) for x in r[1:]])
            else:
                allowed=CASES if kind=='bench' else ('train','validation','test')
                if type(r[1]) is not str or r[1] not in allowed:
                    raise ValueError(kind+': unknown role')
                out.append([ident,r[1]]+[real(x,kind+' numeric') for x in r[2:]])
        if kind=='hand' and len(out)!=4:
            raise ValueError('hand must contain exactly four rows')
        if kind!='hand':
            for group in allowed:
                count=sum(r[1]==group for r in out)
                if count<8:
                    raise ValueError('each case/split needs at least eight rows')
                if kind=='bench' and s['batch_size']>count:
                    raise ValueError('batch_size exceeds case rows')
        return out
    h=rows(hand,'hand'); b=rows(bench,'bench'); d=rows(selection,'selection')
    if observer is not None and not callable(observer):
        raise TypeError('observer must be callable or None')
    return h,b,d,s


def strict_json(raw):
    def pairs(items):
        result={}
        for k,v in items:
            if k in result:
                raise ValueError('duplicate JSON key: '+k)
            result[k]=v
        return result
    def number(token):
        d=Decimal(token); f=float(d)
        if not math.isfinite(f) or d!=0 and (f==0 or abs(f)<1e-150):
            raise ValueError('unsupported JSON magnitude')
        return f
    def bad(token):
        raise ValueError('nonstandard JSON constant: '+token)
    return json.loads(raw,object_pairs_hook=pairs,parse_float=number,parse_constant=bad)


def decimal_token(token):
    try:
        d=Decimal(token)
    except InvalidOperation as e:
        raise ValueError('invalid CSV decimal') from e
    f=float(d)
    if not d.is_finite() or not math.isfinite(f) or d!=0 and (f==0 or abs(f)<1e-150):
        raise ValueError('unsupported CSV number')
    return f


def read_csv(path, kind):
    reader=csv.reader(io.StringIO(Path(path).read_text(encoding='utf-8')))
    header=['id','x1','x2','y'] if kind=='hand' else ['id','case' if kind=='bench' else 'split']+['x'+str(j) for j in range(1,7)]+['y']
    if next(reader,None)!=header:
        raise ValueError('exact CSV header required')
    result=[]
    for row in reader:
        if len(row)!=len(header):
            raise ValueError('incorrect CSV field count')
        if not row[0].isascii() or not row[0].isdigit():
            raise ValueError('ID must be unsigned decimal integer')
        result.append([int(row[0])]+([decimal_token(x) for x in row[1:]] if kind=='hand' else [row[1]]+[decimal_token(x) for x in row[2:]]))
    return result


def load_inputs(data_dir=None, config=None):
    d=ROOT/'data' if data_dir is None else Path(data_dir)
    c=d/'config.json' if config is None else Path(config)
    s=strict_json(c.read_text(encoding='utf-8'))
    h=read_csv(d/'hand.csv','hand'); b=read_csv(d/'benchmark.csv','bench'); v=read_csv(d/'selection.csv','selection')
    return validate(h,b,v,s)


def _objective(X,y,w,lam):
    r=X@w-y
    return float(r@r/(2*len(y))+lam*(w@w)/2), X.T@r/len(y)+lam*w


def _reference(X,y,lam,details=False):
    n,p=X.shape
    if lam==0:
        w,residuals,rank,sv=np.linalg.lstsq(X,y,rcond=None)
    else:
        w,residuals,rank,sv=np.linalg.lstsq(np.vstack([X/np.sqrt(n),np.sqrt(lam)*np.eye(p)]),
                           np.r_[y/np.sqrt(n),np.zeros(p)],rcond=None)
    return {'w':w,'rank':int(rank),'singular_values':sv.tolist()} if details else w


def _hand_report(h,s):
    X=np.array([r[1:3] for r in h]); y=np.array([r[3] for r in h]); w=np.array(s['initial']); lam=s['hand_lambda']; states=[]
    for k in range(3):
        p=X@w; r=p-y; local=[]
        for i in range(4):
            local.append({'id':h[i][0],'x':X[i].tolist(),'y':float(y[i]),'prediction':float(p[i]),'residual':float(r[i]),
                          'half_square':float(r[i]**2/2),'mean_loss':float(r[i]**2/8),'d_loss_d_residual':float(r[i]),
                          'd_residual_d_prediction':1.,'d_prediction_d_w':X[i].tolist(),'gradient_contribution':(r[i]*X[i]/4).tolist()})
        data_grad=X.T@r/4; penalty_grad=lam*w; total=data_grad+penalty_grad
        st={'state':k,'w':w.tolist(),'rows':local,'data_loss':float(r@r/8),'penalty':float(lam*(w@w)/2),
            'objective':float(r@r/8+lam*(w@w)/2),'data_gradient':data_grad.tolist(),'penalty_gradient':penalty_grad.tolist(),'gradient':total.tolist()}
        if k<2:
            st['update']=(-s['hand_step']*total).tolist(); w=w-s['hand_step']*total; st['next_w']=w.tolist()
        states.append(st)
    refs={}
    for name,l in [('ols',0.),('ridge',lam)]:
        ref=_reference(X,y,l); f,g=_objective(X,y,ref,l)
        refs[name]={'lambda':l,'w':ref.tolist(),'objective':f,'gradient_norm':float(np.linalg.norm(g))}
    checks=[]
    for point in ([.2,-.4],[.8,.1],[-.3,.7]):
        w=np.array(point); f,g=_objective(X,y,w,lam)
        for step in s['fd_steps']:
            fd=[]
            for j in range(2):
                plus=w.copy(); minus=w.copy(); plus[j]+=step; minus[j]-=step
                fd.append((_objective(X,y,plus,lam)[0]-_objective(X,y,minus,lam)[0])/(2*step))
            checks.append({'point':point,'h':step,'analytic':g.tolist(),'finite_difference':fd,
                           'relative_error':float(np.linalg.norm(g-fd)/max(1.,np.linalg.norm(g))),
                           'wrong_sign_error':float(np.linalg.norm(-g-fd)/max(1.,np.linalg.norm(g)))})
    return {'states':states,'H':(X.T@X/4).tolist(),'c':(X.T@y/4).tolist(),'references':refs,'gradient_checks':checks}


def _solve(X,y,lam,method,s,seed):
    """Prevalidated internal numerical runner, includes all diagnostic setup."""
    n,p=X.shape; H=X.T@X/n+lam*np.eye(p); c=X.T@y/n
    eig=np.linalg.eigvalsh(H); L=float(eig[-1]); mu=float(eig[0]); reference=_reference(X,y,lam,details=True); ref=reference['w']
    fstar,_=_objective(X,y,ref,lam); w=np.zeros(p); m=np.zeros(p); v=np.zeros(p)
    scale=max(1.,float(np.linalg.norm(c)))
    if L<=0:
        raise ValueError('nonpositive maximum curvature')
    # Diagonal preconditioning preserves the original objective including lambda.
    D=np.maximum(np.diag(H),1e-15)
    Lpre=float(np.linalg.eigvalsh(H/np.sqrt(D[:,None]*D[None,:]))[-1])
    rng=np.random.default_rng(seed) if method=='minibatch' else None
    cost={'gram_formations':1,'gram_rows':n,'spectrum_solves':2,'reference_lstsq':1,
          'reference_rows':n if lam==0 else n+p,'objective_calls':1,'objective_rows':n,
          'diagnostic_prediction_rows':0,
          'full_gradient_calls':1,'full_gradient_rows':n,'batch_gradient_calls':0,'batch_gradient_rows':0,
          'linear_solves':0,'attempted_updates':0,'committed_updates':0}
    def monitor(w,epoch):
        cost['objective_calls']+=1; cost['objective_rows']+=n
        cost['full_gradient_calls']+=1; cost['full_gradient_rows']+=n
        f,g=_objective(X,y,w,lam); e=w-ref; gap=float(e@H@e/2)
        cost['diagnostic_prediction_rows']+=n
        rec={'epoch':epoch,'objective':f,'gradient_norm':float(np.linalg.norm(g)),
             'relative_gradient':float(np.linalg.norm(g)/scale),'quadratic_gap':gap,
             'subtractive_gap':float(f-fstar),'coefficient_distance':float(np.linalg.norm(e)),
             'prediction_distance_rms':float(np.linalg.norm(X@e)/np.sqrt(n)),
             'gradient_rows':cost['full_gradient_rows']+cost['batch_gradient_rows'],'w':w.tolist()}
        return rec,g
    old,g=monitor(w,0); trace=[old]; status='budget_exhausted'; failure=None
    for epoch in range(1,s['max_epochs']+1):
        if old['relative_gradient']<=s['gradient_tolerance']:
            status='gradient_tolerance'; break
        if method=='minibatch':
            # Independent uniform subsets at each step (not random reshuffling).
            for b in range(math.ceil(n/s['batch_size'])):
                ix=rng.choice(n,size=s['batch_size'],replace=False)
                cost['batch_gradient_calls']+=1; cost['batch_gradient_rows']+=len(ix)
                gg=X[ix].T@(X[ix]@w-y[ix])/len(ix)+lam*w
                eta=.2/L/(1+.05*(epoch-1))
                cost['attempted_updates']+=1; w=w-eta*gg; cost['committed_updates']+=1
        else:
            cost['attempted_updates']+=1
            if method=='gd': delta=-g/L
            elif method=='diagonal_gd': delta=-g/D/Lpre
            elif method=='adam':
                m=.9*m+.1*g; v=.999*v+.001*g*g
                delta=-s['adam_step']*(m/(1-.9**epoch))/(np.sqrt(v/(1-.999**epoch))+1e-8)
            elif method=='newton':
                cost['linear_solves']+=1
                delta=-np.linalg.lstsq(H,g,rcond=None)[0]
            elif method=='gd_unsafe': delta=-3.*g/L
            else: raise ValueError('unknown method')
            proposal=w+delta
            if not np.isfinite(proposal).all():
                status='numerical_failure'; failure={'epoch':epoch,'reason':'nonfinite proposed parameters'}; break
            if method=='gd_unsafe':
                trial,tg=monitor(proposal,epoch)
                if trial['objective']>old['objective']:
                    status='objective_increase_rejected'; failure=trial; break
                w=proposal; cost['committed_updates']+=1; old,g=trial,tg; trace.append(old); continue
            w=proposal; cost['committed_updates']+=1
        new,g=monitor(w,epoch); trace.append(new); old=new
        if not math.isfinite(new['objective']) or not math.isfinite(new['gradient_norm']):
            status='numerical_failure'; break
        if new['relative_gradient']<=s['gradient_tolerance']:
            status='gradient_tolerance'; break
    return {'method':method,'lambda':lam,'status':status,'n':n,'p':p,'mu':mu,'L':L,
            'condition':float(L/mu) if mu>0 else None,'preconditioned_L':Lpre,'reference':ref.tolist(),'reference_rank':reference['rank'],'reference_singular_values':reference['singular_values'],
            'reference_objective':fstar,'final':old,'failure':failure,'cost':cost,'trace':trace}


def _arrays(rows, group):
    selected=[r for r in rows if r[1]==group]
    return np.array([r[2:-1] for r in selected],dtype=float),np.array([r[-1] for r in selected],dtype=float)


def _optimizer_suite(rows,s):
    runs=[]
    for case in CASES:
        X,y=_arrays(rows,case); X=X-X.mean(axis=0); y=y-y.mean()
        for lam in (0.,.1):
            for method in METHODS:
                result=_solve(X,y,lam,method,s,s['seed']); result['case']=case; runs.append(result)
    X,y=_arrays(rows,'scaled'); X=X-X.mean(axis=0); y=y-y.mean()
    failure=_solve(X,y,.1,'gd_unsafe',s,s['seed']); failure['case']='scaled'
    return {'runs':runs,'injected_failure':failure,'fixed_stress_lambdas':[0.,.1]}


def _select(train,validation,s):
    """Test labels/features are not accepted by this selection function."""
    X,y=train; V,z=validation; mu=X.mean(axis=0); scales=X.std(axis=0,ddof=0)
    scales=np.where(scales==0.,1.,scales); Z=(X-mu)/scales; W=(V-mu)/scales; ym=float(y.mean()); yc=y-ym
    candidates=[]
    for lam in s['lambdas']:
        w=_reference(Z,yc,lam); f,g=_objective(Z,yc,w,lam)
        rho=float(np.linalg.norm(g)/max(1.,np.linalg.norm(Z.T@yc/len(y))))
        candidates.append({'lambda':lam,'w_standardized':w.tolist(),'objective':f,'relative_gradient':rho,
                           'admitted':rho<=s['gradient_tolerance'],'validation_mse':float(np.mean((ym+W@w-z)**2))})
    admitted=[r for r in candidates if r['admitted']]
    if not admitted:
        return {'status':'selection_failed','candidates':candidates}
    best=min(admitted,key=lambda r:(r['validation_mse'],-r['lambda']))
    beta=np.array(best['w_standardized'])/scales; intercept=ym-float(mu@beta)
    return {'status':'locked','candidates':candidates,'selected_lambda':best['lambda'],
            'mean':mu.tolist(),'scale':scales.tolist(),'target_mean':ym,
            'w_standardized':best['w_standardized'],'w_original':beta.tolist(),'intercept_original':intercept,
            'selection_rule':'minimum exact validation MSE; then larger lambda; no refit'}


def _selection_report(rows,s):
    train=_arrays(rows,'train'); validation=_arrays(rows,'validation')
    locked=_select(train,validation,s)
    locked_bytes=json.dumps(locked,sort_keys=True,allow_nan=False).encode()
    digest=hashlib.sha256(locked_bytes).hexdigest()
    if locked['status']!='locked': return {'lock':locked,'lock_sha256':digest,'test':None}
    # Only after serialization of the immutable choice do we access test arrays.
    Xtest,ytest=_arrays(rows,'test'); pred=locked['intercept_original']+Xtest@np.array(locked['w_original'])
    ols=next(r for r in locked['candidates'] if r['lambda']==0.)
    baseline=locked['target_mean']+(Xtest-locked['mean'])/locked['scale']@np.array(ols['w_standardized'])
    return {'lock':locked,'lock_sha256':digest,'test':{'n':len(ytest),'selected_mse':float(np.mean((pred-ytest)**2)),
            'ols_mse':float(np.mean((baseline-ytest)**2)),'predictions':pred.tolist(),'residuals':(pred-ytest).tolist()},
            'interval':'No post-selection Ridge interval claimed; separate OLS coverage experiment only.'}


def _coverage(s):
    from scipy.stats import t
    n=32; x=np.linspace(-2,2,n); A=np.column_stack([np.ones(n),x]); x0=np.array([1.,1.5]); beta=np.array([1.,.8]); true_mean=float(x0@beta)
    leverage=float(x0@np.linalg.solve(A.T@A,x0)); crit=float(t.ppf(.975,n-2)); rng=np.random.default_rng(s['seed']+1); rows=[]
    for _ in range(s['coverage_repetitions']):
        y=A@beta+rng.normal(0,.5,n); bhat=np.linalg.lstsq(A,y,rcond=None)[0]; r=y-A@bhat; sd=float(np.sqrt(r@r/(n-2)))
        fitted=float(x0@bhat); mean_half=crit*sd*np.sqrt(leverage); pred_half=crit*sd*np.sqrt(1+leverage)
        new_y=true_mean+rng.normal(0,.5); shifted_y=true_mean+rng.normal(0,1.5)
        rows.append({'fitted':fitted,'s':sd,'mean_half':float(mean_half),'prediction_half':float(pred_half),'new_y':float(new_y),'shifted_y':float(shifted_y),
                     'mean_covered':bool(abs(fitted-true_mean)<=mean_half),'prediction_covered':bool(abs(fitted-new_y)<=pred_half),
                     'wrong_mean_as_prediction':bool(abs(fitted-new_y)<=mean_half),'shifted_prediction_covered':bool(abs(fitted-shifted_y)<=pred_half)})
    keys=('mean_covered','prediction_covered','wrong_mean_as_prediction','shifted_prediction_covered')
    summary={k:{'covered':sum(r[k] for r in rows),'total':len(rows),'rate':sum(r[k] for r in rows)/len(rows)} for k in keys}
    return {'n':n,'p_including_intercept':2,'degrees_freedom':n-2,'x0':x0.tolist(),'true_mean':true_mean,'leverage':leverage,'t_critical':crit,'rows':rows,'summary':summary}


def run_experiment(hand,bench,selection,spec,observer=None):
    h,b,d,s=validate(hand,bench,selection,spec,observer)
    result={'unit':'041','runtime':{'python':sys.version.split()[0],'numpy':np.__version__},'configuration':s,
            'hand':_hand_report(h,s),'optimizers':_optimizer_suite(b,s),'selection':_selection_report(d,s),'coverage':_coverage(s)}
    if observer is not None:
        observer(copy.deepcopy(result))
    return result


def check_destination(path,inputs=()):
    p=Path(path).absolute()
    if not p.parent.is_dir(): raise ValueError('output parent must already exist')
    # Reject symlink ancestors too; do not use resolve to silently accept them.
    if any(q.is_symlink() for q in [p,*p.parents]): raise ValueError('symlink output or parent forbidden')
    if p.exists() and (not p.is_file() or p.stat().st_nlink!=1): raise ValueError('regular singly-linked output required')
    protected=[q for q in ROOT.rglob('*') if q.is_file() and '__pycache__' not in q.parts and 'notebook_results' not in q.parts]
    protected.extend(Path(q) for q in inputs)
    for q in protected:
        if p.resolve()==q.resolve() or p.exists() and q.exists() and os.path.samefile(p,q):
            raise ValueError('refuse overwriting input or shipped asset')
    return p


def atomic_json(path,value,inputs=()):
    p=check_destination(path,inputs)
    payload=json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n'
    temp=None
    try:
        with tempfile.NamedTemporaryFile('w',encoding='utf-8',dir=p.parent,prefix='.unit041-',suffix='.tmp',delete=False) as f:
            temp=Path(f.name); f.write(payload); f.flush(); os.fsync(f.fileno())
        # Recheck destination immediately before commit (single-user local contract).
        check_destination(p,inputs); os.replace(temp,p)
    finally:
        if temp is not None and temp.exists(): temp.unlink()


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--data-dir',type=Path); p.add_argument('--config',type=Path); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    h,b,d,s=load_inputs(a.data_dir,a.config)
    directory=ROOT/'data' if a.data_dir is None else a.data_dir
    inputs=[directory/n for n in ['hand.csv','benchmark.csv','selection.csv','config.json']]+([a.config] if a.config else [])
    check_destination(a.output,inputs)
    result=run_experiment(h,b,d,s); atomic_json(a.output,result,inputs)
    print(json.dumps({'unit':'041','selected_lambda':result['selection']['lock'].get('selected_lambda'),'failed_runs':sum(r['status']!='gradient_tolerance' for r in result['optimizers']['runs']),'injected_failure':result['optimizers']['injected_failure']['status']},sort_keys=True))

if __name__=='__main__': main()
