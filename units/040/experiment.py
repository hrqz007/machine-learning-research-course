"""ML040: fixed-objective regularization with validation-only selection.

All data are synthetic. No hidden test score participates in fitting or choice.
The two-dimensional branch solver is a reference, not a high-dimensional solver.
"""
from pathlib import Path
from decimal import Decimal
import argparse, csv, hashlib, io, itertools, json, math, os, tempfile, warnings
import numpy as np

ROOT=Path(__file__).resolve().parent
FIELDS={'kind','hand_ridge_lambda','hand_learning_rate','hand_steps','hand_initial',
 'hand_lasso_lambda','hand_elastic_lambda','hand_elastic_l1_ratio','path_lambdas',
 'selection_lambdas','selection_l1_ratios','selection_seed','train_rows','validation_rows',
 'test_rows','features','correlation','noise_std','solver_tolerance','solver_max_iter',
 'monte_carlo_seed','monte_carlo_repetitions','monte_carlo_noise_std','monte_carlo_truth','monte_carlo_lambdas'}
FAMILIES=('ridge','lasso','elastic')


def scalar(x,name,lo=-1e6,hi=1e6,integer=False):
    if isinstance(x,(bool,np.bool_)) or not isinstance(x,(int,float,np.integer,np.floating)):
        raise ValueError(name+' requires a real numeric scalar, not bool/string/complex')
    if integer and not isinstance(x,(int,np.integer)):raise ValueError(name+' requires integer')
    try:v=float(x)
    except (ValueError,OverflowError) as ex:raise ValueError(name+' unrepresentable scalar') from ex
    if v==0 and x!=0:raise ValueError(name+' nonzero lost during float conversion')
    if not math.isfinite(v) or not lo<=v<=hi or (v!=0 and abs(v)<1e-100):raise ValueError(name+' outside teaching range')
    return int(x) if integer else v


def vector(a,name,d=None):
    if not isinstance(a,(list,tuple,np.ndarray)) or (isinstance(a,np.ndarray) and a.ndim!=1):raise ValueError(name+' must be vector')
    if d is not None and len(a)!=d:raise ValueError(name+' dimension mismatch')
    return np.array([scalar(v,name) for v in a],dtype=float)


def design(a,name,d=None):
    if not isinstance(a,(list,tuple,np.ndarray)) or not 1<=len(a)<=5000:raise ValueError(name+' invalid rows')
    if isinstance(a,np.ndarray) and a.ndim!=2:raise ValueError(name+' requires matrix')
    rows=[vector(row,name,d) for row in a]
    width=len(rows[0])
    if not 1<=width<=64 or any(len(row)!=width for row in rows):raise ValueError(name+' invalid width')
    return np.array(rows)


def xy(X,y,d=None):
    X=design(X,'X',d); y=vector(y,'y',len(X));return X,y


def finite(a):
    if not np.isfinite(a).all():raise ValueError('nonfinite arithmetic result')
    return a


def parse_number(s):
    d=Decimal(s);v=float(d)
    if not math.isfinite(v) or (d!=0 and (v==0 or abs(v)<1e-100)):raise ValueError('nonfinite/underflow numeric literal')
    return v


def pairs_object(items):
    result={}
    for k,v in items:
        if k in result:raise ValueError('duplicate JSON key')
        result[k]=v
    return result


def read_json(path):
    return json.loads(Path(path).read_text(),parse_float=parse_number,object_pairs_hook=pairs_object,
      parse_constant=lambda s:(_ for _ in ()).throw(ValueError('nonfinite JSON')))


def validate_config(c):
    if not isinstance(c,dict) or set(c)!=FIELDS or c['kind']!='regularization_shrinkage_v1':raise ValueError('configuration fields/kind mismatch')
    q=dict(c)
    for k in ('hand_ridge_lambda','hand_lasso_lambda','hand_elastic_lambda'):
        q[k]=scalar(c[k],k,0,100)
    q['hand_elastic_l1_ratio']=scalar(c['hand_elastic_l1_ratio'],'rho',0,1)
    q['hand_learning_rate']=scalar(c['hand_learning_rate'],'learning rate',1e-8,1)
    for k,lo,hi in [('hand_steps',1,10),('selection_seed',0,2**32-1),('train_rows',2,1000),
      ('validation_rows',1,1000),('test_rows',1,1000),('features',1,64),('solver_max_iter',1,100000),
      ('monte_carlo_seed',0,2**32-1),('monte_carlo_repetitions',2,10000)]:
        q[k]=scalar(c[k],k,lo,hi,True)
    for k,lo,hi in [('correlation',0,.999),('noise_std',0,100),('solver_tolerance',1e-14,.01),('monte_carlo_noise_std',0,100)]:
        q[k]=scalar(c[k],k,lo,hi)
    for k,lo,hi in [('path_lambdas',0,100),('selection_lambdas',1e-8,100),('selection_l1_ratios',0,1),('monte_carlo_lambdas',0,100)]:
        if not isinstance(c[k],list) or not 1<=len(c[k])<=30:raise ValueError(k+' invalid list')
        q[k]=[scalar(v,k,lo,hi) for v in c[k]]
        if q[k]!=sorted(set(q[k])):raise ValueError(k+' must strictly increase')
    for k in ('hand_initial','monte_carlo_truth'):q[k]=vector(c[k],k,2).tolist()
    return q


def _csv(path,header):
    reader=csv.DictReader(io.StringIO(Path(path).read_text()))
    if reader.fieldnames!=header:raise ValueError('CSV header mismatch')
    result=[]; seen=set()
    for row in reader:
        if set(row)!=set(header) or any(v is None for v in row.values()):raise ValueError('malformed CSV row')
        sid=row['id']
        if not sid or len(sid)>40 or not sid.isascii() or not all(ch.isalnum() or ch in '_-' for ch in sid) or sid in seen:raise ValueError('bad/duplicate sample ID')
        seen.add(sid);result.append(row)
    return result


def load_inputs(main_path=None,selection_path=None,config_path=None):
    c=validate_config(read_json(config_path or ROOT/'data/model_spec.json'))
    rows=_csv(main_path or ROOT/'data/regression.csv',['id','x1','x2','y'])
    X,y=xy([[parse_number(r['x1']),parse_number(r['x2'])] for r in rows],[parse_number(r['y']) for r in rows],2)
    header=['split','id']+[f'x{i:02}' for i in range(1,c['features']+1)]+['y']
    rows=_csv(selection_path or ROOT/'data/selection.csv',header);splits={}
    if any(r['split'] not in ('train','validation','test') for r in rows):raise ValueError('unknown split')
    for split,key in [('train','train_rows'),('validation','validation_rows'),('test','test_rows')]:
        rr=[r for r in rows if r['split']==split]
        if len(rr)!=c[key]:raise ValueError('split row count mismatch')
        a,b=xy([[parse_number(r[k]) for k in header[2:-1]] for r in rr],[parse_number(r['y']) for r in rr],c['features'])
        splits[split]={'X':a,'y':b,'ids':[r['id'] for r in rr]}
    return X,y,splits,c


def row_ledger(X,y,beta,lam=0.,rho=0.):
    X,y=xy(X,y);beta=vector(beta,'beta',X.shape[1]);lam=scalar(lam,'lambda',0,100);rho=scalar(rho,'rho',0,1)
    pred=finite(X@beta);r=finite(pred-y);n=len(y);contrib=r[:,None]*X/n
    F=float(r@r/(2*n));l1=float(lam*rho*np.abs(beta).sum());l2=float(lam*(1-rho)*(beta@beta)/2)
    g=finite(contrib.sum(axis=0)); smooth_g=g+lam*(1-rho)*beta
    residual=np.where(beta!=0,np.abs(smooth_g+lam*rho*np.sign(beta)),np.maximum(np.abs(smooth_g)-lam*rho,0))
    return {'beta':beta.tolist(),'data_loss':F,'l1_penalty':l1,'l2_penalty':l2,'objective':F+l1+l2,
      'data_gradient':g.tolist(),'l2_gradient':(lam*(1-rho)*beta).tolist(),'smooth_gradient':smooth_g.tolist(),
      'kkt_components':residual.tolist(),'kkt_inf':float(max(residual)),
      'rows':[{'id':i+1,'x':X[i].tolist(),'y':float(y[i]),'prediction':float(pred[i]),'residual':float(r[i]),
       'half_square':float(r[i]**2/2),'loss_contribution':float(r[i]**2/(2*n)),
       'local_dloss_dr':float(r[i]),'jacobian':X[i].tolist(),'gradient_contribution':contrib[i].tolist()} for i in range(n)]}


def ridge_solve(X,y,lam):
    X,y=xy(X,y);lam=scalar(lam,'lambda',0,100)
    if lam==0:return finite(np.linalg.lstsq(X,y,rcond=None)[0])
    H=X.T@X/len(y);c=X.T@y/len(y)
    return finite(np.linalg.solve(H+lam*np.eye(X.shape[1]),c))


def ridge_augmented(X,y,lam):
    X,y=xy(X,y);lam=scalar(lam,'lambda',0,100);n,d=X.shape
    return finite(np.linalg.lstsq(np.vstack((X/np.sqrt(n),np.sqrt(lam)*np.eye(d))),np.r_[y/np.sqrt(n),np.zeros(d)],rcond=None)[0])


def enumerate_2d(X,y,lam,rho):
    X,y=xy(X,y,2);lam=scalar(lam,'lambda',0,100);rho=scalar(rho,'rho',0,1)
    H=X.T@X/len(y)+lam*(1-rho)*np.eye(2);c=X.T@y/len(y);valid=[]
    # Require nonsingular active blocks: full-rank main example or positive L2.
    for signs in itertools.product((-1,0,1),repeat=2):
        active=np.flatnonzero(signs);b=np.zeros(2)
        try:
            if len(active):b[active]=np.linalg.solve(H[np.ix_(active,active)],c[active]-lam*rho*np.array(signs)[active])
        except np.linalg.LinAlgError:continue
        if any(b[j]*signs[j]<=0 for j in active):continue
        g=H@b-c
        if any(abs(g[j])>lam*rho+1e-10 for j in range(2) if signs[j]==0):continue
        k=row_ledger(X,y,b,lam,rho)
        if k['kkt_inf']<=1e-8:valid.append({'signs':list(signs),'beta':b.tolist(),'objective':k['objective'],'kkt_inf':k['kkt_inf']})
    if not valid:raise ValueError('no unique nonsingular 2D branch; use mature solver for degenerate L1')
    return min(valid,key=lambda r:r['objective']),valid


def fit_library(X,y,lam,family,cfg,rho=.5):
    # Validate all raw arguments before importing/calling any solver.
    c=validate_config(cfg);X,y=xy(X,y);lam=scalar(lam,'lambda',0,100);rho=scalar(rho,'rho',0,1)
    if family not in FAMILIES:raise ValueError('unknown model family')
    from sklearn.linear_model import Ridge,Lasso,ElasticNet
    from sklearn.exceptions import ConvergenceWarning
    effective_rho=0. if family=='ridge' else 1. if family=='lasso' else rho
    alpha=len(y)*lam if effective_rho==0 else lam
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            if lam==0:
                beta=np.linalg.lstsq(X,y,rcond=None)[0];n_iter=None;gap=None;solver='numpy.lstsq minimum-norm endpoint';alpha=0.
            elif effective_rho==0:
                m=Ridge(alpha=alpha,fit_intercept=False,solver='svd',tol=c['solver_tolerance']);m.fit(X,y)
                beta=m.coef_;n_iter=None if m.n_iter_ is None else int(m.n_iter_);gap=None;solver='sklearn.Ridge(svd); tol ignored'
            else:
                cls=Lasso if effective_rho==1 else ElasticNet
                kw=dict(alpha=alpha,fit_intercept=False,tol=c['solver_tolerance'],max_iter=c['solver_max_iter'],selection='cyclic',positive=False,precompute=False,warm_start=False)
                if effective_rho!=1:kw['l1_ratio']=effective_rho
                m=cls(**kw);m.fit(X,y);beta=m.coef_;n_iter=int(m.n_iter_);gap=float(m.dual_gap_);solver='sklearn.'+cls.__name__+'(cyclic)'
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as ex:
        return {'family':family,'lambda':lam,'l1_ratio':effective_rho,'library_alpha':alpha,
          'status':'solver_error','error_type':type(ex).__name__,'error':str(ex),
          'beta':None,'n_iter':None,'dual_gap':None,'warnings':[], 'kkt_inf':None}
    finite(beta);ledger=row_ledger(X,y,beta,lam,effective_rho)
    # Common absolute KKT gate is a teaching acceptance rule, distinct from tol.
    gate=1e-7;warning_records=[{'category':type(w.message).__name__,'message':str(w.message)} for w in caught]
    status='converged' if ledger['kkt_inf']<=gate and not any(issubclass(w.category,ConvergenceWarning) for w in caught) else 'not_converged'
    return {'family':family,'lambda':lam,'l1_ratio':effective_rho,'library_alpha':alpha,'solver':solver,
      'status':status,'warnings':warning_records,'n_iter':n_iter,'dual_gap':gap,'kkt_acceptance':gate,
      'beta':beta.tolist(),'data_loss':ledger['data_loss'],'l1_penalty':ledger['l1_penalty'],
      'l2_penalty':ledger['l2_penalty'],'objective':ledger['objective'],'kkt_inf':ledger['kkt_inf'],
      'kkt_components':ledger['kkt_components'],'stored_exact_zeros':int(np.sum(beta==0))}


def select_candidate(trainX,trainy,valX,valy,cfg):
    c=validate_config(cfg);X,y=xy(trainX,trainy,c['features']);V,v=xy(valX,valy,c['features'])
    # No test or truth parameter. The one scaler is fit ONLY on training rows.
    from sklearn.preprocessing import StandardScaler
    scaler=StandardScaler();Z=scaler.fit_transform(X);W=scaler.transform(V);ym=float(y.mean());yc=y-ym
    mu=scaler.mean_;scale=scaler.scale_;candidates=[]
    for family in FAMILIES:
        ratios=c['selection_l1_ratios'] if family=='elastic' else [0. if family=='ridge' else 1.]
        for lam in c['selection_lambdas']:
            for rho in ratios:
                fit=fit_library(Z,yc,lam,family,c,rho)
                if fit['beta'] is None:
                    fit.update(coefficient_standardized=fit.pop('beta'),coefficient_raw=None,intercept_raw=None,validation_mse=None,train_mse=None)
                    candidates.append(fit);continue
                gamma=np.array(fit['beta']);raw=gamma/scale;intercept=ym-mu@raw
                validation_prediction=finite(intercept+V@raw);standard_prediction=ym+W@gamma
                fit.update(coefficient_standardized=fit.pop('beta'),coefficient_raw=raw.tolist(),intercept_raw=float(intercept),
                  validation_mse=float(np.mean((validation_prediction-v)**2)),
                  train_mse=2*fit['data_loss'],prediction_coordinate_max_error=float(np.max(np.abs(validation_prediction-standard_prediction))))
                candidates.append(fit)
    eligible=[r for r in candidates if r['status']=='converged']
    def key(r):return (r['validation_mse'],-r['lambda'],FAMILIES.index(r['family']),-r['l1_ratio'])
    chosen=min(eligible,key=key) if eligible else None
    baseline=fit_library(Z,yc,0.,'ridge',c);gamma=np.array(baseline['beta']);raw=gamma/scale
    baseline.update(coefficient_standardized=baseline.pop('beta'),coefficient_raw=raw.tolist(),intercept_raw=float(ym-mu@raw))
    baseline['validation_mse']=float(np.mean((baseline['intercept_raw']+V@raw-v)**2))
    out={'status':'selected' if chosen else 'selection_failed','tie_rule':'validation MSE, then descending lambda, ridge/lasso/elastic, descending rho; exact ties only',
      'preprocessing':{'mean':mu.tolist(),'scale':scale.tolist(),'variance_ddof0':scaler.var_.tolist(),'y_mean':ym,'fit_rows':len(X)},
      'candidates':candidates,'chosen':chosen,'baseline_ols':baseline,
      'family_validation_winners':{f:min([r for r in eligible if r['family']==f],key=key) if any(r['family']==f for r in eligible) else None for f in FAMILIES}}
    out['choice_fit_sha256']=hashlib.sha256(canonical_bytes(out)).hexdigest()
    return out


def final_evaluate(chosen,testX,testy):
    if not isinstance(chosen,dict):raise ValueError('a frozen selected model is required')
    beta=vector(chosen['coefficient_raw'],'chosen beta');intercept=scalar(chosen['intercept_raw'],'intercept')
    X,y=xy(testX,testy,len(beta));pred=finite(intercept+X@beta)
    return {'rows':len(y),'mse':float(np.mean((pred-y)**2)),'prediction':pred.tolist(),'squared_errors':((pred-y)**2).tolist()}


def monte_carlo(X,cfg):
    c=validate_config(cfg);X=design(X,'fixed X',2);truth=vector(c['monte_carlo_truth'],'truth',2)
    sigma=c['monte_carlo_noise_std'];n=len(X);H=X.T@X/n;R=c['monte_carlo_repetitions']
    if np.linalg.matrix_rank(X)<2:raise ValueError('MC analytical zero-lambda case requires full column rank')
    rng=np.random.default_rng(c['monte_carlo_seed']);noise=rng.normal(0,sigma,size=(R,n));Y=X@truth+noise
    records=[]
    for lam in c['monte_carlo_lambdas']:
        A=np.linalg.solve(H+lam*np.eye(2),np.eye(2));mean=A@H@truth;bias=mean-truth;cov=sigma**2/n*A@H@A
        B=Y@X@A.T/n;emp_mean=B.mean(axis=0);centered=B-emp_mean;emp_cov=centered.T@centered/R
        coefficient_mse=float(bias@bias+np.trace(cov));mean_prediction=float(bias@H@bias+np.trace(H@cov))
        records.append({'lambda':lam,'analytic_mean':mean.tolist(),'analytic_bias':bias.tolist(),'analytic_covariance':cov.tolist(),
          'analytic_coefficient_mse':coefficient_mse,'analytic_noiseless_prediction_mse_on_fixed_X':mean_prediction,
          'analytic_new_label_mse_on_fixed_X':mean_prediction+sigma**2,'empirical_mean':emp_mean.tolist(),
          'empirical_covariance_divisor_R':emp_cov.tolist(),'empirical_coefficient_mse':float(np.mean(np.sum((B-truth)**2,axis=1))),
          'empirical_noiseless_prediction_mse_on_fixed_X':float(np.mean(((B-truth)@X.T)**2)),
          'all_fitted_coefficients':B.tolist()})
    return {'seed':c['monte_carlo_seed'],'repetitions':R,'fixed_design':X.tolist(),'truth':truth.tolist(),'noise_std':sigma,
      'paired_noise_same_across_lambda':noise.tolist(),'records':records,'role':'mechanism diagnostic, never used to select model'}


def main_report(X,y,splits,cfg):
    c=validate_config(cfg);X,y=xy(X,y,2)
    # Full split validation occurs before any solver/RNG, including the last test field.
    if not isinstance(splits,dict) or set(splits)!= {'train','validation','test'}:raise ValueError('split keys mismatch')
    checked={}
    for name in ('train','validation','test'):
        if not isinstance(splits[name],dict) or not {'X','y'}<=set(splits[name]):raise ValueError('malformed split')
        checked[name]=xy(splits[name]['X'],splits[name]['y'],c['features'])
    beta=vector(c['hand_initial'],'hand beta',2);trace=[]
    for k in range(c['hand_steps']+1):
        state=row_ledger(X,y,beta,c['hand_ridge_lambda'],0);state['k']=k;trace.append(state)
        if k<c['hand_steps']:beta=finite(beta-c['hand_learning_rate']*np.array(state['smooth_gradient']))
    main_fits={}
    for family,lam,rho in [('ridge',c['hand_ridge_lambda'],0),('lasso',c['hand_lasso_lambda'],1),('elastic',c['hand_elastic_lambda'],c['hand_elastic_l1_ratio'])]:
        fit=fit_library(X,y,lam,family,c,rho);fit['full_row_ledger']=row_ledger(X,y,fit['beta'],lam,rho)
        if family=='ridge':
            fit['solve_reference']=ridge_solve(X,y,lam).tolist();fit['augmented_reference']=ridge_augmented(X,y,lam).tolist()
        else:fit['enumerated_reference'],fit['valid_sign_branches']=enumerate_2d(X,y,lam,rho)
        main_fits[family]=fit
    paths={family:[fit_library(X,y,lam,family,c,c['hand_elastic_l1_ratio']) for lam in c['path_lambdas']] for family in FAMILIES}
    selected=select_candidate(*checked['train'],*checked['validation'],c)
    test={'chosen':final_evaluate(selected['chosen'],*checked['test']) if selected['chosen'] else None,
      'predeclared_ols_baseline':final_evaluate(selected['baseline_ols'],*checked['test'])}
    import sklearn,scipy
    return {'unit':'040','versions':{'numpy':np.__version__,'scipy':scipy.__version__,'scikit_learn':sklearn.__version__},
      'config':c,'hand_trace':trace,'main_fits':main_fits,'paths':paths,'selection':selected,'final_test':test,'monte_carlo':monte_carlo(X,c)}


def canonical_bytes(obj):
    return (json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def verify_fixture(base=ROOT):
    base=Path(base);contract=read_json(base/'data_integrity.json')
    for rel,h in contract['sha256'].items():
        if hashlib.sha256((base/rel).read_bytes()).hexdigest()!=h:raise ValueError('teaching fixture changed: '+rel)
    return True


def safe_write(path,payload,protected=()):
    """Serialize first; reject assets, symlink components and multiply linked files."""
    data=canonical_bytes(payload);p=Path(path).absolute()
    if any(part.is_symlink() for part in [p,*p.parents]):raise ValueError('output symlink forbidden')
    q=p.resolve();allowed=ROOT/'outputs'
    if q==ROOT or (ROOT in q.parents and allowed not in q.parents):raise ValueError('output would overwrite public asset')
    if q in [Path(v).resolve() for v in protected]:raise ValueError('output is an input')
    if p.exists() and (not p.is_file() or p.stat().st_nlink>1):raise ValueError('output must be ordinary singly-linked file')
    made=[];parent=p.parent
    while not parent.exists():made.append(parent);parent=parent.parent
    temporary=None
    try:
        p.parent.mkdir(parents=True,exist_ok=True)
        fd,temporary=tempfile.mkstemp(prefix='.ml040-',dir=p.parent)
        with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        os.replace(temporary,p);temporary=None
    finally:
        if temporary is not None and Path(temporary).exists():Path(temporary).unlink()
        for directory in made:
            try:directory.rmdir()
            except OSError:pass


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default=str(ROOT/'outputs/result.json'))
    ap.add_argument('--data');ap.add_argument('--selection');ap.add_argument('--config');args=ap.parse_args()
    # Default fixture check catches accidental edits; custom paths still get full schema checks.
    if args.data is None and args.selection is None and args.config is None:verify_fixture()
    X,y,s,c=load_inputs(args.data,args.selection,args.config);report=main_report(X,y,s,c)
    safe_write(args.out,report,[v for v in (args.data,args.selection,args.config) if v])
    print(json.dumps({'status':report['selection']['status'],'choice':({k:report['selection']['chosen'][k] for k in ('family','lambda','l1_ratio')} if report['selection']['chosen'] else None),
      'core_sha256':hashlib.sha256(canonical_bytes(report)).hexdigest()},ensure_ascii=False))

if __name__=='__main__':main()
