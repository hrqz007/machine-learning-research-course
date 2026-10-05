"""ML050: splits define different estimands. No tuning or test-to-fit path."""
from pathlib import Path
import argparse, hashlib, json, os, tempfile
import numpy as np
ROOT=Path(__file__).resolve().parent
MODELS=('pooled','personalized','trend')
SPLITS=('random','group','time','group_time')

def require(ok, message):
    if not ok:raise ValueError(message)

def safe_output(path):
    path=Path(os.path.abspath(path))
    for part in (path,*path.parents):require(not part.is_symlink(),'symlink output is not supported')
    resolved=path.resolve()
    if resolved.is_relative_to(ROOT):require(resolved.is_relative_to(ROOT/'outputs'),'output must not overwrite course files')
    if path.exists():require(path.is_file() and path.stat().st_nlink==1,'output must be a single regular file')
    return path

def atomic_json(path,value):
    path=safe_output(path);payload=json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n'
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,delete=False,prefix='.ml050-') as f:
        name=f.name;f.write(payload)
    try:os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)

def load_data():
    hashes=json.loads((ROOT/'data_integrity.json').read_text())
    for rel,expected in hashes.items():require(hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==expected,'data hash mismatch: '+rel)
    c=json.loads((ROOT/'data/protocol.json').read_text())
    with np.load(ROOT/'data/draws.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
    with np.load(ROOT/'data/splits.npz',allow_pickle=False) as z:s={k:(z[k+'_train'],z[k+'_test']) for k in SPLITS}
    return c,d,s

def panel(config,draws,rep):
    require(isinstance(rep,(int,np.integer)) and 0<=rep<len(draws['a']),'rep out of range')
    G,T=config['n_devices'],config['n_days'];g=np.repeat(np.arange(G),T);t=np.tile(np.arange(T),G)
    x=draws['xnoise'][rep].ravel()+config['x_drift_per_day']*t
    mu=config['intercept']+config['signal_slope']*x+draws['a'][rep,g]+config['response_drift_per_day']*t
    y=mu+draws['eps'][rep].ravel()
    return {'group':g,'day':t,'available':t+config['label_delay_days'],'x':x,'y':y,'mu':mu}

def finite_vector(v,name):
    a=np.asarray(v)
    require(a.ndim==1 and 1<=a.size<=200000,name+' must be a nonempty 1D vector')
    require(a.dtype.kind in 'ifu' and np.isfinite(a).all(),name+' must be finite real numeric')
    return a

def validate_features(x,group,day):
    x=finite_vector(x,'x').astype(float);group=finite_vector(group,'group');day=finite_vector(day,'day').astype(float)
    require(len(x)==len(group)==len(day),'feature lengths differ')
    require(np.all(group>=0) and np.all(group==np.floor(group)),'group must be nonnegative integer ids')
    return x,group.astype(int),day

def design(x,group,day,known_groups,model):
    require(model in MODELS,'unknown model')
    x,group,day=validate_features(x,group,day)
    cols=[np.ones(len(x)),x]
    if model!='pooled':cols.extend((group==g).astype(float) for g in known_groups)
    if model=='trend':cols.append(day/47.) # fixed physical time unit; not fitted on test
    return np.column_stack(cols)

def fit(x,group,day,y,model,alpha=1.):
    x,group,day=validate_features(x,group,day);y=finite_vector(y,'y').astype(float)
    require(len(y)==len(x),'target length differs')
    require(np.isscalar(alpha) and not isinstance(alpha,(bool,np.bool_)) and np.isfinite(alpha) and alpha>0,'alpha must be finite and positive')
    known=np.unique(group);A=design(x,group,day,known,model);p=A.shape[1]
    P=np.eye(p);P[0,0]=0.
    aug=np.vstack((A,np.sqrt(alpha)*P));target=np.r_[y,np.zeros(p)]
    beta,_,rank,svals=np.linalg.lstsq(aug,target,rcond=None)
    require(rank==p,'augmented design is rank deficient')
    require(np.isfinite(beta).all(),'nonfinite fit')
    grad=(A.T@(A@beta-y)+alpha*P@beta)/len(y)
    return {'beta':beta,'groups':known,'model':model,'alpha':alpha,'gradient_max':float(np.max(np.abs(grad))),'condition_augmented':float(svals[0]/svals[-1])}

def predict(fitted,x,group,day):
    return design(x,group,day,fitted['groups'],fitted['model'])@fitted['beta']

def validate_split(p,train,test,kind,cutoff):
    for arr in (train,test):
        require(np.asarray(arr).ndim==1 and len(arr)>0,'empty or nonvector split')
        require(np.asarray(arr).dtype.kind in 'iu','split indices must be integer')
        require(np.min(arr)>=0 and np.max(arr)<len(p['x']),'split index out of bounds')
        require(len(np.unique(arr))==len(arr),'duplicate split index')
    require(not np.intersect1d(train,test).size,'train and test overlap')
    if kind in ('group','group_time'):require(not np.intersect1d(p['group'][train],p['group'][test]).size,'group leakage')
    if kind in ('time','group_time'):
        require(np.max(p['available'][train])<=cutoff,'unavailable training label')
        require(np.min(p['day'][test])>cutoff,'test does not follow fit cutoff')

def metrics(y,pred,mu,noise_sd,group):
    loss=(np.asarray(pred)-y)**2
    return {'mse':float(loss.mean()),'rmse':float(np.sqrt(loss.mean())),
      'group_mean_mse':float(np.mean([loss[group==g].mean() for g in np.unique(group)])),
      'oracle_conditional_mse':float(np.mean((pred-mu)**2)+noise_sd**2)}

def one_run(c,p,splits,keep_predictions=False):
    out={}
    for kind,(tr,te) in splits.items():
        validate_split(p,tr,te,kind,c['fit_cutoff_day'])
        b={'n_train':len(tr),'n_test':len(te),'n_unused':len(p['x'])-len(tr)-len(te),
           'n_train_groups':len(np.unique(p['group'][tr])),'n_test_groups':len(np.unique(p['group'][te])),
           'group_overlap':len(np.intersect1d(p['group'][tr],p['group'][te])),
           'train_day_range':[int(p['day'][tr].min()),int(p['day'][tr].max())],
           'test_day_range':[int(p['day'][te].min()),int(p['day'][te].max())],
           'train_max_label_available':int(p['available'][tr].max()),'models':{}}
        for model in MODELS:
            f=fit(p['x'][tr],p['group'][tr],p['day'][tr],p['y'][tr],model,c['ridge_alpha'])
            pr=predict(f,p['x'][te],p['group'][te],p['day'][te])
            q=metrics(p['y'][te],pr,p['mu'][te],c['noise_sd'],p['group'][te]);q.update({'gradient_max':f['gradient_max'],'condition_augmented':f['condition_augmented']})
            if keep_predictions:q.update({'predictions':pr.tolist(),'coefficients':f['beta'].tolist(),'known_groups':f['groups'].tolist(),'test_row_ids':te.tolist()})
            b['models'][model]=q
        out[kind]=b
    return out

def hand():
    X=np.array([[1,0,1,0],[1,1,1,0],[1,0,0,1],[1,1,0,1]],float);y=np.array([1,3,2,4.]);theta=np.zeros(4);eta=.2;n=4
    pred=X@theta;r=pred-y;local=r[:,None]*X;grad=local.mean(0);updated=theta-eta*grad;nxt=X@updated
    rows=[{'row':f'H{i+1}','features':X[i].tolist(),'y':float(y[i]),'prediction':float(pred[i]),'residual':float(r[i]),'half_squared_loss':float(.5*r[i]**2),'local_derivatives':local[i].tolist(),'next_prediction':float(nxt[i]),'next_half_squared_loss':float(.5*(nxt[i]-y[i])**2)} for i in range(4)]
    return {'rows':rows,'gradient_mean':grad.tolist(),'initial_objective':3.75,'learning_rate':eta,'theta_next':updated.tolist(),'next_objective':float(((nxt-y)@(nxt-y)+updated[1:]@updated[1:])/(2*n)),
      'test_next_predictions':[float(np.array([1,2,1,0])@updated),float(np.array([1,2,0,0])@updated)]}

def cluster_bootstrap(p,main,c):
    q=main['group']['models']['personalized'];te=np.array(q['test_row_ids']);loss=(np.array(q['predictions'])-p['y'][te])**2
    groups=p['group'][te];means=np.array([loss[groups==g].mean() for g in np.unique(groups)])
    rng=np.random.default_rng(c['bootstrap_seed']);B=c['bootstrap_repetitions']
    row=np.array([loss[rng.integers(0,len(loss),len(loss))].mean() for _ in range(B)])
    grp=np.array([means[rng.integers(0,len(means),len(means))].mean() for _ in range(B)])
    return {'estimand':'conditional on fitted model; equal-device empirical test risk; no refit, calendar fixed',
       'row_draws':row.tolist(),'group_draws':grp.tolist(),'row_sd':float(row.std(ddof=1)),'group_sd':float(grp.std(ddof=1)),
       'row_percentile_95':np.quantile(row,[.025,.975]).tolist(),'group_percentile_95':np.quantile(grp,[.025,.975]).tolist(),
       'independent_groups':len(means),'rows':len(loss)}

def run():
    c,d,s=load_data();p=panel(c,d,0);main=one_run(c,p,s,True)
    repeated=[{'rep':r,'results':one_run(c,panel(c,d,r),s)} for r in range(1,c['independent_repetitions']+1)]
    summary={}
    for k in SPLITS:
        summary[k]={}
        for model in MODELS:
            vals=np.array([r['results'][k]['models'][model]['mse'] for r in repeated])
            summary[k][model]={'mean_mse':float(vals.mean()),'sd_between_panels':float(vals.std(ddof=1)),'mcse_mean':float(vals.std(ddof=1)/np.sqrt(len(vals))),'q10':float(np.quantile(vals,.1)),'q90':float(np.quantile(vals,.9)),'min':float(vals.min()),'max':float(vals.max())}
    return {'protocol':c,'hand':hand(),'main':main,'replications':repeated,'summary':summary,'bootstrap':cluster_bootstrap(p,main,c),
      'negative_results':{k:{'personalized_worse_than_pooled':sum(r['results'][k]['models']['personalized']['mse']>r['results'][k]['models']['pooled']['mse'] for r in repeated),'trend_worse_than_personalized':sum(r['results'][k]['models']['trend']['mse']>r['results'][k]['models']['personalized']['mse'] for r in repeated)} for k in SPLITS},
      'weight_example':{'group_sizes':[8,2],'model_A_group_losses':[1,9],'model_B_group_losses':[4,4],'A_row_mse':2.6,'B_row_mse':4.,'A_group_mse':5.,'B_group_mse':4.}}

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(ROOT/'outputs/experiment-result.json'));a=p.parse_args();safe_output(a.out)
    result=run();atomic_json(a.out,result)
    print(json.dumps({'saved':str(a.out),'main_mse':{k:{m:q['mse'] for m,q in v['models'].items()} for k,v in result['main'].items()},'negative_results':result['negative_results']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
