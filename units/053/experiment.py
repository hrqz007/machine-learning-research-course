"""ML053: equal requested-tree budgets; no test data enter choose()."""
from pathlib import Path
import argparse, hashlib, json, time, platform
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
ROOT = Path(__file__).resolve().parent
METHODS = ['grid', 'random', 'halving']
# All methods search the SAME finite domain; their sampling policies differ.
DOMAIN = [(d, f) for d in [2, 4, 6, 8, None] for f in [.2, .4, .6, .8, 1.]]
GRID = [(d, f) for d in [2, 6, None] for f in [.4, 1.]]

def safe_output(path):
    path = Path(path).absolute()
    for part in (path, *path.parents):
        if part.is_symlink(): raise ValueError('Refusing a symlink output path')
    if path.exists() and (not path.is_file() or path.stat().st_nlink > 1):
        raise ValueError('Output must be an ordinary file')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def dump(obj, path):
    path = safe_output(path)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))

def mse(y, p):
    y, p = np.asarray(y), np.asarray(p)
    if y.shape != p.shape or y.size == 0 or not np.isfinite(y-p).all():
        raise ValueError('MSE requires equal, nonempty, finite shapes')
    return float(np.mean((y-p)**2))

def generate_data(directory):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for rep in range(6):
        rng = np.random.default_rng(5300+rep)
        X = rng.normal(size=(600, 10))
        # x0, x1, x2 carry signal; x3..x9 are independent noise.
        y = 2*X[:,0] + 1.5*(X[:,1]**2-1) + X[:,0]*X[:,2] + rng.normal(size=600)
        arrays[f'X{rep}'], arrays[f'y{rep}'] = X, y
    np.savez_compressed(directory/'draws.npz', **arrays)
    dump({'seeds':list(range(5300,5306)), 'n_development':240,'n_test':360,
          'folds':3,'fold_seed':531,'domain':DOMAIN,'grid':GRID,
          'random_candidates':6,'halving_candidates':[12,4,1],
          'halving_resources':[15,30,60],'search_trees_per_method':1080,
          'final_refit_trees_per_method':60,'restart_each_stage':True,
          'comparison':'equal requested trees, not equal time/FLOPs',
          'test_policy':'Only after winner and all search records are fixed'},directory/'protocol.json')

def fit_record(X, y, train, valid, cfg, resource, seed, rep, method, stage, cid):
    """One fresh fit. Log attempted cost even on failure; no silent retry."""
    start = time.perf_counter()
    record = dict(rep=rep, method=method, stage=stage, candidate=cid,
                  depth=cfg[0], max_features=cfg[1], resource=resource,
                  train_ids=train.tolist(), valid_ids=valid.tolist(),
                  requested_trees=resource, completed_trees=0, status='failed')
    try:
        model = RandomForestRegressor(n_estimators=resource, max_depth=cfg[0],
                    max_features=cfg[1], min_samples_leaf=2, n_jobs=1, random_state=seed)
        model.fit(X[train], y[train]); pred = model.predict(X[valid])
        record.update(status='ok', completed_trees=len(model.estimators_),
                      prediction=pred.tolist(), mse=mse(y[valid],pred))
    except (ValueError, FloatingPointError) as exc:
        record.update(error_type=type(exc).__name__, error=str(exc), mse=None)
    record['seconds'] = time.perf_counter()-start
    return record

def choose(X, y, rep, method):
    """Signature deliberately accepts DEVELOPMENT data only."""
    if method not in METHODS: raise ValueError('Unknown search method')
    if len(y) != 240 or X.shape != (240,10) or not np.isfinite(X).all():
        raise ValueError('Expected the fixed 240-row development design')
    folds = list(KFold(3,shuffle=True,random_state=531).split(X))
    rng = np.random.default_rng(5350+rep)
    order = rng.permutation(len(DOMAIN))
    # Random's six are the first six of halving's twelve: a documented coupling.
    cfgs = GRID if method == 'grid' else [DOMAIN[i] for i in order[:6 if method=='random' else 12]]
    alive = list(range(len(cfgs))); logs=[]; stages=[]
    resources = [15,30,60] if method=='halving' else [60]
    for stage, resource in enumerate(resources):
        scores=[]
        for cid in alive:
            rows = [fit_record(X,y,tr,va,cfgs[cid],resource,5380+rep*10+fold,
                              rep,method,stage,cid) for fold,(tr,va) in enumerate(folds)]
            logs.extend(rows)
            score = float(np.mean([r['mse'] for r in rows])) if all(r['status']=='ok' for r in rows) else float('inf')
            scores.append((score,cid))
        scores.sort() # Deterministic tie: smaller candidate id wins.
        stages.append({'resource':resource,'ranking':[{'candidate':c,'mse':s if np.isfinite(s) else None} for s,c in scores]})
        if not np.isfinite(scores[0][0]): raise RuntimeError('All candidates failed; report, do not use test data')
        alive = [c for _,c in scores[:[4,1,1][stage]]] if method=='halving' else [scores[0][1]]
    winner = alive[0]
    return dict(method=method, configurations=cfgs, winner=winner, selected=cfgs[winner],
                cv_mse=stages[-1]['ranking'][0]['mse'], stages=stages, fits=logs,
                requested_trees=sum(r['requested_trees'] for r in logs),
                completed_trees=sum(r['completed_trees'] for r in logs),
                fit_seconds=sum(r['seconds'] for r in logs))

def hand_examples():
    # A toy learning-curve counterexample, not fitted data or a research result.
    curves = {'fast':[.4,.35,.34], 'slow':[.7,.3,.2], 'poor':[.9,.8,.75]}
    rng=np.random.default_rng(5399); errors=rng.normal(0,.15,size=(10000,64))
    return {'budget_grid':6*60*3,'budget_halving':(12*15+4*30+1*60)*3,
            'curves':curves,'noise_candidates':[1,4,16,64],
            'selection_noise_mean':[float(errors[:,:k].min(axis=1).mean()) for k in [1,4,16,64]],
            'noise_true_risk':1.0,'noise_sd':.15,'noise_draws':10000,
            'validation_min':[float((1+errors[:,:k].min(axis=1)).mean()) for k in [1,4,16,64]]}

def run():
    if not (ROOT/'data/draws.npz').exists(): raise FileNotFoundError('Run generate_data.py first')
    data=np.load(ROOT/'data/draws.npz'); all_runs=[]; start=time.perf_counter()
    for rep in range(6):
        X,y=data[f'X{rep}'],data[f'y{rep}']; searches=[]
        # Finish all search choices before accessing any test labels.
        for method in METHODS: searches.append(choose(X[:240],y[:240],rep,method))
        for s in searches:
            d,f=s['selected']; t=time.perf_counter()
            model=RandomForestRegressor(n_estimators=60,max_depth=d,max_features=f,
                       min_samples_leaf=2,n_jobs=1,random_state=5390+rep)
            model.fit(X[:240],y[:240]); pred=model.predict(X[240:])
            s.update(rep=rep,test_mse=mse(y[240:],pred),test_prediction=pred.tolist(),
                     final_refit_trees=60,final_refit_seconds=time.perf_counter()-t)
            all_runs.append(s)
    # Invalid configuration is an explicit separate diagnostic, never a main competitor.
    X,y=data['X0'][:240],data['y0'][:240]
    failure=fit_record(X,y,np.arange(160),np.arange(160,240),(4,0),15,0,-1,'failure_demo',0,0)
    summary={m:{'test_mse':float(np.mean([s['test_mse'] for s in all_runs if s['method']==m])),
                'test_mse_sd':float(np.std([s['test_mse'] for s in all_runs if s['method']==m],ddof=1)),
                'total_search_seconds':float(sum(s['fit_seconds'] for s in all_runs if s['method']==m)),
                'search_trees_per_rep':1080,'refit_trees_per_rep':60} for m in METHODS}
    for m in ['random','halving']:
        diffs=[next(s['test_mse'] for s in all_runs if s['rep']==r and s['method']==m)-next(s['test_mse'] for s in all_runs if s['rep']==r and s['method']=='grid') for r in range(6)]
        summary[m].update(paired_difference_vs_grid=diffs,worse_than_grid_count=sum(v>0 for v in diffs))
    return {'unit':'053','runs':all_runs,'summary':summary,'hand':hand_examples(),
            'failure_demo':failure,'elapsed_seconds':time.perf_counter()-start,
            'data_sha256':hashlib.sha256((ROOT/'data/draws.npz').read_bytes()).hexdigest(),
            'python':platform.python_version(),'counts':{'main_search_fits':522,'main_refit_fits':18,'diagnostic_failed_attempts':1}}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(ROOT/'outputs/result.json'));a=p.parse_args()
    result=run();dump(result,a.out);print(json.dumps(result['summary'],ensure_ascii=False,indent=2))
