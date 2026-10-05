"""ML054: nested selection, deliberate leakage, interaction and dimension controls."""
from pathlib import Path
import argparse,hashlib,json,itertools,platform,time
import numpy as np
from sklearn.linear_model import Ridge,LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
ROOT=Path(__file__).resolve().parent
KS=[2,5,15,40]
METHODS=['nested','global_leaky','all_features']

def safe_output(path):
    path=Path(path).absolute()
    for part in (path,*path.parents):
        if part.is_symlink():raise ValueError('Refusing symlink output')
    if path.exists() and (not path.is_file() or path.stat().st_nlink>1):raise ValueError('Unsafe output')
    path.parent.mkdir(parents=True,exist_ok=True);return path

def dump(obj,path):safe_output(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False))

def rank_features(X,y):
    """Absolute Pearson correlation, computed using only supplied rows."""
    X=np.asarray(X,float);y=np.asarray(y,float)
    if X.ndim!=2 or y.shape!=(len(X),) or len(y)<3 or not np.isfinite(X).all() or not np.isfinite(y).all():raise ValueError('Finite 2D X and matching 1D y required')
    xc=X-X.mean(0);yc=y-y.mean();den=np.sqrt((xc*xc).sum(0)*(yc*yc).sum())
    scores=np.divide(np.abs(xc.T@yc),den,out=np.zeros(X.shape[1]),where=den>0)
    # Explicit stable tie: lowest original column number.
    return np.lexsort((np.arange(X.shape[1]),-scores)),scores

def predict(X,y,Z,cols):
    sc=StandardScaler().fit(X[:,cols]);a=sc.transform(X[:,cols]);b=sc.transform(Z[:,cols])
    model=Ridge(alpha=1.,solver='svd').fit(a,y)
    return model.predict(b),{'mean':sc.mean_.tolist(),'scale':sc.scale_.tolist(),'coef':model.coef_.tolist(),'intercept':float(model.intercept_)}

def mse(y,p):return float(np.mean((np.asarray(y)-np.asarray(p))**2))

def choose_k(X,y,global_rank=None):
    """Choose k inside an outer training fold. Global rank is deliberate bad control."""
    folds=list(KFold(3,shuffle=True,random_state=541).split(X));trials=[]
    for k in KS:
        records=[]
        for tr,va in folds:
            rank,_=rank_features(X[tr],y[tr]) if global_rank is None else (global_rank,None)
            cols=rank[:k];pred,params=predict(X[tr],y[tr],X[va],cols)
            records.append({'train_local':tr.tolist(),'valid_local':va.tolist(),'selected':cols.tolist(),'prediction':pred.tolist(),'mse':mse(y[va],pred)})
        trials.append({'k':k,'mean_mse':float(np.mean([z['mse'] for z in records])),'folds':records})
    best=min(trials,key=lambda z:(z['mean_mse'],z['k']))['k']
    return best,trials

def generate_data(directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);a={}
    for mode in ['signal','null']:
        for rep in range(6):
            rng=np.random.default_rng((5400 if mode=='signal' else 5450)+rep)
            X=rng.normal(size=(360,80));X[:,2]=X[:,0]+.08*rng.normal(size=360)
            y=(2*X[:,0]+1.5*X[:,1] if mode=='signal' else np.zeros(360))+rng.normal(size=360)
            a[f'{mode}_X{rep}'],a[f'{mode}_y{rep}']=X,y
    np.savez_compressed(directory/'draws.npz',**a)
    dump({'n_development':160,'n_confirmation':200,'features':80,'outer_folds':4,'inner_folds':3,'k_candidates':KS,'ridge_alpha':1.,'repeats_per_mechanism':6,'global_leakage_scope':'uses all 160 development labels, never 200 confirmation labels','split_seed_outer':540,'split_seed_inner':541,'signal_seeds':list(range(5400,5406)),'null_seeds':list(range(5450,5456))},directory/'protocol.json')

def controls():
    X=np.array([[-1,-1],[-1,1],[1,-1],[1,1]],float);y=X[:,0]*X[:,1]
    add=LinearRegression().fit(X,y);interaction=np.column_stack([X,X[:,0]*X[:,1]])
    full=LinearRegression().fit(interaction,y)
    rng=np.random.default_rng(5490);distance=[]
    for d in [2,10,50,200]:
        # Independent pairs, not distances sharing one random anchor.
        delta=rng.normal(size=(4000,d))-rng.normal(size=(4000,d));sq=(delta*delta).sum(1)
        distance.append({'d':d,'n_pairs':4000,'mean_squared_distance':float(sq.mean()),'cv_squared_distance':float(sq.std(ddof=1)/sq.mean()),'theory_mean':2*d,'theory_cv':float(np.sqrt(2/d)),'normalized_squared_distance':(sq/(2*d)).tolist()})
    return {'interaction':{'X':X.tolist(),'y':y.tolist(),'correlations':np.corrcoef(np.column_stack([X,y]),rowvar=False)[:2,2].tolist(),'additive_prediction':add.predict(X).tolist(),'interaction_prediction':full.predict(interaction).tolist(),'additive_mse':mse(y,add.predict(X)),'interaction_mse':mse(y,full.predict(interaction))},'distance':distance}

def jaccard(a,b):
    a,b=set(a),set(b)
    return len(a&b)/len(a|b) if a|b else 1.

def run():
    start=time.perf_counter();data=np.load(ROOT/'data/draws.npz');runs=[]
    for mode in ['signal','null']:
        for rep in range(6):
            X,y=data[f'{mode}_X{rep}'],data[f'{mode}_y{rep}'];D,Y=X[:160],y[:160]
            global_rank,_=rank_features(D,Y);outer=list(KFold(4,shuffle=True,random_state=540).split(D))
            for method in METHODS:
                folds=[]
                for fold,(tr,va) in enumerate(outer):
                    if method=='all_features':k=80;trials=[];cols=np.arange(80)
                    else:
                        k,trials=choose_k(D[tr],Y[tr],global_rank if method=='global_leaky' else None)
                        rank,_=rank_features(D[tr],Y[tr]) if method=='nested' else (global_rank,None)
                        cols=rank[:k]
                    pred,params=predict(D[tr],Y[tr],D[va],cols)
                    folds.append({'fold':fold,'train_ids':tr.tolist(),'valid_ids':va.tolist(),'k':k,'selected':cols.tolist(),'inner_trials':trials,'prediction':pred.tolist(),'mse':mse(Y[va],pred),'params':params})
                # The eventual deployment pipeline is clean for all methods: full development
                # can legitimately train the final selector. Never include confirmation y.
                if method=='all_features':k=80;trials=[];cols=np.arange(80)
                else:
                    k,trials=choose_k(D,Y,global_rank if method=='global_leaky' else None)
                    cols=global_rank[:k]
                pred,params=predict(D,Y,X[160:],cols)
                sets=[z['selected'] for z in folds]
                runs.append({'mode':mode,'rep':rep,'method':method,'folds':folds,'outer_mse':float(np.mean([z['mse'] for z in folds])),'confirmation_mse':mse(y[160:],pred),'confirmation_prediction':pred.tolist(),'final_k':k,'final_selected':cols.tolist(),'final_inner_trials':trials,'final_params':params,'stability_mean_jaccard':float(np.mean([jaccard(a,b) for a,b in itertools.combinations(sets,2)])),'selection_frequency':[sum(j in z for z in sets)/4 for j in range(80)]})
    summary={mode:{m:{key:float(np.mean([s[key] for s in runs if s['mode']==mode and s['method']==m])) for key in ['outer_mse','confirmation_mse','stability_mean_jaccard']} for m in METHODS} for mode in ['signal','null']}
    return {'unit':'054','runs':runs,'summary':summary,'controls':controls(),'elapsed_seconds':time.perf_counter()-start,'data_sha256':hashlib.sha256((ROOT/'data/draws.npz').read_bytes()).hexdigest(),'python':platform.python_version()}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(ROOT/'outputs/result.json'));a=p.parse_args();r=run();dump(r,a.out);print(json.dumps(r['summary'],ensure_ascii=False,indent=2))
