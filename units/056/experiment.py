"""ML056: nested selection, immutable decisions, then a single final test evaluation.
All fold-specific imputers/scalers/models are fitted only on that fold's training rows.
The final fit is represented as plain arrays, so no unsafe pickle is needed.
"""
import argparse, hashlib, json, warnings
from pathlib import Path
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score, average_precision_score
from common import ROOT, write_json, require, wilson

def load_rows(path):
    a=np.loadtxt(path,delimiter=',',skiprows=1,ndmin=2)
    require(a.shape[1]==6 and len(a)>0,'expected six columns and nonempty rows')
    require(np.isfinite(a[:,[0,1,2,4,5]]).all(),'only x2 may be missing')
    require(not np.isinf(a[:,3]).any(),'infinite x2 not allowed')
    require(set(a[:,5])=={0,1},'both binary classes required')
    require(set(a[:,4])<={0,1},'group must be binary')
    require(len(np.unique(a[:,0]))==len(a),'row_id must be unique')
    return a[:,1:4],a[:,5].astype(int),a[:,4].astype(int),a[:,0].astype(int)

def cost_vector(y,p,t):
    pred=np.asarray(p)>=t;y=np.asarray(y)
    return 4*((y==1)&~pred)+((y==0)&pred)

def metrics(y,p,t):
    pred=np.asarray(p)>=t;y=np.asarray(y);n=len(y)
    tp=int(((y==1)&pred).sum());fn=int(((y==1)&~pred).sum())
    fp=int(((y==0)&pred).sum());tn=n-tp-fn-fp
    return dict(n=n,tp=tp,fn=fn,fp=fp,tn=tn,cost=float((4*fn+fp)/n),accuracy=float((tp+tn)/n),recall=None if tp+fn==0 else tp/(tp+fn),recall_wilson=wilson(tp,tp+fn),precision=None if tp+fp==0 else tp/(tp+fp),brier=float(np.mean((p-y)**2)),auc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,ap=float(average_precision_score(y,p)) if y.sum()>0 else None)

def model(degree,C):
    # Fit scaling BEFORE the degree expansion; the candidate's transformation is fixed.
    return make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),PolynomialFeatures(degree=degree,include_bias=False),LogisticRegression(C=C,max_iter=1200,solver='lbfgs',tol=1e-9))

def fitted_state(m):
    im,sc,poly,lr=m.steps[0][1],m.steps[1][1],m.steps[2][1],m.steps[3][1]
    return dict(imputer=im.statistics_.tolist(),mean=sc.mean_.tolist(),scale=sc.scale_.tolist(),powers=poly.powers_.tolist(),coef=lr.coef_[0].tolist(),intercept=float(lr.intercept_[0]),iterations=int(lr.n_iter_[0]))

def forward(X,state):
    X=np.where(np.isnan(X),state['imputer'],X)
    Z=(X-np.array(state['mean']))/np.array(state['scale'])
    features=np.prod(Z[:,None,:]**np.array(state['powers'])[None,:,:],axis=2)
    from scipy.special import expit
    return expit(features@np.array(state['coef'])+state['intercept'])

def select(X,y,ids,seed,protocol):
    require(np.bincount(y,minlength=2).min()>=3,'each class needs at least three rows for inner CV')
    folds=list(StratifiedKFold(3,shuffle=True,random_state=seed).split(X,y))
    candidates=[];ledger=[]
    for degree in protocol['degrees']:
        for C in protocol['Cs']:
            p=np.zeros(len(y));fits=[]
            for fold,(tr,va) in enumerate(folds):
                m=model(degree,C)
                with warnings.catch_warnings():
                    warnings.simplefilter('error',ConvergenceWarning);m.fit(X[tr],y[tr])
                p[va]=m.predict_proba(X[va])[:,1]
                fits.append(dict(fold=fold,train_ids=ids[tr].tolist(),valid_ids=ids[va].tolist(),state=fitted_state(m),status='ok'))
            scores=[dict(threshold=t,cost=float(cost_vector(y,p,t).mean())) for t in protocol['thresholds']]
            chosen=min(scores,key=lambda r:(r['cost'],r['threshold']))
            candidates.append(dict(degree=degree,C=C,threshold=chosen['threshold'],cost=chosen['cost']))
            ledger.append(dict(degree=degree,C=C,oof_probability=p.tolist(),scores=scores,fits=fits))
    best=min(candidates,key=lambda r:(r['cost'],r['degree'],r['C'],r['threshold']))
    return dict(best=best,candidates=candidates,ledger=ledger,row_ids=ids.tolist(),y=y.tolist())

def develop(X,y,ids,protocol):
    require(np.bincount(y,minlength=2).min()>=9,'each class needs at least nine development examples')
    outer=[];prob=np.zeros(len(y));threshold=np.zeros(len(y));base=np.zeros(len(y))
    for fold,(tr,va) in enumerate(StratifiedKFold(3,shuffle=True,random_state=5610).split(X,y)):
        selection=select(X[tr],y[tr],ids[tr],5620+fold,protocol);b=selection['best']
        m=model(b['degree'],b['C']).fit(X[tr],y[tr]);p=m.predict_proba(X[va])[:,1]
        prob[va]=p;threshold[va]=b['threshold'];base[va]=y[tr].mean()
        outer.append(dict(fold=fold,train_ids=ids[tr].tolist(),valid_ids=ids[va].tolist(),selection=selection,state=fitted_state(m),metrics=metrics(y[va],p,b['threshold'])))
    final=select(X,y,ids,5650,protocol);b=final['best'];m=model(b['degree'],b['C']).fit(X,y)
    return dict(outer=outer,outer_metrics=metrics(y,prob,threshold),outer_baseline=metrics(y,base,.2),outer_probability=prob.tolist(),outer_threshold=threshold.tolist(),outer_baseline_probability=base.tolist(),final_selection=final,final_state=fitted_state(m),baseline_probability=float(y.mean()),development_ids=ids.tolist(),development_y=y.tolist())

def evaluate_frozen(development,X,y,group,ids,protocol):
    b=development['final_selection']['best'];p=forward(X,development['final_state'])
    baseline=np.full(len(y),development['baseline_probability']);delta=cost_vector(y,p,b['threshold'])-cost_vector(y,baseline,.2)
    rng=np.random.default_rng(protocol['bootstrap_seed'])
    boot=np.array([rng.choice(delta,size=len(delta),replace=True).mean() for _ in range(protocol['bootstrap_repetitions'])])
    return dict(ids=ids.tolist(),y=y.tolist(),group=group.tolist(),probability=p.tolist(),baseline_probability=baseline.tolist(),threshold=b['threshold'],selected=metrics(y,p,b['threshold']),baseline=metrics(y,baseline,.2),paired_difference=float(delta.mean()),paired_bootstrap_95=np.quantile(boot,[.025,.975]).tolist(),bootstrap_distribution=boot.tolist(),groups={str(g):metrics(y[group==g],p[group==g],b['threshold']) for g in protocol['groups']})

def run(data_dir=ROOT/'data'):
    data_dir=Path(data_dir);protocol=json.loads((data_dir/'protocol.json').read_text())
    X,y,g,ids=load_rows(data_dir/'development.csv')
    with warnings.catch_warnings():
        warnings.simplefilter("error",ConvergenceWarning)
        development=develop(X,y,ids,protocol)
    # This digest commits choices BEFORE reading any final-test rows below.
    frozen_json=json.dumps(development,sort_keys=True,separators=(',',':')).encode()
    frozen_sha=hashlib.sha256(frozen_json).hexdigest()
    Xt,yt,gt,it=load_rows(data_dir/'test.csv')
    require(not np.intersect1d(ids,it).size,'development and test IDs overlap')
    test=evaluate_frozen(development,Xt,yt,gt,it,protocol)
    return dict(unit='056',protocol=protocol,development=development,frozen_development_sha256=frozen_sha,test=test,fit_count=76,failed_fits=[],test_evaluations=1)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',default=str(ROOT/'data'));ap.add_argument('--out',default=str(ROOT/'outputs/result.json'));a=ap.parse_args()
    r=run(a.data);write_json(a.out,r)
    print(json.dumps({'selected':r['development']['final_selection']['best'],'test':r['test']['selected'],'difference':r['test']['paired_difference'],'ci':r['test']['paired_bootstrap_95']},ensure_ascii=False,indent=2))
