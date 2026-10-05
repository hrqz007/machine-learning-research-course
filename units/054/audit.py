"""Independent Pearson, ridge normal-equation and selection provenance audit."""
from pathlib import Path
import argparse,json,itertools
import numpy as np
from scipy.linalg import solve
import experiment as e

def require(ok,msg):
    if not ok:raise AssertionError(msg)

def manual_rank(X,y):
    scores=[]
    for col in X.T:
        a=col-col.mean();b=y-y.mean();den=np.sqrt(np.dot(a,a)*np.dot(b,b))
        scores.append(abs(np.dot(a,b))/den if den else 0.)
    return np.array(sorted(range(X.shape[1]),key=lambda j:(-scores[j],j)))

def manual_predict(X,y,Z,cols):
    A=X[:,cols];B=Z[:,cols];mu=A.mean(0);scale=A.std(0);scale[scale==0]=1
    A=(A-mu)/scale;B=(B-mu)/scale;center=y.mean()
    # alpha=1, unpenalized intercept. Independent solve rather than sklearn SVD.
    coef=solve(A.T@A+np.eye(len(cols)),A.T@(y-center),assume_a='pos')
    return B@coef+center

def audit(r):
    data=np.load(e.ROOT/'data/draws.npz');outer_count=0;inner_count=0;final_count=0
    for s in r['runs']:
        X,y=data[f"{s['mode']}_X{s['rep']}"],data[f"{s['mode']}_y{s['rep']}"]
        D,Y=X[:160],y[:160];global_rank=manual_rank(D,Y)
        for f in s['folds']:
            tr=np.array(f['train_ids']);va=np.array(f['valid_ids']);require(set(tr).isdisjoint(va),'outer overlap');require(set(tr)|set(va)==set(range(160)),'outer membership')
            rank=global_rank if s['method']=='global_leaky' else manual_rank(D[tr],Y[tr])
            cols=list(range(80)) if s['method']=='all_features' else rank[:f['k']].tolist()
            require(cols==f['selected'],'outer selection provenance')
            pred=manual_predict(D[tr],Y[tr],D[va],cols);require(np.allclose(pred,f['prediction'],rtol=1e-9,atol=1e-9),'outer ridge prediction');require(abs(e.mse(Y[va],pred)-f['mse'])<1e-9,'outer MSE');outer_count+=1
            for trial in f['inner_trials']:
                for z in trial['folds']:
                    it=np.array(z['train_local']);iv=np.array(z['valid_local']);require(set(it).isdisjoint(iv),'inner overlap')
                    rank=global_rank if s['method']=='global_leaky' else manual_rank(D[tr][it],Y[tr][it])
                    require(rank[:trial['k']].tolist()==z['selected'],'inner ranking provenance')
                    p=manual_predict(D[tr][it],Y[tr][it],D[tr][iv],z['selected']);require(np.allclose(p,z['prediction'],atol=1e-9,rtol=1e-9),'inner ridge');inner_count+=1
                require(abs(np.mean([z['mse'] for z in trial['folds']])-trial['mean_mse'])<1e-12,'inner mean')
            if f['inner_trials']:require(f['k']==min(f['inner_trials'],key=lambda z:(z['mean_mse'],z['k']))['k'],'chosen k')
        p=manual_predict(D,Y,X[160:],s['final_selected']);require(np.allclose(p,s['confirmation_prediction'],atol=1e-9,rtol=1e-9),'confirmation prediction');require(abs(e.mse(y[160:],p)-s['confirmation_mse'])<1e-9,'confirmation MSE');final_count+=1
        jj=[len(set(a['selected'])&set(b['selected']))/len(set(a['selected'])|set(b['selected'])) for a,b in itertools.combinations(s['folds'],2)]
        require(abs(np.mean(jj)-s['stability_mean_jaccard'])<1e-12,'Jaccard')
    h=r['controls']['interaction'];require(h['additive_mse']==1 and h['interaction_mse']<1e-25,'interaction');require(h['correlations']==[0.,0.],'zero marginal correlation')
    # Typical constant feature and bad-input cases; no irrelevant stress tests.
    rank,score=e.rank_features(np.column_stack([np.arange(4),np.ones(4)]),np.arange(4));require(score[1]==0 and rank[0]==0,'constant feature')
    rejected=0
    for X,y in [(np.ones((4,2)),np.ones(3)),(np.array([[1.,2.],[3.,np.nan],[4.,5.]]),np.ones(3))]:
        try:e.rank_features(X,y)
        except ValueError:rejected+=1
    require(rejected==2,'input validation')
    return {'status':'passed','independent_outer_predictions':outer_count,'independent_inner_predictions':inner_count,'independent_confirmation_predictions':final_count,'checks':['manual Pearson ranks at correct boundaries','independent ridge solve','disjoint splits','selected k','Jaccard','interaction table','constant feature','shape and NaN errors'],'limits':['no proof that selected variables cause y','Jaccard not chance adjusted','fold stability descriptive, folds dependent']}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',default=str(e.ROOT/'experiment-result.json'));p.add_argument('--out',default=str(e.ROOT/'outputs/audit.json'));a=p.parse_args();out=audit(json.loads(Path(a.report).read_text()));e.dump(out,a.out);print(json.dumps(out,ensure_ascii=False))
