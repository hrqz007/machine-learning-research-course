"""Independent arithmetic, split/cost accounting, survival, and refit checks."""
import argparse,json,itertools
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import experiment as e

def require(ok,message):
    if not ok: raise AssertionError(message)

def audit(r):
    data=np.load(e.ROOT/'data/draws.npz'); checked=0; refits=0
    require(len(r['runs'])==18,'18 independent rep/method runs expected')
    for s in r['runs']:
        rep=s['rep']; X,y=data[f'X{rep}'],data[f'y{rep}']
        require(sum(z['requested_trees'] for z in s['fits'])==1080,'unequal budget')
        require(s['requested_trees']==s['completed_trees']==1080,'cost mismatch')
        require(s['final_refit_trees']==60,'refit budget')
        for z in s['fits']:
            tr,va=z['train_ids'],z['valid_ids']; require(set(tr).isdisjoint(va),'split leak')
            require(set(tr)|set(va)==set(range(240)),'unexpected sample ids')
            require(z['status']=='ok','unexpected main failure')
            manual=sum((float(y[i])-p)**2 for i,p in zip(va,z['prediction']))/len(va)
            require(abs(manual-z['mse'])<1e-10,'MSE mismatch'); checked+=1
        previous=None
        for st,stage in enumerate(s['stages']):
            ranks=stage['ranking']; reconstructed=[]
            cids=sorted(set(z['candidate'] for z in s['fits'] if z['stage']==st))
            if previous is not None: require(cids==sorted(previous),'survival identity')
            for cid in cids:
                rows=[z for z in s['fits'] if z['stage']==st and z['candidate']==cid]
                require(len(rows)==3,'expected three folds')
                reconstructed.append((sum(z['mse'] for z in rows)/3,cid))
            reconstructed.sort()
            require([c for _,c in reconstructed]==[q['candidate'] for q in ranks],'ranking mismatch')
            previous=[c for _,c in reconstructed[:[4,1,1][st]]] if s['method']=='halving' else [reconstructed[0][1]]
        require(s['winner']==previous[0],'winner mismatch')
        d,f=s['selected']; model=RandomForestRegressor(n_estimators=60,max_depth=d,max_features=f,
                   min_samples_leaf=2,n_jobs=1,random_state=5390+rep).fit(X[:240],y[:240])
        pred=model.predict(X[240:]); require(np.allclose(pred,s['test_prediction'],rtol=0,atol=1e-12),'refit predictions')
        require(abs(np.mean((y[240:]-pred)**2)-s['test_mse'])<1e-10,'test MSE');refits+=1
    require(r['failure_demo']['status']=='failed','failure not logged')
    require(r['failure_demo']['requested_trees']==15 and r['failure_demo']['completed_trees']==0,'failed attempt cost')
    require(r['hand']['budget_grid']==r['hand']['budget_halving']==1080,'hand budget')
    # A held-out label is deliberately changed. The selector receives only [:240].
    y2=data['y0'].copy();y2[240:]=1e6
    a=e.choose(data['X0'][:240],y2[:240],0,'random')
    b=next(s for s in r['runs'] if s['rep']==0 and s['method']=='random')
    require(list(a['selected'])==list(b['selected']) and abs(a['cv_mse']-b['cv_mse'])<1e-12,'test intervention')
    common=[]
    for label,call in [('MSE unequal shapes',lambda:e.mse([1,2],[1])),
                       ('MSE NaN',lambda:e.mse([float('nan')],[1])),
                       ('unknown method',lambda:e.choose(data['X0'][:240],data['y0'][:240],0,'unknown'))]:
        try:call()
        except ValueError:common.append(label)
        else:raise AssertionError(label)
    return {'status':'passed','checked_search_predictions':checked,'independent_final_refits':refits,
            'checks':['cost ledger','disjoint folds','stage survival','all MSE arithmetic','test-label intervention','separate failed attempt']+common,
            'not_claimed':['global optimum','equal wall-clock time','statistical superiority across real datasets']}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',default=str(e.ROOT/'experiment-result.json'));p.add_argument('--out',default=str(e.ROOT/'outputs/audit.json'));a=p.parse_args()
    result=audit(json.loads(Path(a.report).read_text()));e.dump(result,a.out);print(json.dumps(result,ensure_ascii=False))
