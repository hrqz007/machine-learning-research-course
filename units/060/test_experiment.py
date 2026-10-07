"""Independent bootstrap, vote, OOB, covariance and report audit, also under -O."""
from pathlib import Path
import argparse,json,hashlib,numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from common import ROOT,write_json,require
from experiment import load
from bagging import BootstrapEnsemble,positive_probability
CHECKS=[]
def check(v,name):
    if not bool(v):raise RuntimeError('FAILED: '+name)
    CHECKS.append(name)
def close(a,b,name,tol=1e-10):check(np.allclose(a,b,atol=tol,rtol=tol),name)
def reject(fn,name):
    try:fn()
    except (ValueError,TypeError):CHECKS.append(name);return
    raise RuntimeError('FAILED rejection '+name)
def probability(tree,X):
    # Independent column alignment rather than indexing column1 unconditionally.
    q=tree.predict_proba(X);out=np.zeros(len(X))
    for k,label in enumerate(tree.classes_):
        if label==1:out=q[:,k]
    return out
def verify_score(row,y,p,name):
    close(row['accuracy'],np.mean((np.asarray(p)>.5)==y),name+' accuracy');close(row['brier'],np.mean((np.asarray(p)-y)**2),name+' Brier')
def audit(r):
    CHECKS.clear();X,y,_=load('train');T,t,_=load('test');rng=np.random.default_rng(60060);samples=rng.integers(0,360,size=(80,360));seeds=rng.integers(0,2**31-1,size=80)
    computed_oob={}
    for m in [1,3,8]:
        row=r['ensembles'][str(m)];close(row['bootstrap_samples'],samples,f'm{m} exact bootstrap seed');close(row['tree_seeds'],seeds,f'm{m} tree seeds')
        trainp=[];testp=[];mask=[]
        for b,ids in enumerate(samples):
            check(len(ids)==360 and min(ids)>=0 and max(ids)<360,f'm{m} tree{b} bootstrap range');tree=DecisionTreeClassifier(max_features=m,min_samples_leaf=3,random_state=int(seeds[b])).fit(X[ids],y[ids]);testp.append(probability(tree,T));trainp.append(probability(tree,X));mask.append(np.array([i not in set(ids.tolist()) for i in range(360)]));close(row['test_individual_probability'][b],testp[-1],f'm{m} tree{b} independent prediction')
        trainp=np.array(trainp);testp=np.array(testp);mask=np.array(mask)
        for k,B in enumerate([1,5,20,80]):
            o=row['oob'][k];counts=mask[:B].sum(axis=0);valid=counts>0;manual=[]
            # Explicit rowwise leave-out voters catch in-bag leakage or wrong denominator.
            for i in np.flatnonzero(valid):manual.append(sum(trainp[b,i] for b in range(B) if mask[b,i])/counts[i])
            close(o['counts'],counts,f'm{m} B{B} OOB counts');close(o['coverage'],valid.mean(),f'm{m} B{B} coverage');check(all(o['probability'][i] is None for i in np.flatnonzero(~valid)),f'm{m} B{B} uncovered stays missing');close([o['probability'][i] for i in np.flatnonzero(valid)],manual,f'm{m} B{B} OOB values');verify_score(o,y[valid],manual,f'm{m} B{B} OOB');verify_score(row['curves'][k],t,testp[:B].mean(axis=0),f'm{m} B{B} test')
            if B==80:computed_oob[m]=float(np.mean((np.array(manual)-y[valid])**2))
        p=testp.mean(axis=0);close(row['test_probability'],p,f'm{m} soft vote');err=testp-t[None,:];cov=np.cov(err,bias=True);close(err.mean(axis=0).var(),cov.sum()/80**2,f'm{m} exact covariance identity');corr=np.corrcoef(err);close(row['empirical_error_correlation'],corr[np.triu_indices(80,1)].mean(),f'm{m} descriptive correlation');close(row['mean_individual_brier'],(err**2).mean(),f'm{m} individual Brier');check(np.mean((p-t)**2)<=np.mean(err**2)+1e-12,f'm{m} convexity average Brier bound')
        close(row['mean_unique_bootstrap'],np.mean([len(set(q)) for q in samples]),f'm{m} unique rows')
    selected=min(computed_oob,key=lambda m:(computed_oob[m],m));check(r['selection']['max_features']==selected,'OOB-only selected feature count');check(r['selection']['B']==80,'predeclared 80 trees')
    model=DecisionTreeClassifier(min_samples_leaf=3,random_state=60060).fit(X,y);p=probability(model,T);close(r['single']['probability'],p,'single predictions');verify_score(r['single'],t,p,'single')
    rf=RandomForestClassifier(n_estimators=80,max_features=selected,min_samples_leaf=3,bootstrap=True,oob_score=True,random_state=60060,n_jobs=1).fit(X,y);p=np.mean([probability(tree,T) for tree in rf.estimators_],axis=0);rr=r['library_random_forest'];close(rr['probability'],p,'library aggregation');verify_score(rr,t,p,'library test');total=np.zeros(360);counts=np.zeros(360)
    for tree,ids in zip(rf.estimators_,rf.estimators_samples_):
        held=np.array([i not in set(ids.tolist()) for i in range(360)]);total[held]+=probability(tree,X[held]);counts[held]+=1
    check(np.all(counts>0),'library full OOB coverage');oob=total/counts;close(rr['oob_brier'],np.mean((oob-y)**2),'library OOB Brier');close(rr['manual_oob_maxdiff'],np.max(np.abs(oob-rf.oob_decision_function_[:,1])),'library OOB difference');close(rr['manual_aggregation_maxdiff'],0,'library aggregation difference')
    GX,Gy,g=load('group_train');GT,Gt,gt=load('group_test');groups=g['group_id'].astype(int);check(set(groups).isdisjoint(set(gt['group_id'].astype(int))),'train and new groups disjoint');check(len(set(groups))==60 and len(Gy)==360,'group dimensions')
    for group in set(groups):
        ids=np.flatnonzero(groups==group);check(len(ids)==6,f'group{group} size');close(GX[ids],np.tile(GX[ids[0]],(6,1)),f'group{group} duplicate inputs');check(np.all(Gy[ids]==Gy[ids[0]]),f'group{group} shared label')
    # Reconstruct row-bootstrap group result from seeds, without ensemble.oob.
    rng=np.random.default_rng(60160);ids_all=rng.integers(0,360,size=(80,360));seeds=rng.integers(0,2**31-1,size=80);total=np.zeros(360);count=np.zeros(360,int);tp=[];leaked=eligible=0
    for b,ids in enumerate(ids_all):
        tree=DecisionTreeClassifier(max_features=3,min_samples_leaf=1,random_state=int(seeds[b])).fit(GX[ids],Gy[ids]);held=np.bincount(ids,minlength=360)==0;total[held]+=probability(tree,GX[held]);count[held]+=1;tp.append(probability(tree,GT));eligible+=int(held.sum());leaked+=sum(groups[i] in set(groups[ids]) for i in np.flatnonzero(held))
    gr=r['group_counterexample'];verify_score(gr['row_oob'],Gy[count>0],total[count>0]/count[count>0],'group row OOB');verify_score(gr['new_group_test'],Gt,np.mean(tp,axis=0),'new groups');close(gr['row_oob_sibling_fraction'],leaked/eligible,'sibling leakage fraction');check(gr['oob_with_sibling']==leaked and gr['row_oob_count']==eligible,'sibling counts')
    rng=np.random.default_rng(60260);total=np.zeros(360);count=np.zeros(360,int)
    for b in range(80):
        drawn=rng.integers(0,60,size=60);ids=np.concatenate([np.flatnonzero(groups==q) for q in drawn]);tree=DecisionTreeClassifier(random_state=b).fit(GX[ids],Gy[ids]);held=~np.isin(groups,drawn);total[held]+=probability(tree,GX[held]);count[held]+=1
    valid=count>0;verify_score(gr['group_oob'],Gy[valid],total[valid]/count[valid],'whole group OOB');close(gr['group_oob_coverage'],valid.mean(),'whole group coverage')
    rng=np.random.default_rng(60360);k=0
    for rho in [0.,.2,.8]:
        z=rng.normal(size=(12000,1));e=rng.normal(size=(12000,100));a=np.sqrt(rho)*z+np.sqrt(1-rho)*e
        for B in [1,5,20,100]:
            row=r['variance_simulation'][k];k+=1;close(row['theory'],rho+(1-rho)/B,f'rho{rho} B{B} theory');close(row['empirical'],a[:,:B].mean(axis=1).var(),f'rho{rho} B{B} simulation');check(abs(row['empirical']-row['theory'])<.05,'simulation agrees within sampling tolerance')
    close(r['bootstrap_theory']['row_absent_probability'],(359/360)**360,'finite n absent probability');close(r['bootstrap_theory']['expected_unique'],360*(1-(359/360)**360),'expected unique draws')
    for name in ['train','test','group_train','group_test']:check(r['data_sha256'][name]==hashlib.sha256((ROOT/'data'/f'{name}.csv').read_bytes()).hexdigest(),name+' hash')
    for label in [0,1]:
        small=BootstrapEnsemble(3,max_features=1,min_samples_leaf=1).fit([[0],[1],[2]],[label]*3);close(small.predict_probability([[4],[5]]),[label,label],f'single class{label} alignment')
    one=BootstrapEnsemble(1,max_features=1,min_samples_leaf=1).fit([[1]],[1]);p,c=one.oob();check(c[0]==0 and np.isnan(p[0]),'no voter remains missing')
    for kwargs in [{'n_estimators':0},{'n_estimators':True},{'n_estimators':1.5},{'max_features':0},{'min_samples_leaf':0}]:reject(lambda k=kwargs:BootstrapEnsemble(**k),'bad parameter '+repr(kwargs))
    reject(lambda:BootstrapEnsemble().predict_probability([[0]]),'unfitted model');reject(lambda:BootstrapEnsemble(max_features=3).fit([[0,1]],[1]),'too many features');reject(lambda:BootstrapEnsemble().fit([[0],[1]],[0,2]),'invalid labels');reject(lambda:BootstrapEnsemble().fit([[float('nan')]],[0]),'nonfinite training');reject(lambda:one.predict_probability([[1,2]]),'feature mismatch');reject(lambda:one.predict_probability([[1]],B=0),'empty prefix')
    return {'status':'passed','checks':len(CHECKS),'names':CHECKS,'optimized':not __debug__}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',default='experiment-result.json');p.add_argument('--out',default='outputs/test-result.json');a=p.parse_args();r=audit(json.loads(Path(a.report).read_text()));write_json(a.out,r);print(json.dumps({'status':r['status'],'checks':r['checks'],'optimized':r['optimized']}))
