"""OOB selection is frozen before independent test data are read."""
from pathlib import Path
import argparse,hashlib,warnings,numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from common import ROOT,write_json
from bagging import BootstrapEnsemble,positive_probability

def load(name):
    z=np.genfromtxt(ROOT/'data'/f'{name}.csv',delimiter=',',names=True);features=[k for k in z.dtype.names if k.startswith('x')]
    return np.column_stack([z[k] for k in features]),z['y'].astype(int),z
def score(y,p):return {'accuracy':float(np.mean((p>.5)==y)),'brier':float(np.mean((p-y)**2))}
def oob_record(model,y,B):
    p,count=model.oob(B);valid=count>0
    return {'B':B,'coverage':float(valid.mean()),'counts':count.tolist(),'probability':[None if np.isnan(v) else float(v) for v in p],**score(y[valid],p[valid])}
def run():
    X,y,_=load('train');models={};oob={}
    for m in [1,3,8]:
        model=BootstrapEnsemble(80,max_features=m).fit(X,y);models[m]=model;oob[m]=[oob_record(model,y,B) for B in [1,5,20,80]]
    selected=min(models,key=lambda m:(oob[m][-1]['brier'],m))
    selection={'max_features':selected,'B':80,'criterion':'minimum full-coverage OOB Brier among m=1,3,8; ties lower m','selected_before_test_read':True}
    # The independent test file is first read only after the decision above.
    T,t,_=load('test');results={}
    for m,model in models.items():
        individual=model.individual_probabilities(T);errors=individual-t[None,:]
        corr=np.corrcoef(errors);avg=float(corr[np.triu_indices(80,1)].mean())
        results[str(m)]={'bootstrap_samples':model.samples_.tolist(),'tree_seeds':model.seeds_.tolist(),'oob':oob[m],'curves':[{**score(t,individual[:B].mean(axis=0)),'B':B} for B in [1,5,20,80]],'test_individual_probability':individual.tolist(),'test_probability':individual.mean(axis=0).tolist(),'empirical_error_correlation':avg,'mean_individual_brier':float(np.mean(errors**2)),'mean_unique_bootstrap':float(np.mean([len(set(q)) for q in model.samples_])), 'min_samples_leaf':3}
    single=DecisionTreeClassifier(min_samples_leaf=3,random_state=60060).fit(X,y);singlep=positive_probability(single,T)
    rf=RandomForestClassifier(n_estimators=80,max_features=selected,min_samples_leaf=3,bootstrap=True,oob_score=True,random_state=60060,n_jobs=1).fit(X,y)
    rfp=rf.predict_proba(T)[:,1];rfavg=np.mean([positive_probability(tree,T) for tree in rf.estimators_],axis=0)
    # Audit library OOB independently from its saved sample membership.
    rf_sum=np.zeros(len(y));rf_count=np.zeros(len(y),dtype=int)
    for tree,ids in zip(rf.estimators_,rf.estimators_samples_):
        mask=np.bincount(ids,minlength=len(y))==0;rf_sum[mask]+=positive_probability(tree,X[mask]);rf_count[mask]+=1
    rfmanual=rf_sum/rf_count
    GX,Gy,g=load('group_train');GT,Gt,gt=load('group_test');gm=BootstrapEnsemble(80,max_features=3,min_samples_leaf=1,seed=60160).fit(GX,Gy);gp,gc=gm.oob();groups=g['group_id'].astype(int)
    leaked=eligible=0
    for b,ids in enumerate(gm.samples_):
        included=set(groups[ids]);held=np.flatnonzero(gm.oob_mask_[b]);eligible+=len(held);leaked+=sum(groups[i] in included for i in held)
    # Group-bootstrap OOB excludes whole objects; it is a contrast, not a tune.
    rng=np.random.default_rng(60260);group_sum=np.zeros(len(Gy));group_count=np.zeros(len(Gy),int)
    for b in range(80):
        drawn=rng.integers(0,60,size=60);ids=np.concatenate([np.flatnonzero(groups==q) for q in drawn]);tree=DecisionTreeClassifier(random_state=b).fit(GX[ids],Gy[ids]);held=~np.isin(groups,drawn);group_sum[held]+=positive_probability(tree,GX[held]);group_count[held]+=1
    valid=group_count>0;group_oob=group_sum[valid]/group_count[valid]
    # Equal-variance, common-component simulation illustrates the algebra.
    rng=np.random.default_rng(60360);sim=[]
    for rho in [0.,.2,.8]:
        z=rng.normal(size=(12000,1));e=rng.normal(size=(12000,100));a=np.sqrt(rho)*z+np.sqrt(1-rho)*e
        for B in [1,5,20,100]:sim.append({'rho':rho,'B':B,'theory':rho+(1-rho)/B,'empirical':float(a[:,:B].mean(axis=1).var())})
    return {'lesson':'060','sizes':{'train':360,'test':360},'selection':selection,'ensembles':results,'single':{**score(t,singlep),'probability':singlep.tolist()},'library_random_forest':{**score(t,rfp),'probability':rfp.tolist(),'oob_brier':float(np.mean((rfmanual-y)**2)),'manual_aggregation_maxdiff':float(np.max(np.abs(rfp-rfavg))),'manual_oob_maxdiff':float(np.max(np.abs(rf.oob_decision_function_[:,1]-rfmanual))),'same_bootstrap_as_custom':False},'group_counterexample':{'row_oob':score(Gy[gc>0],gp[gc>0]),'new_group_test':score(Gt,gm.predict_probability(GT)),'group_oob':score(Gy[valid],group_oob),'group_oob_coverage':float(valid.mean()),'row_oob_sibling_fraction':leaked/eligible,'row_oob_count':eligible,'oob_with_sibling':int(leaked),'train_groups':60,'test_groups':60,'rows_per_group':6},'variance_simulation':sim,'bootstrap_theory':{'row_absent_probability':float((1-1/360)**360),'expected_unique':float(360*(1-(1-1/360)**360))},'data_sha256':{n:hashlib.sha256((ROOT/'data'/f'{n}.csv').read_bytes()).hexdigest() for n in ['train','test','group_train','group_test']}}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='outputs/result.json');a=p.parse_args();write_json(a.out,run())
