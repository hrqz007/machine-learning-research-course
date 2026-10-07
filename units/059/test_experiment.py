"""Independent risk arithmetic, paths, selection, reports and failure contracts.

Uses explicit exceptions rather than assert, so -O does not disable checks.
"""
from pathlib import Path
import argparse,json,hashlib,itertools
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from common import ROOT,write_json
from experiment import load,fit
from pruning import TreeArrays,export_tree,weakest_link_path,predict_probability,select_validation
CHECKS=[]
def check(v,name):
    if not bool(v):raise RuntimeError('FAILED: '+name)
    CHECKS.append(name)
def close(a,b,name,tol=1e-10):check(np.allclose(a,b,atol=tol,rtol=tol),name)
def reject(fn,name):
    try:fn()
    except (ValueError,TypeError):CHECKS.append(name);return
    raise RuntimeError('FAILED rejection: '+name)
def leaves_under(t,v,stops):
    if v in stops or t.children_left[v]<0:return [v]
    return leaves_under(t,int(t.children_left[v]),stops)+leaves_under(t,int(t.children_right[v]),stops)
def independent_predict(m,X,stops):
    out=[];t=m.tree_
    for row in np.asarray(X,dtype=np.float32):
        k=0
        while k not in stops:
            if t.children_left[k]<0:raise RuntimeError('unreachable report leaves')
            k=int(t.children_left[k] if row[t.feature[k]]<=t.threshold[k] else t.children_right[k])
        out.append(float(t.value[k,0,1]/t.value[k,0,:].sum()))
    return np.array(out)
def audit(report):
    CHECKS.clear();X,y=load('train');V,v=load('validation');T,t=load('test');m=fit(X,y);tr=m.tree_;membership=m.decision_path(X).toarray().astype(bool)
    risks=[]
    for k in range(tr.node_count):
        ys=y[membership[:,k]];p=float(sum(ys))/len(ys);risks.append(len(ys)/len(y)*2*p*(1-p))
        close(tr.n_node_samples[k],len(ys),f'node {k} sample count');close(tr.impurity[k],2*p*(1-p),f'node {k} Gini from labels')
    path=report['path'];allnodes=set(range(tr.node_count));previous=None
    for k,row in enumerate(path):
        stops=set(row['leaves']);check(stops<=allnodes and len(stops)==row['n_leaves'],f'path {k} leaf ids')
        check(set(leaves_under(tr,0,stops))==stops,f'path {k} partition')
        counts=np.sum(membership[:,list(stops)],axis=1);check(np.all(counts==1),f'path {k} exact row coverage')
        close(row['risk'],sum(risks[l] for l in stops),f'path {k} root-normalized risk')
        independent_candidates=[]
        def visit(node):
            if node in stops:return
            sub=leaves_under(tr,node,stops);independent_candidates.append((node,(risks[node]-sum(risks[l] for l in sub))/(len(sub)-1)))
            visit(int(tr.children_left[node]));visit(int(tr.children_right[node]))
        visit(0);check(len(independent_candidates)==len(row['candidates']),f'path {k} all candidates')
        by_id={q['node']:q['alpha'] for q in row['candidates']}
        for node,alpha in independent_candidates:close(by_id[node],alpha,f'path {k} alpha {node}')
        if k+1<len(path):close(path[k+1]['alpha'],min(a for _,a in independent_candidates),f'path {k} weakest link')
        if previous is not None:
            check(row['n_leaves']<previous['n_leaves'],f'path {k} smaller tree');check(row['risk']>=previous['risk']-1e-12,f'path {k} risk increases');check(row['alpha']>=previous['alpha']-1e-12,f'path {k} alpha monotone')
        previous=row
        p=independent_predict(m,V,stops);close(row['validation_brier'],np.mean((p-v)**2),f'path {k} validation Brier')
        p=independent_predict(m,X,stops);close(row['train_brier'],np.mean((p-y)**2),f'path {k} train Brier')
        close(row['risk'],2*row['train_brier'],f'path {k} Gini Brier identity')
        model=fit(X,y,ccp_alpha=report['library']['interior_comparisons'][k]['alpha']);pred=independent_predict(m,T,stops)
        close(model.predict_proba(T)[:,1],pred,f'path {k} sklearn interior predictions');close(report['library']['interior_comparisons'][k]['max_prediction_difference'],0,f'path {k} reported library comparison')
    check(path[-1]['leaves']==[0],'ends at root');check(path[0]['alpha']==0,'starts at alpha zero')
    best=min(range(len(path)),key=lambda k:(path[k]['validation_brier'],path[k]['n_leaves'],path[k]['depth'],k));check(report['choices']['post_index']==best,'post choice validation only')
    grid=report['pre_grid'];expected=[(d,l) for d in [1,2,3,4,6,None] for l in [1,5,15]]
    check(len(grid)==18,'all pre candidates')
    for k,(d,l) in enumerate(expected):
        q=grid[k];pre=fit(X,y,max_depth=d,min_samples_leaf=l);check((q['max_depth'],q['min_samples_leaf'])==(d,l),f'grid {k} settings');close(q['validation_brier'],np.mean((pre.predict_proba(V)[:,1]-v)**2),f'grid {k} Brier');check(q['n_leaves']==pre.get_n_leaves(),f'grid {k} leaves')
    prebest=min(range(len(grid)),key=lambda k:(grid[k]['validation_brier'],grid[k]['n_leaves'],grid[k]['depth'],k));check(report['choices']['pre_index']==prebest,'pre choice validation only')
    q=grid[prebest];pre=fit(X,y,max_depth=q['max_depth'],min_samples_leaf=q['min_samples_leaf'])
    preds={'unrestricted':m.predict_proba(T)[:,1],'pre':pre.predict_proba(T)[:,1],'post':independent_predict(m,T,set(path[best]['leaves'])),'constant':np.full(len(t),y.mean())}
    for name,p in preds.items():
        e=report['evaluation'][name];close(e['probability'],p,f'{name} held-out prediction');close(e['brier'],np.mean((p-t)**2),f'{name} held-out Brier');close(e['accuracy'],np.mean((p>.5)==t),f'{name} held-out accuracy')
    for name in ['train','validation','test']:check(report['data_sha256'][name]==hashlib.sha256((ROOT/'data'/f'{name}.csv').read_bytes()).hexdigest(),name+' data hash')
    sets=report['perturbation']['row_sets'];check(len(sets)==25,'25 perturbations')
    axis=np.array(report['perturbation']['probe_axis']);a,b=np.meshgrid(axis,axis);probe=np.column_stack([a.ravel(),b.ravel(),a.ravel(),np.zeros(a.size)])
    for name,kw in [('unrestricted',{}),('pre',{'max_depth':q['max_depth'],'min_samples_leaf':q['min_samples_leaf']}),('post',{'ccp_alpha':path[best]['alpha']})]:
        ps=[];r=report['stability'][name]
        for i,ids in enumerate(sets):
            check(len(ids)==427 and len(set(ids))==427 and min(ids)>=0 and max(ids)<450,f'{name} perturbation {i} rows')
            model=fit(X[ids],y[ids],**kw);ps.append(model.predict_proba(probe)[:,1]);check(model.get_n_leaves()==r['n_leaves'][i],f'{name} perturbation {i} leaves');check(int(model.tree_.feature[0])==r['root_features'][i],f'{name} perturbation {i} root')
        ps=np.asarray(ps);close(r['probability_variance_mean'],ps.var(axis=0).mean(),name+' variance')
        pair=np.mean([np.mean((ps[i]>.5)!=(ps[j]>.5)) for i in range(25) for j in range(i+1,25)])
        close(r['pair_disagreement'],pair,name+' direct pair disagreement');close(r['mean_probability'],ps.mean(axis=0),name+' probe means');close(r['sd_probability'],ps.std(axis=0),name+' probe sd')
    # A small tree allows exhaustive enumeration of every possible rooted subtree.
    toy=TreeArrays(np.array([1,2,-1,-1,5,-1,-1]),np.array([4,3,-1,-1,6,-1,-1]),np.array([0,1,-2,-2,1,-2,-2]),np.zeros(7),np.array([100,50,40,10,50,10,40]),np.array([.5,.32,.095,.32,.32,.32,.095]),np.array([.5,.2,.05,.8,.8,.2,.95]),2)
    hp=weakest_link_path(toy);close([s['risk'] for s in hp],[.14,.32,.5],'hand path risks');close([s['alpha'] for s in hp],[0,.09,.18],'hand path alpha');check([s['n_leaves'] for s in hp]==[4,2,1],'simultaneous equal links')
    candidates=[(.14,4),(.23,3),(.23,3),(.32,2),(.5,1)]
    for alpha in [0,.05,.09,.10,.15,.18,.2,.8]:
        idx=max(k for k,s in enumerate(hp) if s['alpha']<=alpha+1e-12);s=hp[idx];close(s['risk']+alpha*s['n_leaves'],min(r+alpha*l for r,l in candidates),'exhaustive optimality '+str(alpha))
    pure=fit(np.array([[0],[1],[2]]),np.zeros(3,dtype=int));puretree=export_tree(pure);check(len(weakest_link_path(puretree))==1,'pure stump path');close(predict_probability(puretree,[[4]],[0]),[0],'single class probability')
    tree=export_tree(m)
    for data in [[],[[float('nan')]*4],[[float('inf')]*4],[[1,2]],[[1e100]*4]]:reject(lambda d=data:predict_probability(tree,d,path[0]['leaves']),'invalid prediction '+repr(data))
    reject(lambda:predict_probability(tree,[[0]*4],[]),'empty terminal set');reject(lambda:predict_probability(tree,[[0]*4],[999999]),'invalid terminal index');reject(lambda:predict_probability(tree,[[0]*4],[0,1]),'overlapping terminals');reject(lambda:select_validation([]),'empty selection')
    rows=[{'validation_brier':.2,'n_leaves':3,'depth':2},{'validation_brier':.2,'n_leaves':2,'depth':1}];check(select_validation(rows)==1,'tie selects simpler')
    return {'status':'passed','checks':len(CHECKS),'names':CHECKS,'optimized':not __debug__}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',default='experiment-result.json');p.add_argument('--out',default='outputs/test-result.json');a=p.parse_args();r=audit(json.loads(Path(a.report).read_text()));write_json(a.out,r);print(json.dumps({'status':r['status'],'checks':r['checks'],'optimized':r['optimized']}))
