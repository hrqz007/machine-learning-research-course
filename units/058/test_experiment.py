"""Independent arithmetic and structural audit; checks survive python -O."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from common import ROOT,write_json
from tree import ShallowTree,impurity,candidates,best_split,midpoint
from experiment import load
from generate_data import make_data

CHECKS=[]
def check(condition,name):
    if not bool(condition):raise RuntimeError('FAILED: '+name)
    CHECKS.append(name)
def close(a,b,name,tol=1e-10):check(np.allclose(a,b,rtol=tol,atol=tol),name)
def rejects(fn,name):
    try:fn()
    except (ValueError,TypeError):CHECKS.append(name);return
    raise RuntimeError('FAILED: expected rejection: '+name)

def independent_impurity(y,criterion):
    if criterion=='squared_error':
        mean=sum(float(v) for v in y)/len(y)
        return sum((float(v)-mean)**2 for v in y)/len(y)
    p=sum(float(v) for v in y)/len(y)
    if criterion=='gini':return 1-p*p-(1-p)*(1-p)
    return sum(-q*np.log2(q) for q in [p,1-p] if q>0)

def manual_prediction(nodes,X):
    ids=[]
    for row in X:
        i=0
        while nodes[i]['feature'] is not None:
            n=nodes[i];i=n['left'] if row[n['feature']]<=n['threshold'] else n['right']
        ids.append(i)
    return np.array(ids),np.array([nodes[i]['value'] for i in ids])

def audit_model(state,X,y):
    criterion=state['criterion'];nodes=state['nodes'];leaf_ids=[];global_gain=0.
    for i,n in enumerate(nodes):
        prefix=f'{criterion} node {i} '
        ids=np.asarray(n['train_indices']); yy=y[ids]
        check(n['id']==i,prefix+'consecutive ID')
        check(n['n']==len(ids) and len(ids)==len(set(ids)),prefix+'unique support')
        close(n['value'],sum(yy)/len(yy),prefix+'leaf mean/probability')
        close(n['impurity'],independent_impurity(yy,criterion),prefix+'independent impurity')
        if i==0:check(np.array_equal(ids,np.arange(len(y))),prefix+'full training support')
        check(n['depth']<=state['max_depth'],prefix+'depth bound')
        feasible=[]
        for j in range(X.shape[1]):
            vals=sorted(set(float(v) for v in X[ids,j]))
            for a,b in zip(vals[:-1],vals[1:]):
                t=a/2+b/2
                if t>=b or t<a:t=a
                mask=X[ids,j]<=t
                if min(mask.sum(),(~mask).sum())<state['min_samples_leaf']:continue
                w=mask.mean()*independent_impurity(yy[mask],criterion)+(~mask).mean()*independent_impurity(yy[~mask],criterion)
                feasible.append((n['impurity']-w,j,t))
        if n['feature'] is None:
            check(n['left'] is None and n['right'] is None,prefix+'leaf children absent')
            leaf_ids.extend(ids.tolist())
            if n['depth']<state['max_depth']:
                check(not feasible or max(a[0] for a in feasible)<=1e-12 or n['impurity']<=1e-12,prefix+'valid early stopping')
            continue
        best=feasible[0]
        for row in feasible[1:]:
            if row[0]>best[0]+1e-12:best=row
        check(n['feature']==best[1],prefix+'best feature including tie rule')
        close(n['threshold'],best[2],prefix+'best threshold')
        close(n['gain'],best[0],prefix+'maximal local gain')
        left=nodes[n['left']];right=nodes[n['right']];mask=X[ids,n['feature']]<=n['threshold']
        check(np.array_equal(left['train_indices'],ids[mask]),prefix+'left routing')
        check(np.array_equal(right['train_indices'],ids[~mask]),prefix+'right routing')
        check(min(left['n'],right['n'])>=state['min_samples_leaf'],prefix+'minimum leaf support')
        check(left['depth']==right['depth']==n['depth']+1,prefix+'child depth')
        global_gain+=len(ids)/len(y)*n['gain']
    check(sorted(leaf_ids)==list(range(len(y))),criterion+' partition exactly once')
    leaf_loss=sum(n['n']/len(y)*n['impurity'] for n in nodes if n['feature'] is None)
    close(nodes[0]['impurity']-leaf_loss,global_gain,criterion+' telescoping impurity identity')

def main(report):
    CHECKS.clear();r=json.loads(Path(report).read_text())
    check(r['unit']=='058','unit identity')
    regenerated=make_data()
    for name,arr in regenerated.items():
        x,y=load(name);close(np.column_stack([x,y]),arr,name+' generated offline values',1e-14)
    for criterion,state in r['models'].items():
        stem='regression' if criterion=='squared_error' else 'classification'
        X,y=load(stem+'_train');tx,ty=load(stem+'_test')
        audit_model(state,X,y)
        ids,values=manual_prediction(state['nodes'],tx)
        pred=values if criterion=='squared_error' else (values>.5).astype(int)
        out=r['test'][criterion]
        check(out['n_test']==len(ty),criterion+' held-out denominator')
        close(out['target'],ty,criterion+' held-out labels');close(out['leaf_ids'],ids,criterion+' held-out leaf IDs')
        close(out['prediction'],pred,criterion+' independent held-out prediction')
        if criterion=='squared_error':
            ref=DecisionTreeRegressor(max_depth=2,min_samples_leaf=12,random_state=58).fit(X,y)
            close(out['mse'],np.mean((pred-ty)**2),'regression held-out MSE')
            close(out['baseline_mse'],np.mean((y.mean()-ty)**2),'regression baseline uses training mean')
        else:
            ref=DecisionTreeClassifier(criterion=criterion,max_depth=2,min_samples_leaf=12,random_state=58).fit(X,y)
            close(out['probability'],values,criterion+' independent probability')
            close(out['accuracy'],np.mean(pred==ty),criterion+' held-out accuracy')
            close(out['brier'],np.mean((values-ty)**2),criterion+' held-out Brier')
            close(out['baseline_accuracy'],np.mean((y.mean()>.5)==ty),criterion+' training-only baseline')
        close(out['library_prediction'],ref.predict(tx),criterion+' fresh sklearn comparison')
        close(out['max_prediction_difference'],np.max(np.abs(pred-ref.predict(tx))),criterion+' reported parity')
        fresh=ShallowTree(criterion,2,12).fit(X,y)
        check(json.dumps(fresh.state(),sort_keys=True)==json.dumps(state,sort_keys=True),criterion+' deterministic refit')
    frozen={k:r[k] for k in ['protocol','models','baseline_class','baseline_regression']}
    digest=hashlib.sha256(json.dumps(frozen,sort_keys=True,allow_nan=False).encode()).hexdigest()
    check(digest==r['frozen_sha256_before_test_read'],'frozen state hash')
    close(impurity([0,0,1,1]),.5,'Gini hand .5');close(impurity([0,0,1,1],'entropy'),1.,'entropy hand 1 bit')
    close(impurity([1,2,2,8,9,8],'squared_error'),34/3,'regression parent variance')
    x=np.arange(1,7)[:,None]; yc=np.array([0,0,1,0,1,1]);yr=np.array([1,2,2,8,9,8])
    for criterion,y in [('gini',yc),('entropy',yc),('squared_error',yr)]:
        got=candidates(x,y,criterion);saved=r['hand'][criterion]
        check(got==saved,criterion+' hand candidate report')
        for row in saved:
            left=x[:,0]<=row['threshold'];weighted=left.mean()*independent_impurity(y[left],criterion)+(~left).mean()*independent_impurity(y[~left],criterion)
            close(row['weighted_impurity'],weighted,criterion+' independent hand threshold '+str(row['threshold']))
    check(best_split(x,yc)['threshold']==2.5,'equal-gain threshold tie chooses 2.5')
    dup=np.column_stack([x,x]);check(best_split(dup,yc)['feature']==0,'equal-gain feature tie chooses column 0')
    close(best_split(x,yr,'squared_error')['gain'],100/9,'regression best gain')
    # Boundary convention is verified in float64 without sklearn float32 casting.
    boundary=ShallowTree('gini',1,1).fit([[0],[2]],[0,1]);t=boundary.nodes_[0]['threshold']
    q=np.array([[np.nextafter(t,-np.inf)],[t],[np.nextafter(t,np.inf)]])
    check(np.array_equal(boundary.predict(q),[0,0,1]),'nextafter and equal threshold routing')
    close(boundary.predict_proba(q),[[1,0],[1,0],[0,1]],'boundary probability')
    leaf=ShallowTree('gini',0).fit([[0],[1]],[0,1]);check(leaf.predict([[99]])[0]==0,'class tie chooses 0')
    close(leaf.predict_proba([[99]]),[[.5,.5]],'constant leaf frequency')
    pure=ShallowTree().fit([[1],[1]],[1,1]);check(len(pure.nodes_)==1,'pure node no split')
    constant=ShallowTree().fit([[1],[1]],[0,1]);check(len(constant.nodes_)==1,'constant feature no split')
    repeat=ShallowTree().fit([[0],[0],[1],[1]],[0,0,1,1]);check(len(repeat.nodes_)==3,'duplicate values move together')
    tiny=ShallowTree(min_samples_leaf=4).fit(x,yc);check(len(tiny.nodes_)==1,'minimum leaf constraint excludes all candidates')
    xor=np.array([[0,0],[0,1],[1,0],[1,1]]);xy=np.array([0,1,1,0]);tree=ShallowTree().fit(xor,xy)
    check(len(tree.nodes_)==1 and r['xor']['greedy_accuracy']==.5,'XOR strictly-positive greedy stopping')
    check(np.array_equal((xor[:,0]!=xor[:,1]).astype(int),r['xor']['depth2_constructed_prediction']),'XOR depth2 exact construction')
    for a,b in [(1e308,np.nextafter(1e308,np.inf)),(-1e308,1e308),(0.,np.nextafter(0.,1.))]:
        check(a<=midpoint(a,b)<b,'finite separating midpoint '+str(a))
    for kw in [{'max_depth':-1},{'max_depth':1.5},{'max_depth':True},{'max_depth':21},{'min_samples_leaf':0},{'min_samples_leaf':1.1},{'min_samples_leaf':False},{'criterion':'invalid'}]:rejects(lambda kw=kw:ShallowTree(**kw),'invalid parameter '+str(kw))
    for xx,yy in [([],[]),([1,2],[0,1]),([[1],[2]],[0]),([[np.nan],[2]],[0,1]),([[np.inf],[2]],[0,1]),([[1],[2]],[0,np.nan]),([[1],[2]],[0,2]),([[1],[2]],[[0],[1]])]:rejects(lambda xx=xx,yy=yy:ShallowTree().fit(xx,yy),'invalid training input '+str(len(CHECKS)))
    rejects(lambda:ShallowTree().predict([[1]]),'unfitted prediction')
    rejects(lambda:boundary.predict([[1,2]]),'wrong feature count')
    rejects(lambda:boundary.predict([[np.nan]]),'missing prediction input')
    rejects(lambda:ShallowTree('squared_error').fit([[0],[1]],[1,2]).predict_proba([[0]]),'regression probability rejection')
    rejects(lambda:impurity([1e308,-1e308],'squared_error'),'regression numerical overflow rejection')
    return {'status':'passed','checks':len(CHECKS),'optimized_python':not __debug__,'report':Path(report).name,'check_names':CHECKS}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',default=str(ROOT/'experiment-result.json'));p.add_argument('--out',default=str(ROOT/'outputs/test-result.json'));a=p.parse_args()
    result=main(a.report);write_json(a.out,result);print(json.dumps({k:v for k,v in result.items() if k!='check_names'}))
