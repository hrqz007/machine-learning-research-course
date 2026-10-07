"""Independent analytical, finite-difference, edge and report-integrity tests."""
import argparse,copy,json,hashlib,tempfile,shutil,os
from pathlib import Path
import numpy as np
from gbdt import GBDT,RegressionTree,sigmoid,loss,negative_gradient
from common import ROOT,load_data,write_json,safe_output
from experiment import run
checks=[]
def check(ok,name):
    if not bool(ok):raise RuntimeError('FAILED: '+name)
    checks.append(name)
def close(a,b,name,atol=1e-11):check(np.allclose(a,b,atol=atol,rtol=1e-10),name)
def rejects(action,name):
    try:action()
    except (ValueError,RuntimeError,TypeError,KeyError):checks.append(name);return
    raise RuntimeError('FAILED to reject: '+name)
def same(a,b,path='report'):
    if isinstance(a,dict):
        if set(a)!=set(b):raise ValueError(path+' keys changed')
        for k in a:same(a[k],b[k],path+'.'+k)
    elif isinstance(a,list):
        if len(a)!=len(b):raise ValueError(path+' length changed')
        for i,(u,v) in enumerate(zip(a,b)):same(u,v,path+f'[{i}]')
    elif isinstance(a,(int,float)):
        if not np.isfinite(b) or not np.isclose(a,b,rtol=1e-10,atol=1e-11):raise ValueError(path+' numerical mismatch')
    elif a!=b:raise ValueError(path+' mismatch')
def main():
 p=argparse.ArgumentParser();p.add_argument('--report',default=str(ROOT/'experiment-result.json'));p.add_argument('--out',default=str(ROOT/'outputs/tests.json'));a=p.parse_args()
 X=np.arange(1,5.)[:,None];yr=np.array([1,1,3,3.]);yc=np.array([0,0,1,1.])
 reg=GBDT('squared',2,.5,1,1).fit(X,yr);close(reg.init_,2,'squared optimal constant');close(reg.losses_,[.5,.125,.03125],'hand squared losses');close(reg.history_[0]['pseudo_residual'],[-1,-1,1,1],'hand squared residual');close(reg.predict(X),[1.25,1.25,2.75,2.75],'hand two-round predictions')
 clf=GBDT('logistic',2,.5,1,1).fit(X,yc);close(clf.init_,0,'logit initial constant');close(clf.history_[0]['pseudo_residual'],[-.5,-.5,.5,.5],'logit hand residual');close(clf.history_[0]['leaf_values'],[-2,2],'Newton first leaf values');close(clf.history_[1]['leaf_values'],[-(1+np.exp(-1)),1+np.exp(-1)],'Newton second leaf values');close(clf.losses_,[np.log(2),np.log1p(np.exp(-1)),np.log1p(np.exp(-1-.5*(1+np.exp(-1))))],'logistic independent hand losses')
 close(clf.predict_proba(X).sum(axis=1),np.ones(4),'probabilities sum to one');check(np.all((clf.predict_proba(X)>=0)&(clf.predict_proba(X)<=1)),'probabilities bounded');close(sigmoid(np.array([-1000,0,1000])),[0,.5,1],'stable sigmoid extremes');check(np.isfinite(loss(yc,np.array([-1000,1000,-1000,1000.]),'logistic')),'stable logistic extremes')
 for kind,y in [('squared',yr),('logistic',yc)]:
  F=np.array([-.7,.2,1.1,-1.3]);eps=1e-5;num=[]
  for i in range(4):
   plus=F.copy();minus=F.copy();plus[i]+=eps;minus[i]-=eps;num.append(-4*(loss(y,plus,kind)-loss(y,minus,kind))/(2*eps))
  close(negative_gradient(y,F,kind),num,'finite difference per-sample gradient '+kind,1e-8)
  model=GBDT(kind,8,.3,2,1).fit(X,y);previous=np.full(4,model.init_)
  for i,h in enumerate(model.history_):
   close(h['pseudo_residual'],negative_gradient(y,previous,kind),'stored residual '+kind+str(i));previous+=h['step']*np.array(h['direction']);close(h['score'],previous,'score accumulation '+kind+str(i));close(h['loss'],loss(y,previous,kind),'stage loss '+kind+str(i))
  check(np.max(np.diff(model.losses_))<=1e-12,'descent '+kind)
 gradient=GBDT('logistic',1,.5,1,1,'gradient').fit(X,yc);close(gradient.history_[0]['direction'],[-.5,-.5,.5,.5],'gradient-only leaf interpretation');close(gradient.decision_function(X),[-.25,-.25,.25,.25],'gradient-only actual step')
 tree=RegressionTree(1,1).fit(X,yr);check(tree.root_.threshold==2.5,'tree threshold');close(tree.predict([[2.5],[2.500001]]),[1,3],'left equality semantics');close(RegressionTree(0,1).fit(X,yr).predict(X),[2]*4,'depth-zero constant');close(RegressionTree(3,3).fit(X,yr).predict(X),[2]*4,'minimum leaf constraint');close(GBDT('squared',3).fit([[1],[1]],[2,2]).predict([[1]]),[2],'constant features and targets')
 for bad in ['unknown','MSE',None]:rejects(lambda bad=bad:GBDT(bad),'bad loss '+str(bad))
 for bad in [0,-1,1.2,True]:rejects(lambda bad=bad:GBDT(n_estimators=bad),'bad rounds '+str(bad))
 for bad in [0,-1,np.nan,np.inf,1.2]:rejects(lambda bad=bad:GBDT(learning_rate=bad),'bad rate '+str(bad))
 for bad in [-1,1.2,True]:rejects(lambda bad=bad:GBDT(max_depth=bad),'bad depth '+str(bad))
 for bad in [0,-1,True]:rejects(lambda bad=bad:GBDT(min_samples_leaf=bad),'bad leaf '+str(bad))
 for bad in [[],[[np.nan]],[[np.inf]],[1,2]]:rejects(lambda bad=bad:GBDT().fit(bad,[1]),'bad X '+repr(bad))
 for bad in [[0,0,0,0],[1,1,1,1],[0,1,2,1],[0,1,np.nan,0],[[0],[1],[0],[1]]]:rejects(lambda bad=bad:GBDT('logistic').fit(X,bad),'bad logistic y '+repr(bad))
 rejects(lambda:GBDT().predict(X),'unfitted rejected');rejects(lambda:reg.predict([[1,2]]),'wrong feature count');rejects(lambda:reg.predict_proba(X),'regression probability rejected');rejects(lambda:reg.fit(X,[1,2]),'wrong target length');rejects(lambda:reg.predict([[np.inf]]),'nonfinite predict rejected');rejects(lambda:loss(yr,yr,'bad'),'unknown loss helper');rejects(lambda:negative_gradient(yr,yr,'bad'),'unknown gradient helper')
 reg.fit(X,yr);check(len(reg.history_)==2,'refit resets');close(reg.predict(X),[1.25,1.25,2.75,2.75],'refit deterministic')
 data=load_data();check([len(data[k]['id']) for k in ['train','validation','test']]==[160,100,220],'frozen counts')
 with tempfile.TemporaryDirectory() as td:
  temp=Path(td)/'data';shutil.copytree(ROOT/'data',temp);f=temp/'train.csv';f.write_text(f.read_text().replace('train_0000','train_9000'));rejects(lambda:load_data(temp),'tampered CSV rejected');manifest=json.loads((temp/'generation.json').read_text());manifest['sha256']['train.csv']=hashlib.sha256(f.read_bytes()).hexdigest();(temp/'generation.json').write_text(json.dumps(manifest));rejects(lambda:load_data(temp),'co-tampered digest rejected')
  target=Path(td)/'target';target.write_text('preserve');link=Path(td)/'link';link.symlink_to(target);rejects(lambda:safe_output(link),'symlink rejected');hard=Path(td)/'hard';os.link(target,hard);rejects(lambda:safe_output(hard),'hardlink rejected');check(target.read_text()=='preserve','unsafe target unchanged')
 fresh=run();stored=json.loads(Path(a.report).read_text());same(fresh,stored);check(True,'entire report fresh recomputation')
 for kind in ['squared','logistic']:check(fresh['results'][kind]['max_stage_difference']<1e-10,'all sklearn stage raw scores '+kind)
 for label,path in [('metric',('results','squared','metrics','test','from_zero_loss')),('gradient',('toy','logistic',0,'pseudo_residual',0)),('step',('toy','squared',0,'step'))]:
  mutant=copy.deepcopy(stored);node=mutant
  for key in path[:-1]:node=node[key]
  node[path[-1]]+=.1;rejects(lambda mutant=mutant:same(fresh,mutant),'report mutation caught '+label)
 result={'lesson':'062','status':'passed','check_count':len(checks),'checks':checks,'optimization_flag':not __debug__};write_json(safe_output(a.out),result);print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
