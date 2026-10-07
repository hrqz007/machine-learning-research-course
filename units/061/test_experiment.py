"""Executable tests independent of Python's -O assertion removal."""
import argparse,copy,json,hashlib,tempfile,shutil,os
from pathlib import Path
import numpy as np
from adaboost import AdaBoost,best_stump,Stump,midpoint
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
 X=np.arange(1,7.)[:,None];y=np.array([-1,-1,1,-1,1,1]);m=AdaBoost(2).fit(X,y)
 close([h['error'] for h in m.history_],[1/6,.1],'two hand errors')
 close(m.alphas_,[.5*np.log(5),.5*np.log(9)],'half-log coefficients')
 close(m.history_[0]['weights_after'],[.1,.1,.1,.5,.1,.1],'round one weights')
 close(m.history_[1]['weights_after'],[1/18,1/18,.5,5/18,1/18,1/18],'round two weights')
 close([h['exp_loss'] for h in m.history_],[np.sqrt(5)/3,1/np.sqrt(5)],'independent exponential losses')
 check(m.stumps_[0]==Stump(0,2.5,-1) and m.stumps_[1]==Stump(0,4.5,-1),'deterministic stump rules')
 # Verify the algebra using recorded predictions, without the model's updater.
 log_product=0
 for h,s in zip(m.history_,m.stumps_):
  before=np.array(h['weights_before']);after=np.array(h['weights_after']);raw=before*np.exp(-h['alpha']*y*s.predict(X));close(after,raw/raw.sum(),'weight recursion '+str(h['round']));close(after.sum(),1,'weight normalization '+str(h['round']));log_product+=np.log(h['Z']);close(h['exp_loss'],np.exp(log_product),'product Z identity '+str(h['round']))
 # Exhaustive independent oracle over known thresholds/prediction signs.
 w=np.array([.04,.12,.21,.11,.23,.29]);stump,error=best_stump(X,y,w)
 candidates=[np.full(6,c) for c in [-1,1]]+[np.where(X[:,0]<=t,c,-c) for t in [1.5,2.5,3.5,4.5,5.5] for c in [-1,1]]
 close(error,min(sum(w[i] for i in range(6) if p[i]!=y[i]) for p in candidates),'brute-force weighted oracle')
 close(Stump(0,2.5,-1).predict(np.array([[2.5],[2.50001]])),[-1,1],'threshold equality goes left')
 perfect=AdaBoost(5).fit([[0],[1]],[-1,1]);check(perfect.stop_reason_=='perfect_stump' and len(perfect.stumps_)==1,'perfect stop');check(np.isfinite(perfect.alphas_).all(),'perfect finite cap')
 noedge=AdaBoost(5).fit([[0],[0]],[-1,1]);check(noedge.stop_reason_=='no_edge' and len(noedge.stumps_)==0,'no edge stop');close(noedge.predict([[0]]),[1],'zero score tie policy')
 single=AdaBoost(3).fit([[0],[1]],[1,1]);close(single.predict([[.5]]),[1],'single class handled by constant')
 for bad in [0,-1,1.5,True]:rejects(lambda bad=bad:AdaBoost(bad),'invalid rounds '+str(bad))
 for bad in [0,-.1,1.1,np.nan,np.inf]:rejects(lambda bad=bad:AdaBoost(2,bad),'invalid rate '+str(bad))
 for bad in [[],[[np.nan]],[[np.inf]],[1,2]]:rejects(lambda bad=bad:AdaBoost().fit(bad,[1]),'invalid X '+repr(bad))
 for bad in [[0,1],[-1,np.nan],[[1],[1]],[1]]:rejects(lambda bad=bad:AdaBoost().fit([[0],[1]],bad),'invalid y '+repr(bad))
 rejects(lambda:AdaBoost().predict([[0]]),'unfitted rejected');rejects(lambda:m.predict([[0,1]]),'dimension rejected');rejects(lambda:m.predict([[np.nan]]),'nonfinite predict rejected')
 check(np.isfinite(midpoint(1e308,1.7e308)),'large midpoint finite')
 repeat=AdaBoost(2).fit(X,y);close(repeat.decision_function(X),m.decision_function(X),'deterministic refit');m.fit(X,-y);check(len(m.history_)==2,'refit resets history')
 data=load_data();check(len(data['train']['id'])==180 and data['train']['flipped'].sum()==18,'frozen counts');check(sum(len(v['id']) for v in data.values())==540,'disjoint split row total')
 with tempfile.TemporaryDirectory() as td:
  temp=Path(td)/'data';shutil.copytree(ROOT/'data',temp)
  f=temp/'train.csv';f.write_text(f.read_text().replace('train_0000','train_9000'))
  rejects(lambda:load_data(temp),'tampered CSV rejected')
  manifest=json.loads((temp/'generation.json').read_text());manifest['sha256']['train.csv']=hashlib.sha256(f.read_bytes()).hexdigest();(temp/'generation.json').write_text(json.dumps(manifest))
  rejects(lambda:load_data(temp),'co-tampered CSV and digest rejected')
  target=Path(td)/'target';target.write_text('preserve');link=Path(td)/'link';link.symlink_to(target);rejects(lambda:safe_output(link),'symlink output rejected');hard=Path(td)/'hard';os.link(target,hard);rejects(lambda:safe_output(hard),'hardlink output rejected');check(target.read_text()=='preserve','unsafe target unchanged')
 fresh=run();stored=json.loads(Path(a.report).read_text());same(fresh,stored);check(True,'entire report recomputed and matched')
 for label,path in [('metric',('results','clean','metrics','test','from_zero_accuracy')),('coefficient',('toy',0,'alpha')),('weight',('toy',0,'weights_after',0))]:
  mutant=copy.deepcopy(stored);node=mutant
  for key in path[:-1]:node=node[key]
  node[path[-1]]+=.1;rejects(lambda mutant=mutant:same(fresh,mutant),'report mutation caught '+label)
 for mode in ['clean','noisy']:
  history=fresh['results'][mode]['history'];check(all(history[i]['exp_loss']<=history[i-1]['exp_loss']+1e-12 for i in range(1,len(history))),'exponential loss descent '+mode)
 result={'lesson':'061','status':'passed','check_count':len(checks),'checks':checks,'optimization_flag':not __debug__};write_json(safe_output(a.out),result);print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
