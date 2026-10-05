"""Independently reconstruct metrics, threshold selection, logits and provenance."""
from pathlib import Path
import argparse,json
import numpy as np
from scipy.special import expit
from scipy.stats import norm
import experiment as e

def require(ok,msg):
    if not ok:raise AssertionError(msg)

def check_metrics(y,p,m):
    pred=np.array(p)>=m['threshold'];y=np.array(y)
    tp=sum(int(a and b==1) for a,b in zip(pred,y));fn=sum(int(not a and b==1) for a,b in zip(pred,y));fp=sum(int(a and b==0) for a,b in zip(pred,y));tn=len(y)-tp-fn-fp
    require((tp,fn,fp,tn)==(m['tp'],m['fn'],m['fp'],m['tn']),'confusion counts')
    require(abs((8*fn+fp)/len(y)-m['cost_per_case'])<1e-12,'cost')
    require(abs(sum((float(a)-int(b))**2 for a,b in zip(p,y))/len(y)-m['brier'])<1e-12,'Brier')
    n=tp+fn;z=norm.ppf(.975);rate=tp/n
    # Equivalent Wilson expression using counts, independently written.
    center=(tp+z*z/2)/(n+z*z);half=z*np.sqrt(tp*fn/n+z*z/4)/(n+z*z)
    require(np.allclose([center-half,center+half],m['recall_wilson95'],atol=1e-12),'Wilson')

def audit(r):
    d=np.load(e.ROOT/'data/draws.npz');checked=0
    for s in r['runs']:
        rep=s['rep'];X,y=d[f'X{rep}'],d[f'y{rep}'];yt=y[:2000];yv=y[2000:3200];ys=y[3200:]
        require(np.array_equal(s['test_labels'],ys),'test labels')
        for method,z in s['methods'].items():check_metrics(ys,z['probability'],z['metrics']);checked+=1
        for model,key in [('natural','natural_05'),('weighted','weighted_05'),('oversampled','oversampled_05')]:
            p=expit(X[3200:]@np.array(s['models'][model]['coef'])+s['models'][model]['intercept'])
            require(np.allclose(p,s['methods'][key]['probability'],rtol=0,atol=1e-12),'logit reconstruction')
        v=np.array(s['validation_probability']);cost=[]
        for t in e.THRESHOLDS:
            pred=v>=t;value=(8*np.sum((~pred)&(yv==1))+np.sum(pred&(yv==0)))/len(yv);cost.append(float(value))
        winner=float(e.THRESHOLDS[int(np.argmin(cost))]);require(winner==s['selected_threshold'],'validation threshold')
        require(np.allclose(cost,[t['cost'] for t in s['threshold_trials']],atol=1e-12),'threshold ledger')
        ids=np.array(s['oversample_source_ids']);require(ids.min()>=0 and ids.max()<2000,'resampling crossed split')
        require(np.sum(yt[ids]==1)==np.sum(yt[ids]==0),'balance training only')
        require(np.array_equal(np.sort(ids[yt[ids]==0]),np.flatnonzero(yt==0)),'negative originals preserved')
        pi=s['training_prevalence']
        for raw,corrected in [('weighted_05','weighted_corrected_05'),('oversampled_05','oversampled_corrected_05')]:
            q=np.array(s['methods'][raw]['probability']);p=pi*q/(pi*q+(1-pi)*(1-q))
            require(np.allclose(p,s['methods'][corrected]['probability'],atol=1e-12),'prior correction')
        require(s['test_positives']==int(ys.sum()),'natural prevalence unchanged')
    for s in r['label_runs']:
        rep=s['rep'];y=d[f'y{rep}'];obs=d[f'observed{rep}'];require(np.array_equal(s['observed_ids'],np.flatnonzero(obs)),'label observation provenance')
        for z in s['methods'].values():check_metrics(y[3200:],z['probability'],z['metrics']);checked+=1
    # Exact hand examples and ordinary absent-positive/invalid-input behavior.
    require(abs(r['hand']['weighted_q_at_p_point1']-.5)<1e-12,'weight example')
    require(abs(r['hand']['corrected_p_at_q_half']-.1)<1e-12,'prior example')
    require(e.metrics(np.zeros(3,dtype=int),np.array([.1,.2,.3]),.5)['recall'] is None,'no positive class handling')
    for source,target in [(0,.1),(.5,1)]:
        try:e.prior_correct([.2],source,target)
        except ValueError:pass
        else:raise AssertionError('invalid prior accepted')
    try:e.metrics([0,1],[.2,float('nan')],.5)
    except ValueError:pass
    else:raise AssertionError('NaN accepted')
    return {'status':'passed','metric_records_independently_checked':checked,'checks':['confusion matrix','cost','Brier','Wilson recall interval','stored coefficient logits','validation-only threshold ledger','training-only resample IDs','prior correction','observation IDs','absent minority and invalid inputs'],'limits':['oracle propensities known only in simulation','Wilson conditions on fixed classifier and observed positive count','no guarantee correction repairs arbitrary covariate or concept shift']}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',default=str(e.ROOT/'experiment-result.json'));p.add_argument('--out',default=str(e.ROOT/'outputs/audit.json'));a=p.parse_args();z=audit(json.loads(Path(a.report).read_text()));e.dump(z,a.out);print(json.dumps(z,ensure_ascii=False))
