"""ML055: natural test prevalence, weights, thresholds, resampling, biased labels."""
from pathlib import Path
import argparse,json,hashlib,platform,time,warnings
import numpy as np
from scipy.special import expit,logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,roc_auc_score
from sklearn.exceptions import ConvergenceWarning
ROOT=Path(__file__).resolve().parent
PI=.08; C_FN=8.; C_FP=1.
THRESHOLDS=np.linspace(.01,.60,60)
MAIN=['natural_05','natural_cost','natural_tuned','weighted_05','weighted_corrected_05','oversampled_05','oversampled_corrected_05','oracle_cost']

def safe_output(path):
    path=Path(path).absolute()
    for part in (path,*path.parents):
        if part.is_symlink():raise ValueError('Refusing symlink output')
    if path.exists() and (not path.is_file() or path.stat().st_nlink>1):raise ValueError('Unsafe output')
    path.parent.mkdir(parents=True,exist_ok=True);return path

def dump(obj,path):safe_output(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False))

def wilson(success,total,z=1.959963984540054):
    if total<=0 or not 0<=success<=total:raise ValueError('Wilson interval requires 0 <= success <= positive total')
    p=success/total;den=1+z*z/total;mid=(p+z*z/(2*total))/den
    half=z*np.sqrt(p*(1-p)/total+z*z/(4*total**2))/den
    return [float(mid-half),float(mid+half)]

def metrics(y,p,threshold):
    y=np.asarray(y);p=np.asarray(p)
    if y.ndim!=1 or y.shape!=p.shape or not np.isin(y,[0,1]).all() or not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('binary labels and matching finite probabilities in [0,1] required')
    pred=p>=threshold;tp=int(np.sum(pred&(y==1)));fn=int(np.sum(~pred&(y==1)));fp=int(np.sum(pred&(y==0)));tn=int(np.sum(~pred&(y==0)))
    return {'threshold':float(threshold),'tp':tp,'fn':fn,'fp':fp,'tn':tn,'n':len(y),'positives':tp+fn,
            'recall':tp/(tp+fn) if tp+fn else None,'recall_wilson95':wilson(tp,tp+fn) if tp+fn else None,
            'precision':tp/(tp+fp) if tp+fp else None,'accuracy':(tp+tn)/len(y),
            'cost_per_case':(C_FN*fn+C_FP*fp)/len(y),'brier':float(np.mean((p-y)**2)),
            'average_precision':float(average_precision_score(y,p)) if tp+fn else None,
            'roc_auc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
            'mean_probability':float(p.mean()),'prevalence':float(y.mean())}

def fit_model(X,y,weight=None):
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        model=LogisticRegression(C=1.,solver='lbfgs',max_iter=1000,tol=1e-9)
        model.fit(X,y,sample_weight=weight)
    return model

def prior_correct(q,source_prior,target_prior):
    """Odds correction needs invariant X|Y and meaningful source probabilities."""
    if not 0<source_prior<1 or not 0<target_prior<1:raise ValueError('Priors must lie strictly between 0 and 1')
    q=np.asarray(q,float)
    if not np.isfinite(q).all() or np.any((q<0)|(q>1)):raise ValueError('Invalid probabilities')
    # Rational form preserves exact endpoint probabilities without log(0).
    a=(target_prior/(1-target_prior))/(source_prior/(1-source_prior))
    return a*q/(1-q+a*q)

def choose_threshold(y,p):
    rows=[{'threshold':float(t),'cost':metrics(y,p,t)['cost_per_case']} for t in THRESHOLDS]
    winner=min(rows,key=lambda z:(z['cost'],z['threshold']))['threshold']
    return winner,rows

def true_probability(X,prior=PI):return expit(logit(prior)+1.4*X[:,0]+.8*X[:,1]-.5*(1.4**2+.8**2))

def generate_data(directory):
    d=Path(directory);d.mkdir(parents=True,exist_ok=True);arrays={}
    for rep in range(6):
        rng=np.random.default_rng(5500+rep);y=(rng.random(7200)<PI).astype(int);X=rng.normal(size=(7200,4));X[:,0]+=1.4*y;X[:,1]+=.8*y
        # Label observation depends jointly on X and true Y; oracle propensities
        # are available only because this is a known simulation mechanism.
        obs_prob=np.where(y[:2000]==1,np.where(X[:2000,0]<1.,.15,.8),.6)
        observed=rng.random(2000)<obs_prob
        flip_prob=np.where(y[:2000]==1,.25,.02);yn=y[:2000].copy();flips=rng.random(2000)<flip_prob;yn[flips]=1-yn[flips]
        arrays.update({f'X{rep}':X,f'y{rep}':y,f'observed{rep}':observed,f'obs_prob{rep}':obs_prob,f'noisy_y{rep}':yn})
    np.savez_compressed(d/'draws.npz',**arrays)
    dump({'seeds':list(range(5500,5506)),'population_prior':PI,'n_train':2000,'n_validation':1200,'n_test':4000,'features':4,'class1_mean':[1.4,.8,0,0],'class0_mean':[0,0,0,0],'covariance':'identity','cost_false_negative':C_FN,'cost_false_positive':C_FP,'threshold_grid':THRESHOLDS.tolist(),'correction_target_prior':'training sample prevalence estimate; not test labels','noise_rates':{'positive_to_negative':.25,'negative_to_positive':.02},'selection_propensity':'y=1: .15 if x0<1 else .8; y=0: .6','ipw':'oracle 1/propensity; not learned from unavailable real labels','resampling':'training positives only, with replacement, to match count of negatives'},d/'protocol.json')

def run():
    start=time.perf_counter();d=np.load(ROOT/'data/draws.npz');runs=[];label_runs=[]
    for rep in range(6):
        X,y=d[f'X{rep}'],d[f'y{rep}'];Xt,yt=X[:2000],y[:2000];Xv,yv=X[2000:3200],y[2000:3200];Xs,ys=X[3200:],y[3200:]
        pi=float(yt.mean());n1=int(yt.sum());n0=len(yt)-n1
        if min(n1,n0)==0:raise ValueError('Both classes required in training')
        natural=fit_model(Xt,yt);nv=natural.predict_proba(Xv)[:,1];npred=natural.predict_proba(Xs)[:,1]
        threshold,trials=choose_threshold(yv,nv) # No test labels in threshold selection.
        weights=np.where(yt==1,len(yt)/(2*n1),len(yt)/(2*n0));weighted=fit_model(Xt,yt,weights);wp=weighted.predict_proba(Xs)[:,1]
        rng=np.random.default_rng(5580+rep);pos=np.flatnonzero(yt==1);neg=np.flatnonzero(yt==0)
        sample=np.r_[neg,rng.choice(pos,size=n0,replace=True)];over=fit_model(Xt[sample],yt[sample]);op=over.predict_proba(Xs)[:,1]
        predictions={'natural_05':(npred,.5),'natural_cost':(npred,1/9),'natural_tuned':(npred,threshold),'weighted_05':(wp,.5),'weighted_corrected_05':(prior_correct(wp,.5,pi),.5),'oversampled_05':(op,.5),'oversampled_corrected_05':(prior_correct(op,.5,pi),.5),'oracle_cost':(true_probability(Xs),1/9)}
        records={m:{'metrics':metrics(ys,p,t),'probability':p.tolist()} for m,(p,t) in predictions.items()}
        runs.append({'rep':rep,'training_positives':n1,'training_prevalence':pi,'validation_positives':int(yv.sum()),'test_positives':int(ys.sum()),'test_labels':ys.tolist(),'selected_threshold':threshold,'threshold_trials':trials,'validation_probability':nv.tolist(),'oversample_source_ids':sample.tolist(),'class_weights':{'0':float(len(yt)/(2*n0)),'1':float(len(yt)/(2*n1))},'models':{'natural':{'coef':natural.coef_[0].tolist(),'intercept':float(natural.intercept_[0])},'weighted':{'coef':weighted.coef_[0].tolist(),'intercept':float(weighted.intercept_[0])},'oversampled':{'coef':over.coef_[0].tolist(),'intercept':float(over.intercept_[0])}},'methods':records})
        obs=d[f'observed{rep}'];pr=d[f'obs_prob{rep}'];noisy=d[f'noisy_y{rep}']
        biased=fit_model(Xt[obs],yt[obs]);ipw=fit_model(Xt[obs],yt[obs],1/pr[obs]);noise=fit_model(Xt,noisy)
        # Treating missing labels as zero is an intentional incorrect control.
        missing_as_negative=yt.copy();missing_as_negative[~obs]=0;wrong=fit_model(Xt,missing_as_negative)
        label_predictions={'selected_only':biased.predict_proba(Xs)[:,1],'oracle_ipw':ipw.predict_proba(Xs)[:,1],'noisy_labels':noise.predict_proba(Xs)[:,1],'missing_as_negative':wrong.predict_proba(Xs)[:,1]}
        label_runs.append({'rep':rep,'observed_count':int(obs.sum()),'observed_positives':int(yt[obs].sum()),'observed_ids':np.flatnonzero(obs).tolist(),'methods':{m:{'metrics':metrics(ys,p,1/9),'probability':p.tolist()} for m,p in label_predictions.items()}})
    summary={m:{key:float(np.mean([r['methods'][m]['metrics'][key] for r in runs])) for key in ['recall','precision','accuracy','cost_per_case','brier','average_precision','roc_auc','mean_probability']} for m in MAIN}
    label_summary={m:{key:float(np.mean([r['methods'][m]['metrics'][key] for r in label_runs])) for key in ['recall','cost_per_case','brier']} for m in label_predictions}
    return {'unit':'055','runs':runs,'summary':summary,'label_runs':label_runs,'label_summary':label_summary,'hand':{'weighted_q_at_p_point1':(9*.1)/(9*.1+.9),'corrected_p_at_q_half':float(prior_correct(np.array([.5]),.5,.1)[0]),'cost_threshold':1/9,'wilson_16_of_20':wilson(16,20),'noise_observed_at_p_point1':.02+(.75-.02)*.1},'elapsed_seconds':time.perf_counter()-start,'data_sha256':hashlib.sha256((ROOT/'data/draws.npz').read_bytes()).hexdigest(),'python':platform.python_version()}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(ROOT/'outputs/result.json'));a=p.parse_args();r=run();dump(r,a.out);print(json.dumps({'main':r['summary'],'labels':r['label_summary']},ensure_ascii=False,indent=2))
