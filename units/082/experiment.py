"""合成恢复、真实数据边界与负对照；选择与评价数据严格分工。"""
import argparse,json,platform
from pathlib import Path
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
from generate_data import make_data
from latent_project import *
ROOT=Path(__file__).resolve().parent

def run(return_artifacts=False):
    data=make_data();result={'seed':data['seed'],'versions':{'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__},
         'selection':'K in [1,2,3,4], full covariance, reg .03; maximum validation mean log density; no refit on validation',
         'predictive_check_scope':'fixed fitted parameters, not full Bayesian posterior predictive','wine_features':data['wine_features']};artifacts={}
    for name in ('synthetic','wine','negative'):
        source=data[name];raw={k:np.asarray(source[k]['x']) for k in ('train','validation','test')}
        scaler=StandardScaler().fit(raw['train']);x={k:scaler.transform(v) for k,v in raw.items()}
        model,candidates=select_model(x['train'],x['validation']);r,logp=responsibilities(x['test'],model.weights_,model.means_,model.covariances_)
        joint=log_joint(x['test'],model.weights_,model.means_,model.covariances_);gap=hard_variational_gap(joint)
        entropy=-np.sum(r*np.log(np.maximum(r,np.finfo(float).tiny)),axis=1)
        report={'split_sizes':{k:len(v) for k,v in raw.items()},'candidates':candidates,'selected_k':model.n_components,'test_log_density':float(logp.mean()),
          'responsibility_max_error':float(np.max(np.abs(r-model.predict_proba(x['test'])))),
          'log_density_max_error':float(np.max(np.abs(logp-model.score_samples(x['test'])))),
          'hard_q_mean_ELBO_gap':float(gap.mean()),'mean_component_entropy':float(entropy.mean()),
          'uncertain_fraction_max_r_below_0.8':float(np.mean(r.max(axis=1)<.8)),
          'minimum_covariance_eigenvalue':float(min(np.linalg.eigvalsh(c).min() for c in model.covariances_))}
        pred=model.predict(x['test']);report['test_component_counts']={str(k):int(np.sum(pred==k)) for k in range(model.n_components)}
        if name!='negative':
            report['restart_stability']=stability(x['train'],x['test'],pred,model.n_components)
            report['bootstrap_stability']=stability(x['train'],x['test'],pred,model.n_components,bootstrap=True)
            checks,reps=predictive_check(model,x['test']);report['fixed_parameter_predictive_check']=checks
        else:reps=[]
        # 类别名称仅在冻结分析后做外部审计，不能当无监督模型的训练目标。
        if name=='synthetic':
            truth=source['truth'];report['truth_ari']=float(adjusted_rand_score(source['test']['z'],pred))
            true_means=scaler.transform(truth['means'])
            report['mean_recovery']=(match_means(model.means_,true_means) if model.n_components==3 else {'matched_mean_rmse':None,'reason':'selected K differs from truth; do not force one-to-one match'})
            true_cov=np.asarray(truth['covariances'])/scaler.scale_[None,:,None]/scaler.scale_[None,None,:]
            tr,tlog=responsibilities(x['test'],truth['weights'],true_means,true_cov)
            report['oracle_test_log_density']=float(tlog.mean());report['oracle_component_entropy']=float((-tr*np.log(np.maximum(tr,1e-300))).sum(axis=1).mean())
        elif name=='wine':
            report['external_cultivar_ari']=float(adjusted_rand_score(source['test']['cultivar'],pred))
            report['boundary']='two measured features from a small historical wine dataset; components are fitted density pieces, not proved biological kinds'
        result[name]=report;artifacts[name]={'x':x,'raw':source,'scaler':scaler,'model':model,'responsibilities':r,'predictive_replicates':reps}
    result['conjugate_posterior_predictive']=conjugate_predictive_check(np.asarray(data['negative']['train']['x'])[:,0],np.asarray(data['negative']['test']['x'])[:,0])
    if return_artifacts:return result,artifacts
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(ROOT/'outputs/result.json'));a=p.parse_args();r=run();out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False)+'\n');print(json.dumps(r,indent=2,ensure_ascii=False))
