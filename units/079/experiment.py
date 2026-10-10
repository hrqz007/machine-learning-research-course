"""可解析真值、独立 ELBO 计算、逐坐标轨迹和方向相关的不确定性检查。"""
from pathlib import Path
import argparse,json
import numpy as np
from variational import posterior,coordinate_mean_field,elbo,gaussian_kl,linear_summary
ROOT=Path(__file__).resolve().parent

def run():
    model=json.loads((ROOT/'data/correlated.json').read_text());p=posterior(**model)
    q=coordinate_mean_field(p['precision'],p['natural_mean'])
    records=[]
    for row in q['trace']:
        C=np.diag(row['variance']);value=elbo(**model,q_mean=row['mean'],q_cov=C)
        kl=gaussian_kl(row['mean'],C,p['mean'],p['covariance'])
        records.append({**row,'elbo':value,'kl_q_p':kl,'decomposition_error':abs(p['log_evidence']-value-kl)})
    Q=np.diag(q['variance']);forward_cov=np.diag(np.diag(p['covariance']))
    independent=json.loads((ROOT/'data/independent.json').read_text());ip=posterior(**independent);iq=coordinate_mean_field(ip['precision'],ip['natural_mean'])
    variance_sweep=[]
    for noise in [.05,.1,.25,.5,1.,2.,10.]:
        case={**model,'noise_cov':[[noise]]};truth=posterior(**case);fit=coordinate_mean_field(truth['precision'],truth['natural_mean'])
        variance_sweep.append({'noise_variance':noise,'correlation':float(truth['covariance'][0,1]/truth['covariance'][0,0]),'true_variance':float(truth['covariance'][0,0]),'vi_variance':float(fit['variance'][0]),'variance_ratio':float(fit['variance'][0]/truth['covariance'][0,0])})
    result={'model':model,'posterior_mean':p['mean'].tolist(),'posterior_covariance':p['covariance'].tolist(),'precision':p['precision'].tolist(),'log_evidence':p['log_evidence'],'q_mean':q['mean'].tolist(),'q_variance':q['variance'].tolist(),'converged':q['converged'],'sweeps':q['sweeps'],'max_mean_error':float(np.max(np.abs(q['mean']-p['mean']))),'variance_ratio':(q['variance']/np.diag(p['covariance'])).tolist(),'elbo':records[-1]['elbo'],'kl_q_p':records[-1]['kl_q_p'],'family_gap':float(.5*np.log(np.prod(np.diag(p['precision']))/np.linalg.det(p['precision']))),'max_decomposition_error':max(r['decomposition_error'] for r in records),'min_elbo_increment':min(b['elbo']-a['elbo'] for a,b in zip(records,records[1:])),'trace':records,'forward_kl_optimum':{'mean':p['mean'].tolist(),'variance':np.diag(forward_cov).tolist(),'kl_p_q':gaussian_kl(p['mean'],p['covariance'],p['mean'],forward_cov),'kl_q_p':gaussian_kl(p['mean'],forward_cov,p['mean'],p['covariance'])},'linear_functionals':{},'independent_control':{'kl_q_p':gaussian_kl(iq['mean'],np.diag(iq['variance']),ip['mean'],ip['covariance']),'q_mean':iq['mean'].tolist(),'q_variance':iq['variance'].tolist()},'noise_sweep':variance_sweep}
    for name,weights,noise in [('sum',[1.,1.],0.),('difference',[1.,-1.],0.),('future_sum_observation',[1.,1.],.25)]:
        result['linear_functionals'][name]={'exact':linear_summary(p['mean'],p['covariance'],weights,noise),'mean_field':linear_summary(q['mean'],Q,weights,noise)}
    return result
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',default=str(ROOT/'outputs/result.json'));args=parser.parse_args();out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True);r=run();out.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)+'\n');print(json.dumps({k:r[k] for k in ['converged','sweeps','max_mean_error','variance_ratio','elbo','kl_q_p','max_decomposition_error']},indent=2))
