"""同一已知答案的问题：比较独立MC、IS、MH与Gibbs的计算误差。"""
from pathlib import Path
import argparse,json
import numpy as np
from samplers import *
ROOT=Path(__file__).resolve().parent

def run(return_chains=False):
    data=json.loads((ROOT/'data/model.json').read_text()); y=np.array(data['observations'])
    mean,var=analytic_posterior(y);target=lambda x: log_posterior(x,y)
    n=12000;warmup=2000; starts=[-5.,-1.,3.,7.]; records={}; all_chains={}
    for name,scale in [('tiny',.005),('balanced',.65),('huge',12.)]:
        chains=[];acceptance=[]
        for i,start in enumerate(starts):
            chain,accept=metropolis(target,start,scale,n+warmup,7800+i)
            chains.append(chain[warmup:]);acceptance.append(accept)
        chains=np.array(chains);all_chains[name]=chains
        effective=[ess(x) for x in chains]
        records[name]={'proposal_sd':scale,'mean':float(chains.mean()),'mean_error':float(chains.mean()-mean),
            'acceptance':acceptance,'ess_per_chain':effective,'split_rhat':split_rhat(chains),
            'naive_se':float(chains.std(ddof=1)/np.sqrt(chains.size)),
            'mcse_heuristic':float(np.sqrt(sum(np.var(x,ddof=1)/e for x,e in zip(chains,effective)))/len(chains))}
    rng=np.random.default_rng(7830);direct=rng.normal(mean,np.sqrt(var),48000)
    iid={'mean':float(direct.mean()),'se':float(direct.std(ddof=1)/np.sqrt(len(direct)))}
    integral_samples=rng.uniform(0,1,20000)**2
    integral={'estimate':float(integral_samples.mean()),'exact':1/3,'se':float(integral_samples.std(ddof=1)/np.sqrt(len(integral_samples)))}
    importance={str(s):dict(zip(['mean','weight_ess','max_weight'],importance_normal(y,12000,s,7840))) for s in [.2,2.]}
    gibbs={}
    for rho in [.2,.95]:
        g=gibbs_gaussian(rho,22000,7850)[2000:]
        gibbs[str(rho)]={'mean':g.mean(axis=0).tolist(),'covariance':np.cov(g.T).tolist(),'ess_x':ess(g[:,0])}
    result={'unit':'078','posterior':{'mean':float(mean),'variance':float(var)},'retained_per_chain':n,'warmup':warmup,
            'monte_carlo_integral':integral,'iid':iid,'importance_sampling':importance,'mh':records,'gibbs':gibbs,
            'diagnostic_scope':'classic split Rhat; single-chain positive-pair truncated ESS; heuristic MCSE meaningful only after mixing'}
    return (result,all_chains) if return_chains else result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(ROOT/'outputs/result.json'));a=p.parse_args();out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(run(),ensure_ascii=False,indent=2,allow_nan=False)+'\n');print(out)
