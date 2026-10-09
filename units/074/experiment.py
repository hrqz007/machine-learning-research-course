"""EM手算、重启、对称驻点、塌缩路径与独立实现对照。"""
import argparse
import numpy as np
from sklearn.mixture import GaussianMixture
from common import ROOT,load_data,write_json,safe_output
from mixture import expectation,maximization,em

def run():
    data=load_data();x=data['train']['x'];audit=data['audit']['x'];runs=[]
    initial=[[-3,0,3],[-3,-2.8,-2.6],[0,0,0]]
    initial += [np.random.default_rng(s).choice(x,3,replace=False).tolist() for s in range(5)]
    for means in initial:
        result=em(x,means);result['initial_means']=means;result['audit_mean_log_density']=expectation(audit,result['weights'],result['means'],result['variances'])[1]/len(audit);runs.append(result)
    best=max(range(len(runs)),key=lambda i:runs[i]['history'][-1]);b=runs[best]
    small=np.array([-1.,0.,1.]);r,old=expectation(small,[.5,.5],[-1,1],[1,1]);w,m,v=maximization(small,r);new=expectation(small,w,m,v)[1]
    collapse=[]
    for variance in [1.,.1,.01,.001,.0001,1e-6,1e-8]:
        ll=expectation(np.array([0.,2.,4.]),[1/3,2/3],[0.,3.],[variance,1.])[1];collapse.append({'variance':variance,'log_likelihood':ll})
    library=GaussianMixture(n_components=3,means_init=np.array([-3,0,3])[:,None],weights_init=np.ones(3)/3,precisions_init=np.ones((3,1,1))/np.var(x),reg_covar=0,tol=1e-10,max_iter=500,n_init=1,random_state=0).fit(x[:,None])
    own=runs[0]
    return {'runs':runs,'best_run':best,'hand':{'responsibilities':r.tolist(),'weights':w.tolist(),'means':m.tolist(),'variances':v.tolist(),'old_ll':old,'new_ll':new},'collapse':collapse,'library_mean_log_density':float(library.score(x[:,None])),'own_mean_log_density':own['history'][-1]/len(x),'floor':1e-4}
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(ROOT/'outputs/result.json'));a=p.parse_args();r=run();write_json(safe_output(a.out),r);print({'best_run':r['best_run'],'restart_final_ll':[v['history'][-1] for v in r['runs']],'hand':r['hand']})
if __name__=='__main__':main()
