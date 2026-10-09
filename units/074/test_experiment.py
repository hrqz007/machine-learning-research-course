"""真实似然、已知答案、条件概率与退化防线。"""
import argparse,json
import numpy as np
from common import ROOT,write_json,safe_output,require
from mixture import expectation,maximization

def main():
    p=argparse.ArgumentParser();p.add_argument('--report',default=str(ROOT/'experiment-result.json'));p.add_argument('--out',default=str(ROOT/'outputs/tests.json'));a=p.parse_args();r=json.loads(open(a.report).read());checks=[]
    def check(ok,name):require(ok,name);checks.append(name)
    for i,run in enumerate(r['runs']):check(min(np.diff(run['history']))>=-1e-8,'monotonic run '+str(i))
    hand=r['hand'];check(np.allclose(hand['responsibilities'][1],[.5,.5]),'midpoint responsibility')
    check(np.isclose(hand['responsibilities'][0][0],1/(1+np.exp(-2))),'hand logistic identity')
    check(np.allclose(hand['means'],[-.5077294373038433,.5077294373038433]),'weighted mean hand result')
    check(hand['new_ll']>=hand['old_ll'],'hand likelihood increase')
    rr,ll=expectation(np.array([-1000.,1000.]),[.5,.5],[-2,2],[1,1]);check(np.isfinite(ll) and np.allclose(rr.sum(axis=1),1),'extreme values log space')
    _,ll2=expectation(np.array([-1000.,1000.]),[.5,.5],[2,-2],[1,1]);check(ll==ll2,'component permutation invariance')
    _,_,v=maximization(np.array([0.,0.]),np.ones((2,1)),floor=.01);check(v[0]==.01,'exact constrained floor')
    check(abs(r['library_mean_log_density']-r['own_mean_log_density'])<1e-7,'sklearn independent implementation')
    check(np.ptp([x['history'][-1] for x in r['runs']])>10,'symmetric initialization counterexample')
    check(np.all(np.diff([x['log_likelihood'] for x in r['collapse']])>0),'unbounded collapse path tendency')
    try:expectation(np.array([0.]),[1.],[0.],[0.])
    except ValueError:checks.append('reject zero variance')
    else:raise RuntimeError('invalid variance accepted')
    write_json(safe_output(a.out),{'status':'passed','checks':checks,'count':len(checks)})
if __name__=='__main__':main()
