"""数学性质、边界与独立库交叉核验；不用assert语句。"""
import argparse,json
import numpy as np
from sklearn.metrics import adjusted_rand_score
from common import ROOT, require, safe_output, write_json
from validation import silhouette, adjusted_rand

def main():
    p=argparse.ArgumentParser();p.add_argument('--report',default=str(ROOT/'experiment-result.json'));p.add_argument('--out',default=str(ROOT/'outputs/tests.json'));a=p.parse_args();r=json.loads(open(a.report).read());checks=[]
    def check(ok,name):require(ok,name);checks.append(name)
    X=np.array([[0.],[1.],[5.],[6.]])
    check(np.allclose(silhouette(X,[0,0,1,1]),[1-1/5.5,1-1/4.5,1-1/4.5,1-1/5.5]),'hand silhouette')
    check(adjusted_rand([0,0,1,1],[7,7,3,3])==1,'label permutation')
    check(np.isclose(adjusted_rand([0,0,1,1],[0,1,0,1]),-.5),'negative ARI hand result')
    for seed in range(20):
        rng=np.random.default_rng(seed);a1=rng.integers(0,4,30);b=rng.integers(0,3,30)
        check(np.isclose(adjusted_rand(a1,b),adjusted_rand_score(a1,b)),'random ARI '+str(seed))
    check(silhouette([[0],[1],[4]],[0,0,1])[-1]==0,'singleton convention')
    check(np.allclose(silhouette(X*8,[0,0,1,1]),silhouette(X,[0,0,1,1])),'uniform scaling invariance')
    try:silhouette(X,[0,0,0,0])
    except ValueError:checks.append('reject one cluster')
    else:raise RuntimeError('invalid labels accepted')
    check(r['library_silhouette_max_error']<1e-12 and r['library_ari_error']<1e-12,'independent sklearn oracle')
    check(r['chosen_k']==3 and len(r['stability'])==20,'protocol completed')
    check(r['downstream_mse']['linear_features']<r['downstream_mse']['cluster_mean'],'linear response needs continuous features')
    write_json(safe_output(a.out),{'status':'passed','checks':checks,'count':len(checks)})
if __name__=='__main__':main()
