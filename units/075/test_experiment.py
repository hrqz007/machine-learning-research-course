"""代数不变性、PCA关系、独立密度实现和无效输入。"""
import argparse,json
import numpy as np
from scipy.stats import multivariate_normal
from common import ROOT,load_data,write_json,safe_output,require
from latent import ppca,posterior,log_density,projector
from experiment import matrix

def main():
    p=argparse.ArgumentParser();p.add_argument('--report',default=str(ROOT/'experiment-result.json'));p.add_argument('--out',default=str(ROOT/'outputs/tests.json'));a=p.parse_args();r=json.loads(open(a.report).read());checks=[]
    def check(ok,name):require(ok,name);checks.append(name)
    for name in ['covariance_max_difference','rotation_reconstruction_max_difference','rotation_log_density_max_difference','score_rotation_max_difference','scale_covariance_max_difference']:check(r[name]<1e-10,name)
    X=matrix(load_data()['train']);mu,W,s,values,U=ppca(X,2);C=W@W.T+s*np.eye(4)
    check(np.allclose(log_density(X[:10],mu,C),multivariate_normal.logpdf(X[:10],mean=mu,cov=C)),'scipy independent density')
    check(np.allclose(projector(W),U[:,:2]@U[:,:2].T),'PPCA and PCA same principal subspace')
    check(np.allclose(projector(np.zeros((4,2))),np.zeros((4,4))),'zero-rank projector')
    check(np.isclose(np.trace(projector(np.ones((4,2)))),1),'rank-deficient projector')
    try:log_density(X[:2],np.full(4,np.nan),C)
    except ValueError:checks.append('reject nonfinite mean')
    else:raise RuntimeError('nonfinite mean accepted')
    check(np.isclose(s,np.mean(values[2:])),'noise from discarded eigenvalues')
    check(r['aligned_factor_mse']<r['raw_factor_mse'],'alignment needed for coordinate comparison')
    check(r['pca_reconstruction_mse']<=r['ppca_signal_reconstruction_mse']+1e-12,'PCA minimizes observed rank-q reconstruction')
    check(r['heteroscedastic']['fa_iterations']<2000,'FA convergence within cap')
    check(r['heteroscedastic']['fa_audit_mean_log_density']>r['heteroscedastic']['ppca_audit_mean_log_density'],'unequal-noise counterexample')
    try:ppca(X,4)
    except ValueError:checks.append('reject q equals dimension')
    else:raise RuntimeError('invalid dimension accepted')
    try:ppca(np.ones((10,4)),2)
    except ValueError:checks.append('reject zero residual variance')
    else:raise RuntimeError('singular model accepted')
    write_json(safe_output(a.out),{'status':'passed','checks':checks,'count':len(checks)})
if __name__=='__main__':main()
