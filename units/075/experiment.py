"""精确等价旋转、因子恢复对齐和噪声假设的真实核验。"""
import argparse
import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import FactorAnalysis
from common import ROOT,load_data,write_json,safe_output
from generate_data import TRUE_W
from latent import ppca,posterior,log_density,projector

def matrix(d):return np.column_stack([d[f'x{i}'] for i in range(4)])
def truth(d):return np.column_stack([d['z0'],d['z1']])
def run():
    data=load_data();X=matrix(data['train']);A=matrix(data['audit']);mu,W,noise,values,U=ppca(X,2)
    scores,rec=posterior(A,mu,W,noise);train_scores,_=posterior(X,mu,W,noise);z=truth(data['audit'])
    angle=np.pi/3;R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    Wr=W@R;rot_scores,rot_rec=posterior(A,mu,Wr,noise);C=W@W.T+noise*np.eye(4);Cr=Wr@Wr.T+noise*np.eye(4)
    alignment,_=orthogonal_procrustes(train_scores,truth(data['train']))  # 仅合成数据的训练真值用于对齐。
    pca_rec=(A-mu)@U[:,:2]@U[:,:2].T+mu
    H=matrix(data['hetero_train']);HA=matrix(data['hetero_audit']);hm,hW,hn,*_=ppca(H,2)
    fa=FactorAnalysis(n_components=2,svd_method='lapack',tol=1e-6,max_iter=2000,random_state=0).fit(H)
    faC=fa.components_.T@fa.components_+np.diag(fa.noise_variance_)
    hreport={'ppca_audit_mean_log_density':float(log_density(HA,hm,hW@hW.T+hn*np.eye(4)).mean()),'fa_audit_mean_log_density':float(log_density(HA,fa.mean_,faC).mean()),'fa_noise':fa.noise_variance_.tolist(),'ppca_noise':hn,'fa_iterations':fa.n_iter_}
    scale=np.diag([2.,.5]);scaled_W=W@scale;latent_cov=np.linalg.inv(scale)@np.linalg.inv(scale).T
    return {'mean':mu.tolist(),'W':W.tolist(),'rotated_W':Wr.tolist(),'rotation':R.tolist(),'noise_variance':noise,'eigenvalues':values.tolist(),'covariance_max_difference':float(np.max(np.abs(C-Cr))),'rotation_reconstruction_max_difference':float(np.max(np.abs(rec-rot_rec))),'rotation_log_density_max_difference':float(np.max(np.abs(log_density(A,mu,C)-log_density(A,mu,Cr)))),'score_rotation_max_difference':float(np.max(np.abs(rot_scores-scores@R))),'raw_factor_mse':float(np.mean((scores-z)**2)),'aligned_factor_mse':float(np.mean((scores@alignment-z)**2)),'subspace_projector_distance':float(np.linalg.norm(projector(W)-projector(TRUE_W))),'pca_reconstruction_mse':float(np.mean((A-pca_rec)**2)),'ppca_signal_reconstruction_mse':float(np.mean((A-rec)**2)),'audit_mean_log_density':float(log_density(A,mu,C).mean()),'scale_covariance_max_difference':float(np.max(np.abs(scaled_W@latent_cov@scaled_W.T-W@W.T))),'heteroscedastic':hreport}
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(ROOT/'outputs/result.json'));a=p.parse_args();r=run();write_json(safe_output(a.out),r);print(r)
if __name__=='__main__':main()
