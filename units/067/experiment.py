"""手算、库对照、长度尺度、噪声、边际似然及外推的真实实验。"""
from pathlib import Path
import argparse,json,platform
import numpy as np
import scipy,sklearn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF,WhiteKernel
from threadpoolctl import threadpool_limits
from common import ROOT,load_data,safe_output,write_json
from gaussian_process import kernel,fit,predict

def summary(model,Z,truth,y):
    mean,cov=predict(model,Z);variance=np.diag(cov);observed=variance+model['noise_variance']
    return {'mean':mean.tolist(),'latent_variance':variance.tolist(),'observation_variance':observed.tolist(),'latent_rmse':float(np.sqrt(np.mean((mean-truth)**2))),'latent_pointwise_95_fraction':float(np.mean(abs(mean-truth)<=1.96*np.sqrt(variance))),'observed_pointwise_95_fraction':float(np.mean(abs(mean-y)<=1.96*np.sqrt(observed)))}

def run():
    with threadpool_limits(limits=1):return _run()

def _run():
    data=load_data();X=data['train']['x'][:,None];y=data['train']['y'];grid=np.linspace(-5,6,241)[:,None]
    # 二点可手算 RBF；测试点恰在训练点中点。
    hx=np.array([[-1.],[1.]]);hy=np.array([1.,-1.]);hz=np.array([[0.]])
    ell=float(np.sqrt(2/np.log(2)));hand=fit(hx,hy,ell,.25,0.);hm,hc=predict(hand,hz)
    lib=GaussianProcessRegressor(kernel=RBF(ell,length_scale_bounds='fixed'),alpha=.25,optimizer=None,normalize_y=False).fit(hx,hy);lm,lc=lib.predict(hz,return_cov=True)
    hand_result={'length_scale':ell,'K':kernel(hx,hx,ell).tolist(),'C':hand['C'].tolist(),'cross':kernel(hx,hz,ell).ravel().tolist(),'coefficient':hand['a'].tolist(),'mean':float(hm[0]),'latent_variance':float(hc[0,0]),'observation_variance':float(hc[0,0]+.25),'sklearn_mean':float(lm[0]),'sklearn_variance':float(lc[0,0]),'lml':hand['lml'],'sklearn_lml':float(lib.log_marginal_likelihood()),'lml_parts':hand['lml_parts']}
    variants=[];max_mean=0.;max_cov=0.;max_lml=0.
    for ell,nv,label in [(.25,.0324,'short length'),(1.,.0324,'reference'),(3.,.0324,'long length'),(1.,.0001,'too little noise'),(1.,.36,'large noise')]:
        model=fit(X,y,ell,nv,1e-10);mu,cov=predict(model,grid)
        lib=GaussianProcessRegressor(kernel=RBF(ell,length_scale_bounds='fixed'),alpha=nv+1e-10,optimizer=None,normalize_y=False).fit(X,y);lm,lc=lib.predict(grid,return_cov=True)
        errors={'mean':float(np.max(abs(mu-lm))),'covariance':float(np.max(abs(cov-lc))),'lml':abs(model['lml']-float(lib.log_marginal_likelihood()))}
        max_mean=max(max_mean,errors['mean']);max_cov=max(max_cov,errors['covariance']);max_lml=max(max_lml,errors['lml'])
        row={'label':label,'length_scale':ell,'noise_variance':nv,'jitter':1e-10,'grid_mean':mu.tolist(),'grid_variance':np.diag(cov).tolist(),'lml':model['lml'],'library_errors':errors}
        for name in ['interpolation','extrapolation']:
            part=data[name];row[name]=summary(model,part['x'][:,None],part['f_true'],part['y'])
        variants.append(row)
    # 每一格只使用 train。此图不是测试误差曲线。
    lengths=np.geomspace(.12,4.,46);noises=np.geomspace(.002,.5,35);surface=[]
    for noise in noises:surface.append([fit(X,y,float(length),float(noise))['lml'] for length in lengths])
    slice_parts=[fit(X,y,float(length),.0324)['lml_parts'] for length in lengths]
    # 固定噪声，仅拟合长度尺度；多初始点并不保证全局最优。
    optimized=GaussianProcessRegressor(kernel=RBF(1.,length_scale_bounds=(.12,4.)),alpha=.0324+1e-10,normalize_y=False,n_restarts_optimizer=2,random_state=67).fit(X,y)
    optell=float(optimized.kernel_.length_scale);optmodel=fit(X,y,optell,.0324);om,oc=predict(optmodel,grid)
    # alpha 与 WhiteKernel：同一训练协方差，却有不同的测试对角。
    a=GaussianProcessRegressor(kernel=RBF(1.,length_scale_bounds='fixed'),alpha=.0324+1e-10,optimizer=None).fit(X,y)
    w=GaussianProcessRegressor(kernel=RBF(1.,length_scale_bounds='fixed')+WhiteKernel(.0324,noise_level_bounds='fixed'),alpha=1e-10,optimizer=None).fit(X,y)
    am,ac=a.predict(grid,return_cov=True);wm,wc=w.predict(grid,return_cov=True)
    noise_api={'max_mean_difference':float(np.max(abs(am-wm))),'min_variance_difference':float(np.min(np.diag(wc-ac))),'max_variance_difference':float(np.max(np.diag(wc-ac))),'expected_added_noise_variance':.0324}
    # 重复输入制造严格奇异矩阵；无 jitter 的 Cholesky 应失败。
    duplicate=np.array([[0.],[0.],[1.]]);dy=np.array([.2,.2,.7]);failed=False
    try:fit(duplicate,dy,1.,0.,0.)
    except np.linalg.LinAlgError:failed=True
    stabilized=fit(duplicate,dy,1.,0.,1e-10)
    # 先验样本固定随机种子，选择 90 点以便轻量执行。
    prior_x=np.linspace(-4,4,90)[:,None];rng=np.random.default_rng(67);priors=[]
    for ell in [.3,1.,3.]:
        covariance=kernel(prior_x,prior_x,ell)+1e-10*np.eye(len(prior_x));draw=rng.multivariate_normal(np.zeros(len(prior_x)),covariance,size=3,method='cholesky')
        priors.append({'length_scale':ell,'draws':draw.tolist()})
    return {'unit':'067','seed':67021,'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__},'protocol':{'train_n':len(X),'mean_prior':0.,'signal_variance':1.,'noise_std_in_generator':.18,'normalize_y':False,'evaluation_not_used_for_fit':True,'blas_threads':1},'hand':hand_result,'variants':variants,'grid_x':grid.ravel().tolist(),'library_max_errors':{'mean':max_mean,'covariance':max_cov,'lml':max_lml},'lml_grid':{'length_scales':lengths.tolist(),'noise_variances':noises.tolist(),'values':surface,'fixed_noise':.0324,'fixed_noise_parts':slice_parts},'optimized':{'length_scale':optell,'lml':optmodel['lml'],'grid_mean':om.tolist(),'grid_variance':np.diag(oc).tolist()},'noise_api':noise_api,'jitter_check':{'singular_without_jitter_failed':failed,'jitter':1e-10,'condition_number':float(np.linalg.cond(stabilized['C'])),'finite_coefficients':bool(np.isfinite(stabilized['a']).all())},'prior':{'x':prior_x.ravel().tolist(),'variants':priors},'cost':[{'n':n,'one_dense_matrix_bytes':8*n*n,'cholesky_leading_flops':n**3/3} for n in [100,1000,10000,100000]]}

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(ROOT/'outputs/result.json'));a=p.parse_args();out=safe_output(a.out);report=run();write_json(out,report)
    print(json.dumps({'status':'complete','out':str(out.relative_to(ROOT)) if out.is_relative_to(ROOT) else out.name,'hand':report['hand'],'library_max_errors':report['library_max_errors']},ensure_ascii=False))
if __name__=='__main__':main()
