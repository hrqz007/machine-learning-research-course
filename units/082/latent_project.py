"""独立概率审计工具。全协方差高斯混合；数据校验在-O下仍执行。"""
import math
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score


def matrix(x,d=None):
    a=np.asarray(x,dtype=float)
    if a.ndim!=2 or min(a.shape)==0 or not np.isfinite(a).all():raise ValueError('nonempty finite matrix required')
    if d is not None and a.shape[1]!=d:raise ValueError('dimension mismatch')
    return a


def log_joint(x,weights,means,covariances):
    """log p(x,z=k)，不用sklearn私有API，便于独立对照。"""
    means=matrix(means);x=matrix(x,means.shape[1]);w=np.asarray(weights,dtype=float);cov=np.asarray(covariances,dtype=float)
    k,d=means.shape
    if w.shape!=(k,) or not np.isfinite(w).all() or np.any(w<=0) or not np.isclose(w.sum(),1):raise ValueError('positive normalized weights required')
    if cov.shape!=(k,d,d) or not np.isfinite(cov).all():raise ValueError('covariance shape or entries invalid')
    values=[]
    for j in range(k):
        if not np.allclose(cov[j],cov[j].T):raise ValueError('covariance not symmetric')
        try:l=np.linalg.cholesky(cov[j])
        except np.linalg.LinAlgError as exc:raise ValueError('covariance not positive definite') from exc
        z=np.linalg.solve(l,(x-means[j]).T)
        values.append(math.log(w[j])-.5*(d*math.log(2*math.pi)+2*np.log(np.diag(l)).sum()+np.sum(z*z,axis=0)))
    return np.column_stack(values)


def responsibilities(x,weights,means,covariances):
    """先减去logsumexp，避免远尾概率直接下溢。返回后验责任度与log p(x)。"""
    joint=log_joint(x,weights,means,covariances);evidence=logsumexp(joint,axis=1)
    return np.exp(joint-evidence[:,None]),evidence


def hard_variational_gap(joint):
    """q只在最大联合概率的类别取1：ELBO=max log joint，缺口=-log max r。"""
    a=matrix(joint);log_evidence=logsumexp(a,axis=1);hard_elbo=a.max(axis=1)
    return log_evidence-hard_elbo


def match_means(estimated,true):
    """匈牙利匹配只用于已知合成真值；不赋予真实数据的分量编号语义。"""
    e=matrix(estimated);t=matrix(true,e.shape[1])
    if len(e)!=len(t):raise ValueError('same number of components required')
    squared=((e[:,None,:]-t[None,:,:])**2).sum(axis=2)
    rows,cols=linear_sum_assignment(squared)
    return {'estimated_order':rows.tolist(),'true_order':cols.tolist(),'matched_mean_rmse':float(np.sqrt(squared[rows,cols].mean()))}


def fit_model(x,k,seed=0,n_init=5):
    a=matrix(x)
    if isinstance(k,bool) or not isinstance(k,int) or k<1 or k>len(a):raise ValueError('invalid component count')
    model=GaussianMixture(n_components=k,covariance_type='full',reg_covar=.03,tol=1e-6,max_iter=500,n_init=n_init,random_state=seed)
    model.fit(a)
    if not model.converged_:raise RuntimeError('EM failed to converge; do not hide this candidate')
    return model


def select_model(train,validation):
    """仅验证平均对数密度用于选择K；BIC记录但不混用另一选择准则。"""
    train=matrix(train);validation=matrix(validation,train.shape[1]);models=[];records=[]
    for k in (1,2,3,4):
        m=fit_model(train,k,seed=82000+k);models.append(m)
        records.append({'k':k,'train_log_density':float(m.score(train)),'validation_log_density':float(m.score(validation)),
                        'training_bic':float(m.bic(train)),'converged':bool(m.converged_),'iterations':int(m.n_iter_)})
    index=max(range(len(records)),key=lambda i:(records[i]['validation_log_density'],-records[i]['k']))
    return models[index],records


def draw_mixture(model,n,rng):
    if not isinstance(n,int) or n<1:raise ValueError('n positive integer required')
    labels=rng.choice(model.n_components,size=n,p=model.weights_);x=np.empty((n,model.means_.shape[1]))
    for k in range(model.n_components):
        mask=labels==k;x[mask]=rng.multivariate_normal(model.means_[k],model.covariances_[k],size=int(mask.sum()))
    return x


def diagnostic_statistics(x):
    """两项预声明二维统计量：相关系数、至少一个坐标超出训练标准化±2的比例。"""
    a=matrix(x,2)
    if len(a)<3 or np.any(a.std(axis=0)==0):raise ValueError('statistics need nonconstant columns and at least three rows')
    return {'correlation':float(np.corrcoef(a.T)[0,1]),'tail_fraction':float(np.mean(np.max(np.abs(a),axis=1)>2))}


def predictive_check(model,observed,seed=82100,replicates=200):
    """参数固定的拟合模型复制检查，不是对参数后验积分的完整后验预测。"""
    if not isinstance(replicates,int) or replicates<20:raise ValueError('need at least 20 replicate datasets')
    observed=matrix(observed,2);rng=np.random.default_rng(seed);stats=diagnostic_statistics(observed)
    reps=[diagnostic_statistics(draw_mixture(model,len(observed),rng)) for _ in range(replicates)]
    report={}
    for name,value in stats.items():
        values=np.array([r[name] for r in reps]);lo,hi=np.quantile(values,[.025,.975]);center=np.median(values)
        report[name]={'observed':value,'replicate_interval95':[float(lo),float(hi)],'replicate_median':float(center),
                      'outside_interval':bool(value<lo or value>hi),'descriptive_tail_fraction':float((1+np.sum(np.abs(values-center)>=abs(value-center)))/(replicates+1))}
    return report,reps


def stability(train,evaluation,reference,k,seeds=range(12),bootstrap=False):
    """全部模型在同一固定评价点集比较ARI；不用训练行对齐错误地比较。"""
    train=matrix(train);evaluation=matrix(evaluation,train.shape[1]);ref=np.asarray(reference)
    if ref.shape!=(len(evaluation),):raise ValueError('reference labels must match evaluation rows')
    out=[]
    for seed in seeds:
        rng=np.random.default_rng(82300+seed);sample=train[rng.integers(0,len(train),len(train))] if bootstrap else train
        model=fit_model(sample,k,seed=seed,n_init=1)
        out.append({'seed':seed,'ari_against_reference':float(adjusted_rand_score(ref,model.predict(evaluation))),
                    'evaluation_log_density':float(model.score(evaluation)),'train_log_density':float(model.score(train))})
    return out


def normal_mean_posterior(train,noise_variance=1.,prior_mean=0.,prior_variance=4.):
    """单独的共轭基准：已知观测方差，未知总体均值。不是GMM参数后验。"""
    y=np.asarray(train,dtype=float)
    if y.ndim!=1 or not y.size or not np.isfinite(y).all():raise ValueError('finite nonempty observations required')
    if not np.isfinite([noise_variance,prior_mean,prior_variance]).all() or min(noise_variance,prior_variance)<=0:raise ValueError('finite mean and positive variances required')
    variance=1/(1/prior_variance+len(y)/noise_variance)
    mean=variance*(prior_mean/prior_variance+y.sum()/noise_variance)
    return float(mean),float(variance)


def conjugate_predictive_check(train,test,replicates=5000,seed=82400):
    """真正的参数后验预测小基准：每份复制数据先共享抽一次theta。

    已知合成噪声方差1、先验N(0,4)。返回完整posterior predictive
    与固定后验均值plug-in的复制均值方差；不能替代主GMM的贝叶斯推断。
    """
    y=np.asarray(test,dtype=float)
    if y.ndim!=1 or len(y)<2 or not np.isfinite(y).all():raise ValueError('test needs at least two finite observations')
    if not isinstance(replicates,int) or replicates<100:raise ValueError('at least 100 replicates required')
    mean,var=normal_mean_posterior(train);rng=np.random.default_rng(seed);n=len(y)
    theta=rng.normal(mean,np.sqrt(var),replicates)
    # 广播同一个theta到整份复制数据；不能为每个观测独立重新抽theta。
    full=rng.normal(size=(replicates,n))+theta[:,None]
    plugin=rng.normal(size=(replicates,n))+mean
    full_means=full.mean(axis=1);plugin_means=plugin.mean(axis=1)
    full_var=var+1/n;plugin_var=1/n
    return {'scope':'scalar normal mean with known variance 1; prior N(0,4); raw first feature of synthetic negative control only',
            'train_n':len(train),'test_n':n,'replicates':replicates,'posterior_mean':mean,'posterior_variance':var,
            'observed_test_mean':float(y.mean()),'full_predictive_mean_variance_exact':full_var,'plugin_mean_variance_exact':plugin_var,
            'variance_ratio_exact':full_var/plugin_var,'full_predictive_mean_variance_mc':float(full_means.var(ddof=1)),
            'plugin_mean_variance_mc':float(plugin_means.var(ddof=1)),
            'full_predictive_mean_interval95_exact':[mean-1.959963984540054*np.sqrt(full_var),mean+1.959963984540054*np.sqrt(full_var)],
            'plugin_mean_interval95_exact':[mean-1.959963984540054*np.sqrt(plugin_var),mean+1.959963984540054*np.sqrt(plugin_var)],
            'full_predictive_mean_interval95_mc':np.quantile(full_means,[.025,.975]).tolist()}
