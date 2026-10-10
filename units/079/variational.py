"""透明的线性高斯推断与坐标上升 VI；矩阵求解代替不必要的显式逆。

模型 z~N(m0,S0), x|z~N(Az,R)。所有输入必须有限且协方差正定。
此模块不读取数据文件、不联网，也不把解析后验均值直接塞入迭代器。
"""
import numpy as np


def vector(value, name):
    x = np.asarray(value, dtype=float)
    if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError(name + ' must be a nonempty finite vector')
    return x.copy()


def spd(value, size, name):
    x = np.asarray(value, dtype=float)
    if x.shape != (size, size) or not np.all(np.isfinite(x)):
        raise ValueError(name + ' has invalid shape or values')
    if not np.allclose(x, x.T, rtol=0, atol=1e-12):
        raise ValueError(name + ' must be symmetric')
    x = (x+x.T)/2
    try:
        np.linalg.cholesky(x)
    except np.linalg.LinAlgError as error:
        raise ValueError(name + ' must be positive definite') from error
    return x


def validate_model(x, A, prior_mean, prior_cov, noise_cov):
    x = vector(x, 'x'); m = vector(prior_mean, 'prior_mean')
    A = np.asarray(A, dtype=float)
    if A.shape != (x.size, m.size) or not np.all(np.isfinite(A)):
        raise ValueError('A must have shape (observations, latent dimensions)')
    return x, A.copy(), m, spd(prior_cov,m.size,'prior_cov'), spd(noise_cov,x.size,'noise_cov')


def logdet(matrix):
    # Cholesky 对角线为正；计算对数避免行列式上溢/下溢。
    return float(2*np.log(np.diag(np.linalg.cholesky(matrix))).sum())


def posterior(x, A, prior_mean, prior_cov, noise_cov):
    x,A,m,S,R = validate_model(x,A,prior_mean,prior_cov,noise_cov)
    prior_precision = np.linalg.solve(S, np.eye(m.size))
    precision = prior_precision + A.T @ np.linalg.solve(R,A)
    natural_mean = prior_precision @ m + A.T @ np.linalg.solve(R,x)
    mean = np.linalg.solve(precision,natural_mean)
    covariance = np.linalg.solve(precision,np.eye(m.size))
    # 证据由边缘生成分布独立计算，不依赖 ELBO 或 KL。
    evidence_cov = R + A @ S @ A.T
    residual = x-A@m
    evidence = -.5*(x.size*np.log(2*np.pi)+logdet(evidence_cov)+residual@np.linalg.solve(evidence_cov,residual))
    return dict(mean=mean,covariance=covariance,precision=precision,natural_mean=natural_mean,log_evidence=float(evidence))


def gaussian_kl(q_mean, q_cov, p_mean, p_cov):
    """返回 KL(q||p)。第一个分布提供期望，绝不能默默交换方向。"""
    qm=vector(q_mean,'q_mean'); pm=vector(p_mean,'p_mean')
    if qm.shape!=pm.shape: raise ValueError('mean dimensions differ')
    Q=spd(q_cov,qm.size,'q_cov'); P=spd(p_cov,pm.size,'p_cov')
    delta=qm-pm
    return float(.5*(np.trace(np.linalg.solve(P,Q))+delta@np.linalg.solve(P,delta)-qm.size+logdet(P)-logdet(Q)))


def elbo(x, A, prior_mean, prior_cov, noise_cov, q_mean, q_cov):
    """从 E_q log p(x,z)+H(q) 计算；不使用 log evidence - KL。"""
    x,A,m,S,R=validate_model(x,A,prior_mean,prior_cov,noise_cov)
    qm=vector(q_mean,'q_mean')
    if qm.shape!=m.shape:raise ValueError('q_mean dimension mismatch')
    Q=spd(q_cov,m.size,'q_cov'); residual=x-A@qm; delta=qm-m
    expected_log_likelihood=-.5*(x.size*np.log(2*np.pi)+logdet(R)+residual@np.linalg.solve(R,residual)+np.trace(np.linalg.solve(R,A@Q@A.T)))
    expected_log_prior=-.5*(m.size*np.log(2*np.pi)+logdet(S)+delta@np.linalg.solve(S,delta)+np.trace(np.linalg.solve(S,Q)))
    entropy=.5*(m.size*(1+np.log(2*np.pi))+logdet(Q))
    return float(expected_log_likelihood+expected_log_prior+entropy)


def coordinate_mean_field(precision,natural_mean,initial_mean=None,initial_variance=None,max_sweeps=500,tol=1e-12):
    """更新一个因子就记录一次；未达到容差时显式返回 converged=False。"""
    h=vector(natural_mean,'natural_mean'); P=spd(precision,h.size,'precision')
    if isinstance(max_sweeps,bool) or not isinstance(max_sweeps,int) or max_sweeps<1:
        raise ValueError('max_sweeps must be a positive integer')
    if not np.isscalar(tol) or not np.isfinite(tol) or tol<=0:raise ValueError('tol must be positive and finite')
    mean=np.zeros_like(h) if initial_mean is None else vector(initial_mean,'initial_mean')
    variance=np.ones_like(h) if initial_variance is None else vector(initial_variance,'initial_variance')
    if mean.shape!=h.shape or variance.shape!=h.shape or np.any(variance<=0):
        raise ValueError('initial factors have incompatible dimensions or nonpositive variance')
    trace=[dict(sweep=0,coordinate=None,mean=mean.tolist(),variance=variance.tolist())]
    converged=False
    for sweep in range(1,max_sweeps+1):
        old_mean=mean.copy();old_variance=variance.copy()
        for j in range(h.size):
            # 使用本轮已经更新的分量，所以这是顺序坐标更新。
            other=P[j]@mean-P[j,j]*mean[j]
            mean[j]=(h[j]-other)/P[j,j]
            variance[j]=1/P[j,j]
            trace.append(dict(sweep=sweep,coordinate=j,mean=mean.tolist(),variance=variance.tolist()))
        change=max(float(np.max(np.abs(mean-old_mean))),float(np.max(np.abs(variance-old_variance))))
        if change<tol:
            converged=True;break
    return dict(mean=mean,variance=variance,trace=trace,converged=converged,sweeps=sweep)


def linear_summary(mean,covariance,weights,noise_variance=0.):
    m=vector(mean,'mean');C=spd(covariance,m.size,'covariance');w=vector(weights,'weights')
    if w.shape!=m.shape or not np.isfinite(noise_variance) or noise_variance<0:raise ValueError('invalid weights or noise variance')
    return dict(mean=float(w@m),variance=float(w@C@w+noise_variance))
