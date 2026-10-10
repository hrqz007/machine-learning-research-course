"""透明的教学采样器；不依赖黑盒概率程序库。"""
import numpy as np

def log_posterior(theta, y, noise=1., prior_sd=2.):
    """已知噪声正态模型的未归一化对数后验，常数对MH无影响。"""
    y=np.asarray(y,dtype=float)
    if y.ndim!=1 or not len(y) or not np.all(np.isfinite(y)):
        raise ValueError('y must be a nonempty finite vector')
    if not np.isfinite([theta,noise,prior_sd]).all() or noise<=0 or prior_sd<=0:
        raise ValueError('finite theta and positive scales required')
    return -.5*(theta/prior_sd)**2-.5*np.sum(((y-theta)/noise)**2)

def analytic_posterior(y, noise=1., prior_sd=2.):
    y=np.asarray(y,dtype=float)
    log_posterior(0.,y,noise,prior_sd) # 统一验证输入。
    variance=1/(len(y)/noise**2+1/prior_sd**2)
    return variance*np.sum(y)/noise**2, variance

def metropolis(log_target, start, proposal_sd, n, seed):
    """对称随机游走MH；每次拒绝也保存旧状态，返回完整链及接受率。"""
    if n<2 or int(n)!=n or proposal_sd<=0 or not np.isfinite(proposal_sd):
        raise ValueError('n>=2 integer and positive finite proposal scale required')
    rng=np.random.default_rng(seed); current=float(start); value=float(log_target(current))
    if not np.isfinite(value): raise ValueError('initial log density must be finite')
    chain=np.empty(n); accepted=0
    for t in range(n):
        candidate=current+rng.normal(0,proposal_sd)
        proposed=float(log_target(candidate))
        if np.isnan(proposed) or proposed==np.inf: raise ValueError('invalid proposed density')
        # log(U)<min(0,log ratio)避免先指数化引起溢出。
        if np.log(rng.uniform())<min(0.,proposed-value):
            current,value=candidate,proposed;accepted+=1
        chain[t]=current # 拒绝必须记录，否则目标分布被改变。
    return chain,accepted/n

def autocorrelation(x, max_lag=1000):
    """FFT计算样本ACF；采用同一分母，滞后越大估计越不稳定。"""
    x=np.asarray(x,dtype=float)
    if x.ndim!=1 or len(x)<4 or not np.isfinite(x).all(): raise ValueError('finite vector length>=4 required')
    z=x-x.mean(); denom=np.dot(z,z)
    if denom<=0: raise ValueError('constant chain has undefined autocorrelation')
    size=1<<(2*len(x)-1).bit_length()
    spectrum=np.fft.rfft(z,n=size)
    cov=np.fft.irfft(spectrum*np.conjugate(spectrum),n=size)[:min(max_lag+1,len(x))]
    return cov/denom

def ess(x, max_lag=1000):
    """教学版正序列截断ESS，非Stan的rank-normalized多链ESS。"""
    acf=autocorrelation(x,max_lag); positive_sum=0.
    # 成对处理rho(1)+rho(2)，遇首个非正对停止，抑制尾部噪声。
    for k in range(1,len(acf)-1,2):
        pair=acf[k]+acf[k+1]
        if pair<=0: break
        positive_sum+=pair
    return float(min(len(x),len(x)/(1+2*positive_sum)))

def split_rhat(chains):
    """经典split-Rhat；明确不冒充rank-normalized/folded Rhat。"""
    a=np.asarray(chains,dtype=float)
    if a.ndim!=2 or a.shape[0]<2 or a.shape[1]<8 or not np.isfinite(a).all():
        raise ValueError('at least two finite chains of length>=8 required')
    n=a.shape[1]//2
    halves=np.concatenate([a[:,:n],a[:,-n:]],axis=0)
    within=np.var(halves,axis=1,ddof=1).mean()
    between=n*np.var(halves.mean(axis=1),ddof=1)
    if within<=0: return float('inf') # 卡死链不能宣称诊断正常。
    variance=(n-1)/n*within+between/n
    return float(np.sqrt(variance/within))

def gibbs_gaussian(rho,n,seed,start=(0.,0.)):
    """单位边缘方差、相关rho的二元正态，系统扫描Gibbs。"""
    if not np.isfinite(rho) or not -1<rho<1 or not isinstance(n,(int,np.integer)) or n<2 or np.asarray(start).shape!=(2,) or not np.isfinite(start).all(): raise ValueError('|rho|<1, finite pair start, integer n>=2 required')
    rng=np.random.default_rng(seed); x,y=map(float,start); out=np.empty((n,2)); sd=np.sqrt(1-rho*rho)
    for t in range(n):
        x=rng.normal(rho*y,sd) # 用当前y更新x。
        y=rng.normal(rho*x,sd) # 用刚更新x更新y，而不是旧x。
        out[t]=x,y
    return out

def importance_normal(y,n,proposal_sd,seed):
    """从零均值正态提议做自归一化重要性采样，报告权重ESS。"""
    if not isinstance(n,(int,np.integer)) or n<2 or not np.isfinite(proposal_sd) or proposal_sd<=0: raise ValueError('invalid importance sampler settings')
    rng=np.random.default_rng(seed); x=rng.normal(0,proposal_sd,n)
    logw=np.array([log_posterior(v,y) for v in x])+0.5*(x/proposal_sd)**2+np.log(proposal_sd)
    weights=np.exp(logw-logw.max()); weights/=weights.sum()
    return float(weights@x),float(1/(weights@weights)),float(weights.max())
