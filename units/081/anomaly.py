"""教学实现：统一高分=更异常；独立干净校准；不从测试标签选择阈值。"""
from dataclasses import dataclass
import math
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, average_precision_score


def matrix(x, dimension=None):
    """显式验证不能用assert；python -O下也必须保留。"""
    a=np.asarray(x,dtype=float)
    if a.ndim!=2 or min(a.shape)==0 or not np.isfinite(a).all():
        raise ValueError('expected a nonempty finite two-dimensional matrix')
    if dimension is not None and a.shape[1]!=dimension:raise ValueError('feature dimension differs')
    return a


def vector(x):
    a=np.asarray(x,dtype=float)
    if a.ndim!=1 or not a.size or not np.isfinite(a).all():raise ValueError('scores must be a finite nonempty vector')
    return a


class GaussianScore:
    """正则化单高斯负对数密度。Cholesky求解代替直接求逆。"""
    def __init__(self,regularization=.02):
        if not np.isfinite(regularization) or regularization<=0:raise ValueError('regularization must be positive')
        self.regularization=float(regularization)
    def fit(self,x):
        a=matrix(x)
        if len(a)<2:raise ValueError('at least two training rows')
        self.mean_=a.mean(axis=0);center=a-self.mean_
        self.covariance_=center.T@center/len(a)+self.regularization*np.eye(a.shape[1])
        self.cholesky_=np.linalg.cholesky(self.covariance_)
        self.logdet_=2*np.log(np.diag(self.cholesky_)).sum()
        return self
    def score(self,x):
        if not hasattr(self,'mean_'):raise ValueError('fit before scoring')
        a=matrix(x,len(self.mean_));z=np.linalg.solve(self.cholesky_,(a-self.mean_).T)
        return .5*(a.shape[1]*math.log(2*math.pi)+self.logdet_+np.sum(z*z,axis=0))


class KNNRadius:
    """到第k个训练邻居的距离。只面向新样本；不是LOF。"""
    def __init__(self,k=20):
        if isinstance(k,bool) or not isinstance(k,(int,np.integer)) or k<1:raise ValueError('k must be a positive integer')
        self.k=int(k)
    def fit(self,x):
        a=matrix(x)
        if self.k>len(a):raise ValueError('k exceeds training rows')
        self.train_=a.copy();return self
    def score(self,x):
        if not hasattr(self,'train_'):raise ValueError('fit before scoring')
        a=matrix(x,self.train_.shape[1]);parts=[]
        # 分块避免一次分配“全部待测点×全部训练点”的大矩阵。
        for start in range(0,len(a),256):
            distances=cdist(a[start:start+256],self.train_)
            parts.append(np.partition(distances,self.k-1,axis=1)[:,self.k-1])
        return np.concatenate(parts)


class LibraryScore:
    """库方法提供正常度；显式取负，统一异常分数方向。"""
    def __init__(self,kind):
        if kind not in ('lof','ocsvm'):raise ValueError('unknown method')
        self.kind=kind
    def fit(self,x):
        a=matrix(x)
        if len(a)<=20:raise ValueError('need more than 20 reference observations')
        self.dimension_=a.shape[1]
        self.estimator_=(LocalOutlierFactor(n_neighbors=20,novelty=True,contamination='auto')
                         if self.kind=='lof' else OneClassSVM(kernel='rbf',gamma='scale',nu=.05))
        self.estimator_.fit(a);return self
    def score(self,x):
        if not hasattr(self,'estimator_'):raise ValueError('fit before scoring')
        return -self.estimator_.score_samples(matrix(x,self.dimension_))


@dataclass(frozen=True)
class Calibration:
    threshold: float
    rank: int
    n: int
    alpha: float
    marginal_bound: float
    def alarms(self,scores):
        # 严格大于阈值，分数相同不报警；并列时更保守。
        return vector(scores)>self.threshold


def calibrate(clean_scores,alpha=.05):
    """固定评分函数后，用正常校准集的修正次序统计量控制边际误报。

    k=ceil((m+1)(1-alpha)); k>m时采用+inf。此时不是失败而是
    样本过少，只能用不报警的规则获得所要求的分布无关界。
    调用者必须保证输入干净、独立，函数不能从数字猜出标签泄漏。
    """
    a=vector(clean_scores)
    if isinstance(alpha,bool) or not np.isfinite(alpha) or not 0<alpha<1:raise ValueError('0 < alpha < 1 required')
    k=math.ceil((len(a)+1)*(1-alpha))
    threshold=float('inf') if k>len(a) else float(np.sort(a)[k-1])
    return Calibration(threshold,k,len(a),float(alpha),max(0.,(len(a)+1-k)/(len(a)+1)))


def wilson(successes,total,z=1.959963984540054):
    """二项比例的Wilson近似区间；不冒充校准条件保证。"""
    if isinstance(total,bool) or not isinstance(total,(int,np.integer)) or total<=0:raise ValueError('total must be positive integer')
    if not isinstance(successes,(int,np.integer)) or not 0<=successes<=total:raise ValueError('invalid count')
    p=successes/total;den=1+z*z/total
    center=(p+z*z/(2*total))/den;half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return [max(0.,center-half),min(1.,center+half)]


def evaluate(normal_scores,anomaly_scores,calibration):
    normal=vector(normal_scores);anomaly=vector(anomaly_scores)
    fp=int(calibration.alarms(normal).sum());tp=int(calibration.alarms(anomaly).sum())
    labels=np.r_[np.zeros(len(normal),dtype=int),np.ones(len(anomaly),dtype=int)]
    scores=np.r_[normal,anomaly]
    return {'false_positives':fp,'normal_n':len(normal),'fpr':fp/len(normal),'fpr_wilson95':wilson(fp,len(normal)),
            'true_positives':tp,'anomaly_n':len(anomaly),'recall':tp/len(anomaly),
            'auroc':float(roc_auc_score(labels,scores)),'average_precision':float(average_precision_score(labels,scores)),
            'test_prevalence':len(anomaly)/len(labels)}


def make_models():
    # 超参数在打开测试标签之前固定；比较所有方法，不据测试分数宣布“最终选型”。
    return {'gaussian':GaussianScore(.02),'knn':KNNRadius(20),'lof':LibraryScore('lof'),'ocsvm':LibraryScore('ocsvm')}
