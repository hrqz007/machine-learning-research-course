"""从定义实现轮廓系数与调整兰德指数；显式输入检查不受-O影响。"""
import numpy as np
from common import finite_matrix, require

def silhouette(X, labels):
    X=finite_matrix(X)  # 确认每一行是有限数值样本。
    labels=np.asarray(labels)  # 簇编号只用于比较相等，不作数值运算。
    require(labels.shape==(len(X),),'one label per sample')
    groups=np.unique(labels);require(1<len(groups)<len(X),'need 2..n-1 clusters')
    D=np.sqrt(np.sum((X[:,None,:]-X[None,:,:])**2,axis=2))  # 两两欧氏距离。
    scores=np.zeros(len(X))  # 单点簇约定得0分，而不是完美的1分。
    for i in range(len(X)):
        own=labels==labels[i];own[i]=False  # 计算簇内平均时排除自身。
        if not own.any():continue
        a=D[i,own].mean()  # 到本簇其他点的平均距离。
        b=min(D[i,labels==g].mean() for g in groups if g!=labels[i])  # 最近的其他簇。
        scores[i]=(b-a)/max(a,b) if max(a,b)>0 else 0.  # 处理重复点的零分母。
    return scores

def adjusted_rand(a,b):
    a=np.asarray(a);b=np.asarray(b)
    require(a.ndim==b.ndim==1 and len(a)==len(b) and len(a)>0,'aligned nonempty labels')
    _,ia=np.unique(a,return_inverse=True);_,ib=np.unique(b,return_inverse=True)
    counts=np.zeros((ia.max()+1,ib.max()+1),dtype=np.int64)
    np.add.at(counts,(ia,ib),1)  # 每个单元格记录同时属于两种分组的样本数。
    choose=lambda v:np.sum(v*(v-1)/2)  # 从每个组中选两个样本，不考虑顺序。
    n=len(a);pairs=n*(n-1)/2
    if pairs==0:return 1.  # 一个对象的两种划分都相同。
    row=choose(counts.sum(axis=1));col=choose(counts.sum(axis=0));agree=choose(counts)
    expected=row*col/pairs  # 固定两边簇大小、随机重排对象身份的期望。
    denominator=(row+col)/2-expected
    return float((agree-expected)/denominator) if denominator else 1.
