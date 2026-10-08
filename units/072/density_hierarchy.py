"""小规模教学实现：DBSCAN 的核心图，以及 single/complete/average 凝聚。"""
import numpy as np
from common import finite_matrix,require

def positive_int(v,name):
    require(isinstance(v,(int,np.integer)) and not isinstance(v,(bool,np.bool_)) and v>0,name+' must be positive integer');return int(v)
def distances(X):
    X=finite_matrix(X)
    with np.errstate(over='ignore',invalid='ignore'):D=np.sqrt(np.sum((X[:,None,:]-X[None,:,:])**2,axis=2))
    require(np.isfinite(D).all(),'distances overflow; rescale data');return D

def dbscan(X,eps=.2,min_samples=5):
    X=finite_matrix(X);require(np.isfinite(eps) and eps>0,'eps must be positive and finite');positive_int(min_samples,'min_samples')
    D=distances(X);neighbors=D<=eps;core=neighbors.sum(axis=1)>=min_samples;labels=np.full(len(X),-1,dtype=int);cluster=0
    # 先只沿核心点边求连通分量。边界点不会成为连接两团的桥。
    for start in range(len(X)):
        if not core[start] or labels[start]>=0:continue
        stack=[start];labels[start]=cluster
        while stack:
            p=stack.pop()
            for q in np.flatnonzero(neighbors[p]&core):
                if labels[q]<0:labels[q]=cluster;stack.append(int(q))
        cluster+=1
    ambiguous=[]
    for i in np.flatnonzero(~core):
        adjacent=np.unique(labels[neighbors[i]&core])
        if len(adjacent):labels[i]=int(adjacent[0])
        if len(adjacent)>1:ambiguous.append(int(i))
    return {'labels':labels,'core':core,'neighbor_counts':neighbors.sum(axis=1),'ambiguous_border_indices':ambiguous,'clusters':cluster,'noise_fraction':float(np.mean(labels<0))}

def agglomerative(X,method='single'):
    X=finite_matrix(X);require(method in ['single','complete','average'],'supported: single, complete, average');D=distances(X);active={i:[i] for i in range(len(X))};Z=[]
    # 直接重算跨簇距离供读者审计；不是生产级最近邻链算法。
    for step in range(len(X)-1):
        options=[];keys=sorted(active)
        for i,a in enumerate(keys):
            for b in keys[i+1:]:
                block=D[np.ix_(active[a],active[b])]
                val={'single':np.min,'complete':np.max,'average':np.mean}[method](block)
                options.append((float(val),a,b))
        height,a,b=min(options);members=active.pop(a)+active.pop(b);Z.append([a,b,height,len(members)]);active[len(X)+step]=members
    return np.asarray(Z,dtype=float).reshape(-1,4)

def cut_k(Z,k):
    Z=np.asarray(Z,dtype=float);require(Z.ndim==2 and Z.shape[1]==4 and np.isfinite(Z).all(),'invalid linkage');n=len(Z)+1;positive_int(k,'k');require(k<=n,'k exceeds sample count');active={i:[i] for i in range(n)}
    # 即使只要前缀切分，也验证整棵树。否则坏的尾部记录会被悄悄忽略。
    selected={i:members.copy() for i,members in active.items()} if k==n else None
    for i,row in enumerate(Z):
        a,b,h,count=row;require(a==int(a) and b==int(b) and a!=b,'invalid child IDs');a=int(a);b=int(b)
        require(a in active and b in active and h>=0,'invalid merge');members=active.pop(a)+active.pop(b);require(count==len(members),'invalid count');active[n+i]=members
        if len(active)==k:selected={j:members.copy() for j,members in active.items()}
    require(selected is not None,'requested cut not found')
    labels=np.empty(n,dtype=int)
    for lab,(_,members) in enumerate(sorted(selected.items())):labels[members]=lab
    return labels
