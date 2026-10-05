"""Explicit folds, global mapping and library splitter demonstrations."""
import numpy as np
from numeric import scalar

def make_folds(n,k,seed):
    n=scalar(n,'rows',2,10000,True);k=scalar(k,'fold count',2,10,True);seed=scalar(seed,'split seed',0,2**40,True)
    if k>n:raise ValueError('fold count exceeds rows')
    order=np.random.default_rng(seed).permutation(n);result=[]
    for val in np.array_split(order,k):
        val=np.sort(val);train=np.setdiff1d(np.arange(n),val);result.append({'training_indices':train.tolist(),'validation_indices':val.tolist()})
    validate_folds(result,n);return result

def validate_folds(folds,n):
    n=scalar(n,'fold rows',2,10000,True)
    if not isinstance(folds,list) or not 2<=len(folds)<=10:raise ValueError('fold list')
    seen=[];out=[]
    for row in folds:
        if not isinstance(row,dict) or set(row)!={'training_indices','validation_indices'}:raise ValueError('fold fields')
        rr={}
        for name in row:
            raw=row[name]
            if not isinstance(raw,list) or not raw:raise ValueError('nonempty index list required')
            ix=[scalar(v,'fold index',0,n-1,True) for v in raw]
            if len(ix)!=len(set(ix)):raise ValueError('duplicate index within fold')
            rr[name]=ix
        tr=set(rr['training_indices']);va=set(rr['validation_indices'])
        if tr&va or tr|va!=set(range(n)):raise ValueError('each fold must use disjoint complement training and validation')
        seen+=rr['validation_indices'];out.append(rr)
    if sorted(seen)!=list(range(n)):raise ValueError('validation folds must partition all rows exactly once')
    return out

def splitter_demo(gap=1):
    from sklearn.model_selection import KFold,StratifiedKFold,GroupKFold,TimeSeriesSplit
    gap=scalar(gap,'time demo gap',0,2,True);X=np.arange(12)[:,None];y=np.array([0,0,1,0,1,1,0,0,1,0,1,1]);groups=np.repeat(np.arange(4),3)
    objects=[('KFold',KFold(3,shuffle=True,random_state=5111)),('StratifiedKFold',StratifiedKFold(3,shuffle=True,random_state=5111)),('GroupKFold',GroupKFold(4)),('TimeSeriesSplit',TimeSeriesSplit(n_splits=3,test_size=2,gap=gap))];out={}
    for name,splitter in objects:
        rows=[]
        for tr,va in splitter.split(X,y,groups if name=='GroupKFold' else None):
            rows.append({'training_indices':tr.tolist(),'validation_indices':va.tolist(),'training_groups':sorted(set(groups[tr].tolist())),'validation_groups':sorted(set(groups[va].tolist())),'training_positive_count':int(y[tr].sum()),'validation_positive_count':int(y[va].sum()),'training_max_time':int(tr.max()),'validation_min_time':int(va.min())})
        out[name]=rows
    return {'row_indices':list(range(12)),'binary_labels':y.tolist(),'groups':groups.tolist(),'time':list(range(12)),'gap':gap,'splitters':out,'scope':'index-design examples only, not comparable model performance estimates; groups may overlap in random/stratified/time split, and GroupKFold may violate time ordering'}
