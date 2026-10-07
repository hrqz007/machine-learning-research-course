"""Small transparent calculations used to explain a Newton tree leaf."""
import numpy as np
from scipy.special import expit
from common import require

def derivatives(y,score):
    y=np.asarray(y,dtype=float);score=np.asarray(score,dtype=float)
    require(y.shape==score.shape and y.ndim==1 and y.size>0,'equal nonempty 1D shapes required')
    require(np.isfinite(score).all() and np.isin(y,[0,1]).all(),'finite score and binary label required')
    p=expit(score);return p-y,p*(1-p)

def leaf_weight(G,H,l2):
    require(np.isfinite([G,H,l2]).all() and H>=0 and l2>=0 and H+l2>0,'nonnegative curvature and penalty, positive denominator required')
    return -G/(H+l2)

def split_gain(GL,HL,GR,HR,l2=0.,gamma=0.):
    require(np.isfinite([GL,HL,GR,HR,l2,gamma]).all() and min(HL,HR,l2,gamma)>=0,'invalid split statistics')
    require(HL+l2>0 and HR+l2>0,'positive child denominator required')
    return .5*(GL*GL/(HL+l2)+GR*GR/(HR+l2)-(GL+GR)**2/(HL+HR+l2))-gamma

def metrics(y,p):
    from sklearn.metrics import log_loss,brier_score_loss,roc_auc_score,accuracy_score
    return {'log_loss':float(log_loss(y,p)),'brier':float(brier_score_loss(y,p)),
            'auc':float(roc_auc_score(y,p)),'accuracy':float(accuracy_score(y,p>=.5))}

def reliability(y,p):
    """Five fixed bins, including p=1 in the last bin; empty bins are explicit."""
    y=np.asarray(y);p=np.asarray(p);edges=np.linspace(0,1,6);rows=[]
    for i in range(5):
        m=(p>=edges[i])&((p<edges[i+1]) if i<4 else (p<=edges[i+1]))
        rows.append({'left':float(edges[i]),'right':float(edges[i+1]),'count':int(m.sum()),'mean_probability':float(p[m].mean()) if m.any() else None,'fraction_positive':float(y[m].mean()) if m.any() else None})
    return rows
