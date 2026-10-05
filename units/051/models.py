"""Fixed Legendre features and mean-penalty ridge; no fitted preprocessing."""
import math
import numpy as np
from numeric import scalar,vector

def candidate(c):
    if not isinstance(c,dict) or set(c)!={'degree','penalty'}:raise ValueError('candidate degree/penalty required')
    d=scalar(c['degree'],'degree',0,9,True);lam=scalar(c['penalty'],'penalty',0,10)
    if d==0 and lam!=0:raise ValueError('constant model has penalty0')
    return {'degree':d,'penalty':lam}
def design(x,degree):
    xx=vector(x,'Legendre input x',-1,1);d=scalar(degree,'degree',0,9,True)
    return np.polynomial.legendre.legvander(xx,d)
def fit(x,y,c):
    c=candidate(c);xx=vector(x,'training x',-1,1);yy=vector(y,'training y',-1000,1000,len(xx));d=c['degree'];lam=c['penalty']
    if len(xx)<d+2:raise ValueError('training rows must exceed coefficient count')
    A=np.polynomial.legendre.legvander(xx,d);D=np.diag([0.]+[1.]*d);aug=np.vstack([A,math.sqrt(len(xx)*lam)*D]);target=np.r_[yy,np.zeros(d+1)];theta,residual,rank,s=np.linalg.lstsq(aug,target,rcond=None)
    if rank<d+1 or not np.all(np.isfinite(theta)) or np.max(abs(theta))>1e8:raise ValueError('numerically unsupported fitted design')
    pred=A@theta;err=pred-yy
    return {'candidate':c,'n_training':len(xx),'coefficients':theta.tolist(),'rank':int(rank),'condition_number':float(s[0]/s[-1]),'training_MSE':float(np.mean(err*err)),'mean_half_loss_plus_penalty':float(.5*np.mean(err*err)+.5*lam*np.dot(theta[1:],theta[1:])),'penalty_convention':'mean half-SSE + lambda/2 * sum(non-intercept Legendre coefficients squared); sklearn alpha=n_train*lambda'}
def predict(model,x):
    if not isinstance(model,dict):raise ValueError('model record required')
    c=candidate(model.get('candidate'));theta=vector(model.get('coefficients'),'model coefficients',-1e8,1e8,c['degree']+1);A=design(x,c['degree']);p=A@theta
    if not np.all(np.isfinite(p)):raise ValueError('nonfinite model prediction')
    return p

def population_risk(model,true_coefficients,noise_sd):
    c=candidate(model.get('candidate'));theta=vector(model.get('coefficients'),'model coefficients',-1e8,1e8,c['degree']+1);truth=vector(true_coefficients,'true coefficients',-10,10);sd=scalar(noise_sd,'noise sd',.05,3)
    if len(truth)>10:raise ValueError('true degree>9')
    n=max(len(theta),len(truth));diff=np.pad(theta,(0,n-len(theta)))-np.pad(truth,(0,n-len(truth)));components=diff**2/(2*np.arange(n)+1)
    return {'total_MSE':float(sd*sd+math.fsum(components)),'noise_variance':sd*sd,'integrated_squared_error_of_this_fitted_function':float(math.fsum(components)),'Legendre_component_terms':components.tolist(),'scope':'conditional risk of this fixed fitted polynomial under declared uniform[-1,1] X and independent zero-mean noise; oracle diagnosis only, never selection criterion'}
