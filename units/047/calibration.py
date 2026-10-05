"""Binary proper scores, explicit reliability definitions and scalar temperature."""
import math
import numpy as np
from numeric import scalar,vector,labels

def sigmoid(z):
    z=np.asarray(z,dtype=float);t=np.exp(-np.abs(z));return np.where(z>=0,1/(1+t),t/(1+t))

def _state(z,y,a,rows=False):
    scaled=a*z;p=sigmoid(scaled);other=sigmoid(-scaled);t=(2*y-1)*scaled
    losses=np.maximum(-t,0)+np.log1p(np.exp(-abs(t)))
    dz=np.where(y==0,p,-other);h=p*other;n=len(z)
    out={'inverse_temperature':float(a),'temperature':None if a==0 else float(1/a),'infinite_temperature':bool(a==0),
         'mean_log_loss':float(math.fsum(losses)/n),'gradient':float(math.fsum(dz*z)/n),
         'hessian':float(math.fsum(h*z*z)/n),'probabilities':p.tolist()}
    if rows:
        entries=[]
        for i in range(n):
            denom=p[i] if y[i]==1 else other[i]
            local=(-1 if y[i] else 1)/denom if denom>=1/np.finfo(float).max else None
            entries.append({'row':i+1,'raw_logit':float(z[i]),'y':int(y[i]),'scaled_logit':float(scaled[i]),'probability':float(p[i]),'loss':float(losses[i]),'mean_loss_contribution':float(losses[i]/n),'local_dloss_dp':None if local is None else float(local),'local_dp_dscaled_logit':float(h[i]),'local_dscaled_logit_da':float(z[i]),'stable_dloss_dscaled_logit':float(dz[i]),'gradient_contribution':float(dz[i]*z[i]/n),'hessian_contribution':float(h[i]*z[i]*z[i]/n)})
        out['rows']=entries
    return out

def temperature_state(z,y,inverse_temperature,rows=True):
    z=vector(z,'logits');y=labels(y,len(z));a=scalar(inverse_temperature,'inverse temperature',0,4)
    return _state(z,y,a,rows)

def fit_temperature(z_cal,y_cal,bounds=(0,4),tol=1e-11):
    z=vector(z_cal,'calibration logits');y=labels(y_cal,len(z));bounds=vector(bounds,'inverse temperature bounds',0,4,size=2);tol=scalar(tol,'temperature tolerance',1e-13,1e-4)
    lo,hi=map(float,bounds)
    if not lo<hi:raise ValueError('temperature bounds must increase')
    left=_state(z,y,lo);right=_state(z,y,hi);trace=[]
    if left['gradient']>=0:a=lo;status='lower_boundary'
    elif right['gradient']<=0:a=hi;status='upper_boundary'
    else:
        status='bracket_tolerance'
        for iteration in range(100):
            mid=(lo+hi)/2;state=_state(z,y,mid);trace.append({'iteration':iteration,'lo':lo,'hi':hi,'a':mid,'gradient':state['gradient'],'loss':state['mean_log_loss']})
            if state['gradient']>0:hi=mid
            else:lo=mid
            if hi-lo<=tol:break
        a=(lo+hi)/2
    final=_state(z,y,a)
    # Projected optimality residual checks the constrained [lower, upper] problem.
    g=final['gradient'];lower,upper=bounds
    residual=max(0.,-g) if a==lower else max(0.,g) if a==upper else abs(g)
    return {'status':status,'bounds':bounds.tolist(),'final':final,'projected_gradient_residual':float(residual),'iterations':len(trace),'bracket_trace':trace,
            'selection_scope':'only supplied calibration logits and labels; no evaluation labels accepted'}

def reliability(scores,outcomes,bins=5):
    s=vector(scores,'probability/confidence',0,1);y=labels(outcomes,len(s));bins=scalar(bins,'bins',1,30,True)
    edges=np.linspace(0,1,bins+1);index=np.minimum(np.searchsorted(edges,s,side='right')-1,bins-1);out=[];weighted=[]
    for b in range(bins):
        mask=index==b;count=int(mask.sum());mean=float(s[mask].mean()) if count else None;rate=float(y[mask].mean()) if count else None
        contribution=(count/len(s))*abs(mean-rate) if count else 0.
        out.append({'bin':b,'lower':float(edges[b]),'upper':float(edges[b+1]),'upper_inclusive':b==bins-1,'count':count,'mean_score':mean,'empirical_outcome_rate':rate,'weighted_absolute_gap':float(contribution)})
        weighted.append(contribution)
    return {'bins':out,'ece':float(math.fsum(weighted)),'n':len(s),'boundary_rule':'[lower,upper), except final upper=1 inclusive; empty bins have null averages'}

def evaluate(z,y,a,bins=5):
    z=vector(z,'evaluation logits');y=labels(y,len(z));a=scalar(a,'inverse temperature',0,4);bins=scalar(bins,'bins',1,30,True)
    state=_state(z,y,a);p=np.array(state['probabilities']);pred=(p>.5).astype(int);confidence=np.maximum(p,1-p);correct=(pred==y).astype(int)
    from sklearn.metrics import brier_score_loss,log_loss
    positive=reliability(p,y,bins);top=reliability(confidence,correct,bins)
    return {'n':len(y),'inverse_temperature':a,'mean_log_loss':state['mean_log_loss'],'binary_Brier':float(np.mean((p-y)**2)),'accuracy':float(np.mean(correct)),
            'positive_class_reliability':positive,'top_label_reliability':top,
            'library_binary_Brier':float(brier_score_loss(y,p,pos_label=1,scale_by_half=True)),
            'library_log_loss':float(log_loss(y,np.column_stack([1-p,p]),labels=[0,1])),
            'log_loss_library_note':'sklearn receives probabilities and may clip extremes; core score is computed directly from logits',
            'predictions':pred.tolist(),'probabilities':p.tolist()}
