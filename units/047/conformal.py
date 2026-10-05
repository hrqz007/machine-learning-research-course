"""Split conformal intervals from independent calibration residuals."""
from fractions import Fraction
import math
import numpy as np
from numeric import scalar,vector

def finite_sample_quantile(scores,miscoverage):
    scores=vector(scores,'nonnegative calibration scores',0,1000);alpha=scalar(miscoverage,'miscoverage',1e-6,1-1e-6);n=len(scores)
    # Nominal alpha is the canonical decimal representation, stated explicitly.
    exact=Fraction(str(alpha));num=(n+1)*(exact.denominator-exact.numerator);den=exact.denominator;k=(num+den-1)//den
    infinite=k>n;q=None if infinite else float(np.partition(scores,k-1)[k-1])
    return {'n_calibration':n,'miscoverage':alpha,'nominal_alpha_fraction':str(exact),'rank':int(k),'quantile':q,'infinite':bool(infinite),
            'quantile_definition':'k-th order statistic, k=ceil((n+1)*(1-alpha)); infinity if k>n; no interpolation',
            'nominal_alpha_semantics':'exact fraction of the canonical decimal str(float(alpha)); avoids intermediate floating-ceil rounding'}

def fit_line(x,y):
    x=vector(x,'training x',-100,100);y=vector(y,'training y',-1000,1000,size=len(x))
    if len(x)<2:raise ValueError('line fit needs at least two training rows')
    A=np.column_stack([np.ones(len(x)),x]);theta,residual,rank,s=np.linalg.lstsq(A,y,rcond=None)
    if rank!=2:raise ValueError('training line design is rank deficient')
    return {'theta':theta.tolist(),'rank':int(rank),'singular_values':s.tolist(),'condition_number':float(s[0]/s[-1]),'training_mse':float(np.mean((A@theta-y)**2))}

def interval_report(prediction,y,quantile):
    prediction=vector(prediction,'prediction',-1000,1000);y=vector(y,'evaluation y',-1000,1000,size=len(prediction))
    if not isinstance(quantile,dict) or not isinstance(quantile.get('infinite'),bool):raise ValueError('quantile record required')
    if quantile['infinite']:
        if quantile.get('quantile') is not None:raise ValueError('infinite interval must have null numerical quantile')
        return {'n':len(y),'coverage':1.,'covered':[True]*len(y),'interval_lower':None,'interval_upper':None,'width':None,'infinite_width':True,'target':'new observed response Y, not conditional mean'}
    q=scalar(quantile['quantile'],'finite interval quantile',0,1000);lower=prediction-q;upper=prediction+q;covered=(y>=lower)&(y<=upper)
    return {'n':len(y),'coverage':float(covered.mean()),'covered':covered.tolist(),'interval_lower':lower.tolist(),'interval_upper':upper.tolist(),'width':2*q,'infinite_width':False,'target':'new observed response Y, not conditional mean'}

def trial(train_x,train_y,cal_x,cal_y,test_x,test_y,shifted_test_y,miscoverage,group_boundary=1):
    # Validate all arrays and configuration before the first fit.
    train_x=vector(train_x,'train x',-100,100);train_y=vector(train_y,'train y',-1000,1000,size=len(train_x))
    cal_x=vector(cal_x,'cal x',-100,100);cal_y=vector(cal_y,'cal y',-1000,1000,size=len(cal_x));test_x=vector(test_x,'test x',-100,100);test_y=vector(test_y,'test y',-1000,1000,size=len(test_x));shifted_test_y=vector(shifted_test_y,'shifted test y',-1000,1000,size=len(test_x))
    alpha=scalar(miscoverage,'miscoverage',1e-6,1-1e-6);boundary=scalar(group_boundary,'group boundary',1e-6,100)
    fit=fit_line(train_x,train_y);b,w=fit['theta'];cal_pred=b+w*cal_x;score=np.abs(cal_y-cal_pred);quantile=finite_sample_quantile(score,alpha);prediction=b+w*test_x
    baseline=interval_report(prediction,test_y,quantile);shifted=interval_report(prediction,shifted_test_y,quantile)
    groups={}
    for name,mask in [('low_abs_x',abs(test_x)<=boundary),('high_abs_x',abs(test_x)>boundary)]:
        count=int(mask.sum());groups[name]={'n':count,'coverage':float(np.asarray(baseline['covered'])[mask].mean()) if count else None,'shifted_coverage':float(np.asarray(shifted['covered'])[mask].mean()) if count else None}
    # Explicitly invalid counterexample: fit 1-NN on calibration outcomes, then reuse them.
    cal_index=np.argmin(abs(cal_x[:,None]-cal_x[None,:]),axis=1);bad_cal_prediction=cal_y[cal_index];bad_scores=np.abs(cal_y-bad_cal_prediction);bad_quantile=finite_sample_quantile(bad_scores,alpha)
    test_index=np.argmin(abs(test_x[:,None]-cal_x[None,:]),axis=1);bad_prediction=cal_y[test_index];bad_result=interval_report(bad_prediction,test_y,bad_quantile)
    return {'fit':fit,'calibration_prediction':cal_pred.tolist(),'calibration_scores':score.tolist(),'quantile':quantile,'test_prediction':prediction.tolist(),'baseline':baseline,'noise_shift':shifted,'groups':groups,
            'invalid_calibration_reuse':{'description':'1-NN fitted to the same calibration labels used to get residual scores; exchangeability proof invalid','calibration_scores':bad_scores.tolist(),'quantile':bad_quantile,'evaluation':bad_result}}
