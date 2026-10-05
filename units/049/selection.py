"""Selection over independent fair-noise candidates, exact CDF formula reference."""
from decimal import Decimal,localcontext
import math
import numpy as np
from numeric import scalar
from bounds import radius

def choose(counts,n,M):
    n=scalar(n,'candidate trial count',1,10000,True);M=scalar(M,'number selected from',1,2048,True)
    raw=np.asarray(counts) if isinstance(counts,np.ndarray) else np.asarray(counts,dtype=object)
    if raw.ndim!=2 or not 1<=raw.shape[0]<=10000 or not M<=raw.shape[1]<=2048 or raw.size>5000000:raise ValueError('candidate count matrix shape')
    if np.issubdtype(raw.dtype,np.integer):
        if np.any(raw<0) or np.any(raw>n):raise ValueError('candidate error count outside0..n')
        values=raw.astype(np.int64,copy=True)
    else:
        values=np.array([[scalar(v,'candidate error count',0,n,True) for v in row] for row in raw],dtype=np.int64)
    index=np.argmin(values[:,:M],axis=1);minimum=values[np.arange(len(values)),index]
    return {'selected_indices':index.tolist(),'selected_error_counts':minimum.tolist(),'tie_rule':'first column with minimum training error; no holdout input accepted'}

def analytic_minimum(n,M,delta):
    n=scalar(n,'analytic binomial n',2,200,True);M=scalar(M,'analytic candidate count',1,2048,True);delta=scalar(delta,'delta',1e-12,1-1e-12)
    with localcontext() as ctx:
        ctx.prec=100;den=Decimal(2)**n;cum=0;cdf=[]
        for k in range(n+1):cum+=math.comb(n,k);cdf.append(Decimal(cum)/den)
        fixed=((Decimal(2)/Decimal(str(delta))).ln()/(2*n)).sqrt();uniform=((Decimal(2*M)/Decimal(str(delta))).ln()/(2*n)).sqrt();masses=[];previous=Decimal(1)
        for k in range(n+1):
            survival=(1-cdf[k])**M;mass=previous-survival;previous=survival;masses.append(mass)
        if abs(sum(masses)-1)>Decimal('1e-95') or any(v<0 for v in masses):raise RuntimeError('minimum distribution normalization')
        fixed_failure=sum(p for k,p in enumerate(masses) if abs(Decimal(k)/n-Decimal('.5'))>fixed)
        uniform_failure=sum(p for k,p in enumerate(masses) if abs(Decimal(k)/n-Decimal('.5'))>uniform)
        expected=sum(Decimal(k)*p/n for k,p in enumerate(masses))
        fixed_one=sum(Decimal(math.comb(n,k))/den for k in range(n+1) if abs(Decimal(k)/n-Decimal('.5'))>fixed)
        return {'n':n,'M':M,'precision_decimal_digits':100,'fixed_radius_decimal':str(fixed),'uniform_radius_decimal':str(uniform),'selected_fixed_radius_violation_probability':float(fixed_failure),'selected_uniform_radius_violation_probability':float(uniform_failure),'predeclared_single_rule_violation_probability':float(fixed_one),'expected_selected_training_risk':float(expected),'expected_optimism':float(Decimal('.5')-expected),'minimum_pmf':[float(v) for v in masses],'formula':'P(min K_j > k) = (1 - BinomialCDF(k;n,1/2))**M; independent candidate errors','evaluation_note':'binomial coefficients are exact integers; probabilities evaluated with 100-digit Decimal, not reported as exact rational fractions'}

def vc_patterns(maximum=5):
    maximum=scalar(maximum,'max VC points',3,8,True);out=[]
    for m in range(1,maximum+1):
        thresholds=sorted({tuple(int(j>=cut) for j in range(m)) for cut in range(m+1)})
        intervals={tuple([0]*m)}
        for left in range(m):
            for right in range(left,m):intervals.add(tuple(int(left<=j<=right) for j in range(m)))
        all_patterns={tuple((mask>>j)&1 for j in range(m)) for mask in range(2**m)}
        out.append({'points':m,'threshold_patterns':[list(v) for v in thresholds],'interval_patterns':[list(v) for v in sorted(intervals)],'threshold_missing':[list(v) for v in sorted(all_patterns-set(thresholds))],'interval_missing':[list(v) for v in sorted(all_patterns-intervals)],'threshold_count':len(thresholds),'interval_count':len(intervals),'all_labelings':2**m})
    return {'right_facing_threshold_VC_dimension':1,'interval_VC_dimension':2,'ordered_distinct_point_patterns':out,'scope':'all real thresholds with fixed increasing direction and all real closed intervals; these are separate example classes, not a shortcut substituting growth into a fixed-class union bound'}
