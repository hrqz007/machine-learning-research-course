"""026 Hypothesis tests and effect sizes: bounded offline teaching implementation.

Read README for statistical and numerical contracts. Inputs are fully validated
before random generators or output files are touched. No assertion is used as
an input gate. This module does not select a test by its observed p value.
"""
from pathlib import Path
from fractions import Fraction
from decimal import Decimal, InvalidOperation
from itertools import product
from collections import Counter
import argparse
import csv
import json
import math
import re
import sys
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
MIN_NORMAL = sys.float_info.min


def require(condition, message):
    if not condition:
        raise ValueError(message)


def real(value, name='value', low=-1e6, high=1e6):
    # Deliberately narrow: do not silently cast Fraction, Decimal, longdouble,
    # objects, bool, text or complex, even if they happen to represent an int.
    permitted = type(value) in (int, float, np.float16, np.float32, np.float64) or isinstance(value, np.integer)
    require(permitted and not isinstance(value, (bool, np.bool_)), name+' requires a supported real scalar')
    require(low <= value <= high, name+' is nonfinite or outside the supported range')
    require(value == 0 or abs(value) >= 1e-100, name+' is below the supported nonzero magnitude')
    result = float(value)
    require(math.isfinite(result) and (value == 0 or result != 0), name+' cannot be represented safely')
    return result


def integer(value, name, low, high):
    require(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)), name+' must be an integer')
    require(low <= value <= high, name+' outside supported range')
    return int(value)


def vector(values, name='values', minimum=1, maximum=10000):
    require(isinstance(values, (list, tuple, np.ndarray)), name+' must be list, tuple or numeric array')
    if isinstance(values, np.ndarray):
        require(values.ndim == 1, name+' must have shape (n,)')
        require(values.dtype.kind in 'iuf' and values.dtype.itemsize <= 8, name+' rejects object/bool/complex/extended arrays')
    require(minimum <= len(values) <= maximum, name+' length outside supported range')
    return np.array([real(v, name+'['+str(i)+']') for i, v in enumerate(values)], dtype=np.float64)


def alpha_value(alpha):
    return real(alpha, 'alpha', 1e-4, .2)


def positive_probability(value, name):
    require(math.isfinite(value) and MIN_NORMAL <= value <= 1, name+' positive probability underflow/overflow; use a log-tail method')
    return float(value)


def moments(values):
    x = vector(values, minimum=2)
    # Original-value compensated summation is important: subtracting an anchor
    # would lose the residual of [-1, 1, 1e-100].
    center = math.fsum(float(v) for v in x) / len(x)
    if np.all(x == x[0]):
        return {'n':len(x), 'mean':float(x[0]), 'sse':0.0, 'variance':0.0, 'sd':0.0, 'constant':True}
    deviations = [float(v)-center for v in x]
    squares = [v*v for v in deviations]
    require(all(d == 0 or q >= MIN_NORMAL for d, q in zip(deviations, squares)), 'nonzero squared deviation is zero/subnormal')
    residual_sum = math.fsum(deviations)
    # Correct the residual mean lost when the representable center rounded.
    sse = math.fsum(squares) - residual_sum*residual_sum/len(x)
    variance = sse/(len(x)-1)
    require(MIN_NORMAL <= variance < math.inf, 'nonconstant variance is zero/subnormal/nonfinite')
    return {'n':len(x), 'mean':center, 'sse':sse, 'variance':variance, 'sd':math.sqrt(variance), 'constant':False}


def paired_summary(score_a, score_b):
    a = vector(score_a, 'score_a', 2)
    b = vector(score_b, 'score_b', 2)
    require(len(a) == len(b), 'paired arrays must have equal lengths')
    d = vector([float(y)-float(x) for x,y in zip(a,b)], 'B-A', 2)
    ma, mb, md = moments(a), moments(b), moments(d)
    da=[float(x)-ma['mean'] for x in a]
    db=[float(y)-mb['mean'] for y in b]
    covariance = (math.fsum(x*y for x,y in zip(da,db))-math.fsum(da)*math.fsum(db)/len(a))/(len(a)-1)
    return {'a':ma, 'b':mb, 'difference':md, 'values':d.tolist(), 'covariance':covariance,
            'paired_mean_variance':md['variance']/len(a),
            'incorrect_independent_mean_variance':(ma['variance']+mb['variance'])/len(a)}


def known_sigma_test(differences, sigma, null=0.0, alpha=.05):
    x = vector(differences, 'differences', 2)
    sigma = real(sigma, 'sigma', .01, 1000)
    null = real(null, 'null')
    alpha = alpha_value(alpha)
    m = moments(x)
    se = sigma/math.sqrt(len(x))
    z = (m['mean']-null)/se
    require(abs(z) <= 30, 'abs(z)>30 outside the supported tail range')
    p = positive_probability(2*stats.norm.sf(abs(z)), 'normal two-sided p')
    critical = float(stats.norm.isf(alpha/2))
    return {'status':'ok', 'method':'external_known_sigma_Gaussian_z', 'n':len(x), 'mean':m['mean'], 'sigma':sigma,
            'null':null, 'alpha':alpha, 'se':se, 'statistic':z, 'p_two_sided':p,
            'critical':critical, 'interval':[m['mean']-critical*se,m['mean']+critical*se], 'reject':p<=alpha}


def paired_t_test(differences, null=0.0, alpha=.05):
    x = vector(differences, 'differences', 2)
    null = real(null, 'null')
    alpha = alpha_value(alpha)
    m = moments(x)
    if m['constant']:
        return {'status':'degenerate_zero_sample_sd', 'method':'paired_Gaussian_t', 'n':len(x), 'mean':m['mean'],
                'sd':0.0, 'null':null, 'alpha':alpha, 'statistic':None, 'p_two_sided':None, 'interval':None, 'reject':None, 'd_z':None}
    se = m['sd']/math.sqrt(len(x))
    t = (m['mean']-null)/se
    require(abs(t)<=30, 'abs(t)>30 outside the supported tail range')
    df = len(x)-1
    p = positive_probability(2*stats.t.sf(abs(t),df), 'Student two-sided p')
    critical = float(stats.t.isf(alpha/2,df))
    return {'status':'ok', 'method':'paired_Gaussian_t', 'n':len(x), 'mean':m['mean'], 'sd':m['sd'], 'null':null,
            'alpha':alpha, 'se':se, 'df':df, 'statistic':t, 'p_two_sided':p, 'critical':critical,
            'interval':[m['mean']-critical*se,m['mean']+critical*se], 'reject':p<=alpha, 'd_z':m['mean']/m['sd']}


def exact_signflip(differences, alpha=.05):
    x = vector(differences, 'differences', 1, 16)
    alpha = alpha_value(alpha)
    # Validate every number first. Integer-only avoids silently moving ties.
    require(all(float(v).is_integer() for v in x), 'exact sign-flip demo requires integer-valued differences')
    values = [int(v) for v in x]
    magnitudes = [abs(v) for v in values]
    observed = abs(sum(values))
    sums = [sum(s*v for s,v in zip(signs,magnitudes)) for signs in product((-1,1),repeat=len(x))]
    tail = sum(abs(s)>=observed for s in sums)
    total = 2**len(x)
    p = tail/total
    return {'method':'exact_absolute_sum_signflip', 'n':len(x), 'observed_absolute_sum':observed,
            'label_count':total, 'tail_count':tail, 'p_fraction':str(Fraction(tail,total)), 'p_two_sided':p,
            'alpha':alpha, 'reject':p<=alpha, 'sums':sums,
            'counts':{str(k):v for k,v in sorted(Counter(sums).items())}}


def gaussian_power(delta, sigma, n, alpha=.05):
    delta = real(delta, 'delta', -100,100)
    sigma = real(sigma, 'sigma', .01,1000)
    n = integer(n, 'n', 2,4096)
    alpha = alpha_value(alpha)
    lam = delta*math.sqrt(n)/sigma
    require(abs(lam)<=12, 'noncentrality>12 outside the teaching probability range')
    c = float(stats.norm.isf(alpha/2))
    left = positive_probability(stats.norm.sf(c+lam), 'power left tail')
    right = positive_probability(stats.norm.sf(c-lam), 'power right tail')
    beta = positive_probability(stats.norm.cdf(c-abs(lam))-stats.norm.cdf(-c-abs(lam)), 'positive type-II probability')
    power = positive_probability(math.fsum([left,right]), 'power')
    return {'delta':delta, 'sigma':sigma, 'n':n, 'alpha':alpha, 'lambda_abs':abs(lam), 'lambda_signed':lam,
            'power':power, 'type_II':beta, 'left_tail':left, 'right_tail':right,
            'rounding_note':'Power may round to 1; separately retained type_II is still positive.'}


def family_error(m, alpha=.05):
    m = integer(m, 'm', 1,1000)
    alpha = alpha_value(alpha)
    raw = -math.expm1(m*math.log1p(-alpha))
    corrected = -math.expm1(m*math.log1p(-alpha/m))
    return {'m':m,'alpha':alpha,'unadjusted':raw,'bonferroni_threshold':alpha/m,'bonferroni':corrected,
            'conditions':'All m null hypotheses true; independent valid continuous p values.'}


def bonferroni(p_values, alpha=.05):
    p = vector(p_values,'p_values',1,1000)
    alpha = alpha_value(alpha)
    require(np.all((p>=0)&(p<=1)), 'p values must be in [0,1]')
    # True p endpoints are accepted; all positions are checked before adjustment.
    return {'adjusted':[min(1.0,len(p)*float(v)) for v in p], 'reject':(p<=alpha/len(p)).tolist(), 'threshold':alpha/len(p)}


def binomial_summary(k, repetitions):
    r = integer(repetitions,'repetitions',1,1000000)
    k = integer(k,'k',0,r)
    rate = k/r
    se = math.sqrt(rate*(1-rate)/r)
    z = float(stats.norm.isf(.025))
    denom = 1+z*z/r
    center = (rate+z*z/(2*r))/denom
    half = z*math.sqrt(rate*(1-rate)/r+z*z/(4*r*r))/denom
    return {'count':k,'repetitions':r,'rate':rate,'mc_se':se,'wilson95':[max(0.,center-half),min(1.,center+half)]}


def validate_spec(spec):
    expected={'kind','seed','repetitions','known_sigma','alpha','null_delta','power_deltas','power_sizes','family_sizes','practical_threshold'}
    require(type(spec) is dict and set(spec)==expected, 'model spec keys must exactly match the documented schema')
    require(spec['kind']=='synthetic_paired_hypothesis_demo','unknown model kind')
    seed=integer(spec['seed'],'seed',0,2**32-1-3000)
    r=integer(spec['repetitions'],'repetitions',100,50000)
    sigma=real(spec['known_sigma'],'known_sigma',.1,100)
    alpha=alpha_value(spec['alpha'])
    null=real(spec['null_delta'],'null_delta',-100,100)
    require(null==0,'this simulation design requires null_delta=0')
    threshold=real(spec['practical_threshold'],'practical_threshold',0,100)
    deltas=vector(spec['power_deltas'],'power_deltas',1,10).tolist()
    sizes_raw=spec['power_sizes']; families_raw=spec['family_sizes']
    require(type(sizes_raw) is list and 1<=len(sizes_raw)<=10,'power_sizes must be a list of length 1..10')
    require(type(families_raw) is list and 1<=len(families_raw)<=10,'family_sizes must be a list of length 1..10')
    sizes=[integer(n,'power_size',2,4096) for n in sizes_raw]
    families=[integer(m,'family_size',1,100) for m in families_raw]
    require(len(set(deltas))==len(deltas) and len(set(sizes))==len(sizes) and len(set(families))==len(families),'duplicate design settings are not permitted')
    # Validate every combination before any RNG, including the last grid value.
    for delta in deltas:
        for n in sizes:
            gaussian_power(delta,sigma,n,alpha)
    require(r*(64+sum(families)+len(deltas)*len(sizes))<=15000000,'simulation work budget exceeded')
    return {'kind':spec['kind'],'seed':seed,'repetitions':r,'known_sigma':sigma,'alpha':alpha,'null_delta':null,
            'power_deltas':deltas,'power_sizes':sizes,'family_sizes':families,'practical_threshold':threshold}


def simulations(spec):
    s=validate_spec(spec)
    r=s['repetitions']; alpha=s['alpha']; seed=s['seed']; sigma=s['known_sigma']
    critical=float(stats.norm.isf(alpha/2))
    power=[]
    for i,(delta,n) in enumerate(product(s['power_deltas'],s['power_sizes'])):
        offset=100+i
        rng=np.random.Generator(np.random.PCG64(seed+offset))
        z=rng.normal(delta*math.sqrt(n)/sigma,1.0,r)
        entry=gaussian_power(delta,sigma,n,alpha)
        entry.update({'seed_offset':offset,'observed':binomial_summary(int(np.count_nonzero(np.abs(z)>=critical)),r)})
        power.append(entry)
    family=[]
    for i,m in enumerate(s['family_sizes']):
        offset=1000+i
        rng=np.random.Generator(np.random.PCG64(seed+offset))
        z=rng.normal(0.,1.,(r,m))
        entry=family_error(m,alpha)
        entry.update({'seed_offset':offset,
             'observed_unadjusted':binomial_summary(int(np.count_nonzero(np.any(np.abs(z)>=critical,axis=1))),r),
             'observed_bonferroni':binomial_summary(int(np.count_nonzero(np.any(np.abs(z)>=stats.norm.isf(alpha/(2*m)),axis=1))),r)})
        family.append(entry)
    rng=np.random.Generator(np.random.PCG64(seed+2000))
    increments=rng.normal(0.,1.,(r,64))
    looks=np.arange(8,65,8)
    z_paths=np.cumsum(increments,axis=1)[:,looks-1]/np.sqrt(looks)
    crossing=np.abs(z_paths)>=critical
    optional={'seed_offset':2000,'looks':looks.tolist(),'critical':critical,
              'final_only':binomial_summary(int(np.count_nonzero(crossing[:,-1])),r),
              'any_look':binomial_summary(int(np.count_nonzero(np.any(crossing,axis=1))),r),
              'first_12_standardized_paths':z_paths[:12].tolist(),
              'conditions':'Shared-prefix Gaussian zero-mean paths; finite simulation, no independence formula across looks.'}
    # Direction choice uses another independent stream and a lower cutoff.
    rng=np.random.Generator(np.random.PCG64(seed+2100))
    z=rng.normal(0.,1.,r)
    direction={'seed_offset':2100,'theory':2*alpha,
               'observed':binomial_summary(int(np.count_nonzero(np.abs(z)>=stats.norm.isf(alpha))),r)}
    return {'power':power,'families':family,'optional_looks':optional,'posthoc_direction':direction,
            'rng':'NumPy Generator(PCG64); separate seed+offset per mechanism'}


def parse_csv_number(text, name):
    require(type(text) is str and re.fullmatch(r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?',text) is not None,
            name+' requires an unpadded finite decimal literal')
    try:
        exact=Decimal(text)
    except InvalidOperation as exc:
        raise ValueError(name+' invalid decimal') from exc
    require(exact.is_finite() and abs(exact)<=Decimal('1e6'),name+' outside supported range')
    require(exact==0 or abs(exact)>=Decimal('1e-100'),name+' nonzero decimal below supported range')
    value=real(float(exact),name)
    require(exact==0 or value!=0,name+' nonzero decimal collapsed')
    require(Fraction(exact)==Fraction.from_float(value),name+' decimal is not exactly representable in float64; teaching CSV requires exact binary values')
    return value


def read_pairs(path):
    with Path(path).open(encoding='utf-8',newline='') as handle:
        reader=csv.DictReader(handle)
        require(reader.fieldnames==['pair_id','score_a','score_b'],'CSV header must be pair_id,score_a,score_b in this order')
        rows=list(reader)
    require(2<=len(rows)<=16,'CSV needs 2..16 paired rows for complete sign-flip enumeration')
    ids=[]; a=[]; b=[]
    for i,row in enumerate(rows):
        require(set(row)=={'pair_id','score_a','score_b'} and None not in row.values(),'CSV missing or extra fields')
        label=row['pair_id']
        require(re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,31}',label) is not None,'pair_id must be a nonempty ASCII identifier')
        ids.append(label)
        a.append(parse_csv_number(row['score_a'],f'row{i+1}.score_a'))
        b.append(parse_csv_number(row['score_b'],f'row{i+1}.score_b'))
    require(len(set(ids))==len(ids),'duplicate pair_id')
    paired_summary(a,b)  # Validate derived differences before returning.
    return ids,a,b


def read_spec(path):
    def parse_float_token(token):
        exact=Decimal(token)
        require(exact.is_finite() and abs(exact)<=Decimal('1e6'),'JSON float outside supported range')
        require(exact==0 or abs(exact)>=Decimal('1e-100'),'JSON nonzero float below supported range before cast')
        value=real(float(exact),'JSON number')
        require(exact==0 or value!=0,'JSON nonzero token collapsed to zero')
        return value
    def reject_constant(token):
        raise ValueError('JSON nonfinite token is forbidden: '+token)
    def unique_keys(pairs):
        result={}
        for k,v in pairs:
            require(k not in result,'duplicate JSON key '+k)
            result[k]=v
        return result
    with Path(path).open(encoding='utf-8') as handle:
        raw=json.load(handle,object_pairs_hook=unique_keys,parse_float=parse_float_token,parse_constant=reject_constant)
    return validate_spec(raw)


def make_report(ids, score_a, score_b, spec):
    # All deterministic computations and validity gates precede simulations.
    s=validate_spec(spec)
    require(type(ids) is list and all(type(i) is str and re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,31}',i) for i in ids),'invalid ids')
    require(len(set(ids))==len(ids),'duplicate ids')
    paired=paired_summary(score_a,score_b)
    require(len(ids)==paired['difference']['n'],'id count mismatch')
    d=paired['values']
    z=known_sigma_test(d,s['known_sigma'],s['null_delta'],s['alpha'])
    t=paired_t_test(d,s['null_delta'],s['alpha'])
    flip=exact_signflip(d,s['alpha'])
    sim=simulations(s)
    return {'unit':'026','data_kind':'synthetic teaching only','spec':s,'pair_ids':ids,'paired':paired,
            'known_sigma':z,'paired_t':t,'signflip':flip,'simulations':sim,
            'warning':'Three different reference models, not a menu from which to select the smallest p value.'}


def write_report(report, output_dir):
    # Serialization also happens before any output mutation. Atomic file replace
    # protects an existing report if validation/computation/serialization fails.
    payload=json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    folder=Path(output_dir); folder.mkdir(parents=True,exist_ok=True)
    import tempfile
    temporary=None
    try:
        with tempfile.NamedTemporaryFile('w',encoding='utf-8',dir=folder,prefix='.report-',suffix='.tmp',delete=False) as handle:
            temporary=Path(handle.name); handle.write(payload)
        temporary.replace(folder/'report.json')
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return folder/'report.json'


def self_test():
    d=[3,2,4,1,-1,0,2,5]
    m=moments(d); z=known_sigma_test(d,2); t=paired_t_test(d); f=exact_signflip(d)
    checks=[m['mean']==2,m['sse']==28,m['variance']==4,math.isclose(z['p_two_sided'],.004677734981047265,rel_tol=1e-13),
            math.isclose(t['p_two_sided'],.025463561683239266,rel_tol=1e-13),f['tail_count']==12,f['label_count']==256,
            f['p_two_sided']==3/64,exact_signflip([0,0])['p_two_sided']==1,
            paired_t_test([2,2])['status']=='degenerate_zero_sample_sd',moments([-1,1,1e-100])['mean']==1e-100/3,
            math.isclose(gaussian_power(1,2,32)['power'],.807430419432,rel_tol=1e-10),
            math.isclose(family_error(20)['unadjusted'],.641514077591,rel_tol=1e-10),
            bonferroni([0.,1.,.01])['adjusted']==[0.,1.,.03],binomial_summary(0,100)['wilson95'][1]>0]
    for i,result in enumerate(checks):
        require(result,'self test failed '+str(i+1))
    return {'passed':len(checks)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'data/paired_scores.csv')
    parser.add_argument('--spec',type=Path,default=ROOT/'data/model_spec.json')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test())); return
    ids,a,b=read_pairs(args.input)
    spec=read_spec(args.spec)
    report=make_report(ids,a,b,spec)
    target=write_report(report,args.output_dir)
    print(json.dumps({'status':'passed','report':str(target),'p_z':report['known_sigma']['p_two_sided'],
                      'p_t':report['paired_t']['p_two_sided'],'p_signflip':report['signflip']['p_two_sided']},ensure_ascii=False))


if __name__=='__main__':
    main()
