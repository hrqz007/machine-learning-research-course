"""027 估计误差与偏差方差：有限范围、离线合成实验。

每个数值输入在转换与RNG之前完整验证。Fraction只由精确枚举接口接受；
通常浮点接口不接受bool、文本、Decimal、Fraction或扩展精度数值。
"""
from pathlib import Path
from fractions import Fraction
from itertools import product
import argparse, csv, json, math, re
import numpy as np
ROOT = Path(__file__).resolve().parent
MAX = 1e6
MIN = 1e-100


def require(condition, message):
    if not condition:
        raise ValueError(message)


def real(value, name='value', low=-MAX, high=MAX):
    allowed = type(value) in (int, float) or isinstance(value, (np.integer, np.floating))
    require(allowed and not isinstance(value, (bool, np.bool_)), name+' requires a supported real scalar')
    if isinstance(value, np.floating):
        require(value.dtype.itemsize <= 8, name+' extended precision is unsupported')
    require(low <= value <= high, name+' outside supported range or nonfinite')
    converted = float(value)
    require(math.isfinite(converted), name+' must be finite')
    require(value == 0 or abs(value) >= MIN, name+' nonzero magnitude below supported floor')
    require(value == 0 or converted != 0, name+' nonzero value collapsed')
    if isinstance(value, (int, np.integer)):
        require(int(converted) == value, name+' integer conversion loses value')
    return converted


def integer(value, name, low, high):
    require((type(value) is int or isinstance(value, np.integer)) and not isinstance(value,(bool,np.bool_)), name+' must be an integer')
    require(low <= value <= high, name+' outside supported integer range')
    return int(value)


def vector(values, name='values', minimum=1, maximum=100000):
    require(isinstance(values,(list,tuple,np.ndarray)), name+' must be a sequence')
    if isinstance(values,np.ndarray):
        require(values.ndim == 1, name+' must have shape (R,)')
    require(minimum <= len(values) <= maximum, name+' unsupported length')
    vals=[real(v,name+' element') for v in values]
    return np.array(vals,dtype=np.float64)


def _checked_square(value, name='term'):
    require(math.isfinite(value) and abs(value) <= 8e12, name+' outside arithmetic range')
    if value == 0:
        return 0.0
    result=value*value
    require(math.isfinite(result) and result >= np.finfo(float).tiny, name+' positive square is subnormal or zero; rescale inputs')
    return result


def mean(values):
    x=vector(values)
    if np.all(x == x[0]):
        return float(x[0])
    # Compensate original values; an anchor subtraction can erase a small mean.
    return math.fsum(x.tolist())/len(x)


def median(values):
    x=vector(values)
    ordered=sorted(x.tolist());n=len(x)
    return ordered[n//2] if n%2 else math.fsum([ordered[n//2-1]/2,ordered[n//2]/2])


def risk_summary(estimates, truth):
    """Finite R identity, variance denominator R, and MCSE of mean squared loss."""
    t=vector(estimates,minimum=2);theta=real(truth,'truth')
    average=mean(t);bias=average-theta
    losses=[_checked_square(float(v)-theta,'estimation error') for v in t]
    deviations=[_checked_square(float(v)-average,'centered estimate') for v in t]
    mse=losses[0] if all(v == losses[0] for v in losses) else math.fsum(losses)/len(t)
    var_R=math.fsum(deviations)/len(t)
    bias_sq=_checked_square(bias,'bias')
    loss_deviations=[_checked_square(v-mse,'centered squared loss') for v in losses]
    loss_var=math.fsum(loss_deviations)/(len(t)-1)
    require(abs(mse-(bias_sq+var_R)) <= 2e-12*max(1,mse,bias_sq+var_R),'finite risk decomposition failed')
    return {'R':len(t),'mean_estimate':average,'bias':bias,'bias_squared':bias_sq,
            'variance_R':var_R,'sample_variance_R_minus_1':var_R*len(t)/(len(t)-1),
            'mse':mse,'identity_residual':mse-bias_sq-var_R,'risk_mcse':math.sqrt(loss_var/len(t))}


def shrinkage_risk(mu, sigma, n, c):
    mu=real(mu,'mu',-100,100);sigma=real(sigma,'sigma',.01,100)
    n=integer(n,'n',1,1000000);c=real(c,'c',0,1)
    bias=(c-1)*mu;variance=_checked_square(c*sigma,'scaled sigma')/n
    b2=_checked_square(bias,'bias')
    return {'bias':bias,'variance':variance,'mse':b2+variance}


def vanishing_risk(mu, sigma, n, kappa):
    # Validate every argument before using the derived coefficient.
    mu=real(mu,'mu',-100,100);sigma=real(sigma,'sigma',.01,100)
    n=integer(n,'n',1,1000000);kappa=real(kappa,'kappa',.01,100)
    c=n/(n+kappa)
    out=shrinkage_risk(mu,sigma,n,c);out['c']=c
    return out


def oracle_risk(mu, sigma, n):
    mu=real(mu,'mu',-100,100);sigma=real(sigma,'sigma',.01,100);n=integer(n,'n',1,1000000)
    a=_checked_square(mu,'mu');v=_checked_square(sigma,'sigma')/n
    c=a/(a+v)
    return {'c_oracle':c,'mse_oracle':a*v/(a+v),'uses_unknown_truth':True}


def exact_two_point(mu=Fraction(1,2), n=3, c=Fraction(1,2)):
    """Exact arithmetic API, distinct from float APIs. Ordered samples have equal mass."""
    require(type(mu) is Fraction and type(c) is Fraction,'exact API requires Fraction mu and c')
    require(abs(mu)<=100 and 0<=c<=1,'exact parameters outside range')
    require(mu.denominator<=10000 and c.denominator<=10000,'exact denominator exceeds teaching budget')
    n=integer(n,'n',1,9);require(n%2==1,'odd n required for this exact median example')
    rows=[];values={name:[] for name in ['mean','median','fixed_shrinkage']}
    for signs in product([-1,1],repeat=n):
        x=[mu+v for v in signs];avg=sum(x,Fraction(0))/n
        estimates={'mean':avg,'median':sorted(x)[n//2],'fixed_shrinkage':c*avg}
        rows.append({'signs':list(signs),'estimates':{k:str(v) for k,v in estimates.items()}})
        for key,value in estimates.items():values[key].append(value)
    stats={}
    for key,t in values.items():
        avg=sum(t,Fraction(0))/len(t);bias=avg-mu
        variance=sum(((v-avg)**2 for v in t),Fraction(0))/len(t)
        mse=sum(((v-mu)**2 for v in t),Fraction(0))/len(t)
        require(mse==bias*bias+variance,'exact identity failed')
        stats[key]={k:str(v) for k,v in {'expectation':avg,'bias':bias,'variance':variance,'mse':mse,'new_y_prediction_mse':mse+1}.items()}
    return {'mu':str(mu),'n':n,'c':str(c),'ordered_sample_count':len(rows),'sample_probability':str(Fraction(1,len(rows))),'rows':rows,'statistics':stats}


SPEC_KEYS={'kind','seed','repetitions','sample_sizes','center','gaussian_sigma','contamination_probability','contamination_sigma','shrinkage_factor','vanishing_shrinkage_strength','risk_centers'}


def validate_spec(spec):
    require(type(spec) is dict and set(spec)==SPEC_KEYS,'spec keys must exactly match schema')
    require(spec['kind']=='synthetic_estimator_risk_demo','wrong model kind')
    s={'kind':spec['kind'],'seed':integer(spec['seed'],'seed',0,2**32-1),'repetitions':integer(spec['repetitions'],'repetitions',20,20000)}
    require(type(spec['sample_sizes']) is list and 1<=len(spec['sample_sizes'])<=8,'sample_sizes must be a nonempty short list')
    s['sample_sizes']=[integer(n,'sample size',3,501) for n in spec['sample_sizes']]
    require(all(n%2 for n in s['sample_sizes']) and len(set(s['sample_sizes']))==len(s['sample_sizes']),'sample sizes must be distinct odd integers')
    for key,lo,hi in [('center',-100,100),('gaussian_sigma',.1,10),('contamination_probability',0,1),('contamination_sigma',.1,30),('shrinkage_factor',0,1),('vanishing_shrinkage_strength',.01,100)]:
        s[key]=real(spec[key],key,lo,hi)
    s['risk_centers']=vector(spec['risk_centers'],'risk_centers',minimum=1,maximum=31).tolist()
    require(all(abs(v)<=100 for v in s['risk_centers']),'risk_centers outside model range')
    require(s['repetitions']*sum(s['sample_sizes'])<=3000000,'simulation draw budget exceeded')
    return s


def unique_object(pairs):
    out={}
    for key,value in pairs:
        require(key not in out,'duplicate JSON key: '+key);out[key]=value
    return out


def decimal_token(token):
    value=float(token)
    require(math.isfinite(value),'JSON number overflow or nonfinite')
    if value==0:
        require(not any(c in '123456789' for c in token.lower().split('e')[0]),'JSON nonzero token underflow')
    return real(value,'JSON number')


def reject_constant(token):
    raise ValueError('nonstandard JSON constant: '+token)


def load_spec(path=None):
    return validate_spec(json.loads(Path(path or ROOT/'data/model_spec.json').read_text(),parse_float=decimal_token,parse_constant=reject_constant,object_pairs_hook=unique_object))


def load_model(path=None):
    header=['state','center_numerator','center_denominator','offset','probability_numerator','probability_denominator']
    with Path(path or ROOT/'data/two_point_model.csv').open(newline='',encoding='utf-8') as f:
        r=csv.DictReader(f);require(r.fieldnames==header,'model CSV header mismatch');rows=list(r)
    require(len(rows)==2,'model requires exactly two states')
    parsed=[]
    for row in rows:
        require(set(row)==set(header) and all(v is not None for v in row.values()),'ragged model CSV')
        require(row['state'] in ['low','high'],'unknown model state')
        values=[]
        for key in header[1:]:
            text=row[key];require(re.fullmatch(r'-?(?:0|[1-9][0-9]{0,5})',text) is not None,'model requires bounded integer tokens')
            values.append(int(text))
        cn,cd,offset,pn,pd=values;require(cd>0 and pd>0,'positive denominators required')
        center=Fraction(cn,cd);prob=Fraction(pn,pd)
        require(abs(center)<=100 and center.denominator<=10000,'center outside exact range')
        require(offset in [-1,1] and prob==Fraction(1,2),'two equiprobable unit offsets required')
        parsed.append((row['state'],center,offset,prob))
    require([r[0] for r in parsed]==['low','high'],'states must be unique and ordered low,high')
    require(parsed[0][1]==parsed[1][1] and [r[2] for r in parsed]==[-1,1],'inconsistent centers/offsets')
    return parsed[0][1]


def _batch_estimates(x,c,kappa):
    # Private: x comes from the checked simulation model, shape (R,n).
    av=np.array([math.fsum(row.tolist())/x.shape[1] for row in x])
    return {'mean':av,'median':np.median(x,axis=1),'fixed_shrinkage':c*av,'vanishing_shrinkage':x.shape[1]/(x.shape[1]+kappa)*av}


def simulation(spec):
    s=validate_spec(spec)  # all final and unused parameters before constructing RNG
    streams=np.random.SeedSequence(s['seed']).spawn(2*len(s['sample_sizes']))
    results=[]
    for model_index,model in enumerate(['gaussian','symmetric_contamination']):
        eps=0 if model=='gaussian' else s['contamination_probability']
        popvar=(1-eps)*s['gaussian_sigma']**2+eps*s['contamination_sigma']**2
        for size_index,n in enumerate(s['sample_sizes']):
            rng=np.random.Generator(np.random.PCG64(streams[model_index*len(s['sample_sizes'])+size_index]))
            R=s['repetitions'];shape=(R,n)
            z=rng.standard_normal(shape)
            high=rng.random(shape)<eps
            scales=np.where(high,s['contamination_sigma'],s['gaussian_sigma'])
            x=s['center']+scales*z
            estimates=_batch_estimates(x,s['shrinkage_factor'],s['vanishing_shrinkage_strength'])
            # Fresh Y is independent of X, including its component choice.
            znew=rng.standard_normal(R);highnew=rng.random(R)<eps
            y=s['center']+np.where(highnew,s['contamination_sigma'],s['gaussian_sigma'])*znew
            for name,t in estimates.items():
                summary=risk_summary(t,s['center'])
                c={'mean':1,'fixed_shrinkage':s['shrinkage_factor'],'vanishing_shrinkage':n/(n+s['vanishing_shrinkage_strength'])}.get(name)
                theory=None if c is None else shrinkage_risk(s['center'],math.sqrt(popvar),n,c)
                pred_losses=(y-t)**2
                summary.update({'model':model,'n':n,'rule':name,'population_variance':popvar,'theory':theory,
                                'prediction_mse_observed':math.fsum(pred_losses.tolist())/R,
                                'prediction_mse_theory':None if theory is None else theory['mse']+popvar})
                results.append(summary)
    return {'spec':s,'rows':results,'independent_batches_per_row':s['repetitions'],'same_batch_methods_are_paired':True,
            'risk_centers_theory':[{'mu':mu,'mean':shrinkage_risk(mu,s['gaussian_sigma'],3,1),'fixed':shrinkage_risk(mu,s['gaussian_sigma'],3,s['shrinkage_factor']),'oracle':oracle_risk(mu,s['gaussian_sigma'],3)} for mu in s['risk_centers']]}


def run(input_path=None,spec_path=None,output_dir=None):
    target=Path(output_dir or ROOT/'outputs')
    require(not target.exists() or target.is_dir(),'output target must be a directory')
    mu=load_model(input_path);spec=load_spec(spec_path)
    exact=exact_two_point(mu)
    report={'exact':exact,'simulation':simulation(spec)}
    target.mkdir(parents=True,exist_ok=True)
    temp=target/'report.json.tmp';temp.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False));temp.replace(target/'report.json')
    return report


def self_test():
    exact=exact_two_point();s=exact['statistics']
    require(s['mean']['mse']=='1/3' and s['median']['mse']=='1' and s['fixed_shrinkage']['mse']=='7/48','exact main risk')
    require(s['fixed_shrinkage']['new_y_prediction_mse']=='55/48','prediction exact anchor')
    require(abs(shrinkage_risk(2,1,3,.5)['mse']-13/12)<1e-14,'bad-center risk')
    require(abs(oracle_risk(.5,1,3)['c_oracle']-3/7)<1e-14,'oracle coefficient')
    require(mean([-1,1,1e-100])>0,'original-value compensated mean')
    require(risk_summary([.1,.1,.1],.1)['mse']==0,'constant exact zero')
    require(median([-1,0,0,1,100])==0 and mean([-1,0,0,1,100])==20,'replacement sensitivity')
    require(abs(risk_summary([0,1,2],.5)['identity_residual'])<1e-14,'finite decomposition')
    return {'core_checks':8,'status':'passed'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input');p.add_argument('--spec');p.add_argument('--output-dir');args=p.parse_args()
    tests=self_test();report=run(args.input,args.spec,args.output_dir)
    print(json.dumps({'self_test':tests,'exact_risks':report['exact']['statistics'],'simulation_rows':len(report['simulation']['rows'])},ensure_ascii=False,indent=2))
