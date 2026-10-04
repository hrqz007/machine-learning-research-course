"""Finite, offline Beta-Bernoulli teaching experiments; no external data access."""
from __future__ import annotations
import argparse, csv, json, math, numbers, sys
from fractions import Fraction
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent
MAX_SHAPE_SUM = 10000.0

def require(condition, message):
    if not condition: raise ValueError(message)

def real(value, name):
    require(isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_)), name + ' must be an ordinary Python/NumPy real scalar, not bool/text/complex/Fraction')
    if isinstance(value, np.floating):
        require(value.dtype.itemsize <= 8, name + ' extended-precision scalars are outside this float64 contract')
    try:
        converted = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(name + ' conversion outside finite float64 range') from error
    require(math.isfinite(converted), name + ' must be finite')
    require(value == 0 or converted != 0, name + ' nonzero input converted to zero')
    if isinstance(value,(int,np.integer)):
        require(converted == int(value), name + ' integer cannot be represented exactly as float64')
    return converted

def count(value, name, maximum=1000):
    require(isinstance(value, numbers.Integral) and not isinstance(value, (bool, np.bool_)), name + ' must be an integer')
    value = int(value); require(0 <= value <= maximum, name + ' outside supported range')
    return value

def shapes(a, b):
    a, b = real(a, 'a'), real(b, 'b')
    require(a >= .25 and b >= .25, 'numerical implementation requires a,b >= 0.25; theory allows all positive shapes')
    require(math.isfinite(a+b) and a+b <= MAX_SHAPE_SUM, 'supported shape sum is <= 10000')
    return a, b

def log_beta(a, b):
    a,b=shapes(a,b); value=math.lgamma(a)+math.lgamma(b)-math.lgamma(a+b)
    require(math.isfinite(value), 'nonfinite log Beta'); return value

def beta_logpdf(x,a,b):
    a,b=shapes(a,b); x=real(x,'x'); require(0 <= x <= 1, 'x must lie in [0,1]')
    normalizer=log_beta(a,b)
    if x == 0:
        return -math.inf if a>1 else math.inf if a<1 else -normalizer
    if x == 1:
        return -math.inf if b>1 else math.inf if b<1 else -normalizer
    value=(a-1)*math.log(x)+(b-1)*math.log1p(-x)-normalizer
    require(math.isfinite(value),'nonfinite interior log density');return value

def beta_pdf(x,a,b):
    value=beta_logpdf(x,a,b)
    if not math.isfinite(value): return 0.0 if value<0 else math.inf
    require(value <= math.log(sys.float_info.max), 'density overflow')
    ans=math.exp(value);require(ans >= sys.float_info.min,'positive density underflow/subnormal: use log density');return ans

def beta_moments(a,b):
    a,b=shapes(a,b);s=a+b;q=a/s;v=q*(b/s)/(s+1)
    return {'mean':q,'second_moment':q*((a+1)/(s+1)),'variance':v}

def beta_mode(a,b):
    a,b=shapes(a,b)
    if a>1 and b>1: return {'kind':'unique_interior','theta':(a-1)/(a+b-2)}
    if a==1 and b==1: return {'kind':'uniform_nonunique','theta':None}
    if a<1 and b<1: return {'kind':'both_boundaries_unbounded','theta':None}
    if a<1: return {'kind':'left_boundary_unbounded','theta':None}
    if b<1: return {'kind':'right_boundary_unbounded','theta':None}
    return {'kind':'continuous_closure_endpoint','theta':0.0 if a==1 else 1.0}

def posterior(a,b,n,k):
    a,b=shapes(a,b);n=count(n,'n');k=count(k,'k');require(k<=n,'k must be <= n')
    return shapes(a+k,b+n-k)

def evidence(a,b,n,k,count_only=False):
    require(type(count_only) is bool, 'count_only must be bool')
    aa,bb=posterior(a,b,n,k);v=log_beta(aa,bb)-log_beta(a,b)
    if count_only:v+=math.log(math.comb(n,k))
    require(math.isfinite(v),'nonfinite log evidence'); p=math.exp(v)
    require(p>=sys.float_info.min and p<=1+1e-9,'evidence outside supported positive range')
    return {'log_evidence':v,'evidence':p,'data_event':'count' if count_only else 'ordered_sequence'}

def predictive_pmf(a,b,m):
    a,b=shapes(a,b);m=count(m,'m');base=log_beta(a,b)
    # Each finite log mass has a separately checked representable probability.
    masses=[]
    for j in range(m+1):
        v=math.log(math.comb(m,j))+log_beta(a+j,b+m-j)-base
        p=math.exp(v);require(math.isfinite(p) and p>=sys.float_info.min,'predictive positive mass underflow/subnormal')
        masses.append(p)
    require(abs(math.fsum(masses)-1)<=1e-9,'predictive normalization failed; no silent renormalization')
    return masses

def predictive_moments(a,b,m):
    a,b=shapes(a,b);m=count(m,'m');mom=beta_moments(a,b);q,v=mom['mean'],mom['variance'];plug=m*q*(1-q)
    return {'mean':m*q,'variance':plug+m*(m-1)*v,'plugin_variance':plug,'pair_covariance':v,'pair_correlation':1/(a+b+1)}

def beta_fraction(a,b):
    a=count(a,'integer a',200);b=count(b,'integer b',200);require(a>0 and b>0,'integer shapes must be positive')
    return Fraction(math.factorial(a-1)*math.factorial(b-1),math.factorial(a+b-1))

def exact_predictive(a,b,m):
    count(m,'m',100);base=beta_fraction(a,b)
    return [math.comb(m,j)*beta_fraction(a+j,b+m-j)/base for j in range(m+1)]

def beta_integer_interval(a,b,lo=Fraction(0),hi=Fraction(1)):
    """Polynomial expansion integral, deliberately separate from lgamma."""
    base=beta_fraction(a,b)
    require(isinstance(lo,Fraction) and isinstance(hi,Fraction) and 0<=lo<=hi<=1,'exact endpoints must be Fractions in [0,1]')
    return sum((Fraction((-1)**j*math.comb(b-1,j),a+j)*(hi**(a+j)-lo**(a+j)) for j in range(b)),Fraction(0))/base

def plugin_pmf(m,q):
    m=count(m,'m');q=real(q,'q');require(0<=q<=1,'q outside [0,1]')
    if q == 0:return [1.0]+[0.0]*m
    if q == 1:return [0.0]*m+[1.0]
    masses=[]
    for j in range(m+1):
        logp=math.log(math.comb(m,j))+j*math.log(q)+(m-j)*math.log1p(-q)
        p=math.exp(logp)
        require(math.isfinite(p) and p>=sys.float_info.min,'positive plugin mass underflow/subnormal')
        masses.append(p)
    require(abs(math.fsum(masses)-1)<=1e-9,'plugin normalization failed')
    return masses

def discrete_update(points,weights,n,k):
    n=count(n,'n');k=count(k,'k');require(k<=n,'k>n')
    require(isinstance(points,(list,tuple)) and isinstance(weights,(list,tuple)) and len(points)==len(weights)>0,'nonempty aligned sequences required')
    pts=[real(x,'point') for x in points];ws=[real(x,'weight') for x in weights]
    require(all(0<=x<=1 for x in pts) and all(w>=0 for w in ws),'invalid point or weight')
    require(len(set(pts))==len(pts),'duplicate support points')
    total=math.fsum(ws);require(abs(total-1)<=1e-12,'weights must sum to one')
    raw=[]
    for x,w in zip(pts,ws):
        if w == 0 or (x == 0 and k > 0) or (x == 1 and n-k > 0):
            raw.append(0.0)
            continue
        log_mass=math.log(w)
        if k:log_mass+=k*math.log(x)
        if n-k:log_mass+=(n-k)*math.log1p(-x)
        mass=math.exp(log_mass)
        require(math.isfinite(mass) and mass>=sys.float_info.min,'positive weighted likelihood underflow/subnormal')
        raw.append(mass)
    z=math.fsum(raw)
    require(z>=sys.float_info.min and math.isfinite(z),'zero evidence or unsupported positive underflow')
    posterior_values=[x/z for x in raw]
    require(all(x == 0 or y>=sys.float_info.min for x,y in zip(raw,posterior_values)),'positive posterior mass underflow/subnormal')
    return {'evidence':z,'posterior':posterior_values}

def validate_observations(rows):
    require(isinstance(rows,(list,tuple)) and 0<len(rows)<=1000,'1..1000 observations required')
    clean=[];seen=set()
    for row in rows:
        require(isinstance(row,dict) and set(row)=={'observation_id','batch','success'},'invalid observation fields')
        oid=row['observation_id'];batch=row['batch'];y=row['success']
        require(isinstance(oid,str) and oid.isascii() and oid.isalnum() and 1<=len(oid)<=30,'invalid observation ID')
        require(oid not in seen,'duplicate observation ID');seen.add(oid)
        require(batch in ('first','second'),'batch must be first or second')
        require(type(y) is int and y in (0,1),'success must be integer 0 or 1, not bool')
        clean.append(dict(row))
    return clean

def load_observations(path=None):
    with Path(path or ROOT/'data'/'observations.csv').open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f);require(reader.fieldnames==['observation_id','batch','success'],'invalid observation CSV header');rows=list(reader)
    for row in rows:
        require(row.get('success') in ('0','1'),'CSV success must be 0 or 1');row['success']=int(row['success'])
    return validate_observations(rows)

def load_priors(path=None):
    with Path(path or ROOT/'data'/'priors.csv').open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f);require(reader.fieldnames==['prior_id','alpha','beta','purpose'],'invalid prior CSV header');rows=list(reader)
    require(0<len(rows)<=20,'1..20 priors required');seen=set();out=[]
    for r in rows:
        require(set(r)=={'prior_id','alpha','beta','purpose'},'invalid prior fields')
        pid=r['prior_id'];require(isinstance(pid,str) and pid.replace('_','').isalnum() and pid.isascii() and pid not in seen,'invalid or duplicate prior ID');seen.add(pid)
        require(isinstance(r['purpose'],str) and r['purpose'].replace('_','').isalnum() and r['purpose'].isascii(),'invalid prior purpose')
        a,b=shapes(float(r['alpha']),float(r['beta']));out.append({'prior_id':pid,'alpha':a,'beta':b,'purpose':r['purpose']})
    require('main' in seen,'one prior named main required');return out

def sequential_update(a,b,rows):
    rows=validate_observations(rows);a,b=shapes(a,b);history=[]
    for batch in ('first','second'):
        values=[r['success'] for r in rows if r['batch']==batch];a,b=posterior(a,b,len(values),sum(values));history.append({'batch':batch,'a':a,'b':b})
    return history

def simulate(a,b,m=4,repetitions=20000,seed=24024):
    a,b=shapes(a,b);m=count(m,'m',100);repetitions=count(repetitions,'repetitions',100000);seed=count(seed,'seed',2**32-1)
    require(repetitions>0 and repetitions*max(m,1)<=5000000,'invalid simulation size')
    # Complete contract validation happens before RNG construction.
    rng=np.random.Generator(np.random.PCG64(seed));theta=rng.beta(a,b,repetitions)
    require(np.all(np.isfinite(theta)) and np.all((theta>=0)&(theta<=1)),'invalid generated parameters')
    counts=rng.binomial(m,theta)
    independent_theta=rng.beta(a,b,(repetitions,m));wrong_counts=(rng.random((repetitions,m))<independent_theta).sum(axis=1)
    return {'theta':theta,'shared_counts':counts,'redrawn_counts':wrong_counts}

def population_moments(values):
    require(isinstance(values,(list,tuple,np.ndarray)) and len(values)>0,'nonempty real sequence required')
    xs=[real(x,'value') for x in values]
    if all(x==xs[0] for x in xs):return {'mean':xs[0],'variance':0.0}
    # Compensated original-value summation preserves small mixed-scale means.
    # Shifting by one extreme sample can erase a representable small value.
    total=math.fsum(xs);mean=total/len(xs)
    require(math.isfinite(mean),'mean overflow')
    require(total == 0 or abs(mean)>=sys.float_info.min,'nonzero mean underflow/subnormal')
    residuals=[x-mean for x in xs];ss=[]
    for residual in residuals:
        require(math.isfinite(residual),'centered residual overflow')
        square=residual*residual
        require(math.isfinite(square),'squared residual overflow')
        require(residual == 0 or square/len(xs)>=sys.float_info.min,'positive variance contribution underflow/subnormal')
        ss.append(square)
    variance=math.fsum(ss)/len(xs)
    require(math.isfinite(variance),'variance overflow')
    require(variance>=sys.float_info.min,'nonconstant sample lost its positive variance')
    return {'mean':mean,'variance':variance}

def quadrature(a,b,n=1000):
    a,b=shapes(a,b);n=count(n,'grid n',100000);require(n>0,'grid must be nonempty')
    # Midpoints avoid infinite endpoints. This does not erase singularity error.
    return math.fsum(beta_pdf((i+.5)/n,a,b) for i in range(n))/n

def reference_checks():
    checks=0
    for a in range(1,7):
        for b in range(1,7):
            base=beta_fraction(a,b)
            require(abs(math.exp(log_beta(a,b))-float(base))<2e-14,'Beta normalization');checks+=1
            require(beta_integer_interval(a,b)==1,'polynomial integral');checks+=1
            mo=beta_moments(a,b)
            for name,ref in [('mean',Fraction(a,a+b)),('variance',Fraction(a*b,(a+b)**2*(a+b+1)))]:
                require(abs(mo[name]-float(ref))<2e-15,'moment identity');checks+=1
            for m in range(7):
                ex=exact_predictive(a,b,m);num=predictive_pmf(a,b,m)
                require(sum(ex)==1,'predictive exact normalization');checks+=1
                for p,q in zip(num,ex):require(abs(p-float(q))<3e-14,'predictive reference');checks+=1
    require(beta_integer_interval(5,7,Fraction(1,2),Fraction(1))==Fraction(281,1024),'posterior tail');checks+=1
    return checks

def boundary_checks():
    bad=[lambda:shapes(0,2),lambda:shapes(-1,2),lambda:shapes(.1,2),lambda:shapes(True,2),lambda:shapes('2',2),lambda:shapes(2+0j,2),lambda:shapes(math.nan,2),lambda:shapes(2,math.inf),lambda:shapes(1e308,1e308),lambda:shapes(6000,6000),lambda:posterior(2,2,8,9),lambda:posterior(2,2,8.0,3),lambda:posterior(2,2,True,0),lambda:posterior(2,2,-1,0),lambda:beta_pdf(-.1,2,2),lambda:beta_pdf(1.1,2,2),lambda:beta_pdf(True,2,2),lambda:beta_logpdf(math.nan,2,2),lambda:predictive_pmf(2,2,2.1),lambda:predictive_pmf(2,2,-1),lambda:predictive_pmf(4999,4999,4),lambda:discrete_update([0,1],[.5,.5],8,3),lambda:discrete_update([0,math.nan],[1,0],1,1),lambda:discrete_update([0,1],[1,1],1,1),lambda:discrete_update([0,0],[.5,.5],1,0),lambda:discrete_update([0,1],[True,0],1,0),lambda:quadrature(2,2,0),lambda:quadrature(2,2,True),lambda:simulate(2,2,4,0),lambda:simulate(2,2,True),lambda:simulate(2,2,4,20000,True),lambda:population_moments([.1,math.nan]),lambda:population_moments([.1,True]),lambda:population_moments([]),lambda:evidence(2,2,8,3,1),lambda:plugin_pmf(4,math.nan),lambda:plugin_pmf(4,2),lambda:beta_pdf(1e-200,5,7)]
    bad += [lambda:discrete_update([.1,.5],[.5,.5],1000,500),lambda:plugin_pmf(1000,.99),lambda:population_moments([0,1e-200]),lambda:real(np.longdouble('1e-400'),'x'),lambda:real(Fraction(1,10**400),'x'),lambda:population_moments([0,1e-154]),lambda:discrete_update([0,Fraction(1,10**400)],[1,0],1,0),lambda:plugin_pmf(4,np.longdouble('.5')),lambda:population_moments([2**53,2**53+1]),lambda:real(np.int64(2**53+1),'x')]
    bad += [lambda:population_moments([-1,1,1e-200]),lambda:population_moments([-1,1,1e-170]),lambda:population_moments([-1,1,1e-154])]
    row={'observation_id':'O1','batch':'first','success':0}
    bad += [lambda:validate_observations([row,row]),lambda:validate_observations([row,dict(row,observation_id='O2',success=True)]),lambda:validate_observations([dict(row,batch='later')]),lambda:validate_observations([])]
    for f in bad:
        try:f()
        except (ValueError,OverflowError):pass
        else:raise ValueError('invalid input escaped')
    accepted=0
    for a,b,kind in [(2,2,'unique_interior'),(1,1,'uniform_nonunique'),(1,3,'continuous_closure_endpoint'),(3,1,'continuous_closure_endpoint'),(.5,2,'left_boundary_unbounded'),(2,.5,'right_boundary_unbounded'),(.5,.5,'both_boundaries_unbounded')]:
        require(beta_mode(a,b)['kind']==kind,'mode semantics');accepted+=1
    require(abs(beta_pdf(0,1,3)-3)<1e-14,'zero exponent boundary');accepted+=1
    require(beta_pdf(1,2,2)==0 and math.isinf(beta_pdf(0,.5,.5)),'endpoint limits');accepted+=1
    require(posterior(2,2,0,0)==(2,2) and predictive_pmf(2,2,0)==[1.0],'zero observations/future');accepted+=1
    for v in [.1,1e-140,3e90]:require(population_moments([v]*7)=={'mean':v,'variance':0.0},'constant exact');accepted+=1
    require(plugin_pmf(4,0)==[1,0,0,0,0] and plugin_pmf(4,1)==[0,0,0,0,1],'true endpoint zeros retained');accepted+=1
    require(discrete_update([0,1],[.5,.5],1,1)['posterior']==[0,1],'impossible endpoint mass stays zero');accepted+=1
    require(discrete_update([.1,.5],[0,1],1000,500)['posterior']==[0,1],'true zero weight stays zero');accepted+=1
    require(population_moments([-1,1,1e-100])['mean']==math.fsum([-1,1,1e-100])/3,'mixed-scale small mean retained');accepted+=1
    return {'expected_rejections':len(bad),'accepted_groups':accepted}

def run(observations=None,priors=None,repetitions=20000):
    rows=load_observations(observations);ps=load_priors(priors);n=len(rows);k=sum(r['success'] for r in rows)
    main=next(p for p in ps if p['prior_id']=='main');a,b=posterior(main['alpha'],main['beta'],n,k)
    # Resolve every prior before stochastic work, so an invalid later prior cannot escape.
    sensitivity=[dict(p,posterior=list(posterior(p['alpha'],p['beta'],n,k)),moments=beta_moments(*posterior(p['alpha'],p['beta'],n,k))) for p in ps]
    pmf=predictive_pmf(a,b,4);sim=simulate(a,b,4,repetitions)
    report={'kind':'synthetic_beta_bernoulli_teaching','n':n,'k':k,'prior':[main['alpha'],main['beta']],'posterior':[a,b],'posterior_moments':beta_moments(a,b),'theta_coordinate_mode':beta_mode(a,b),'eta_coordinate_mode_mapped_to_theta':a/(a+b),'ordered_evidence':evidence(main['alpha'],main['beta'],n,k),'count_evidence':evidence(main['alpha'],main['beta'],n,k,True),'sequential':sequential_update(main['alpha'],main['beta'],rows),'predictive_pmf':pmf,'predictive_mass_sum':math.fsum(pmf),'predictive_moments':predictive_moments(a,b,4),'plugin_pmf':plugin_pmf(4,a/(a+b)),'sensitivity':sensitivity,'grid_midpoint':[{'n':g,'integral':quadrature(a,b,g)} for g in [20,100,1000]],'simulation':{'seed':24024,'repetitions':repetitions,'parameter':population_moments(sim['theta']),'shared_counts':population_moments(sim['shared_counts']),'redrawn_counts':population_moments(sim['redrawn_counts']),'shared_frequencies':(np.bincount(sim['shared_counts'],minlength=5)/repetitions).tolist(),'redrawn_frequencies':(np.bincount(sim['redrawn_counts'],minlength=5)/repetitions).tolist()},'tests':{'references':reference_checks(),'boundary':boundary_checks()}}
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--observations',type=Path);parser.add_argument('--priors',type=Path);parser.add_argument('--output',type=Path,default=ROOT/'outputs');args=parser.parse_args()
    report=run(args.observations,args.priors);payload=json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    args.output.mkdir(parents=True,exist_ok=True);target=args.output/'report.json';temp=args.output/'report.json.tmp';temp.write_text(payload,encoding='utf-8');temp.replace(target)
    print(json.dumps({'n':report['n'],'k':report['k'],'posterior':report['posterior'],'tests':report['tests']},sort_keys=True))
if __name__=='__main__':main()
