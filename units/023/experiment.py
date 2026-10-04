"""Unit 023: likelihood, closed-form MLEs and transparent numerical checks.

All data are synthetic. Core functions have no I/O or global RNG mutation.
The finite numerical domain is deliberately conservative; see README.md.
"""
from pathlib import Path
import argparse, csv, io, json, math, re, sys
import numpy as np

ROOT = Path(__file__).resolve().parent
MAX_N = 100000
MIN_V, MAX_V = 1e-12, 1e12

def require(condition, message):
    """Checks remain active under python -O."""
    if not condition:
        raise ValueError(message)

def scalar(value, name, low=-1e6, high=1e6):
    require(isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_)), name + ': real numeric scalar required')
    require(not isinstance(value,np.floating) or value.dtype.itemsize <= 8, name + ': extended-precision inputs must be explicitly rescaled/converted by caller')
    # Check before converting, including oversized Python integers.
    require(bool(np.isfinite(value)) if not isinstance(value, int) else True, name + ': finite value required')
    require(low <= value <= high, name + ': outside documented range')
    converted = float(value)
    require(math.isfinite(converted), name + ': conversion overflow')
    require(value == 0 or converted != 0, name + ': conversion underflow')
    return converted

def integer(value, name, low, high):
    require(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)), name + ': integer required')
    require(low <= value <= high, name + ': outside documented range')
    return int(value)

def vector(values, name='data', binary=False):
    require(type(binary) is bool, 'binary must be bool')
    require(type(name) is str, 'name must be text')
    require(isinstance(values, (list, tuple, np.ndarray)), name + ': list, tuple or numeric ndarray required')
    if isinstance(values, np.ndarray):
        require(values.ndim == 1, name + ': one-dimensional array required')
        require(values.dtype.kind in 'iuf', name + ': bool, text, complex and object dtypes rejected')
    require(1 <= len(values) <= MAX_N, name + ': length must be 1..100000')
    checked = []
    for value in values:
        z = scalar(value, name)
        require(z == 0 or abs(z) >= 1e-12, name + ': nonzero magnitudes below 1e-12 are unsupported')
        if binary:
            require(z in (0.0, 1.0), name + ': Bernoulli observations must be 0 or 1')
        checked.append(z)
    return np.array(checked, dtype=np.float64)

def counts(values):
    x = vector(values, binary=True)
    return len(x), int(math.fsum(x))

def _bern_ll(n, k, p):
    if p == 0:
        return 0.0 if k == 0 else -math.inf
    if p == 1:
        return 0.0 if k == n else -math.inf
    result = k * math.log(p) + (n-k) * math.log1p(-p)
    require(math.isfinite(result), 'Bernoulli log likelihood overflow')
    return result

def bernoulli_loglik(values, p, *, count_observation=False):
    """Ordered sequence by default; count likelihood adds log(n choose k)."""
    n, k = counts(values)
    p = scalar(p, 'p', 0, 1)
    require(type(count_observation) is bool, 'count_observation must be bool')
    result = _bern_ll(n, k, p)
    if count_observation:
        result += math.log(math.comb(n, k))
    return result

def bernoulli_mle(values):
    n, k = counts(values)
    return {'n': n, 'successes': k, 'p_mle': k/n,
            'location': 'boundary' if k in (0,n) else 'interior'}

def bernoulli_derivatives(values, p):
    n, k = counts(values)
    p = scalar(p, 'interior p', 1e-12, 1-1e-12)
    score = k/p - (n-k)/(1-p)
    curvature = -k/p**2 - (n-k)/(1-p)**2
    require(math.isfinite(score) and math.isfinite(curvature), 'derivative overflow')
    return score, curvature

def bisect_bernoulli(values, tolerance=1e-10, max_iterations=100):
    """Bisect the strictly decreasing score; treat closed-domain endpoints exactly."""
    n,k = counts(values)
    tolerance = scalar(tolerance, 'tolerance', 1e-12, 1e-3)
    max_iterations = integer(max_iterations, 'max_iterations', 1, 200)
    if k in (0,n):
        return {'p': k/n, 'bracket': [k/n,k/n], 'iterations': 0, 'converged': True}
    lo,hi = 0.0,1.0
    for step in range(1,max_iterations+1):
        mid = (lo+hi)/2
        score = k/mid-(n-k)/(1-mid)
        if score == 0:
            lo=hi=mid
        elif score > 0:
            lo=mid
        else:
            hi=mid
        if hi-lo <= tolerance:
            break
    return {'p': (lo+hi)/2, 'bracket': [lo,hi], 'iterations': step,
            'converged': hi-lo <= tolerance}

def _sse(x, center):
    deviations = [float(z)-center for z in x]
    squares = [z*z for z in deviations]
    require(all(math.isfinite(q) for q in squares), 'SSE overflow')
    require(all(d == 0 or q > 0 for d,q in zip(deviations,squares)), 'SSE underflow')
    total = math.fsum(squares)
    require(math.isfinite(total), 'SSE sum overflow')
    return total

def gaussian_fit(values):
    """Unrestricted model mu in R, v>0, subject to documented numeric guardrails."""
    x = vector(values)
    n = len(x)
    if all(z == x[0] for z in x):
        return {'n':n, 'mean':float(x[0]), 'sse':0.0, 'variance_mle':None,
                'variance_unbiased':0.0 if n>1 else None,
                'status':'no_finite_mle_unbounded_likelihood'}
    mean = math.fsum(x)/n
    sse = _sse(x,mean)
    if sse == 0:
        # Distinguish family degeneracy from numerical loss of distinct values.
        require(all(z == x[0] for z in x), 'SSE collapsed for distinct data')
        return {'n':n, 'mean':mean, 'sse':0.0, 'variance_mle':None,
                'variance_unbiased':0.0 if n>1 else None,
                'status':'no_finite_mle_unbounded_likelihood'}
    v = scalar(sse/n, 'fitted variance', MIN_V, MAX_V)
    return {'n':n, 'mean':mean, 'sse':sse, 'variance_mle':v,
            'variance_unbiased':sse/(n-1) if n>1 else None, 'status':'finite_unique_mle'}

def gaussian_loglik(values, mu, variance):
    x = vector(values)
    mu = scalar(mu,'mu')
    variance = scalar(variance,'variance',MIN_V,MAX_V)
    q = _sse(x,mu)
    result = -len(x)/2 * (math.log(2*math.pi)+math.log(variance))-q/(2*variance)
    require(math.isfinite(result), 'Gaussian log likelihood overflow')
    return result

def gaussian_derivatives(values, mu, variance):
    x = vector(values)
    mu = scalar(mu,'mu'); variance = scalar(variance,'variance',MIN_V,MAX_V)
    q = _sse(x,mu)
    dmu = math.fsum(float(z)-mu for z in x)/variance
    dv = -len(x)/(2*variance)+q/(2*variance**2)
    require(math.isfinite(dmu) and math.isfinite(dv), 'Gaussian derivative overflow')
    return dmu,dv

def golden_maximize(function, lower, upper, tolerance=1e-9, max_iterations=200):
    """Golden section search on a finite bracket for a continuous unimodal function.

    All input configuration is validated before the first function call. Return
    status is not a global-optimality proof; the caller supplies the shape argument.
    """
    require(callable(function), 'function must be callable')
    lo=scalar(lower,'lower'); hi=scalar(upper,'upper')
    tolerance=scalar(tolerance,'tolerance',1e-12,1e-2)
    max_iterations=integer(max_iterations,'max_iterations',1,1000)
    require(lo < hi, 'lower must be below upper')
    def evaluate(z):
        return scalar(function(z),'objective value',-1e250,1e250)
    ratio=(math.sqrt(5)-1)/2
    c=hi-ratio*(hi-lo); d=lo+ratio*(hi-lo)
    fl,fh=evaluate(lo),evaluate(hi)
    fc,fd=evaluate(c),evaluate(d)
    candidates=[(fl,lo),(fh,hi)]
    for step in range(1,max_iterations+1):
        if hi-lo <= tolerance:
            break
        if fc > fd:
            hi,d,fd=d,c,fc
            c=hi-ratio*(hi-lo); fc=evaluate(c)
        else:
            lo,c,fc=c,d,fd
            d=lo+ratio*(hi-lo); fd=evaluate(d)
    mid=(lo+hi)/2
    candidates.extend([(fc,c),(fd,d),(evaluate(mid),mid)])
    best=max(candidates)
    return {'x':best[1], 'value':best[0], 'bracket':[lo,hi],
            'iterations':step, 'converged':hi-lo <= tolerance}

def load_csv(path, binary=False):
    require(type(binary) is bool,'binary must be bool')
    path=Path(path)
    require(path.stat().st_size <= 5000000,'CSV too large')
    with path.open(newline='',encoding='utf-8') as stream:
        rows=list(csv.reader(stream))
    require(rows and rows[0] == ['observation_id','value'],'CSV header must be observation_id,value')
    require(1 <= len(rows)-1 <= MAX_N,'CSV row count out of range')
    ids=set(); values=[]
    pattern=r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
    for row in rows[1:]:
        require(len(row)==2,'CSV must have exactly two fields per row')
        name,token=row
        require(bool(re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,39}',name)) and name not in ids,'CSV identifiers must be unique, unpadded ASCII identifiers')
        require(bool(re.fullmatch(pattern,token)),'CSV value must be a finite unpadded decimal token')
        # Decimal string transport is intentional here; numeric APIs reject text.
        value=float(token)
        require(math.isfinite(value),'CSV conversion overflow')
        require(value != 0 or not any(c in '123456789' for c in token.split('e')[0].split('E')[0]),'CSV conversion underflow')
        ids.add(name);values.append(value)
    return vector(values,binary=binary)

def validate_spec(spec):
    keys={'kind','seed','repetitions','sizes','p','mu','variance'}
    require(type(spec) is dict and set(spec)==keys,'Exact simulation specification keys required')
    require(spec['kind']=='synthetic_mle_repeated_sampling','Unknown simulation kind')
    seed=integer(spec['seed'],'seed',0,2**32-513)
    repetitions=integer(spec['repetitions'],'repetitions',2,10000)
    require(type(spec['sizes']) is list and 1 <= len(spec['sizes']) <= 8,'sizes must be a nonempty list, at most 8 entries')
    sizes=[integer(n,'sample size',2,512) for n in spec['sizes']]
    require(sizes==sorted(set(sizes)),'sizes must be increasing and unique')
    require(repetitions*sum(sizes)<=2000000,'simulation exceeds 2 million cells per family')
    p=scalar(spec['p'],'simulation p',0,1)
    mu=scalar(spec['mu'],'simulation mu',-100,100)
    variance=scalar(spec['variance'],'simulation variance',1e-4,10000)
    return {'kind':spec['kind'],'seed':seed,'repetitions':repetitions,'sizes':sizes,'p':p,'mu':mu,'variance':variance}

def load_spec(path=None):
    path=ROOT/'data/model_spec.json' if path is None else Path(path)
    require(path.stat().st_size<=10000,'specification too large')
    def no_duplicate_keys(pairs):
        result={}
        for key,value in pairs:
            require(key not in result,'Duplicate JSON key');result[key]=value
        return result
    return validate_spec(json.loads(path.read_text(encoding='utf-8'),object_pairs_hook=no_duplicate_keys))

def repeat_sampling(spec):
    cfg=validate_spec(spec)  # Complete validation precedes every RNG construction.
    rows=[]; samples={}
    for n in cfg['sizes']:
        rng=np.random.Generator(np.random.PCG64(cfg['seed']+n))
        b=rng.binomial(1,cfg['p'],(cfg['repetitions'],n))
        x=rng.normal(cfg['mu'],math.sqrt(cfg['variance']),(cfg['repetitions'],n))
        require(np.all(np.isfinite(x)) and np.max(np.abs(x))<=1e6,'Generated data outside range')
        means=x.mean(axis=1); d=x-means[:,None]
        with np.errstate(over='raise',under='raise',invalid='raise'):
            sse=np.sum(d*d,axis=1); vmle=sse/n; unbiased=sse/(n-1)
        require(np.all(np.isfinite(vmle)) and np.all(vmle>=MIN_V) and np.all(vmle<=MAX_V),'Simulated variance outside supported range')
        proportions=b.mean(axis=1)
        samples[n]={'p_mle':proportions,'mu_mle':means,'variance_mle':vmle,'variance_unbiased':unbiased}
        rows.append({'n':n,'repetitions':cfg['repetitions'],
          'mean_p_mle':float(proportions.mean()),'mean_mu_mle':float(means.mean()),
          'mean_variance_mle':float(vmle.mean()),'mean_variance_unbiased':float(unbiased.mean()),
          'theory_mean_variance_mle':(n-1)/n*cfg['variance'],
          'theory_mean_variance_unbiased':cfg['variance']})
    return {'spec':cfg,'summaries':rows,'samples':samples}

def reference_checks():
    checked=0
    def close(got,want,tol=1e-9):
        nonlocal checked
        require(math.isclose(got,want,rel_tol=tol,abs_tol=tol),f'Reference mismatch: {got} vs {want}');checked+=1
    x=[1,0,0,1,0,1,0,0]
    close(bernoulli_mle(x)['p_mle'],3/8)
    close(bernoulli_loglik(x,.5),-8*math.log(2))
    close(bernoulli_loglik(x,.375),3*math.log(3/8)+5*math.log(5/8))
    close(bernoulli_loglik(x,.375,count_observation=True)-bernoulli_loglik(x,.375),math.log(56))
    close(bernoulli_derivatives(x,.375)[0],0)
    close(bisect_bernoulli(x)['p'],3/8)
    for n in range(1,21):
        for k in range(n+1):
            y=[1]*k+[0]*(n-k);fit=bernoulli_mle(y);numeric=bisect_bernoulli(y)
            close(fit['p_mle'],k/n);close(numeric['p'],k/n)
            require(numeric['converged'],'Bisection not converged')
    g=gaussian_fit([0,1,2,5])
    for key,value in [('mean',2),('sse',14),('variance_mle',3.5),('variance_unbiased',14/3)]:close(g[key],value)
    close(gaussian_derivatives([0,1,2,5],2,3.5)[0],0)
    close(gaussian_derivatives([0,1,2,5],2,3.5)[1],0)
    sol=golden_maximize(lambda z:gaussian_loglik([0,1,2,5],2,math.exp(z)),-5,5)
    require(sol['converged'],'Golden search not converged');close(math.exp(sol['x']),3.5,1e-6)
    require(gaussian_fit([7,7])['variance_mle'] is None,'Degenerate Gaussian incorrectly fitted');checked+=1
    return {'passed':checked}

def boundary_checks():
    bad=[lambda:bernoulli_mle([]),lambda:bernoulli_mle([0,True]),lambda:bernoulli_mle(['0',1]),
         lambda:bernoulli_mle(np.array([0,1],dtype=object)),lambda:bernoulli_mle([0,1j]),
         lambda:bernoulli_mle([[0,1]]),lambda:bernoulli_mle([.5]),lambda:bernoulli_loglik([0],-1),
         lambda:bernoulli_loglik([0],float('nan')),lambda:bernoulli_loglik([0],.5,count_observation=1),
         lambda:gaussian_fit([0,float('inf')]),lambda:gaussian_fit([0,1e-100]),lambda:gaussian_fit([0,1e7]),
         lambda:gaussian_fit([1,np.nextafter(1.,2.)]),lambda:gaussian_loglik([0],0,0),
         lambda:gaussian_loglik([0],0,True),lambda:gaussian_loglik([0],float('nan'),1),
         lambda:golden_maximize(lambda x:x,0,1,tolerance=True),lambda:bisect_bernoulli([0,1],max_iterations=True),
         lambda:gaussian_fit(np.array([1,2],dtype=complex)),lambda:gaussian_fit(np.array([[1,2]])),
         lambda:gaussian_fit([10**1000]),lambda:gaussian_fit(np.array([True,False]))]
    for call in bad:
        try:call()
        except ValueError:pass
        else:raise ValueError('Invalid input accepted')
    require(bernoulli_loglik([0,0],0)==0,'all-zero boundary')
    require(bernoulli_loglik([1,1],1)==0,'all-one boundary')
    require(bernoulli_loglik([0,1],0)==-math.inf,'impossible event support')
    require(gaussian_fit([3])['status']=='no_finite_mle_unbounded_likelihood','n=1 degeneracy')
    return {'rejected':len(bad),'accepted_edge_cases':4}

def run(bernoulli_path=None, gaussian_path=None, spec_path=None, output=None):
    require(output is None or isinstance(output,(str,Path)), 'output must be a path string or Path')
    require(output is None or str(output).strip() != '', 'output path cannot be empty')
    target=ROOT/'outputs' if output is None else Path(output)
    b=load_csv(ROOT/'data/bernoulli.csv' if bernoulli_path is None else bernoulli_path,binary=True)
    x=load_csv(ROOT/'data/gaussian.csv' if gaussian_path is None else gaussian_path)
    spec=load_spec(spec_path)
    # Validate both fitted datasets before simulation or writing.
    bf=bernoulli_mle(b);gf=gaussian_fit(x)
    checks={'reference':reference_checks(),'boundary':boundary_checks()}
    repeat=repeat_sampling(spec)
    report={'kind':'synthetic_demonstration_not_real_performance','bernoulli':bf,'gaussian':gf,
      'bernoulli_numeric':bisect_bernoulli(b),'spec':spec,'repeated_sampling':repeat['summaries'],'checks':checks}
    if gf['variance_mle'] is not None:
        # Exp parametrization removes positivity violations, not existence issues.
        sol=golden_maximize(lambda z:gaussian_loglik(x,gf['mean'],math.exp(z)),math.log(MIN_V),math.log(MAX_V))
        require(sol['converged'],'Numeric profile optimization failed to converge')
        report['gaussian_numeric']={'mean_fixed_at_sample_mean':gf['mean'],'variance':math.exp(sol['x']),'solver':sol}
    else:
        report['gaussian_numeric']={'status':'not_run_no_finite_mle'}
    # Prepare every output byte before creating/overwriting any result file.
    report_text=json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    buffer=io.StringIO(newline='');writer=csv.writer(buffer,lineterminator='\n')
    writer.writerow(['n','replicate','p_mle','mu_mle','variance_mle','variance_unbiased'])
    for n,d in repeat['samples'].items():
        for i in range(spec['repetitions']):writer.writerow([n,i+1]+[format(float(d[k][i]),'.17g') for k in ['p_mle','mu_mle','variance_mle','variance_unbiased']])
    target.mkdir(parents=True,exist_ok=True)
    for name,text in [('report.json',report_text),('replicate_estimates.csv',buffer.getvalue())]:
        (target/name).write_text(text,encoding='utf-8')
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bernoulli');parser.add_argument('--gaussian');parser.add_argument('--spec');parser.add_argument('--output')
    args=parser.parse_args()
    report=run(args.bernoulli,args.gaussian,args.spec,args.output)
    print(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
if __name__=='__main__':main()
