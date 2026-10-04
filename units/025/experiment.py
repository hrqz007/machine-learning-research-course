"""025 区间估计与Bootstrap：离线、有限范围、可重复的教学实现。

所有公共数值接口先验证完整输入；不用 assert。这里只支持实数标量、
列表/元组/NumPy数组，不隐式接受文本、bool、复数、Decimal或任意迭代器。
float64教学域见README；不声称支持全部有限浮点数。
"""
from pathlib import Path
from fractions import Fraction
from collections import Counter
from statistics import NormalDist
import argparse, csv, json, math, numbers, re
import numpy as np
ROOT = Path(__file__).resolve().parent
Z95 = NormalDist().inv_cdf(.975)
MAX = 1e6
MIN = 1e-100


def require(condition, message):
    if not condition:
        raise ValueError(message)


def real(value, name, low=-MAX, high=MAX):
    require(isinstance(value, (numbers.Real, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_)), name+' must be a real number, not bool/text/complex')
    require(low <= value <= high, name+' is outside the supported range or nonfinite')
    require(math.isfinite(value), name+' must be finite')
    require(value == 0 or abs(value) >= MIN, name+' is below the supported nonzero magnitude')
    converted = float(value)
    require(value == 0 or converted != 0, name+' nonzero value collapsed to zero during float64 conversion')
    return converted


def integer(value, name, low, high):
    require(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)), name+' must be an integer')
    require(low <= value <= high, name+' is outside the supported range')
    return int(value)


def sequence(value, name):
    require(isinstance(value, (list, tuple, np.ndarray)), name+' must be a list, tuple or array')
    if isinstance(value, np.ndarray):
        require(value.ndim == 1, name+' must have shape (n,)')
    return value


def vector(value, name='x', minimum=1):
    raw = sequence(value, name)
    require(minimum <= len(raw) <= 10000, name+' length outside supported range')
    checked = [real(v, f'{name}[{i}]') for i, v in enumerate(raw)]
    if all(v == checked[0] for v in checked):
        require(all(v == raw[0] for v in raw), name+' nonconstant values collapsed to a float64 constant')
    return np.array(checked, dtype=float)


def probability(value, name='q', interior=False):
    q = real(value, name, 0, 1)
    require(q != 1 or value == 1, name+' positive upper-tail probability collapsed to zero during conversion')
    if interior:
        require(1e-6 <= q <= 1-1e-6, name+' must lie in [1e-6,1-1e-6]')
    return q


def mean(x):
    a = vector(x)
    if np.all(a == a[0]):
        return float(a[0])
    return math.fsum(float(v) for v in a) / len(a)


def sample_sd(x):
    a = vector(x, minimum=2)
    if np.all(a == a[0]):
        return 0.0
    center = math.fsum(float(v) for v in a) / len(a)
    deviations = [float(v)-center for v in a]
    require(all(d == 0 or abs(d) >= MIN for d in deviations), 'centered variation below supported range; rescale explicitly')
    squares = [d*d for d in deviations]
    require(all(d == 0 or square > 0 for d,square in zip(deviations,squares)), 'nonzero squared deviation underflowed')
    return math.sqrt(math.fsum(squares)/(len(a)-1))


def weighted_quantile(values, weights, q):
    """Inverse weighted empirical CDF; q=0 is minimum POSITIVE-mass support."""
    x = vector(values, 'values')
    w = vector(weights, 'weights')
    q = probability(q)
    require(len(x) == len(w), 'values and weights must have equal lengths')
    require(np.all(w >= 0), 'weights must be nonnegative')
    total = math.fsum(float(v) for v in w)
    require(total > 0, 'at least one weight must be positive')
    # All entries, including zero-mass values, were checked above.
    order = sorted(range(len(x)), key=lambda i: x[i])
    positive = [i for i in order if w[i] > 0]
    if q == 0:
        return float(x[positive[0]])
    exact_weights = {i: Fraction.from_float(float(w[i])) for i in positive}
    threshold = Fraction(str(q)) * sum(exact_weights.values(), Fraction(0))
    cumulative = Fraction(0)
    for i in positive:
        cumulative += exact_weights[i]
        if cumulative >= threshold:
            return float(x[i])
    return float(x[positive[-1]])  # q=1 roundoff safeguard


def empirical_quantile(values, q):
    x = vector(values, 'values')
    q = probability(q)
    # Exact rank convention: left inverse F_B; never linear interpolation.
    ordered = np.sort(x)
    rank = Fraction(str(q)) * len(x)
    index = max(0, math.ceil(rank)-1)
    return float(ordered[index])


def percentile_interval(replicates, level=.95):
    x = vector(replicates, 'replicates', 2)
    level = probability(level, 'level', True)
    tail = (Fraction(1)-Fraction(str(level)))/2
    return [empirical_quantile(x, float(tail)), empirical_quantile(x, float(1-tail))]


def known_sigma_interval(x, sigma, level=.95):
    a = vector(x)
    sigma = real(sigma, 'sigma', 1e-6, 1000)
    level = probability(level, 'level', True)
    z = NormalDist().inv_cdf((1+level)/2)
    center = mean(a)
    half = z*sigma/math.sqrt(len(a))
    return [center-half, center+half]


def wald_interval(k, n, level=.95):
    n = integer(n, 'n', 1, 10000)
    k = integer(k, 'k', 0, n)
    level = probability(level, 'level', True)
    p = k/n
    half = NormalDist().inv_cdf((1+level)/2)*math.sqrt(p*(1-p)/n)
    return [p-half, p+half]  # deliberately NOT clipped; lesson discusses failure


def _binomial_terms(n, p):
    # n,p have already been validated. True endpoint zero masses are allowed.
    if p == 0:
        return [1.0]+[0.0]*n
    if p == 1:
        return [0.0]*n+[1.0]
    terms=[]
    for k in range(n+1):
        left=p**k;right=(1-p)**(n-k)
        require(left > 0 and right > 0, 'positive binomial factor underflow; use a log-probability implementation')
        term=math.comb(n,k)*left*right
        require(term > 0 and math.isfinite(term), 'positive binomial mass underflow or overflow')
        terms.append(term)
    return terms


def exact_wald_coverage(p, n, level=.95):
    p = probability(p, 'p')
    n = integer(n, 'n', 1, 100)
    level = probability(level, 'level', True)
    masses = _binomial_terms(n,p)  # check even counts whose interval will not cover
    terms = []
    for k in range(n+1):
        lo, hi = wald_interval(k, n, level)
        if lo <= p <= hi:
            terms.append(masses[k])
    return math.fsum(terms)


def beta_cdf_integer(x, a, b):
    """Integer-shape Beta CDF using a positive binomial-tail identity."""
    x = probability(x, 'x')
    a = integer(a, 'a', 1, 40)
    b = integer(b, 'b', 1, 40)
    m = a+b-1
    terms = _binomial_terms(m,x)  # evaluate/check all terms, including outside the tail
    value = min(1.0, math.fsum(terms[a:]))
    require(x == 0 or value > 0, 'Beta CDF underflow; use a log-tail implementation')
    return value


def beta_quantile_integer(q, a, b):
    q = probability(q, 'q', True)
    a = integer(a, 'a', 1, 40)
    b = integer(b, 'b', 1, 40)
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo+hi)/2
        if beta_cdf_integer(mid, a, b) < q:
            lo = mid
        else:
            hi = mid
    return (lo+hi)/2


def exact_bootstrap(x):
    """Enumerate n**n ordered index samples, n<=6; exact Fraction arithmetic."""
    from itertools import product
    a = vector(x, minimum=1)
    require(len(a) <= 6, 'exact enumeration supports n<=6 only')
    # Fractions represent actual binary floats exactly (integers remain exact).
    fractions = [Fraction.from_float(float(v)) for v in a]
    counts = Counter(sum((fractions[i] for i in idx), Fraction(0))/len(a)
                     for idx in product(range(len(a)), repeat=len(a)))
    return [{'mean': float(v), 'exact_mean': str(v), 'count': counts[v],
             'total': len(a)**len(a)} for v in sorted(counts)]


def options(B, seed, count):
    B = integer(B, 'B', 2, 10000)
    seed = integer(seed, 'seed', 0, 2**32-1)
    require(B*count <= 2000000, 'resampling budget exceeds 2 million cells')
    return B, seed


def _row_means(rows):
    # Inputs are already validated; compensate each original-value sum.
    return np.array([float(row[0]) if np.all(row == row[0]) else
                     math.fsum(float(v) for v in row)/len(row) for row in rows])


def bootstrap_mean(x, B=999, seed=25, method='nonparametric', sigma=2.0):
    a = vector(x, minimum=2)
    B, seed = options(B, seed, len(a))
    require(isinstance(method, str) and method in ('nonparametric','parametric_known_sigma'), 'unsupported bootstrap method')
    # Validate even when sigma is unselected by nonparametric branch.
    sigma = real(sigma, 'sigma', 1e-6, 1000)
    center = mean(a)
    rng = np.random.Generator(np.random.PCG64(seed))
    if method == 'nonparametric':
        if np.all(a == a[0]):
            return np.full(B, float(a[0]))
        return _row_means(a[rng.integers(0,len(a),size=(B,len(a)))])
    # Equivalent to drawing n iid normals per resample and averaging.
    return center + sigma/math.sqrt(len(a))*rng.standard_normal(B)


def paired_bootstrap(before, after, B=999, seed=25, mode='paired'):
    x = vector(before, 'before', 2)
    y = vector(after, 'after', 2)
    B, seed = options(B, seed, len(x)+len(y))
    require(len(x) == len(y), 'paired observations require equal lengths')
    require(isinstance(mode, str) and mode in ('paired','independent'), 'mode must be paired or independent')
    differences = vector((y-x).tolist(), 'differences', 2)
    rng = np.random.Generator(np.random.PCG64(seed))
    if mode == 'paired':
        if np.all(differences == differences[0]):
            return np.full(B,float(differences[0]))
        idx = rng.integers(0,len(x),size=(B,len(x)))
        return _row_means(differences[idx])
    idx_x = rng.integers(0,len(x),size=(B,len(x)))
    idx_y = rng.integers(0,len(y),size=(B,len(y)))
    return _row_means(y[idx_y])-_row_means(x[idx_x])


def balanced_groups(groups):
    require(isinstance(groups, (list,tuple,np.ndarray)), 'groups must be a sequence of groups')
    if isinstance(groups,np.ndarray):
        require(groups.ndim == 2, 'group array must have shape (G,m)')
    require(2 <= len(groups) <= 100, 'G must lie in [2,100]')
    rows = [vector(row, f'group[{i}]', 1) for i,row in enumerate(groups)]
    require(all(len(row) == len(rows[0]) for row in rows), 'comparison requires equal group sizes; specify a new estimand for unequal groups')
    require(len(rows[0]) <= 100, 'm must be at most 100')
    return np.stack(rows)


def group_bootstrap(groups, B=999, seed=25, mode='group'):
    a = balanced_groups(groups)
    B, seed = options(B, seed, a.size)
    require(isinstance(mode,str) and mode in ('group','row'), 'mode must be group or row')
    rng = np.random.Generator(np.random.PCG64(seed))
    if np.all(a == a.flat[0]):
        return np.full(B,float(a.flat[0]))
    if mode == 'group':
        centers = _row_means(a)
        return _row_means(centers[rng.integers(0,len(a),size=(B,len(a)))])
    flat = a.ravel()
    return _row_means(flat[rng.integers(0,len(flat),size=(B,len(flat)))])


def coverage_summary(intervals, truth):
    require(isinstance(intervals,(list,tuple,np.ndarray)), 'intervals must be a sequence')
    require(2 <= len(intervals) <= 10000, 'number of intervals outside range')
    rows = [vector(row, f'interval[{i}]',2) for i,row in enumerate(intervals)]
    require(all(len(row)==2 for row in rows), 'each interval needs two endpoints')
    require(all(row[0] <= row[1] for row in rows), 'interval endpoints reversed')
    truth = real(truth,'truth')
    a = np.stack(rows)
    hits = (a[:,0] <= truth) & (truth <= a[:,1])
    rate = float(hits.mean()); R=len(hits)
    se = math.sqrt(rate*(1-rate)/R)
    # Wilson score interval for the Monte Carlo Bernoulli probability only.
    den=1+Z95**2/R
    center=(rate+Z95**2/(2*R))/den
    half=Z95/den*math.sqrt(rate*(1-rate)/R+Z95**2/(4*R*R))
    return {'repetitions':R,'hits':int(hits.sum()),'coverage':rate,
            'mc_se':se,'mc_wilson95':[center-half,center+half],
            'mean_width':float(np.mean(a[:,1]-a[:,0]))}


SPEC_KEYS={'kind','seed','normal_repetitions','group_repetitions','B','n','G','m','mu','sigma','tau'}

def validate_spec(spec):
    require(isinstance(spec,dict) and set(spec)==SPEC_KEYS, 'spec keys must match the documented schema exactly')
    require(spec['kind']=='synthetic_interval_coverage', 'wrong simulation kind')
    out={'kind':spec['kind']}
    for key,low,high in [('seed',0,2**32-1),('normal_repetitions',20,5000),('group_repetitions',20,2000),('B',99,1999),('n',2,100),('G',3,100),('m',1,30)]:
        out[key]=integer(spec[key],key,low,high)
    for key,low,high in [('mu',-100,100),('sigma',.01,10),('tau',0,10)]:
        out[key]=real(spec[key],key,low,high)
    cost=out['normal_repetitions']*out['B']*out['n'] + out['group_repetitions']*out['B']*out['G']*out['m']
    require(cost <= 120000000, 'total simulation budget exceeds 120 million cells')
    return out


def unique_json_object(pairs):
    out = {}
    for key, value in pairs:
        require(key not in out, 'duplicate JSON key: '+key)
        out[key] = value
    return out


def invalid_json_constant(token):
    raise ValueError('nonfinite JSON constant: '+token)


def load_spec(path=None):
    # Inspect decimal tokens before binary64 conversion can erase nonzero data.
    raw = json.loads(Path(path or ROOT/'data/model_spec.json').read_text(),
                     parse_float=lambda token: csv_number(token, 'JSON number'),
                     parse_constant=invalid_json_constant,
                     object_pairs_hook=unique_json_object)
    return validate_spec(raw)


def simulation(spec):
    s=validate_spec(spec)  # complete validation before first RNG
    seed=s['seed']; R=s['normal_repetitions']; n=s['n']; B=s['B']
    # Separate streams keep outer experiments independent from inner resampling.
    streams=np.random.SeedSequence(seed).spawn(4)
    rnormal,rboot,rgroup,rgroupboot=[np.random.Generator(np.random.PCG64(v)) for v in streams]
    normal={k:[] for k in ['known_sigma','nonparametric_percentile','parametric_known_sigma_percentile']}
    for _ in range(R):
        x=rnormal.normal(s['mu'],s['sigma'],n)
        center=float(x.mean())
        normal['known_sigma'].append([center-Z95*s['sigma']/math.sqrt(n),center+Z95*s['sigma']/math.sqrt(n)])
        vals=x[rboot.integers(0,n,size=(B,n))].mean(axis=1)
        normal['nonparametric_percentile'].append(np.quantile(vals,[.025,.975],method='inverted_cdf').tolist())
        vals=center+s['sigma']/math.sqrt(n)*rboot.standard_normal(B)
        normal['parametric_known_sigma_percentile'].append(np.quantile(vals,[.025,.975],method='inverted_cdf').tolist())
    clustered={k:[] for k in ['row_percentile','group_percentile']}; G=s['G']; m=s['m']
    for _ in range(s['group_repetitions']):
        x=s['mu']+rgroup.normal(0,s['tau'],(G,1))+rgroup.normal(0,s['sigma'],(G,m))
        flat=x.ravel(); centers=x.mean(axis=1)
        row=flat[rgroupboot.integers(0,G*m,size=(B,G*m))].mean(axis=1)
        group=centers[rgroupboot.integers(0,G,size=(B,G))].mean(axis=1)
        clustered['row_percentile'].append(np.quantile(row,[.025,.975],method='inverted_cdf').tolist())
        clustered['group_percentile'].append(np.quantile(group,[.025,.975],method='inverted_cdf').tolist())
    return {'spec':s,'normal':{k:coverage_summary(v,s['mu']) for k,v in normal.items()},
            'group':{k:coverage_summary(v,s['mu']) for k,v in clustered.items()},
            'normal_intervals':normal,'group_intervals':clustered,
            'group_theory':{'true_variance':s['tau']**2/G+s['sigma']**2/(G*m),
                            'iid_variance':(s['tau']**2+s['sigma']**2)/(G*m)}}


NUMBER=re.compile(r'^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$')

def csv_number(text, name):
    require(isinstance(text,str) and NUMBER.fullmatch(text) is not None, name+' must be a decimal number token')
    value=float(text)
    # The explicit text file parser is distinct from numerical API coercion.
    if value == 0:
        mantissa=text.lower().split('e')[0]
        require(not any(c in '123456789' for c in mantissa), name+' underflowed to zero')
    return real(value,name)


def load_csv(path):
    with Path(path).open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f)
        require(reader.fieldnames==['id','value'], 'CSV header must be id,value')
        rows=list(reader)
    require(2 <= len(rows) <= 10000, 'CSV requires 2..10000 rows')
    ids=[]; vals=[]
    for row in rows:
        require(set(row)=={'id','value'} and all(v is not None for v in row.values()), 'ragged CSV row')
        require(isinstance(row['id'],str) and re.fullmatch(r'[A-Za-z0-9_-]{1,32}',row['id']) is not None, 'invalid CSV id')
        ids.append(row['id']); vals.append(csv_number(row['value'],'CSV value'))
    require(len(set(ids))==len(ids),'duplicate CSV id')
    return vector(vals)


def run(input_path=None, spec_path=None, output_dir=None):
    target=Path(output_dir or ROOT/'outputs')
    require(not target.exists() or target.is_dir(),'output path must be a directory')
    x=load_csv(input_path or ROOT/'data/tiny_sample.csv')
    s=load_spec(spec_path)
    # Nothing is written or randomly generated until ALL input files validate.
    exact=exact_bootstrap(x)  # n>6 rejected before simulations or output mutation
    result=simulation(s)
    result['tiny_exact']=exact
    result['wald_n20_p002']=exact_wald_coverage(.02,20)
    result['beta_5_7_credible95']=[beta_quantile_integer(.025,5,7),beta_quantile_integer(.975,5,7)]
    result['known_sigma_hand']=known_sigma_interval([9,10,11,10],2)
    target.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)
    temp=target/'report.json.tmp'; temp.write_text(payload); temp.replace(target/'report.json')
    return result


def self_test():
    require(sample_sd([.1,.1,.1])==0,'constant sd must be exact zero')
    require(mean([0,2])==1,'mean hand check')
    exact=exact_bootstrap([0,2]);require([r['count'] for r in exact]==[1,2,1],'exact multiplicities')
    require(percentile_interval([0,1,1,2])==[0,2],'empirical inverse CDF')
    require(weighted_quantile([0,1,2],[1,2,1],.5)==1,'weighted quantile')
    require(known_sigma_interval([9,10,11,10],2)==[10-Z95,10+Z95],'known sigma hand check')
    require(wald_interval(0,20)==[0,0],'Wald boundary')
    require(np.all(paired_bootstrap([1,2,4,8],[2,3,5,9])==1),'constant paired effect')
    require(abs(beta_cdf_integer(beta_quantile_integer(.025,5,7),5,7)-.025)<1e-13,'Beta inverse check')
    return {'status':'passed','checks':9}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path);parser.add_argument('--spec',type=Path)
    parser.add_argument('--output-dir',type=Path);parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test()));return
    r=run(args.input,args.spec,args.output_dir)
    print(json.dumps({k:r[k] for k in ['normal','group','group_theory','wald_n20_p002','beta_5_7_credible95']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
