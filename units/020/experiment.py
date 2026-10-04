"""Unit 020. Small offline distribution experiments with explicit contracts.

Numerical API is deliberately narrower than the mathematical families. Nonzero
subnormal values and meaningful underflow raise ValueError; log probabilities
can remain usable when ordinary probabilities cannot. No validation uses assert.
"""
from __future__ import annotations
import argparse
import csv
from decimal import Decimal, localcontext
from fractions import Fraction
import json
import math
from numbers import Integral, Real
from pathlib import Path
import platform
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
TINY = sys.float_info.min
MAX_SIZE = 200000
MAX_N = 10000
SEEDS = {"bernoulli": 20001, "binomial": 20002, "categorical": 20003,
         "poisson": 20004, "gaussian": 20005, "mixture": 20006}
REPEAT_SEED_BASE = 202000


def require(condition, message):
    if not condition:
        raise ValueError(message)


def real(value, name="value"):
    require(not isinstance(value, (bool, np.bool_)) and isinstance(value, Real),
            name + " must be a real scalar, not bool/text/complex/array")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(name + " cannot be represented") from exc
    require(math.isfinite(result), name + " must be finite")
    require(result != 0 or value == 0, name + " conversion underflow")
    require(result == 0 or abs(result) >= TINY, name + " subnormal unsupported")
    return result


def integer(value, name, low, high):
    require(not isinstance(value, (bool, np.bool_)) and isinstance(value, Integral),
            name + " must be an integer scalar")
    result = int(value)
    require(low <= result <= high, name + " outside implementation range")
    return result


def checked(value, name):
    return real(value, name)


def multiply(a, b, name):
    result = checked(a * b, name)
    require(result != 0 or a == 0 or b == 0, name + " product underflow")
    return result


def divide(a, b, name):
    require(b != 0, name + " zero denominator")
    result = checked(a / b, name)
    require(result != 0 or a == 0, name + " division underflow")
    return result


def total(values, name):
    try:
        result = math.fsum(values)
    except (OverflowError, ValueError) as exc:
        raise ValueError(name + " sum overflow") from exc
    return checked(result, name)


def probability(p):
    p = real(p, "p")
    require(0 <= p <= 1, "p must be in [0,1]")
    return p


def probabilities(values):
    require(isinstance(values, (list, tuple, np.ndarray)), "probabilities need a 1-D list")
    if isinstance(values, np.ndarray):
        require(values.ndim == 1 and values.dtype.kind in "iuf", "probability shape/dtype invalid")
    require(1 <= len(values) <= 10000, "invalid probability count")
    ps = [probability(v) for v in values]
    require(total(ps, "probability sum") == 1.0,
            "probabilities must fsum to exactly 1.0; no silent renormalization")
    return ps


def vector(values, name="values"):
    require(isinstance(values, (list, tuple, np.ndarray)), name + " must be a 1-D sequence")
    if isinstance(values, np.ndarray):
        require(values.ndim == 1 and values.dtype.kind in "iuf", name + " shape/dtype invalid")
    require(1 <= len(values) <= MAX_SIZE, name + " length invalid")
    return [real(v, name) for v in values]


def mass_from_log(logp):
    if logp == -math.inf:  # Deliberate support-zero only, internal callers control this.
        return 0.0
    logp = real(logp, "log probability")
    require(logp <= 1e-10, "log probability unexpectedly positive")
    require(logp >= math.log(TINY), "positive probability underflows: use logpmf")
    result = checked(math.exp(logp), "probability")
    require(0 < result <= 1.0 + 1e-10, "invalid probability result")
    return result


def bernoulli_pmf(y, p):
    p = probability(p)
    y = integer(y, "y", -MAX_N, MAX_N)
    return p if y == 1 else (1.0 - p if y == 0 else 0.0)


def binomial_logpmf(k, n, p):
    n = integer(n, "n", 0, MAX_N)
    k = integer(k, "k", -MAX_N, MAX_N)
    p = probability(p)
    if not 0 <= k <= n:
        return -math.inf
    if p == 0 or p == 1:
        return 0.0 if k == (n if p == 1 else 0) else -math.inf
    # math.comb gives exact integer combinations before logarithm.
    result = total([math.log(math.comb(n, k)),
                    multiply(float(k), math.log(p), "k log p"),
                    multiply(float(n-k), math.log1p(-p), "(n-k) log(1-p)")], "binomial logpmf")
    require(result <= 1e-10, "binomial logpmf positive")
    return result


def binomial_pmf(k, n, p):
    return mass_from_log(binomial_logpmf(k, n, p))


def poisson_logpmf(k, lam):
    k = integer(k, "k", -MAX_N, 1000000)
    lam = real(lam, "lambda")
    require(0 <= lam <= 1000000, "lambda outside [0,1e6] implementation range")
    if k < 0:
        return -math.inf
    if lam == 0:
        return 0.0 if k == 0 else -math.inf
    return total([-lam, multiply(float(k), math.log(lam), "k log lambda"),
                  -math.lgamma(k + 1)], "Poisson logpmf")


def poisson_pmf(k, lam):
    return mass_from_log(poisson_logpmf(k, lam))


def normal_logpdf(x, mu, variance):
    x, mu, v = real(x, "x"), real(mu, "mu"), real(variance, "variance")
    require(v > 0, "variance must be strictly positive")
    delta = checked(x - mu, "x-mu")
    z = divide(delta, math.sqrt(v), "standardized displacement")
    z2 = multiply(z, z, "standardized square")
    # Avoid overflowing 2*pi*v or underflowing the normalizing factor.
    result = total([-0.5 * math.log(2 * math.pi), -0.5 * math.log(v), -0.5 * z2], "normal logpdf")
    return result


def normal_pdf(x, mu, variance):
    logp = normal_logpdf(x, mu, variance)
    require(math.log(TINY) <= logp <= math.log(sys.float_info.max),
            "density outside ordinary float range: use logpdf")
    return checked(math.exp(logp), "normal density")


def finite_expectation(values, weights, callback):
    # Validate ALL values and probabilities before the first user callback.
    xs, ps = vector(values), probabilities(weights)
    require(len(xs) == len(ps), "support and probabilities length mismatch")
    require(callable(callback), "callback must be callable")
    terms = []
    for x, p in zip(xs, ps):
        y = real(callback(x), "callback result")
        terms.append(multiply(p, y, "weighted callback"))
    return total(terms, "expectation")


def moments(values):
    xs = vector(values)
    ref = xs[0]
    ds = [checked(x-ref, "centered displacement") for x in xs]
    # Divide before summing to avoid needless overflow for large constant data.
    shift = total([divide(d, len(xs), "mean term") for d in ds], "mean displacement")
    mean = checked(ref + shift, "sample mean")
    sq = []
    for d in ds:
        r = checked(d-shift, "centered residual")
        sq.append(divide(multiply(r, r, "squared residual"), len(xs), "variance term"))
    var = total(sq, "empirical variance")
    require(var >= 0, "variance negative")
    return {"n": len(xs), "mean": mean, "variance_n": var}


def categorical_sample(ps, size, seed):
    ps = probabilities(ps)
    size = integer(size, "size", 1, MAX_SIZE)
    seed = integer(seed, "seed", 0, 2**63-1)
    cumulative = [total(ps[:i+1], "cumulative mass") for i in range(len(ps))]
    for p, left, right in zip(ps, [0.0] + cumulative[:-1], cumulative):
        require(p == 0 or right > left, "positive mass lost in cumulative representation")
    require(cumulative[-1] == 1, "cumulative total invalid")
    u = np.random.Generator(np.random.PCG64(seed)).random(size)
    result = np.searchsorted(cumulative, u, side="right")
    require(result.shape == (size,) and np.all((result >= 0) & (result < len(ps))), "bad categorical output")
    return result


def sample(family, size, seed, *, p=None, n=None, lam=None, mu=None, variance=None):
    require(isinstance(family, str) and family in {"bernoulli", "binomial", "poisson", "gaussian"}, "unknown family")
    size = integer(size, "size", 1, MAX_SIZE)
    seed = integer(seed, "seed", 0, 2**63-1)
    args = {"p":p, "n":n, "lam":lam, "mu":mu, "variance":variance}
    expected = {"bernoulli":{"p"}, "binomial":{"n","p"}, "poisson":{"lam"}, "gaussian":{"mu","variance"}}[family]
    require(all((value is not None) == (key in expected) for key,value in args.items()), "missing or extraneous parameters")
    # Complete validation before constructing or using the generator.
    if family in {"bernoulli", "binomial"}:
        p = probability(p)
        n = 1 if family == "bernoulli" else integer(n, "n", 0, MAX_N)
    elif family == "poisson":
        lam = real(lam, "lambda")
        require(0 <= lam <= 1000000, "lambda implementation range [0,1e6]")
    else:
        mu, variance = real(mu, "mu"), real(variance, "variance")
        require(variance > 0, "variance must be positive")
    rng = np.random.Generator(np.random.PCG64(seed))
    if family in {"bernoulli", "binomial"}:
        result = rng.binomial(n, p, size)
        require(np.all((result >= 0) & (result <= n)), "binomial support violation")
    elif family == "poisson":
        result = rng.poisson(lam, size)
        require(np.all(result >= 0), "Poisson support violation")
    else:
        z = rng.standard_normal(size)
        # Every intermediate checked; NumPy broadcasting not part of this API.
        result_list = []
        for zi in z:
            delta = multiply(math.sqrt(variance), real(zi), "Gaussian displacement")
            value = checked(mu + delta, "Gaussian sample")
            require(delta == 0 or value != mu, "Gaussian displacement swallowed by addition")
            result_list.append(value)
        result = np.array(result_list, dtype=float)
    require(result.shape == (size,) and np.all(np.isfinite(result)), "generated output invalid")
    return result


def generate_data(size=12000):
    size = integer(size, "size", 1, MAX_SIZE)
    data = {
        "bernoulli":sample("bernoulli",size,SEEDS["bernoulli"],p=.25),
        "binomial":sample("binomial",size,SEEDS["binomial"],n=8,p=.25),
        "categorical":categorical_sample([.5,.3,.2],size,SEEDS["categorical"]),
        "poisson":sample("poisson",size,SEEDS["poisson"],lam=4),
        "gaussian":sample("gaussian",size,SEEDS["gaussian"],mu=1,variance=4),
    }
    groups = categorical_sample([.5,.5], size, SEEDS["mixture"])
    low = sample("poisson", size, SEEDS["mixture"]+100, lam=1)
    high = sample("poisson", size, SEEDS["mixture"]+200, lam=7)
    data["mixture"] = np.where(groups == 0, low, high)
    data["mixture_group"] = groups
    return data


def exact_checks():
    checks = 0
    max_binomial_error = 0.0
    for n in range(0,17):
        for p in [Fraction(0),Fraction(1,10),Fraction(1,4),Fraction(1,2),Fraction(9,10),Fraction(1)]:
            oracle = [Fraction(math.comb(n,k)) * p**k * (1-p)**(n-k) for k in range(n+1)]
            require(sum(oracle) == 1, "rational normalization failed")
            mu = sum(Fraction(k)*v for k,v in enumerate(oracle))
            var = sum((Fraction(k)-mu)**2*v for k,v in enumerate(oracle))
            require(mu == n*p and var == n*p*(1-p), "rational moments failed")
            checks += 3
            for k,v in enumerate(oracle):
                actual = binomial_pmf(k,n,float(p))
                error = abs(actual-float(v))
                max_binomial_error = max(max_binomial_error,error)
                require(math.isclose(actual,float(v),rel_tol=2e-13,abs_tol=3e-15), "binomial oracle mismatch")
                checks += 1
    max_poisson_error = 0.0
    with localcontext() as ctx:
        ctx.prec = 80
        for lam in [Decimal('0.1'),Decimal('0.5'),Decimal(1),Decimal(4),Decimal(7),Decimal(20)]:
            for k in range(0,51):
                expected = (-lam).exp()*lam**k/Decimal(math.factorial(k))
                value = poisson_pmf(k,float(lam))
                err = abs(value-float(expected)); max_poisson_error=max(max_poisson_error,err)
                require(math.isclose(value,float(expected),rel_tol=8e-14,abs_tol=1e-16), "Poisson decimal oracle mismatch")
                checks += 1
    # Discrete normal quadrature: a numerical diagnostic, not a proof.
    grid = np.linspace(-9,9,18001)
    density = np.array([normal_pdf(float(x),0,1) for x in grid])
    quadrature = {"mass":float(np.trapezoid(density,grid)),
                  "mean":float(np.trapezoid(grid*density,grid)),
                  "second_moment":float(np.trapezoid(grid*grid*density,grid))}
    require(abs(quadrature["mass"]-1) < 1e-12 and abs(quadrature["mean"]) < 1e-12
            and abs(quadrature["second_moment"]-1) < 1e-12, "Gaussian grid check")
    checks += 3
    return {"checks":checks,"max_binomial_abs_error":max_binomial_error,
            "max_poisson_abs_error":max_poisson_error,"normal_quadrature":quadrature}


def adversarial_checks():
    failures = []
    def reject(label, call):
        try:
            call()
        except (ValueError, TypeError, OverflowError):
            failures.append(label)
        else:
            raise ValueError("expected rejection: " + label)
    for label,bad in [("bool",True),("numpy_bool",np.bool_(False)),("text","0.5"),
                      ("complex",.5+0j),("nan",float('nan')),("inf",float('inf')),
                      ("array",np.array([.5])),("subnormal",1e-320)]:
        reject("p_"+label,lambda bad=bad: bernoulli_pmf(1,bad))
    for bad in [-.1,1.1]: reject("p_range_"+str(bad),lambda bad=bad:bernoulli_pmf(1,bad))
    for bad in [True,4.0,"4",-1,10001]: reject("n_"+str(bad),lambda bad=bad:binomial_pmf(1,bad,.5))
    for bad in [True,.5,complex(1),float('nan')]: reject("k_"+str(bad),lambda bad=bad:poisson_pmf(bad,2))
    for bad in [-1,1000001]: reject("lambda_"+str(bad),lambda bad=bad:poisson_pmf(1,bad))
    for bad in [0,-1,True,"1",float('nan'),1e-320]: reject("variance_"+str(bad),lambda bad=bad:normal_pdf(0,0,bad))
    reject("binomial_underflow",lambda:binomial_pmf(10000,10000,.5))
    reject("Poisson_underflow",lambda:poisson_pmf(0,1000))
    reject("normal_underflow",lambda:normal_pdf(100,0,1))
    reject("normal_difference_overflow",lambda:normal_logpdf(1e308,-1e308,1))
    reject("normal_square_overflow",lambda:normal_logpdf(1e200,0,1))
    reject("normal_square_underflow",lambda:normal_logpdf(1e-200,0,1))
    for label,ps in [("negative",[1.1,-.1]),("sum",[.2,.7]),("near_sum",[.2,.8000000000001]),
                     ("nested",[[.5],[.5]]),("mixed_bool",[True,0]),("empty",[]),
                     ("object",np.array([.5,.5],dtype=object))]:
        reject("probabilities_"+label,lambda ps=ps:probabilities(ps))
    reject("lost_cumulative_mass",lambda:categorical_sample([1,1e-20],5,1))
    calls=[]
    def spy(x): calls.append(x); return x
    reject("callback_late_invalid",lambda:finite_expectation([0,float('inf')],[.5,.5],spy))
    reject("callback_probability_invalid",lambda:finite_expectation([0,1],[.5,.6],spy))
    reject("callback_shape_invalid",lambda:finite_expectation([0,1],[1],spy))
    require(not calls,"callback ran before complete validation")
    for label,value in [("bool",True),("string","1"),("complex",1j),("nan",float('nan')),("array",np.array([1]))]:
        reject("callback_result_"+label,lambda value=value:finite_expectation([1],[1],lambda x:value))
    reject("callback_product_underflow",lambda:finite_expectation([0,1],[1,1e-200],lambda x:1e-200))
    reject("callback_sum_overflow",lambda:finite_expectation([0,1],[.5,.5],lambda x:1.7e308 if x==0 else 1.7e308*2))
    reject("moments_overflow",lambda:moments([-1e308,1e308]))
    reject("moments_variance_underflow",lambda:moments([0,1e-200]))
    reject("moments_mixed_bool",lambda:moments([1,True]))
    reject("moments_2d",lambda:moments(np.ones((2,2))))
    for label,call in [
        ("sample_size_zero",lambda:sample("poisson",0,1,lam=2)),
        ("sample_size_bool",lambda:sample("poisson",True,1,lam=2)),
        ("sample_seed_bool",lambda:sample("poisson",2,True,lam=2)),
        ("sample_unknown",lambda:sample("bad",2,1,lam=2)),
        ("sample_extra",lambda:sample("poisson",2,1,lam=2,p=.5)),
        ("sample_missing",lambda:sample("gaussian",2,1,mu=0)),
        ("sample_lost_shift",lambda:sample("gaussian",5,1,mu=1e20,variance=1))]:
        reject(label,call)
    accepted = 0
    for condition in [bernoulli_pmf(0,0)==1,bernoulli_pmf(1,1)==1,
                      binomial_pmf(0,0,.25)==1,binomial_pmf(8,8,1)==1,
                      binomial_pmf(9,8,.25)==0,poisson_pmf(-1,2)==0,
                      poisson_pmf(0,0)==1,poisson_pmf(1,0)==0,
                      math.isfinite(poisson_logpmf(0,1000)),
                      np.all(categorical_sample([0,1,0],10,1)==1),
                      moments([1e200,1e200])["variance_n"]==0,
                      finite_expectation([0,2],[.5,.5],lambda x:x)==1]:
        require(condition,"accepted boundary failed"); accepted += 1
    return {"rejections":len(failures),"accepted":accepted,"total":len(failures)+accepted,
            "rejection_labels":failures,"callback_calls_on_invalid_inputs":len(calls)}


def run_experiment():
    data = generate_data()
    summary = {name:moments(data[name]) for name in ["bernoulli","binomial","poisson","gaussian","mixture"]}
    counts = np.bincount(data["categorical"],minlength=3)
    summary["categorical"] = {"n":len(data["categorical"]),"counts":counts.tolist(),
        "frequencies":(counts/len(data["categorical"])).tolist(),
        "indicator_variances_n":[moments((data["categorical"]==j).astype(int))["variance_n"] for j in range(3)]}
    repetitions=[]
    for size in [40,400,4000]:
        for repeat in range(40):
            seed = REPEAT_SEED_BASE+size*100+repeat
            row=moments(sample("poisson",size,seed,lam=4))
            repetitions.append({"size":size,"repeat":repeat,"seed":seed,**row})
    return data, {"unit":"020","python":platform.python_version(),"numpy":np.__version__,
        "generator":"numpy.random.Generator(PCG64)","seeds":SEEDS,
        "mixture_component_seeds":[SEEDS['mixture']+100,SEEDS['mixture']+200],
        "sample_size":12000,"variance_denominator":"n","summary":summary,
        "theory":{"bernoulli":[.25,.1875],"binomial":[2,1.5],"poisson":[4,4],"gaussian":[1,4],"mixture":[4,13]},
        "repetitions":repetitions,"exact_checks":exact_checks(),"boundary_checks":adversarial_checks(),
        "limits":["finite synthetic samples are not a statistical proof", "browser UI, socket transport and learner Anaconda not tested"]}


def write_outputs(data, report, out_dir):
    out = Path(out_dir)
    # Nothing is written until all computations and checks have succeeded.
    out.mkdir(parents=True,exist_ok=True)
    (out/"experiment_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    with (out/"synthetic_samples.csv").open("w",newline="") as f:
        writer=csv.writer(f); names=list(data);writer.writerow(names)
        for row in zip(*(data[n] for n in names)):
            writer.writerow([format(float(v),'.17g') if name=='gaussian' else int(v) for name,v in zip(names,row)])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=ROOT/"outputs")
    args=parser.parse_args()
    data,report=run_experiment();write_outputs(data,report,args.out)
    print(json.dumps({"status":"passed","unit":"020","summary":report['summary'],
                      "exact_checks":report['exact_checks'],"boundary_total":report['boundary_checks']['total']},ensure_ascii=False,sort_keys=True))

if __name__ == '__main__':
    main()
