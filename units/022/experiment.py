"""Offline sampling and limit-theorem experiments. Synthetic data only.

The exact checks use Fraction; Monte Carlo is a finite illustration, not proof.
All validation remains enabled under python -O. No network or paid service.
"""
from pathlib import Path
import argparse
import csv
from fractions import Fraction
import itertools
import json
import math
import numbers
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
MAX_ABS = 1e140
TINY = sys.float_info.min
FAMILIES = ("bernoulli", "exponential", "pareto_1_5", "cauchy")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, name, low=0, high=1_000_000):
    require(isinstance(value, numbers.Integral) and not isinstance(value, (bool, np.bool_)), name + " must be an integer, not bool")
    value = int(value)
    require(low <= value <= high, name + " outside documented range")
    return value


def real(value, name):
    require(isinstance(value, numbers.Real) and not isinstance(value, (bool, np.bool_)), name + " must be a real number, not bool/text/complex")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise ValueError(name + " outside float64 range") from None
    require(math.isfinite(result) and abs(result) <= MAX_ABS, name + " outside finite teaching range")
    require(value == 0 or result != 0, name + " float conversion underflows")
    require(result == 0 or abs(result) >= TINY, name + " subnormal input is unsupported")
    return result


def real_vector(values):
    if isinstance(values, np.ndarray):
        require(values.ndim == 1 and values.dtype.kind in "iuf", "expected one real numeric vector")
        source = values.tolist()
    else:
        require(isinstance(values, (list, tuple)), "expected list, tuple or ndarray")
        source = values
    require(1 <= len(source) <= 2_000_000, "vector length outside teaching range")
    return [real(x, "observation") for x in source]


def moments(values):
    """Return descriptive variance_n, unbiased-under-iid s2 and estimated SE.

    n=1 yields None for quantities requiring a second observation. Range checks
    reject unsafe squaring and do not turn a tiny nonzero square into zero.
    """
    x = real_vector(values)
    n = len(x)
    # Validate the entire vector first. An exactly constant float vector has zero
    # spread; fsum(x)/n can round away from x[0] and create false deviations.
    if all(v == x[0] for v in x):
        return {"n": n, "mean": x[0], "variance_n": 0.0,
                "s2": 0.0 if n > 1 else None,
                "s": 0.0 if n > 1 else None,
                "estimated_se_iid": 0.0 if n > 1 else None}
    mean = math.fsum(x) / n
    deviations = [v - mean for v in x]
    squares = [d * d for d in deviations]
    require(all(math.isfinite(q) for q in squares), "squared deviations overflow")
    require(all(d == 0 or q >= TINY for d, q in zip(deviations, squares)), "nonzero squared deviation underflows")
    ss = math.fsum(squares)
    variance_n = ss / n
    require(ss == 0 or variance_n >= TINY, "variance underflows")
    s2 = ss / (n - 1) if n > 1 else None
    se2 = s2 / n if n > 1 else None
    require(se2 is None or se2 == 0 or se2 >= TINY, "estimated SE squared underflows")
    return {"n": n, "mean": mean, "variance_n": variance_n,
            "s2": s2, "s": math.sqrt(s2) if s2 is not None else None,
            "estimated_se_iid": math.sqrt(se2) if se2 is not None else None}


def fraction_probability(numerator, denominator):
    a = integer(numerator, "numerator", 0, 1000)
    b = integer(denominator, "denominator", 1, 1000)
    require(a <= b, "probability exceeds one")
    return Fraction(a, b)


def binomial_exact(n, numerator=1, denominator=4):
    n = integer(n, "n", 0, 512)
    p = fraction_probability(numerator, denominator)
    return [Fraction(math.comb(n, k)) * p**k * (1 - p)**(n-k) for k in range(n+1)]


def normal_cdf(z):
    z = real(z, "z")
    return 0.5 * math.erfc(-z / math.sqrt(2))


def exact_mean_summary(n, numerator=1, denominator=4, epsilon=Fraction(1, 10)):
    n = integer(n, "n", 1, 512)
    p = fraction_probability(numerator, denominator)
    require(isinstance(epsilon, Fraction) and epsilon > 0, "epsilon must be a positive Fraction")
    pmf = binomial_exact(n, numerator, denominator)
    mean = sum((Fraction(k, n) * q for k, q in enumerate(pmf)), Fraction())
    variance = sum(((Fraction(k, n) - p)**2 * q for k, q in enumerate(pmf)), Fraction())
    tail = sum((q for k, q in enumerate(pmf) if abs(Fraction(k, n)-p) >= epsilon), Fraction())
    bound = min(Fraction(1), p*(1-p)/(n*epsilon**2))
    distance = None
    if 0 < p < 1:
        sd = math.sqrt(float(p*(1-p)/n))
        cdf = Fraction()
        errors = []
        for k, q in enumerate(pmf):
            phi = normal_cdf((float(Fraction(k,n)-p))/sd)
            errors.extend([abs(float(cdf)-phi), abs(float(cdf+q)-phi)])
            cdf += q
        distance = max(errors)
    return {"n": n, "p": str(p), "mean_exact": str(mean),
            "variance_exact": str(variance), "standard_error": math.sqrt(float(variance)),
            "tail_ge_epsilon": float(tail), "tail_exact": str(tail),
            "epsilon": str(epsilon), "chebyshev_bound": float(bound),
            "normal_cdf_max_jump_error": distance}


def mean_variance_equicorrelated(n, variance, rho):
    n = integer(n, "n", 2, 1_000_000)
    variance = real(variance, "variance")
    rho = real(rho, "rho")
    require(variance > 0, "variance must be positive")
    require(-1/(n-1) <= rho <= 1, "equicorrelation matrix is not PSD")
    design_effect = 1 + (n-1)*rho
    value = variance / n * design_effect
    require(math.isfinite(value), "mean variance overflow")
    require(design_effect == 0 or value >= TINY, "mean variance underflows")
    return {"variance_mean": value, "design_effect": design_effect,
            "effective_n_for_mean": n/design_effect if design_effect > 0 else None}


def finite_population_exact(values, n):
    require(isinstance(values, (list, tuple)), "finite population must be a list or tuple")
    require(2 <= len(values) <= 12, "finite population size must be 2..12")
    x = [Fraction(integer(v, "population value", -1000, 1000)) for v in values]
    n = integer(n, "sample size", 1, len(x))
    N = len(x)
    mu = sum(x)/N
    variance_N = sum((v-mu)**2 for v in x)/N
    means = [sum(x[i] for i in indices)/n for indices in itertools.combinations(range(N), n)]
    exact_var = sum((m-mu)**2 for m in means)/len(means)
    theory = variance_N/n * Fraction(N-n, N-1)
    require(exact_var == theory, "finite-population correction mismatch")
    return {"N": N, "n": n, "mean": str(mu), "variance_N": str(variance_N),
            "sample_mean_variance": str(exact_var), "sample_means": [str(m) for m in means]}


def validate_spec(spec):
    require(isinstance(spec, dict), "spec must be a JSON object")
    keys = {"kind", "seed", "repetitions", "sizes", "p_numerator", "p_denominator"}
    require(set(spec) == keys and spec["kind"] == "synthetic_sampling_demo", "unknown or missing specification fields")
    seed = integer(spec["seed"], "seed", 0, 2**32-10000)
    repeats = integer(spec["repetitions"], "repetitions", 2, 8000)
    require(isinstance(spec["sizes"], list) and 1 <= len(spec["sizes"]) <= 8, "sizes must contain 1..8 integers")
    sizes = [integer(n, "sample size", 2, 512) for n in spec["sizes"]]
    require(sizes == sorted(set(sizes)), "sample sizes must be distinct and increasing")
    require(max(sizes)*repeats <= 2_000_000, "simulation matrix exceeds two million cells")
    p = fraction_probability(spec["p_numerator"], spec["p_denominator"])
    require(0 < p < 1, "main CLT experiment requires positive variance: 0<p<1")
    return {"kind": spec["kind"], "seed": seed, "repetitions": repeats, "sizes": sizes,
            "p_numerator": p.numerator, "p_denominator": p.denominator}


def load_spec(path=ROOT / "data/model_spec.json"):
    return validate_spec(json.loads(Path(path).read_text(encoding="utf-8")))


def sample_means(spec):
    spec = validate_spec(spec)  # Complete validation before creating any RNG.
    out = {}
    p = spec["p_numerator"] / spec["p_denominator"]
    r = spec["repetitions"]
    for family_id, family in enumerate(FAMILIES):
        out[family] = {}
        for n in spec["sizes"]:
            rng = np.random.Generator(np.random.PCG64(spec["seed"] + 1000*family_id+n))
            shape = (r, n)  # each row is a new independent dataset
            if family == "bernoulli":
                x = rng.binomial(1, p, shape)
            elif family == "exponential":
                x = rng.exponential(1.0, shape)
            elif family == "pareto_1_5":
                x = 1 + rng.pareto(1.5, shape)  # Pareto I, support [1,infinity)
            else:
                x = rng.standard_cauchy(shape)
            require(np.all(np.isfinite(x)) and np.max(np.abs(x)) <= MAX_ABS, "simulation outside finite teaching range; nothing clipped")
            means = np.mean(x, axis=1, dtype=np.float64)
            require(np.all(np.isfinite(means)), "sample means nonfinite")
            out[family][n] = means
    return out


def cluster_experiment(spec):
    spec = validate_spec(spec)
    p = spec["p_numerator"] / spec["p_denominator"]
    r = spec["repetitions"]
    rng = np.random.Generator(np.random.PCG64(spec["seed"]+8000))
    groups = rng.binomial(1, p, (r, 8))
    repeated = np.repeat(groups, 8, axis=1)
    means = repeated.mean(axis=1)
    naive_se = repeated.std(axis=1, ddof=1)/8
    require(np.array_equal(means, groups.mean(axis=1)), "group replication must preserve every mean")
    return {"means": means, "naive_se": naive_se,
            "true_se": math.sqrt(p*(1-p)/8), "nominal_n": 64, "independent_groups": 8,
            "true_variance": p*(1-p)/8, "iid_variance_if_64": p*(1-p)/64,
            "variance_inflation": 8}


def summaries(samples):
    result = []
    for family, by_n in samples.items():
        for n, values in by_n.items():
            q = np.quantile(values, [.25, .5, .75], method="linear")
            row = {"family": family, "n": n, "repetitions": len(values),
                   "replicate_mean": moments(values)["mean"],
                   "replicate_sd": moments(values)["s"],
                   "q25": float(q[0]), "median": float(q[1]), "q75": float(q[2]),
                   "iqr": float(q[2]-q[0])}
            result.append(row)
    return result


def reference_checks():
    count = 0
    for n in range(1, 9):
        for a, b in [(0,1),(1,4),(1,2),(2,3),(1,1)]:
            p = Fraction(a,b)
            pmf = binomial_exact(n,a,b)
            require(sum(pmf) == 1, "normalization"); count += 1
            require(sum(Fraction(k)*q for k,q in enumerate(pmf)) == n*p, "count mean"); count += 1
            require(sum((Fraction(k)-n*p)**2*q for k,q in enumerate(pmf)) == n*p*(1-p), "count variance"); count += 1
            # Enumerate complete 0/1 datasets; this is independent of comb().
            enum = [Fraction() for _ in range(n+1)]
            expected_s2 = Fraction()
            for xs in itertools.product([0,1], repeat=n):
                k = sum(xs); probability = p**k*(1-p)**(n-k)
                enum[k] += probability
                if n > 1:
                    bar = Fraction(k,n)
                    expected_s2 += sum((Fraction(x)-bar)**2 for x in xs)/(n-1)*probability
            require(enum == pmf, "sequence enumeration"); count += 1
            if n > 1:
                require(expected_s2 == p*(1-p), "unbiased sample variance"); count += 1
    for N in range(2, 9):
        population = [i % 3 for i in range(N)]
        for n in range(1,N+1):
            finite_population_exact(population,n); count += 1
    for n in [2,4,8,16,32,64,128,256]:
        x = exact_mean_summary(n)
        require(x["tail_ge_epsilon"] <= x["chebyshev_bound"] + 1e-15, "Chebyshev bound"); count += 1
        require(Fraction(x["variance_exact"]) == Fraction(3,16*n), "variance / n"); count += 1
    for n in [2,3,7,16,64]:
        for rho in [0,.125,.5,1]:
            mat = np.full((n,n),rho); np.fill_diagonal(mat,1)
            expected = float(np.sum(mat))/n**2
            got = mean_variance_equicorrelated(n,1,rho)["variance_mean"]
            require(math.isclose(got,expected,rel_tol=1e-14,abs_tol=1e-14), "covariance sum"); count += 1
    require(moments([0,0,0,1])["s2"] == .25, "hand sample variance"); count += 1
    require(moments([0,0,0,1])["estimated_se_iid"] == .25, "hand estimated SE"); count += 1
    return count


def boundary_checks():
    bad = [
        ("empty",lambda: moments([])), ("bool",lambda: moments([0,True])),
        ("numpy bool",lambda: moments([0,np.bool_(False)])),
        ("text",lambda: moments([0,"1"])), ("complex",lambda: moments([0,1j])),
        ("nan",lambda: moments([0,float("nan")])), ("infinity",lambda: moments([0,float("inf")])),
        ("matrix",lambda: moments([[0,1]])), ("object array",lambda: moments(np.array([1],dtype=object))),
        ("bool array",lambda: moments(np.array([True]))), ("complex array",lambda: moments(np.array([1j]))),
        ("large",lambda: moments([1e200])), ("subnormal",lambda: moments([1e-320])),
        ("square underflow",lambda: moments([0,1e-200])),
        ("negative n",lambda: binomial_exact(-1)), ("float n",lambda: binomial_exact(2.0)),
        ("bool n",lambda: binomial_exact(True)), ("too large n",lambda: binomial_exact(513)),
        ("zero denominator",lambda: binomial_exact(3,1,0)), ("p>1",lambda: binomial_exact(3,2,1)),
        ("bool numerator",lambda: binomial_exact(3,True,2)),
        ("mean n0",lambda: exact_mean_summary(0)), ("zero epsilon",lambda: exact_mean_summary(8,epsilon=Fraction(0))),
        ("float epsilon",lambda: exact_mean_summary(8,epsilon=.1)),
        ("rho high",lambda: mean_variance_equicorrelated(4,1,1.1)),
        ("rho low",lambda: mean_variance_equicorrelated(4,1,-.34)),
        ("variance0",lambda: mean_variance_equicorrelated(4,0,.2)),
        ("rho bool",lambda: mean_variance_equicorrelated(4,1,True)),
        ("equicorrelation n1",lambda: mean_variance_equicorrelated(1,1,0)),
        ("population bool",lambda: finite_population_exact([0,True],1)),
        ("population singleton",lambda: finite_population_exact([1],1)),
        ("oversample",lambda: finite_population_exact([0,1],3)),
        ("normal nan",lambda: normal_cdf(float("nan"))),
        ("fraction conversion underflow",lambda: moments([Fraction(1,10**400)])),
        ("equicorrelation underflow",lambda: mean_variance_equicorrelated(4,TINY,0)),
        ("constant prefix bool",lambda: moments([.1,.1,True])),
        ("constant prefix text",lambda: moments([1e-140,1e-140,"bad"])),
        ("constant prefix nan",lambda: moments([3e130,3e130,float("nan")])),
        ("constant prefix range",lambda: moments([3e130,3e130,1e200]))
    ]
    good_spec = {"kind":"synthetic_sampling_demo","seed":22022,"repetitions":4000,"sizes":[4,16,64,256],"p_numerator":1,"p_denominator":4}
    for name, replacement in [("spec seed bool",{"seed":True}), ("spec seed negative",{"seed":-1}),
        ("spec repeated sizes",{"sizes":[4,4]}), ("spec descending",{"sizes":[16,4]}),
        ("spec size bool",{"sizes":[True]}), ("spec size text",{"sizes":["4"]}),
        ("spec size float",{"sizes":[4.0]}), ("spec size zero",{"sizes":[0]}),
        ("spec empty sizes",{"sizes":[]}), ("spec too large",{"sizes":[512],"repetitions":8000}),
        ("spec repeat1",{"repetitions":1}), ("spec repeat bool",{"repetitions":True}),
        ("spec p0",{"p_numerator":0}), ("spec p1",{"p_numerator":4}),
        ("spec unknown kind",{"kind":"real_private_data"}), ("spec unknown key",{"unexpected":1})]:
        trial = dict(good_spec); trial.update(replacement)
        bad.append((name,lambda trial=trial:validate_spec(trial)))
    for name, call in bad:
        try: call()
        except ValueError: pass
        else: raise ValueError("expected rejection: "+name)
    accepted = [moments([3]), moments([3,3]), moments(np.array([0,1],dtype=np.int64)),
        binomial_exact(0), binomial_exact(4,0,1), binomial_exact(4,1,1),
        exact_mean_summary(4,0,1), mean_variance_equicorrelated(4,1,-1/3),
        finite_population_exact([0,0,1,1],4), validate_spec(good_spec),
        moments([.1]*3), moments([1e-140]*3), moments([3e130]*3),
        moments(np.full(7,-.2))]
    require(accepted[0]["s"] is None and accepted[0]["estimated_se_iid"] is None, "n1 guard")
    require(accepted[1]["s"] == 0, "constant valid")
    for row, expected in zip(accepted[-4:], [.1,1e-140,3e130,-.2]):
        require(row["mean"] == expected and row["variance_n"] == 0 and row["s2"] == 0 and row["s"] == 0 and row["estimated_se_iid"] == 0, "exact constant regression")
    return {"expected_rejections":len(bad),"accepted_groups":len(accepted),"groups":len(bad)+len(accepted)}


def load_population(path=ROOT / "data/finite_population.csv"):
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        require(reader.fieldnames == ["unit_id", "value"], "population CSV header mismatch")
        rows = list(reader)
    require(2 <= len(rows) <= 12, "population CSV requires 2..12 rows")
    seen = set(); values = []
    for row in rows:
        require(set(row) == {"unit_id", "value"}, "population CSV row width mismatch")
        label = row["unit_id"]
        require(isinstance(label, str) and label.strip() == label and bool(label) and label not in seen, "population IDs must be unique and nonempty")
        seen.add(label)
        raw = row["value"]
        require(isinstance(raw, str) and raw.lstrip("-").isdigit(), "population CSV value must be an integer")
        values.append(integer(int(raw), "population value", -1000, 1000))
    return values


def build_report(spec, population=None):
    spec = validate_spec(spec)
    if population is None:
        population = load_population()
    population_result = finite_population_exact(population, 2)
    exact = [exact_mean_summary(n,spec["p_numerator"],spec["p_denominator"]) for n in spec["sizes"]]
    checks = {"reference_checks":reference_checks(),"boundary":boundary_checks()}
    samples = sample_means(spec)
    cluster = cluster_experiment(spec)
    report = {"spec":spec,"runtime":{"python":sys.version.split()[0],"numpy":np.__version__,"bit_generator":"PCG64"},
              "exact_bernoulli":exact,"finite_simulation":summaries(samples),
              "rare_event_P_zero_n30":float(Fraction(99,100)**30),
              "finite_population":population_result,
              "cluster":{k:v for k,v in cluster.items() if k not in {"means","naive_se"}},
              "cluster_empirical_sd":moments(cluster["means"])["s"],
              "checks":checks,
              "limits":["Synthetic finite simulations, not proof or real-world evidence", "No Cauchy population mean or population variance is claimed", "Pareto I alpha=1.5 has finite mean3 and infinite variance", "Browser Jupyter transport and learner Anaconda installation not tested"]}
    return report,samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec",type=Path,default=ROOT/"data/model_spec.json")
    parser.add_argument("--output",type=Path,default=ROOT/"outputs")
    parser.add_argument("--population",type=Path,default=ROOT/"data/finite_population.csv")
    args = parser.parse_args()
    spec = load_spec(args.spec)
    population = load_population(args.population)
    report,samples = build_report(spec,population)  # All validation/computation before output mutation.
    encoded = json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n"
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"report.json").write_text(encoded,encoding="utf-8")
    with (args.output/"sample_means.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f); writer.writerow(["family","n","replicate","mean"])
        for family,by_n in samples.items():
            for n,values in by_n.items():
                for i,value in enumerate(values): writer.writerow([family,n,i,format(float(value),".17g")])
    print(json.dumps({"checks":report["checks"],"exact_bernoulli":report["exact_bernoulli"],
                      "cluster_empirical_sd":report["cluster_empirical_sd"]},ensure_ascii=False,sort_keys=True))


if __name__ == "__main__":
    main()
