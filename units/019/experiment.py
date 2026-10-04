"""Finite random variables, moments and conditional expectations.

Offline synthetic teaching models. Run with Python 3.12; no third-party import.
Validated float routines deliberately reject nonzero subnormal values/products.
Exact Fraction routines are separate reference computations, not float wrappers.
"""
from pathlib import Path
from fractions import Fraction
import argparse
import csv
import json
import math
import random
import statistics
import sys

BASE = Path(__file__).resolve().parent
MIN_NORMAL = sys.float_info.min


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def real(value, name="value"):
    if isinstance(value, bool) or not isinstance(value, (int, float, Fraction)):
        raise TypeError(f"{name} must be an ordinary finite real number")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} is not representable") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if (number != 0 and abs(number) < MIN_NORMAL) or (number == 0 and value != 0):
        raise ValueError(f"{name} is below the supported normal-float range")
    return number


def product(a, b, name="product"):
    value = a * b
    if a != 0 and b != 0 and value == 0:
        raise ValueError(f"{name} underflowed")
    return real(value, name)


def total(values, name="sum"):
    try:
        return real(math.fsum(values), name)
    except OverflowError as exc:
        raise ValueError(f"{name} overflowed") from exc


def vector(values, name="values"):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{name} must be a nonempty flat list or tuple")
    return [real(x, name) for x in values]


def model(values, probabilities):
    x = vector(values)
    p = vector(probabilities, "probabilities")
    if len(x) != len(p) or any(w < 0 for w in p):
        raise ValueError("lengths must match and probabilities must be nonnegative")
    if total(p) != 1.0:
        raise ValueError("math.fsum(probabilities) must equal 1.0; no renormalization")
    return x, p


def pmf(values, probabilities):
    x, p = model(values, probabilities)
    groups = {}
    for value, weight in zip(x, p):
        groups.setdefault(value, []).append(weight)
    return {value: total(groups[value]) for value in sorted(groups)}


def cdf(values, probabilities, threshold, include=True):
    x, p = model(values, probabilities)
    t = real(threshold, "threshold")
    if not isinstance(include, bool):
        raise TypeError("include must be bool")
    return total([w for value, w in zip(x, p) if value <= t] if include else
                 [w for value, w in zip(x, p) if value < t])


def expectation(values, probabilities, function=lambda x: x):
    x, p = model(values, probabilities)
    if not callable(function):
        raise TypeError("function must be callable")
    # Entire model is checked before any callback. Zero-mass points contribute zero.
    contributions = [product(real(function(value), "callback output"), weight)
                     for value, weight in zip(x, p) if weight > 0]
    return total(contributions)


def moments(values, probabilities):
    x, p = model(values, probabilities)
    reference = x[next(i for i, w in enumerate(p) if w > 0)]
    offsets = [real(value - reference, "offset") if w > 0 else 0.0
               for value, w in zip(x, p)]
    center = total(product(delta, w) for delta, w in zip(offsets, p))
    mean = real(reference + center, "mean")
    variance = total(product(product(real(delta - center, "centered offset"),
                                    real(delta - center, "centered offset")), w)
                     for delta, w in zip(offsets, p) if w > 0)
    return {"mean": mean, "variance": variance,
            "standard_deviation": math.sqrt(variance)}


def square_risk(values, probabilities, prediction):
    a = real(prediction, "prediction")
    return expectation(values, probabilities,
                       lambda x: product(real(x-a, "residual"), real(x-a, "residual")))


def condition(values, probabilities, selected):
    x, p = model(values, probabilities)
    if not isinstance(selected, (list, tuple)) or len(selected) != len(x):
        raise ValueError("selected must match the model")
    if any(not isinstance(flag, bool) for flag in selected):
        raise TypeError("selected must contain bool only")
    mass = total(w for w, flag in zip(p, selected) if flag)
    if mass == 0:
        raise ValueError("conditioning event has zero model probability")
    xx = [v for v, flag, w in zip(x, selected, p) if flag and w > 0]
    pp = [real(w/mass, "conditional probability") for w, flag in zip(p, selected) if flag and w > 0]
    # Conditional weights are divided by the already validated event mass.
    # Rounding can make their fsum adjacent to 1. Compute conditional moments by
    # weighted unnormalized numerator / mass instead of silently renormalizing.
    reference = xx[0]
    numerator = total(product(real(v-reference, "conditional offset"), w)
                      for v, w, flag in zip(x, p, selected) if flag and w > 0)
    offset_mean = real(numerator/mass, "conditional mean offset")
    mean = real(reference+offset_mean, "conditional mean")
    return {"mass": mass, "mean": mean, "values": xx, "probabilities": pp}


def empirical(values, groups):
    x = vector(values)
    if not isinstance(groups, (list, tuple)) or len(groups) != len(x):
        raise ValueError("groups must match sample length")
    if any(g not in ("quiet", "active") for g in groups):
        raise ValueError("group labels must be quiet or active")
    n = len(x)
    reference = x[0]
    offsets = [real(v-reference, "sample offset") for v in x]
    average_offset = total(offsets)/n
    mean = real(reference+average_offset, "sample mean")
    variance = real(total(product(real(d-average_offset), real(d-average_offset)) for d in offsets)/n)
    result = {"n": n, "mean": mean, "variance_n": variance, "groups": {}}
    reconstruction = []
    for group in ("quiet", "active"):
        members = [v for v, g in zip(x, groups) if g == group]
        estimate = total(members)/len(members) if members else None
        result["groups"][group] = {"count": len(members), "mean": estimate}
        if members:
            reconstruction.append(product(len(members)/n, estimate))
    result["empirical_tower"] = total(reconstruction)
    return result


def density(t):
    t = real(t, "t")
    return (1+t)/4 if 0 <= t <= 2 else 0.0


def continuous_cdf(t):
    t = real(t, "t")
    if t < 0:
        return 0.0
    if t >= 2:
        return 1.0
    return real(product(t, .25+t/8), "continuous CDF")


def inverse_cdf(u):
    u = real(u, "u")
    if not 0 <= u <= 1:
        raise ValueError("u must lie in [0,1]; random.random uses [0,1)")
    return real((8*u)/(math.sqrt(1+8*u)+1), "inverse CDF")


def mixed_cdf(t):
    t = real(t, "t")
    if t < 0:
        return 0.0
    if t >= 1:
        return 1.0
    return 1/3+2*t/3


def trapezoid(function, lower, upper, n):
    a, b = real(lower), real(upper)
    if b <= a or type(n) is not int or not 1 <= n <= 100000:
        raise ValueError("ordered bounds and integer 1 <= n <= 100000 required")
    span = real(b-a, "span")
    h = real(span/n, "step")
    if h == 0:
        raise ValueError("step underflowed")
    grid = [real(a+i*h, "grid point") for i in range(n)] + [b]
    if any(right <= left for left, right in zip(grid, grid[1:])):
        raise ValueError("grid points did not move strictly forward")
    if not callable(function):
        raise TypeError("function must be callable")
    values = [real(function(point), "integrand") for point in grid]
    areas = [product(real(right-left, "cell width"),
                     total([product(yl, .5), product(yr, .5)]), "cell area")
             for left, right, yl, yr in zip(grid, grid[1:], values, values[1:])]
    return total(areas)


def load_model(path=BASE/"data/scenarios.csv"):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["scenario_id", "probability_numerator", "probability_denominator", "loss_units", "group"]:
            raise ValueError("unexpected CSV schema")
        rows = list(reader)
    if not rows or len({r["scenario_id"] for r in rows}) != len(rows):
        raise ValueError("empty or repeated scenario id")
    exact_x, exact_p, groups = [], [], []
    for row in rows:
        if not row["scenario_id"] or row["group"] not in ("quiet", "active"):
            raise ValueError("invalid id or group")
        try:
            numerator = int(row["probability_numerator"])
            denominator = int(row["probability_denominator"])
            value = Fraction(row["loss_units"])
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError("invalid numeric CSV entry") from exc
        if denominator <= 0 or numerator < 0:
            raise ValueError("invalid probability ratio")
        exact_x.append(value)
        exact_p.append(Fraction(numerator, denominator))
        groups.append(row["group"])
    if sum(exact_p) != 1:
        raise ValueError("exact CSV probabilities must sum to one")
    model(exact_x, exact_p)
    # The executable demonstrations below are specifically the published model.
    if exact_x != list(map(Fraction, [0,0,1,1,2,4])) or exact_p != [Fraction(1,6)]*6 or groups != ["quiet"]*2+["active"]*4:
        raise ValueError("CSV differs from the stated six-state teaching model")
    return exact_x, exact_p, groups


def exact_audit():
    checks = 0
    generator = random.Random(19041)
    for _ in range(160):
        count = generator.randint(2,8)
        x = [Fraction(generator.randint(-20,20), generator.randint(1,7)) for _ in range(count)]
        counts = [generator.randint(1,10) for _ in x]
        p = [Fraction(c, sum(counts)) for c in counts]
        mu = sum(v*w for v,w in zip(x,p))
        variance = sum((v-mu)**2*w for v,w in zip(x,p))
        a = Fraction(generator.randint(-9,9), 3)
        require(sum((v-a)**2*w for v,w in zip(x,p)) == variance+(a-mu)**2, "exact risk identity")
        require(sum(v*v*w for v,w in zip(x,p))-mu*mu == variance, "exact variance identity")
        aggregated = {v: sum(w for z,w in zip(x,p) if z == v) for v in set(x)}
        require(sum(v*w for v,w in aggregated.items()) == mu, "exact aggregation")
        mass = sum(p[::2]); other = 1-mass
        conditional = sum(v*w for v,w in zip(x[::2],p[::2]))/mass
        complement = sum(v*w for v,w in zip(x[1::2],p[1::2]))/other
        require(mass*conditional+other*complement == mu, "exact tower identity")
        threshold = x[0]
        require(sum(w for v,w in zip(x,p) if v <= threshold)-sum(w for v,w in zip(x,p) if v < threshold)==aggregated[threshold], "exact CDF jump")
        checks += 5
        if math.fsum(map(float,p)) == 1.0:
            actual = moments(x,p)
            require(math.isclose(actual["mean"],float(mu),rel_tol=2e-13,abs_tol=2e-13), "float mean oracle")
            require(math.isclose(actual["variance"],float(variance),rel_tol=2e-13,abs_tol=2e-13), "float variance oracle")
            checks += 2
    return checks


def boundary_audit():
    rejected = 0
    cases = [lambda: model([],[]), lambda: model([1],[.5]), lambda: model([1,2],[1.1,-.1]),
             lambda: model([1,2],[1]), lambda: model([True],[1]), lambda: model(["1"],[1]),
             lambda: model([[1]],[1]), lambda: model([complex(1,0)],[1]), lambda: model([float("nan")],[1]),
             lambda: model([float("inf")],[1]), lambda: model([1],[True]), lambda: model([1],["1"]),
             lambda: model([1],[0]), lambda: model([1],[1.000000000001]),
             lambda: condition([1,2],[1,0],[False,True]), lambda: condition([1],[1],[1]),
             lambda: condition([1],[1],[]), lambda: expectation([1],[1],lambda x: float("nan")),
             lambda: expectation([1],[1],lambda x: True), lambda: expectation([1],[1],lambda x: "1"),
             lambda: moments([-1e308,1e308],[.5,.5]), lambda: moments([0,1e200],[.5,.5]),
             lambda: expectation([1e308,1e308],[.5,.5],lambda x: x*x),
             lambda: expectation([1e-200],[1],lambda x: product(x,x)),
             lambda: model([1],[Fraction(1,10**400)]), lambda: real(5e-324),
             lambda: product(1e-200,1e-200), lambda: product(1e-150,1e-160),
             lambda: inverse_cdf(-.01), lambda: inverse_cdf(1.01), lambda: inverse_cdf(True),
             lambda: empirical([],[]), lambda: empirical([1],["bad"]), lambda: empirical([1],[]),
             lambda: square_risk([1],[1],True), lambda: cdf([1],[1],1,1),
             lambda: trapezoid(lambda x:x,0,1,True), lambda: trapezoid(lambda x:x,1,0,10),
             lambda: trapezoid(lambda x:x,0,1,0), lambda: trapezoid(lambda x:x,0,1,100001),
             lambda: trapezoid(lambda x:x,0,1,1.5), lambda: trapezoid(lambda x:float("inf"),0,1,4),
             lambda: trapezoid(lambda x:True,0,1,4), lambda: trapezoid(lambda x:1e-200,0,1e-200,2),
             lambda: trapezoid(lambda x:x,1e16,1e16+2,8),
             lambda: simulate(1,0,[0,0,1,1,2,float("nan")],["quiet"]*2+["active"]*4),
             lambda: simulate(1,0,[0,0,1,1,2,True],["quiet"]*2+["active"]*4),
             lambda: simulate(1,0,[0,0,1,1,2,"4"],["quiet"]*2+["active"]*4),
             lambda: simulate(1,0,[0,0,1,1,2,4],["quiet"]*2+["active"]*3+["bad"])]
    for operation in cases:
        try:
            operation()
        except (ValueError, TypeError):
            rejected += 1
        else:
            raise AssertionError("an invalid input was accepted")
    calls = []
    try:
        expectation([1,float("nan")],[.5,.5],lambda x:calls.append(x) or x)
    except ValueError:
        pass
    require(not calls, "invalid model must be rejected before callback")
    try:
        trapezoid(lambda x:calls.append(x) or x,1e16,1e16+2,8)
    except ValueError:
        pass
    require(not calls, "invalid grid must be rejected before callback")
    require(inverse_cdf(0)==0 and inverse_cdf(1)==2, "inverse endpoints")
    require(cdf([0,1],[.5,.5],1)-cdf([0,1],[.5,.5],1,False)==.5, "CDF endpoint")
    missing = empirical([0,0],["quiet","quiet"])
    require(missing["groups"]["active"]["mean"] is None, "absent observed group is undefined")
    require(mixed_cdf(0)==1/3 and mixed_cdf(-1)==0 and mixed_cdf(1)==1,"mixed CDF")
    original_random = random.Random
    random_calls = []
    def forbidden_random(*args, **kwargs):
        random_calls.append((args, kwargs))
        raise AssertionError("RNG must not be created before model validation")
    try:
        random.Random = forbidden_random
        for operation in cases[-4:]:
            try:
                operation()
            except (ValueError, TypeError):
                pass
            else:
                raise AssertionError("invalid unsampled state escaped validation")
    finally:
        random.Random = original_random
    require(not random_calls, "invalid simulation model must not touch RNG")
    return {"expected_rejections":rejected, "accepted_or_sentinel_checks":7}


def simulate(n, seed, x, groups):
    if type(n) is not int or not 1 <= n <= 100000 or type(seed) is not int:
        raise ValueError("bounded positive integer n and integer seed required")
    # Validate the entire positive-probability model BEFORE creating the RNG.
    values = vector(x, "simulation values")
    if (not isinstance(groups, (list, tuple)) or len(values) != 6 or len(groups) != 6):
        raise ValueError("simulation uses six equal-probability states")
    if any(type(label) is not str or label not in ("quiet", "active") for label in groups):
        raise ValueError("all six group labels must be quiet or active")
    rng = random.Random(seed)
    index = [rng.randrange(6) for _ in range(n)]
    result = empirical([values[i] for i in index],[groups[i] for i in index])
    result["seed"] = seed
    require(math.isclose(result["empirical_tower"],result["mean"],rel_tol=1e-14,abs_tol=1e-14), "sample tower check")
    return result


def run():
    x, p, groups = load_model()
    main = moments(x,p)
    main["second_moment"] = expectation(x,p,lambda v:v*v)
    main["pmf"] = pmf(x,p)
    main["conditionals"] = {g:condition(x,p,[label==g for label in groups]) for g in ("quiet","active")}
    main["constant_risk"] = square_risk(x,p,4/3)
    main["group_risk"] = total(product((float(v)-(0 if g=="quiet" else 2))**2,float(w)) for v,w,g in zip(x,p,groups))
    main["wrong_unweighted_group_mean"] = 1.0
    n_values = [30,300,3000,30000]
    repeats = 24
    simulations = [{"n":n,"runs":[simulate(n,190000+n*100+r,x,groups) for r in range(repeats)]} for n in n_values]
    small = [simulate(2,199000+r,x,groups) for r in range(64)]
    quadrature=[]
    for n in [4,16,64,256]:
        vals=[trapezoid(lambda t,k=k:product(t**k,density(t)),0,2,n) for k in range(3)]
        quadrature.append({"n":n,"mass":vals[0],"first_moment":vals[1],"second_moment":vals[2],"variance_by_moments":vals[2]-vals[1]**2})
    rng=random.Random(190777)
    continuous=[inverse_cdf(rng.random()) for _ in range(30000)]
    shifted=[1e12+float(v) for v in x]
    shifted_naive=math.fsum(v*v for v in shifted)/6-(math.fsum(shifted)/6)**2
    shifted_stable=moments(shifted,p)["variance"]
    huge=[1e18+float(v) for v in x]
    boundary=boundary_audit()
    return {"unit":"019","model":main,"exact_and_float_crosschecks":exact_audit(),
            "boundary_checks":boundary,"simulation_repetitions_per_n":repeats,"simulations":simulations,
            "small_samples":small,"small_sample_absent_active":sum(r["groups"]["active"]["count"]==0 for r in small),
            "continuous_quadrature":quadrature,
            "continuous_sample":{"n":len(continuous),"seed":190777,"mean":statistics.fmean(continuous),"variance_n":statistics.pvariance(continuous)},
            "cancellation":{"offset":1e12,"naive_second_moment_difference":shifted_naive,"reference_shifted_variance":shifted_stable,"exact_variance":"17/9","huge_offset":1e18,"huge_distinct_values":len(set(huge))},
            "heavy_tail":[{"B":B,"mass":1-1/B,"truncated_first_moment":math.log(B)} for B in [2,10,100,10000,1e8]],
            "runtime":{"python":sys.version.split()[0],"core":"standard library","random_note":"Fixed seeds; cross-version random helper behavior is not promised"}}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,default=BASE/"outputs");args=parser.parse_args()
    report=run()  # Validate all inputs and finish computation before replacing an output.
    args.output.mkdir(parents=True,exist_ok=True)
    content=json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n"
    temporary=args.output/"report.json.tmp";temporary.write_text(content,encoding="utf-8");temporary.replace(args.output/"report.json")
    print(json.dumps({"unit":"019","status":"passed","mean":report["model"]["mean"],"variance":report["model"]["variance"],"crosschecks":report["exact_and_float_crosschecks"],"boundary":report["boundary_checks"],"missing_active_in_64_small_runs":report["small_sample_absent_active"]},ensure_ascii=False,sort_keys=True))

if __name__ == "__main__":
    main()
