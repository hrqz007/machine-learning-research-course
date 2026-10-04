"""Unit 017: finite quadrature, normalization and two separate error sources.

Standard-library core. All inputs are finite scalar real values; booleans and
text are rejected. Grids must be strictly increasing in floating representation.
Finite checks do not guarantee correct quadrature for an arbitrary function.
"""
from pathlib import Path
from fractions import Fraction
from numbers import Real
import argparse
import csv
import json
import math
import sys

BASE = Path(__file__).resolve().parent
MAX_SEGMENTS = 100_000


def real(value, name="value"):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(name + " must be a real number, not boolean or text")
    try:
        out = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(name + " is not a representable finite float") from error
    if not math.isfinite(out):
        raise ValueError(name + " must be finite")
    return out


def positive_integer(n, limit=MAX_SEGMENTS):
    if type(n) is not int or not 1 <= n <= limit:
        raise ValueError("segment count must be an integer in [1, " + str(limit) + "]")
    return n


def finite_sum(values, name="sum"):
    try:
        result = math.fsum(values)
    except (OverflowError, ValueError) as error:
        raise ValueError(name + " overflowed") from error
    return real(result, name)


def vector(values, name):
    if not isinstance(values, (list, tuple)):
        raise ValueError(name + " must be a one-dimensional list or tuple")
    return [real(value, name + " element") for value in values]


def uniform_grid(a, b, n):
    a, b = real(a, "lower bound"), real(b, "upper bound")
    n = positive_integer(n)
    if not a < b:
        raise ValueError("bounds must satisfy lower < upper")
    width = real(b-a, "interval width")
    grid = [a + width*(i/n) for i in range(n+1)]
    grid[0], grid[-1] = a, b
    # Validate the entire grid before any user callback can be invoked.
    grid = vector(grid, "grid")
    if any(not left < right for left, right in zip(grid, grid[1:])):
        raise ValueError("grid has collapsed or reversed floating-point coordinates")
    return grid


def evaluate(function, grid):
    if not callable(function):
        raise ValueError("integrand must be callable")
    result = []
    for x in grid:
        x = real(x, "callback coordinate")
        try:
            value = function(x)
        except (ArithmeticError, ValueError, TypeError) as error:
            raise ValueError("integrand failed at a valid coordinate") from error
        result.append(real(value, "integrand return value"))
    return result


def trapezoid_samples(xs, ys):
    xs, ys = vector(xs, "x"), vector(ys, "y")
    if len(xs) < 2 or len(xs) != len(ys):
        raise ValueError("x and y must have equal lengths of at least two")
    if any(not a < b for a, b in zip(xs, xs[1:])):
        raise ValueError("sample coordinates must be strictly increasing")
    terms = []
    for a, b, u, v in zip(xs, xs[1:], ys, ys[1:]):
        width = real(b-a, "sample interval width")
        average = real(u/2 + v/2, "endpoint average")
        terms.append(real(width*average, "trapezoid contribution"))
    return finite_sum(terms, "quadrature total")


def integrate(function, a, b, n, rule="trapezoid"):
    if rule not in ("left", "right", "midpoint", "trapezoid"):
        raise ValueError("unknown integration rule")
    grid = uniform_grid(a, b, n)
    widths = [real(right-left, "segment width") for left, right in zip(grid, grid[1:])]
    if rule == "trapezoid":
        return trapezoid_samples(grid, evaluate(function, grid))
    if rule == "left":
        points = grid[:-1]
    elif rule == "right":
        points = grid[1:]
    else:
        points = [left + width/2 for left, width in zip(grid[:-1], widths)]
        if any(not left < mid < right for left, mid, right in zip(grid, points, grid[1:])):
            raise ValueError("a midpoint cannot be represented strictly inside its segment")
    values = evaluate(function, points)
    return finite_sum([real(w*y, "rectangle contribution") for w, y in zip(widths, values)], "quadrature total")


def shape(x):
    x = real(x, "x")
    return 1+x if 0 <= x <= 2 else 0.0


def density(x):
    return shape(x)/4


def cumulative(x):
    x = real(x, "x")
    if x < 0:
        return 0.0
    if x > 2:
        return 1.0
    return x/4 + x*x/8


def interval_mass(a, b):
    a, b = real(a, "lower"), real(b, "upper")
    if a > b:
        raise ValueError("probability interval must not be reversed")
    return real(cumulative(b)-cumulative(a), "interval mass")


def normalized_samples(xs, nonnegative_values):
    xs, values = vector(xs, "x"), vector(nonnegative_values, "shape")
    if any(v < 0 for v in values):
        raise ValueError("density samples must be nonnegative")
    area = trapezoid_samples(xs, values)
    if not area > 0:
        raise ValueError("normalizer must be finite and strictly positive")
    normalized = [real(v/area, "normalized sample") for v in values]
    return area, normalized


def transformed_density(y):
    y = real(y, "y")
    return density((y-1)/3)/3


def exponential_audit(B, n):
    B = real(B, "cutoff")
    if B <= 0:
        raise ValueError("cutoff must be positive")
    Q = integrate(lambda x: math.exp(-x), 0, B, n)
    retained = -math.expm1(-B)
    tail = math.exp(-B)
    discretization = Q-retained
    total = Q-1
    return {"B": B, "n": n, "h": B/n, "estimate": Q,
            "exact_retained_float": retained, "tail_float": tail,
            "discretization_signed": discretization, "total_signed": total,
            "decomposition_residual": total-(discretization-tail)}


def triangle_grid(n, include_boundary=True):
    n = positive_integer(n, 512)
    if type(include_boundary) is not bool:
        raise ValueError("include_boundary must be a boolean")
    # Equal axis grids mean comparing exact midpoint coordinates equals j<=i.
    count = sum(1 for i in range(n) for j in range(n)
                if (j <= i if include_boundary else j < i))
    return {"n": n, "include_boundary": include_boundary,
            "selected_cells": count, "total_cells": n*n, "mass": 2*count/(n*n)}


def narrow_peak(x):
    x = real(x, "x")
    # Fixed finite coefficients; no arbitrary width is accepted by this helper.
    if x < 0.52 or x > 0.54:
        return 0.0
    return max(0.0, 1.0-abs(x-0.53)/0.01)/0.01


def load_data(path=None):
    path = Path(path) if path is not None else BASE/"data/density_samples.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["x", "q"]:
            raise ValueError("expected CSV columns x,q")
        rows = list(reader)
    try:
        xs, qs = [float(r["x"]) for r in rows], [float(r["q"]) for r in rows]
    except (TypeError, ValueError) as error:
        raise ValueError("CSV must contain numeric values") from error
    trapezoid_samples(xs, qs)
    if any(q < 0 for q in qs):
        raise ValueError("CSV shape contains a negative value")
    return xs, qs


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def exact_checks():
    checks = 0
    for n in range(1, 65):
        h = Fraction(1, n)
        left = h*sum((i*h)**2 for i in range(n))
        right = h*sum((i*h)**2 for i in range(1, n+1))
        midpoint = h*sum(((Fraction(i)+Fraction(1,2))*h)**2 for i in range(n))
        trap = (left+right)/2
        answers = [Fraction(1,3)-Fraction(1,2*n)+Fraction(1,6*n*n),
                   Fraction(1,3)+Fraction(1,2*n)+Fraction(1,6*n*n),
                   Fraction(1,3)-Fraction(1,12*n*n),
                   Fraction(1,3)+Fraction(1,6*n*n)]
        for value, answer in zip([left,right,midpoint,trap], answers):
            require(value == answer, "exact polynomial sum differs")
            checks += 1
    # Arbitrary rational, nonuniform grids: trapezoids integrate linear q exactly.
    for k in range(1, 41):
        xs = [Fraction(0), Fraction(k,100), Fraction(3*k+50,100), Fraction(2)]
        vals = [1+x for x in xs]
        area = sum((xs[i+1]-xs[i])*(vals[i]+vals[i+1])/2 for i in range(3))
        require(area == 4, "nonuniform linear integral differs")
        require(abs(trapezoid_samples(xs, vals)-4) < 2e-15, "float nonuniform check")
        checks += 2
    for n in range(1, 41):
        points = [(Fraction(2*i+1,2*n), Fraction(2*j+1,2*n))
                  for i in range(n) for j in range(n)]
        for include in [True, False]:
            count = sum(1 for x,y in points if (y<=x if include else y<x))
            exact = Fraction(2*count, n*n)
            require(exact == 1 + (Fraction(1,n) if include else -Fraction(1,n)), "triangle count")
            require(abs(triangle_grid(n,include)["mass"]-float(exact)) < 1e-15, "triangle implementation")
            checks += 2
    lo, hi = Fraction(1,2), Fraction(3,2)
    mass = (hi-lo)/4+(hi**2-lo**2)/8
    require(mass == Fraction(1,2), "main interval")
    checks += 1
    return {"passed": checks, "exact_rational_checks": 377, "floating_comparisons": 120, "main_normalizer": "4", "main_interval_mass": str(mass),
            "polynomial_n": [1,64], "nonuniform_grids":40, "triangle_n":[1,40]}


def boundary_checks():
    cases = [
        ("boolean bound", lambda: uniform_grid(False,1,4)),
        ("text bound", lambda: uniform_grid("0",1,4)),
        ("nan bound", lambda: uniform_grid(0,float("nan"),4)),
        ("infinite bound", lambda: uniform_grid(0,float("inf"),4)),
        ("complex bound", lambda: uniform_grid(0,1j,4)),
        ("boolean n", lambda: uniform_grid(0,1,True)),
        ("float n", lambda: uniform_grid(0,1,4.0)),
        ("zero n", lambda: uniform_grid(0,1,0)),
        ("negative n", lambda: uniform_grid(0,1,-2)),
        ("resource bound", lambda: uniform_grid(0,1,MAX_SEGMENTS+1)),
        ("reversed bounds", lambda: uniform_grid(1,0,4)),
        ("equal bounds", lambda: uniform_grid(1,1,4)),
        ("overflow width", lambda: uniform_grid(-1e308,1e308,4)),
        ("collapsed grid", lambda: uniform_grid(1e20,math.nextafter(1e20,math.inf),4)),
        ("unrepresentable midpoint", lambda: integrate(lambda x:x,1,math.nextafter(1,math.inf),1,"midpoint")),
        ("unknown rule", lambda: integrate(lambda x:x,0,1,4,"simpson")),
        ("noncallable", lambda: integrate(3,0,1,4)),
        ("callback nan", lambda: integrate(lambda x:float("nan"),0,1,4)),
        ("callback bool", lambda: integrate(lambda x:True,0,1,4)),
        ("callback text", lambda: integrate(lambda x:"1",0,1,4)),
        ("callback domain", lambda: integrate(math.log,0,1,4)),
        ("callback overflow", lambda: integrate(math.exp,800,801,4)),
        ("contribution overflow", lambda: integrate(lambda x:1e308,0,4,2)),
        ("summation overflow", lambda: integrate(lambda x:1e308,0,3,3)),
        ("empty samples", lambda: trapezoid_samples([],[])),
        ("one sample", lambda: trapezoid_samples([1],[2])),
        ("length mismatch", lambda: trapezoid_samples([0,1],[1])),
        ("unsorted x", lambda: trapezoid_samples([0,2,1],[1,3,2])),
        ("duplicate x", lambda: trapezoid_samples([0,0],[1,2])),
        ("mixed bool", lambda: trapezoid_samples([0,1],[1,True])),
        ("mixed text", lambda: trapezoid_samples([0,1],[1,"2"])),
        ("nested shape", lambda: trapezoid_samples([[0],[1]],[1,2])),
        ("nonfinite y", lambda: trapezoid_samples([0,1],[1,float("inf")])),
        ("negative density", lambda: normalized_samples([0,1],[1,-1])),
        ("zero normalizer", lambda: normalized_samples([0,1],[0,0])),
        ("underflow normalizer", lambda: normalized_samples([0,1e-308],[1e-308,1e-308])),
        ("negative cutoff", lambda: exponential_audit(-1,4)),
        ("zero cutoff", lambda: exponential_audit(0,4)),
        ("triangle too large", lambda: triangle_grid(513)),
        ("nonboolean boundary", lambda: triangle_grid(4,1)),
        ("reversed probability interval", lambda: interval_mass(1,0)),
    ]
    passed = []
    for name, call in cases:
        try:
            call()
        except ValueError:
            passed.append(name)
        else:
            raise AssertionError("expected rejection: "+name)
    calls = []
    try:
        integrate(lambda x:calls.append(x) or 1,-1e308,1e308,4)
    except ValueError:
        pass
    require(not calls, "invalid coordinates reached callback")
    passed.append("invalid grid rejected before callback")
    require(abs(integrate(lambda x:x*x,0,1,4)-11/32) < 1e-15,"known trapezoid")
    passed.append("known trapezoid 11/32")
    require(abs(integrate(lambda x:x*x,0,1,4,"midpoint")-21/64) < 1e-15,"known midpoint")
    passed.append("known midpoint 21/64")
    require(interval_mass(-10,10)==1 and interval_mass(1,1)==0,"support clipping")
    passed.append("support clipping and zero probability interval")
    require(-math.expm1(-1e-18)>0 and 1-math.exp(-1e-18)==0,"small exponential cancellation")
    passed.append("expm1 avoids demonstrable subtraction loss")
    return {"passed":len(passed), "cases":passed}


def run(output=None):
    xs,qs=load_data();normalizer,ps=normalized_samples(xs,qs)
    polynomials=[]
    for n in [2,4,8,16,32,64,128,256]:
        for rule in ["left","right","midpoint","trapezoid"]:
            q=integrate(lambda x:x*x,0,1,n,rule)
            polynomials.append({"n":n,"rule":rule,"estimate":q,"signed_error":q-1/3})
    exponentials=[exponential_audit(B,n) for B in [1,2,4,8] for n in [8,32,128,512]]
    aliases=[]
    for n in [8,16,32,64,128,256,512,1024]:
        aliases.append({"n":n,"oscillation_estimate":integrate(lambda x:math.sin(16*math.pi*x)**2,0,1,n),
                        "peak_estimate":integrate(narrow_peak,0,1,n)})
    report={"unit":"017","python":sys.version.split()[0],"synthetic":True,
            "main":{"normalizer":normalizer,"normalized_mass":trapezoid_samples(xs,ps),
                    "interval_mass":interval_mass(.5,1.5),
                    "transformed_mass":integrate(transformed_density,1,7,8),
                    "transformed_interval_mass":integrate(transformed_density,2.5,5.5,8),
                    "nonlinear_correct":integrate(lambda u:2*u**5,0,1,2048),
                    "nonlinear_missing_factor":integrate(lambda u:u**4,0,1,2048)},
            "polynomial_scan":polynomials,"exponential_scan":exponentials,
            "cancellation":exponential_audit(4,8),
            "triangle":[triangle_grid(n,b) for n in [4,8,16,32,64] for b in [True,False]],
            "missed_structure":aliases,"exact_checks":exact_checks(),"boundary_checks":boundary_checks()}
    target=Path(output) if output is not None else BASE/"outputs"
    target.mkdir(parents=True,exist_ok=True)
    (target/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"normalizer":normalizer,"interval_mass":report["main"]["interval_mass"],
                      "exact_checks":report["exact_checks"]["passed"],
                      "boundary_checks":report["boundary_checks"]["passed"],
                      "cancellation_estimate":report["cancellation"]["estimate"]},sort_keys=True))
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path)
    run(parser.parse_args().output)
