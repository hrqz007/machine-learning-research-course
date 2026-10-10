"""Lesson 083: deterministic, offline representation experiments (stdlib only)."""
from pathlib import Path
import argparse, csv, hashlib, json, math, platform, random

ROOT = Path(__file__).resolve().parent

def affine(x, weights, bias):
    if not weights or len(weights) != len(bias):
        raise ValueError('one bias is required for each nonempty output row')
    if not x or any(len(row) != len(x) for row in weights):
        raise ValueError('weight rows must match input dimension')
    if not all(math.isfinite(float(v)) for v in list(x) + list(bias) + [v for row in weights for v in row]):
        raise ValueError('finite values required')
    return [sum(w * v for w, v in zip(row, x)) + b for row, b in zip(weights, bias)]

def relu(x):
    return max(0.0, x)

def parameter_count(widths):
    if len(widths) < 2 or any(type(v) is not int or v <= 0 for v in widths):
        raise ValueError('at least two positive integer widths required')
    return sum((a + 1) * b for a, b in zip(widths, widths[1:]))

def fixed_xor(x, activation=True):
    hidden = affine(x, [[1., 1.], [1., 1.]], [0., -1.])
    if activation:
        hidden = [relu(v) for v in hidden]
    return hidden, affine(hidden, [[1., -2.]], [0.])[0]

def alternative_xor(x):
    hidden = [relu(x[0] - x[1]), relu(x[1] - x[0])]
    return sum(hidden)

def collapse(w1, b1, w2, b2):
    if not w1 or not w2 or len(w1) != len(b1) or len(w2) != len(b2):
        raise ValueError('invalid layer dimensions')
    d = len(w1[0])
    if d == 0 or any(len(r) != d for r in w1) or any(len(r) != len(w1) for r in w2):
        raise ValueError('incompatible matrix shapes')
    w = [[sum(w2[j][k] * w1[k][i] for k in range(len(w1))) for i in range(d)] for j in range(len(w2))]
    b = [sum(w2[j][k] * b1[k] for k in range(len(w1))) + b2[j] for j in range(len(w2))]
    return w, b

def interpolate_square(x, segments):
    if type(segments) is not int or segments <= 0:
        raise ValueError('segments must be positive integer')
    # On [0,1], first slope 1/n, each knot raises it by 2/n.
    return x / segments + sum((2 / segments) * relu(x - k / segments) for k in range(1, segments))

def generate_data(directory):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'xor.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f); writer.writerow(['x1', 'x2', 'y'])
        writer.writerows([(0,0,0),(0,1,1),(1,0,1),(1,1,0)])
    with (directory / 'square-grid.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f); writer.writerow(['x', 'y'])
        writer.writerows((i / 1024, (i / 1024)**2) for i in range(1025))
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.glob('*.csv'))}

def run():
    rows = []
    with (ROOT / 'data/xor.csv').open(encoding='utf-8') as f:
        for r in csv.DictReader(f):
            x = [float(r['x1']), float(r['x2'])]; y = int(r['y'])
            h, score = fixed_xor(x); _, linear_score = fixed_xor(x, False)
            rows.append(dict(x=x, y=y, hidden=h, score=score, prediction=int(score >= .5), linear_score=linear_score, linear_prediction=int(linear_score >= .5)))
    rng = random.Random(83021)
    w1 = [[rng.uniform(-1, 1) for _ in range(2)] for _ in range(3)]; b1 = [rng.uniform(-1, 1) for _ in range(3)]
    w2 = [[rng.uniform(-1, 1) for _ in range(3)] for _ in range(2)]; b2 = [rng.uniform(-1,1) for _ in range(2)]
    w, b = collapse(w1, b1, w2, b2)
    points = [[rng.uniform(-2,2), rng.uniform(-2,2)] for _ in range(100)]
    differences = [abs(a-c) for x in points for a,c in zip(affine(affine(x,w1,b1),w2,b2), affine(x,w,b))]
    best = 0; witness = None
    for a in range(-4,5):
        for b_ in range(-4,5):
            for c in range(-4,5):
                correct = sum(int(a*r['x'][0] + b_*r['x'][1] + c >= 0) == r['y'] for r in rows)
                if correct > best: best, witness = correct, [a,b_,c]
    square_rows = list(csv.DictReader((ROOT/'data/square-grid.csv').open(encoding='utf-8')))
    interpolation = []
    for n in [2,4,8,16]:
        err = max(abs(interpolate_square(float(r['x']),n)-float(r['y'])) for r in square_rows)
        interpolation.append(dict(segments=n, hidden_hinges=n-1, max_grid_error=err, exact_interval_max_error=1/(4*n*n)))
    return dict(lesson='083', python=platform.python_version(), seed=83021, protocol='Fixed weights; no training or generalization estimate.', xor=rows,
        fixed_accuracy=sum(r['prediction']==r['y'] for r in rows)/4,
        removed_activation_accuracy=sum(r['linear_prediction']==r['y'] for r in rows)/4,
        affine_collapse_max_abs_error=max(differences), affine_trials=100,
        finite_linear_search=dict(triples=9**3, best_accuracy=best/4, witness=witness, warning='Finite search is a check, not the impossibility proof.'),
        parameter_counts={'2-2-1':parameter_count([2,2,1]), '2-4-3-1':parameter_count([2,4,3,1])},
        same_vertices_different_extensions=[dict(x=x, triangular=fixed_xor(x)[1], absolute_difference=alternative_xor(x)) for x in [[.5,.5],[.25,.25],[2.,2.]]],
        square_interpolation=interpolation,
        data_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'data').glob('*.csv'))})

if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--out',default='outputs/result.json'); args=parser.parse_args()
    result=run(); target=Path(args.out); target.parent.mkdir(parents=True,exist_ok=True); target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
