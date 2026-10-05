"""ML046: fixed-score classification metrics. No fitting or threshold selection.

Numerical input contract is documented in input_contract.json. Paths are relative
only to this file, never to the caller's working directory. Import has no writes.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import platform
import numpy as np
import sklearn
from sklearn import metrics as skm

ROOT = Path(__file__).resolve().parent

def require(condition, message):
    if not bool(condition):
        raise ValueError(message)

def validate(y, scores, weights=None):
    y0, s0 = np.asarray(y), np.asarray(scores)
    require(y0.ndim == s0.ndim == 1, 'labels and scores must be one-dimensional')
    require(2 <= len(y0) <= 100_000 and len(y0) == len(s0), 'lengths must match, 2..100000')
    require(y0.dtype.kind in 'biuf' and s0.dtype.kind in 'iuf', 'real numeric arrays required')
    require(np.isfinite(y0).all() and np.isfinite(s0).all(), 'finite labels and scores required')
    require(np.isin(y0, [0, 1]).all() and len(np.unique(y0)) == 2, 'both binary classes required')
    require(((s0 >= 0) & (s0 <= 1)).all(), 'scores must be in [0,1]')
    if weights is None:
        w = np.ones(len(y0), dtype=float)
    else:
        raw = np.asarray(weights)
        require(raw.ndim == 1 and len(raw) == len(y0) and raw.dtype.kind in 'iuf', 'invalid weights shape/type')
        require(np.isfinite(raw).all() and ((raw >= 1e-6) & (raw <= 1e6)).all(), 'weights must be in [1e-6,1e6]')
        w = raw.astype(float)
    return y0.astype(int), s0.astype(float), w

def validate_threshold(t):
    require(isinstance(t, (int, float, np.integer, np.floating)) and not isinstance(t, (bool, np.bool_)), 'numeric scalar threshold required')
    require(not math.isnan(float(t)) and (0 <= float(t) <= 1 or math.isinf(float(t))), 'threshold must be in [0,1] or +/-inf')
    return float(t)

def load_case():
    path = ROOT / 'data/review_scores.csv'
    integrity = json.loads((ROOT / 'data_integrity.json').read_text())
    require(hashlib.sha256(path.read_bytes()).hexdigest() == integrity['sha256'], 'fixed data hash mismatch')
    with path.open(newline='') as f:
        records = list(csv.DictReader(f))
    ids = [r['id'] for r in records]
    require(len(ids) == len(set(ids)), 'duplicate sample id')
    y = np.array([int(r['label']) for r in records])
    scores = np.array([int(r['score_percent']) / 100 for r in records])
    validate(y, scores)
    return ids, y, scores

def safe_ratio(numerator, denominator, default=0.0):
    return float(numerator / denominator) if denominator else float(default)

def counts_at(y, scores, threshold, weights=None):
    y, scores, w = validate(y, scores, weights)
    t = validate_threshold(threshold)
    pred = scores >= t
    tp = float(w[(y == 1) & pred].sum())
    fp = float(w[(y == 0) & pred].sum())
    fn = float(w[(y == 1) & ~pred].sum())
    tn = float(w[(y == 0) & ~pred].sum())
    return {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
            'precision': safe_ratio(tp, tp + fp),
            'precision_defined': bool(tp + fp),
            'recall': tp / (tp + fn), 'fpr': fp / (fp + tn),
            'specificity': tn / (fp + tn),
            'f1': safe_ratio(2 * tp, 2 * tp + fp + fn),
            'accuracy': (tp + tn) / w.sum()}

def sweep(y, scores, weights=None):
    y, scores, w = validate(y, scores, weights)
    # Ties move as one group; no intermediate point from arbitrary row order.
    thresholds = np.r_[np.inf, np.unique(scores)[::-1]]
    return [{'threshold': 'inf' if np.isinf(t) else float(t),
             **counts_at(y, scores, t, w)} for t in thresholds]

def prior_weights(y, scores, prevalence):
    y, _, _ = validate(y, scores)
    require(isinstance(prevalence, (int, float)) and not isinstance(prevalence, bool), 'numeric prevalence required')
    require(math.isfinite(prevalence) and .01 <= prevalence <= .99, 'prevalence must be in [.01,.99]')
    original = float(y.mean())
    return np.where(y == 1, prevalence / original, (1 - prevalence) / (1 - original))

def library_metrics(y, scores, weights=None):
    y, s, w = validate(y, scores, weights)
    fpr, tpr, rt = skm.roc_curve(y, s, pos_label=1, sample_weight=w, drop_intermediate=False)
    precision, recall, pt = skm.precision_recall_curve(y, s, pos_label=1, sample_weight=w, drop_intermediate=False)
    return {'roc': {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'thresholds': ['inf' if np.isinf(x) else float(x) for x in rt]},
            'pr': {'precision': precision.tolist(), 'recall': recall.tolist(), 'thresholds': pt.tolist()},
            'auc': float(skm.roc_auc_score(y, s, sample_weight=w)),
            'ap': float(skm.average_precision_score(y, s, pos_label=1, sample_weight=w)),
            'pr_trapezoid': float(skm.auc(recall, precision))}

def average_metrics(y, scores, threshold=.8):
    y, s, _ = validate(y, scores)
    pred = s >= validate_threshold(threshold)
    result = {}
    for average in ['binary', 'macro', 'micro', 'weighted', None]:
        p, r, f, support = skm.precision_recall_fscore_support(
            y, pred, labels=[0, 1], pos_label=1, beta=1.0,
            average=average, zero_division=0)
        result[str(average)] = {k: np.asarray(v).tolist() if v is not None else None
                                for k, v in zip(['precision', 'recall', 'f1', 'support'], [p, r, f, support])}
    return result

def run():
    ids, y, scores = load_case()
    base = library_metrics(y, scores)
    shifted = {}
    for pi in [.1, .25, .5]:
        weights = prior_weights(y, scores, pi)
        shifted[str(pi)] = {'weights': weights.tolist(),
                            'at_080': counts_at(y, scores, .8, weights),
                            'curves': library_metrics(y, scores, weights)}
    # Monotone transform is an illustration, not a fitted calibrator.
    transformed = scores ** 3
    calibration = {}
    for name, s in [('original', scores), ('cubed', transformed)]:
        calibration[name] = {'auc': skm.roc_auc_score(y, s),
                             'ap': skm.average_precision_score(y, s, pos_label=1),
                             'log_loss_as_probability': skm.log_loss(y, s, labels=[0, 1]),
                             'brier_as_probability': skm.brier_score_loss(y, s, pos_label=1, scale_by_half=True),
                             'at_080': counts_at(y, s, .8)}
    calibration['mapped_threshold'] = counts_at(y, transformed, .8 ** 3)
    return {'unit': '046', 'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'sklearn': sklearn.__version__},
            'protocol': 'fixed teaching scores; >= threshold; ties grouped; no fitting/selection; label 1 positive',
            'ids': ids, 'labels': y.tolist(), 'scores': scores.tolist(),
            'sweep': sweep(y, scores), 'curves': base,
            'at_080': counts_at(y, scores, .8), 'at_055': counts_at(y, scores, .55),
            'averages_080': average_metrics(y, scores), 'prior_shift': shifted,
            'monotone': calibration}

def output_dir(path):
    raw = Path(path).absolute()
    for component in (raw, *raw.parents):
        require(not component.is_symlink(), 'symlink output component')
    resolved = raw.resolve()
    require(resolved != ROOT, 'choose outputs/ or a separate output directory')
    if resolved.is_relative_to(ROOT):
        require(resolved.is_relative_to(ROOT / 'outputs'), 'inside unit, outputs must stay under outputs/')
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved

def write_json(path, value):
    require(not path.is_symlink() and (not path.exists() or (path.is_file() and path.stat().st_nlink == 1)), 'unsafe output file')
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=str(ROOT / 'outputs'))
    args = parser.parse_args()
    dest = output_dir(args.out)
    result = run()
    write_json(dest / 'experiment-result.json', result)
    rows = result['sweep']
    path = dest / 'threshold-sweep.csv'
    require(not path.is_symlink() and (not path.exists() or (path.is_file() and path.stat().st_nlink == 1)), 'unsafe csv output')
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({'auc': result['curves']['auc'], 'ap': result['curves']['ap'],
                      'pr_trapezoid': result['curves']['pr_trapezoid'], 'output': str(dest)}, ensure_ascii=False))
if __name__ == '__main__':
    main()
