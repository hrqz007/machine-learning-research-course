"""Independent exact Fraction references + finite, curriculum-relevant cases.

No Python assert statements: python -O runs the same checks. This is a learner
self-check, not an independent publication-quality review.
"""
from fractions import Fraction as F
from decimal import Decimal, localcontext
from itertools import product
from pathlib import Path
import argparse, csv, json
import numpy as np
from sklearn import metrics as skm
from experiment import ROOT, load_case, validate, counts_at, library_metrics, prior_weights, average_metrics, output_dir, write_json

def check(ok, message):
    if not bool(ok):
        raise RuntimeError(message)

def close(actual, expected, label):
    check(np.allclose(actual, np.asarray(expected, dtype=float), rtol=2e-13, atol=2e-14), label)

def fraction_reference(y, integer_scores, weights=None):
    # Enumeration from raw integer scores, not experiment.sweep or sklearn.
    w = [F(1) for _ in y] if weights is None else list(weights)
    positive = sum((w[i] for i in range(len(y)) if y[i] == 1), F())
    negative = sum((w[i] for i in range(len(y)) if y[i] == 0), F())
    thresholds = sorted(set(integer_scores), reverse=True)
    rows = []
    for t in [None] + thresholds:
        selected = [False if t is None else s >= t for s in integer_scores]
        tp = sum((wi for yi, pi, wi in zip(y, selected, w) if yi == 1 and pi), F())
        fp = sum((wi for yi, pi, wi in zip(y, selected, w) if yi == 0 and pi), F())
        fn, tn = positive - tp, negative - fp
        p = tp / (tp + fp) if tp + fp else F(1)  # PR plotting sentinel, not scalar precision.
        r = tp / positive
        rows.append({'t': t, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'p': p, 'r': r, 'fpr': fp / negative})
    ap = sum(((b['r'] - a['r']) * b['p'] for a, b in zip(rows, rows[1:])), F())
    roc = sum(((b['fpr'] - a['fpr']) * (a['r'] + b['r']) / 2 for a, b in zip(rows, rows[1:])), F())
    pr_trapezoid = sum(((b['r'] - a['r']) * (a['p'] + b['p']) / 2 for a, b in zip(rows, rows[1:])), F())
    pair = F()
    for i, yi in enumerate(y):
        if yi != 1: continue
        for j, yj in enumerate(y):
            if yj != 0: continue
            merit = F(1) if integer_scores[i] > integer_scores[j] else F(1, 2) if integer_scores[i] == integer_scores[j] else F(0)
            pair += w[i] * w[j] * merit
    pair /= positive * negative
    check(pair == roc, 'independent pair count vs exact ROC integration')
    return {'rows': rows, 'ap': ap, 'auc': pair, 'pr_trapezoid': pr_trapezoid}

def check_case(y, integer_scores, weights=None):
    s = np.array(integer_scores, dtype=float) / 100
    w = None if weights is None else np.array(weights, dtype=float)
    reference = fraction_reference(y, integer_scores, weights)
    got = library_metrics(y, s, w)
    close([got[k] for k in ['auc', 'ap', 'pr_trapezoid']], [reference[k] for k in ['auc', 'ap', 'pr_trapezoid']], 'areas')
    close(got['roc']['fpr'], [r['fpr'] for r in reference['rows']], 'ROC FPR full sequence')
    close(got['roc']['tpr'], [r['r'] for r in reference['rows']], 'ROC TPR full sequence')
    check(got['roc']['thresholds'][0] == 'inf', 'ROC sentinel')
    close(got['roc']['thresholds'][1:], [r['t'] / 100 for r in reference['rows'][1:]], 'ROC thresholds')
    close(got['pr']['thresholds'], sorted(set(s)), 'PR thresholds increase')
    close(got['pr']['precision'], [r['p'] for r in reference['rows'][:0:-1]] + [F(1)], 'PR precision and sentinel')
    close(got['pr']['recall'], [r['r'] for r in reference['rows'][:0:-1]] + [F(0)], 'PR recall and sentinel')
    for r in reference['rows']:
        t = np.inf if r['t'] is None else r['t'] / 100
        c = counts_at(y, s, t, w)
        close([c[k] for k in ['TP', 'FP', 'FN', 'TN']], [r[k] for k in ['tp','fp','fn','tn']], 'all threshold counts')
        cm = skm.confusion_matrix(y, s >= t, labels=[0, 1], sample_weight=w)
        close(cm, [[r['tn'], r['fp']], [r['fn'], r['tp']]], 'library matrix orientation')
    return reference

def run_audit(exhaustive=True):
    ids, y, s = load_case()
    with (ROOT / 'data/review_scores.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    raw_s = [int(r['score_percent']) for r in rows]
    ref = check_case(y.tolist(), raw_s)
    check(ref['auc'] == F(43, 54) and ref['ap'] == F(9, 14) and ref['pr_trapezoid'] == F(79, 126), 'paper anchors')
    anchor = counts_at(y, s, .8)
    close([anchor[k] for k in ['TP','FP','FN','TN','precision','recall','f1','accuracy']], [2,2,1,7,F(1,2),F(2,3),F(4,7),F(3,4)], 'threshold .8 anchors')
    avg = average_metrics(y, s)
    close([avg[a]['f1'] for a in ['binary','macro','micro','weighted']], [F(4,7),F(83,119),F(3,4),F(181,238)], 'four F1 averages')
    prior_refs = {}
    for pi in [F(1,10),F(1,4),F(1,2)]:
        w = [pi/F(1,4) if yi else (1-pi)/F(3,4) for yi in y]
        result = check_case(y.tolist(), raw_s, w)
        close(prior_weights(y,s,float(pi)),w,'analytic prior weights')
        check(result['auc'] == F(43,54), 'class-conditional weighting preserves AUC')
        prior_refs[str(pi)] = {key:str(result[key]) for key in ['auc','ap','pr_trapezoid']}
    # High-precision loss reference, independent of sklearn and NumPy arithmetic.
    loss_reference = {}
    for power in [1,3]:
        probs = [F(si,100)**power for si in raw_s]
        exact_brier = sum(((pi-int(yi))**2 for pi,yi in zip(probs,y)),F()) / len(y)
        with localcontext() as ctx:
            ctx.prec = 60
            decimals = [Decimal(pi.numerator)/Decimal(pi.denominator) for pi in probs]
            exact_logloss = -sum((pi.ln() if yi else (1-pi).ln() for pi,yi in zip(decimals,y)),Decimal(0))/Decimal(len(y))
        got_s = s**power
        close(skm.brier_score_loss(y,got_s,pos_label=1,scale_by_half=True),float(exact_brier),'exact Brier')
        close(skm.log_loss(y,got_s,labels=[0,1]),float(exact_logloss),'Decimal log loss')
        loss_reference[str(power)]={'brier':str(exact_brier),'log_loss':str(exact_logloss)}
    # Tied rows can exchange order without changing any metric.
    permutation = [11,6,3,2,10,0,9,1,8,4,7,5]
    perm = check_case(y[permutation].tolist(),np.array(raw_s)[permutation].tolist())
    check(perm['ap'] == ref['ap'] and perm['auc'] == ref['auc'],'permutation invariance')
    a,b = library_metrics(y,s),library_metrics(y,s**3)
    close([a['auc'],a['ap']],[b['auc'],b['ap']],'strictly increasing transform')
    check(counts_at(y,s,.8) == counts_at(y,s**3,.8**3), 'mapped threshold predictions')
    # All-tied scores expose the difference between an attainable point and a line segment.
    tied = check_case([0,0,0,1],[50]*4)
    check(tied['auc'] == F(1,2) and tied['ap'] == F(1,4), 'all ties')
    check_case([0,1],[0,100]);check_case([0,1],[100,0])
    check(counts_at(y,s,np.inf)['precision_defined'] is False,'undefined scalar precision flagged')
    check(counts_at(y,s,-np.inf)['TP'] == 3,'all positive endpoint')
    # Exact equality vs just above a tied threshold changes the entire tie group.
    close([counts_at(y,s,.8)[k]-counts_at(y,s,np.nextafter(.8,np.inf))[k] for k in ['TP','FP']], [1,1],'tie boundary inclusion')
    rejected = []
    invalid = [([],[]),([0,1],[.2]),([0,0],[.2,.4]),([0,2],[.2,.4]),([0,1],[np.nan,.4]),([0,1],[.2,1.1]),([[0,1]],[.2,.4]),([0,1],['.2','.4'])]
    for i,(yi,si) in enumerate(invalid):
        try: validate(yi,si)
        except ValueError: rejected.append('input_'+str(i))
        else: raise RuntimeError('invalid input accepted')
    for wi in [[1,0],[1,-1],[1,np.inf],[1],[1,'a'],[1,1e-9]]:
        try: validate([0,1],[.2,.4],wi)
        except ValueError: rejected.append('weights_'+str(len(rejected)))
        else: raise RuntimeError('invalid weights accepted')
    for ti in [np.nan, -.1, 1.1, True, '.8']:
        try: counts_at(y,s,ti)
        except ValueError: rejected.append('threshold_'+str(len(rejected)))
        else: raise RuntimeError('invalid threshold accepted')
    exhaustive_count = 0
    if exhaustive:
        # Every binary label vector containing both classes and every 3-level
        # score vector at n=4: 14*81=1134. Relevant small tie/order patterns.
        for yi in product([0,1], repeat=4):
            if len(set(yi)) != 2: continue
            for si in product([0,50,100], repeat=4):
                check_case(list(yi),list(si));exhaustive_count += 1
    return {'status':'passed','reference':'Fraction arithmetic, independent threshold enumeration and positive-negative pair enumeration',
            'anchors':{k:str(ref[k]) for k in ['auc','ap','pr_trapezoid']},'prior_exact':prior_refs,
            'probability_loss_references':loss_reference,'exhaustive_small_cases':exhaustive_count,'invalid_inputs_rejected':len(rejected),
            'assert_statements_used':False,'scope':'author self-check only; fixed empirical scores, not population performance proof'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(ROOT/'outputs'));p.add_argument('--quick',action='store_true');a=p.parse_args()
    result=run_audit(not a.quick);dest=output_dir(a.out);write_json(dest/'audit-result.json',result)
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
