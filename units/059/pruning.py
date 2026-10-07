"""Cost-complexity pruning from transparent arrays, independently of sklearn's pruner.

The grower is sklearn. This module owns subtree enumeration, weakest-link
updates and prediction. No validation/test labels enter a pruning decision.
"""
from dataclasses import dataclass
import numpy as np
from common import finite_matrix, require

@dataclass
class TreeArrays:
    left: np.ndarray
    right: np.ndarray
    feature: np.ndarray
    threshold: np.ndarray
    count: np.ndarray
    impurity: np.ndarray
    probability: np.ndarray
    n_features: int

def export_tree(model):
    """Export a fitted binary 0/1 sklearn tree with probabilities aligned to class 1."""
    t=model.tree_; vals=t.value[:,0,:]; p=np.zeros(t.node_count)
    if 1 in model.classes_: p=vals[:,list(model.classes_).index(1)]/vals.sum(axis=1)
    return TreeArrays(t.children_left.copy(),t.children_right.copy(),t.feature.copy(),
                      t.threshold.copy(),t.n_node_samples.copy(),t.impurity.copy(),p,
                      model.n_features_in_)

def terminals(tree,node=0,stops=None):
    """Only reachable leaves count. A pruned node hides all its descendants."""
    stops=set() if stops is None else set(stops)
    if node in stops or tree.left[node]<0:return [int(node)]
    return terminals(tree,int(tree.left[node]),stops)+terminals(tree,int(tree.right[node]),stops)

def node_depths(tree):
    depth=np.zeros(len(tree.left),dtype=int); stack=[0]
    while stack:
        v=stack.pop()
        if tree.left[v]>=0:
            for w in (tree.left[v],tree.right[v]):depth[w]=depth[v]+1;stack.append(int(w))
    return depth

def reachable(tree,stops):
    answer=[];stack=[0]
    while stack:
        v=stack.pop();answer.append(v)
        if v not in stops and tree.left[v]>=0:stack.extend([int(tree.right[v]),int(tree.left[v])])
    return answer

def risk(tree,stops):
    leaves=terminals(tree,stops=stops)
    return float(np.sum(tree.count[leaves]*tree.impurity[leaves])/tree.count[0])

def weakest_link_path(tree,tol=1e-12):
    """Return nested terminal sets. Equal weakest links are removed together.

    Risk uses root-normalized Gini, not misclassification rate. Only subtrees
    of this fixed grown tree are searched; no alternative splits are invented.
    """
    stops=set();depth=node_depths(tree); rows=[];alpha=0.;removed=[]
    while True:
        leaves=terminals(tree,stops=stops);candidates=[]
        for v in reachable(tree,stops):
            if v in leaves:continue
            sub=terminals(tree,v,stops)
            collapsed=tree.count[v]*tree.impurity[v]/tree.count[0]
            current=np.sum(tree.count[sub]*tree.impurity[sub])/tree.count[0]
            candidates.append({'node':v,'leaves':len(sub),'alpha':float((collapsed-current)/(len(sub)-1))})
        rows.append({'alpha':float(alpha),'leaves':leaves,'n_leaves':len(leaves),'depth':int(max(depth[leaves])),
                     'risk':risk(tree,stops),'removed':removed,'candidates':candidates})
        if not candidates:break
        nxt=min(c['alpha'] for c in candidates)
        require(nxt>=alpha-1e-10,'decreasing effective alpha indicates invalid tree')
        alpha=max(alpha,nxt);removed=[]
        # Parent-first makes overlapping ties unambiguous.
        for c in sorted(candidates,key=lambda q:(depth[q['node']],q['node'])):
            v=c['node']
            if abs(c['alpha']-nxt)<=tol and v in reachable(tree,stops):
                stops.difference_update(terminals(tree,v,stops));stops.add(v);removed.append(v)
    return rows

def predict_probability(tree,X,leaves):
    X=finite_matrix(X);require(X.shape[1]==tree.n_features,'feature count mismatch')
    # sklearn's grower uses float32 input, so the exported predictor does too.
    with np.errstate(over='ignore'):X=X.astype(np.float32)
    require(np.isfinite(X).all(),'input not representable as finite float32')
    stops=set(map(int,leaves));valid=set(range(len(tree.left)))
    require(stops and stops<=valid,'invalid terminal indices')
    require(set(terminals(tree,stops=stops))==stops,'terminals must form a reachable partition')
    out=[]
    for x in X:
        v=0
        while v not in stops:
            v=int(tree.left[v] if x[tree.feature[v]]<=tree.threshold[v] else tree.right[v])
        out.append(tree.probability[v])
    return np.asarray(out)

def select_validation(rows):
    require(len(rows)>0,'empty model candidates')
    # No test score is present in this function; ties prefer fewer leaves.
    return min(range(len(rows)),key=lambda k:(rows[k]['validation_brier'],rows[k]['n_leaves'],rows[k]['depth'],k))
