"""Audit published/rebuilt ML056 results independently of their summary fields."""
import argparse,json,copy
import numpy as np
from numpy.testing import assert_allclose
from common import ROOT,require,write_json
from experiment import load_rows,forward,cost_vector,metrics

def audit(r,data_dir=ROOT/'data',negative_checks=True):
    X,y,g,ids=load_rows(data_dir/'development.csv');lookup={v:i for i,v in enumerate(ids)}
    fits=0;scores=0
    selections=[a['selection'] for a in r['development']['outer']]+[r['development']['final_selection']]
    for s in selections:
        sy=np.array(s['y']);sids=np.array(s['row_ids']);where={v:i for i,v in enumerate(sids)}
        require(len(sids)==len(set(sids.tolist())), 'selection IDs must be unique')
        require(set(sids.tolist()) <= set(ids.tolist()), 'selection IDs outside development')
        require(len(sy)==len(sids), 'selection label length mismatch')
        assert_allclose(sy, y[[lookup[i] for i in sids]], atol=0, rtol=0)
        all_choices=[]
        for c in s['ledger']:
            p=np.array(c['oof_probability']);seen=[]
            for f in c['fits']:
                train_ids=f['train_ids'];valid_ids=f['valid_ids']
                require(len(train_ids)==len(set(train_ids)) and len(valid_ids)==len(set(valid_ids)), 'inner duplicate IDs')
                require(set(train_ids) <= set(sids.tolist()) and set(valid_ids) <= set(sids.tolist()), 'inner IDs outside selection')
                require(set(train_ids) | set(valid_ids) == set(sids.tolist()), 'inner fold does not partition selection')
                tr=np.array([lookup[i] for i in f['train_ids']]);va=np.array([lookup[i] for i in f['valid_ids']]);seen.extend(f['valid_ids'])
                require(not set(f['train_ids'])&set(f['valid_ids']),'inner overlap')
                assert_allclose(np.nanmedian(X[tr],axis=0),f['state']['imputer'],atol=1e-12)
                filled=np.where(np.isnan(X[tr]),f['state']['imputer'],X[tr])
                assert_allclose(filled.mean(0),f['state']['mean'],atol=1e-10)
                assert_allclose(filled.std(0),f['state']['scale'],atol=1e-10)
                assert_allclose(forward(X[va],f['state']),p[[where[i] for i in f['valid_ids']]],atol=1e-12)
                fits+=1
            require(sorted(seen)==sorted(sids.tolist()),'OOF rows must each be predicted once')
            for score in c['scores']:
                actual=float(cost_vector(sy,p,score['threshold']).mean());assert_allclose(actual,score['cost'],atol=1e-15)
                all_choices.append((actual,c['degree'],c['C'],score['threshold']));scores+=1
        best=min(all_choices);b=s['best'];require(best==(b['cost'],b['degree'],b['C'],b['threshold']),'selection/tie mismatch')
    seen=[]
    for out in r['development']['outer']:
        require(not set(out['train_ids'])&set(out['valid_ids']),'outer overlap')
        require(set(out['selection']['row_ids'])==set(out['train_ids']),'inner sees outer holdout')
        seen.extend(out['valid_ids']);fits+=1
    require(sorted(seen)==sorted(ids.tolist()),'outer OOF coverage')
    Xt,yt,gt,it=load_rows(data_dir/'test.csv');t=r['test'];p=forward(Xt,r['development']['final_state']);assert_allclose(p,t['probability'],atol=1e-12)
    for key,val in metrics(yt,p,t['threshold']).items():
        if val is not None:assert_allclose(val,t['selected'][key],atol=1e-12)
    require(not set(ids)&set(it),'test overlap');require(fits+1==r['fit_count']==76,'fit budget mismatch')
    delta=cost_vector(yt,p,t['threshold'])-cost_vector(yt,np.array(t['baseline_probability']),.2)
    assert_allclose(delta.mean(),t['paired_difference'],atol=1e-15)
    require(t['paired_bootstrap_95'][0]<=t['paired_bootstrap_95'][1],'invalid interval')
    # A five-row hand example exercises the >= boundary and asymmetric costs.
    require(cost_vector(np.array([1,1,0,0,1]),np.array([.1,.2,.3,.0,.9]),.2).tolist()==[4,0,1,0,0],'decision boundary failure')
    for group in [0,1]:
        actual=metrics(yt[gt==group],p[gt==group],t['threshold'])
        for key in ['n','tp','fn','fp','tn','cost','recall']:assert_allclose(actual[key],t['groups'][str(group)][key])
    if negative_checks:
        # Alter a held-out training ID and a saved label independently.
        for mutation in ['outer_holdout_in_inner_training','selection_label_mismatch']:
            bad=copy.deepcopy(r);outer=bad['development']['outer'][0]
            if mutation=='outer_holdout_in_inner_training':
                outer['selection']['ledger'][0]['fits'][0]['train_ids'][0]=outer['valid_ids'][0]
            else:
                outer['selection']['y'][0]=1-outer['selection']['y'][0]
            try: audit(bad,data_dir,negative_checks=False)
            except (ValueError,AssertionError): pass
            else: raise AssertionError('negative mutation was not rejected: '+mutation)
    return dict(status='passed',negative_mutations_rejected=2 if negative_checks else 0,independently_reconstructed_inner_fits=72,total_fits=76,threshold_candidates_checked=scores,checks=['fold isolation','OOF coverage','train-only median/mean/scale','plain-array probability forward','global tie-breaking','test metrics','paired difference','group counts','hand boundary'])
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--report',default=str(ROOT/'experiment-result.json'));a.add_argument('--out',default=str(ROOT/'outputs/test-result.json'));v=a.parse_args()
    out=audit(json.loads(open(v.report).read()));write_json(v.out,out);print(json.dumps(out,ensure_ascii=False))
