"""Hand examples, ordinary failures, library agreement and independent report audit."""
import argparse,json
import numpy as np
from numpy.testing import assert_allclose,assert_array_equal
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from knn import KNN
from common import ROOT,require,write_json

def audit(r):
    # x=1 chooses x=0 and x=2; inverse-distance weights are equal.
    X=np.array([[0.],[2.],[5.]]);q=np.array([[1.]])
    assert_allclose(KNN(2,task='regression').fit(X,[0,4,10]).predict(q),[2.])
    assert_allclose(KNN(3,'distance','regression').fit(X,[0,4,10]).predict(q),[26/9])
    # Exact duplicates: zero-distance labels alone vote, with no divide-by-zero.
    z=KNN(3,'distance').fit([[0.],[0.],[2.]],[0,1,1]);assert_allclose(z.predict_proba([[0.]]),[[.5,.5]]);assert_array_equal(z.predict([[0.]]),[0])
    tie=KNN(1).fit([[-1.],[1.]],[1,0]);assert_array_equal(tie.predict([[0.]]),[1])
    rng=np.random.default_rng(5711);X=rng.normal(size=(35,4));Q=rng.normal(size=(13,4));y=rng.integers(0,2,35);yr=rng.normal(size=35)
    comparisons=0
    for k in [1,4,35]:
        for weights in ['uniform','distance']:
            a=KNN(k,weights).fit(X,y);b=KNeighborsClassifier(n_neighbors=k,weights=weights,algorithm='brute').fit(X,y)
            assert_allclose(a.predict_proba(Q),b.predict_proba(Q),atol=1e-9,rtol=0)
            a=KNN(k,weights,'regression').fit(X,yr);b=KNeighborsRegressor(n_neighbors=k,weights=weights,algorithm='brute').fit(X,yr)
            assert_allclose(a.predict(Q),b.predict(Q),atol=1e-9,rtol=0);comparisons+=2
    failures=[lambda:KNN(0),lambda:KNN(1.5),lambda:KNN(True),lambda:KNN(2).fit([[1]],[0]),lambda:KNN().predict([[0]]),lambda:KNN(1).fit([[np.nan]],[0]),lambda:KNN(1).fit([[0]],[2]),lambda:KNN(1).fit([[0],[1]],[0]),lambda:KNN(1).fit([[0]],[0]).predict([[0,1]])]
    for f in failures:
        try:f()
        except ValueError:pass
        else:raise AssertionError('common invalid input not rejected')
    arrays=np.load(ROOT/'data/draws.npz');X=arrays['X'];labels=arrays['y'];noise=arrays['noise']
    require(np.array_equal(np.asarray(r['classification_test_y']),labels[540:]), 'test labels differ from data')
    require(np.array_equal(np.asarray(r['regression_test_y']),arrays['yr'][180:]), 'regression labels differ from data')
    for row in r['classification']:
        route=row['route'];Z=np.column_stack([X,noise]) if route=='scaled_noise20' else X.copy()
        if route!='raw':
            scaler=StandardScaler().fit(Z[:360])
            assert_allclose(scaler.mean_,r['scalers'][route]['mean'],atol=1e-12,rtol=0)
            assert_allclose(scaler.scale_,r['scalers'][route]['scale'],atol=1e-12,rtol=0)
            Z=scaler.transform(Z)
        for candidate in row['candidates']:
            reference=KNeighborsClassifier(n_neighbors=candidate['k'],weights=row['weights'],algorithm='brute').fit(Z[:360],labels[:360])
            probability=reference.predict_proba(Z[360:540])[:,1]
            assert_allclose(probability,candidate['validation_probability'],atol=1e-9,rtol=0)
            assert_allclose(np.mean((probability>.5)!=labels[360:540]),candidate['validation_error'],atol=1e-15)
        reference=KNeighborsClassifier(n_neighbors=row['selected_k'],weights=row['weights'],algorithm='brute').fit(Z[:360],labels[:360])
        assert_allclose(reference.predict_proba(Z[540:])[:,1],row['test_probability'],atol=1e-9,rtol=0)
    y=np.array(r['classification_test_y'])
    for row in r['classification']:
        best=min(row['candidates'],key=lambda v:(v['validation_error'],v['k']));require(best['k']==row['selected_k'],'test-driven or wrong k choice')
        assert_allclose(np.mean((np.array(row['test_probability'])>.5)!=y),row['test_error'],atol=1e-15);require(row['library_max_error']<1e-9,'library classification discrepancy')
    for row in r['regression']:
        best=min(row['candidates'],key=lambda v:(v['validation_mse'],v['k']));require(best['k']==row['selected_k'],'wrong regression k')
        assert_allclose(np.mean((np.array(row['test_prediction'])-r['regression_test_y'])**2),row['test_mse'],atol=1e-15);require(row['library_max_error']<1e-9,'library regression discrepancy')
    truth=np.sin(2*np.array(r['bias_variance_grid']))
    for row in r['bias_variance']:
        p=np.array(row['predictions']);b=np.mean((p.mean(0)-truth)**2);v=np.mean(p.var(0));err=np.mean((p-truth)**2)
        assert_allclose([b,v,err],[row['squared_bias'],row['variance'],row['mean_squared_error_to_truth']],atol=1e-12);assert_allclose(b+v,err,atol=1e-12)
    return dict(status='passed',independently_reconstructed_validation_candidates=24,training_only_scalers=2,hand_cases=4,random_library_comparisons=comparisons,common_invalid_inputs=len(failures),report_routes=8,bias_variance_identities=4,library_tolerance=1e-9)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',default=str(ROOT/'experiment-result.json'));p.add_argument('--out',default=str(ROOT/'outputs/test-result.json'));a=p.parse_args();r=audit(json.loads(open(a.report).read()));write_json(a.out,r);print(r)
