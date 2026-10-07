"""Analytic dual solutions, bias edge case, PSD counterexample and independent replay."""
import argparse,copy,hashlib,json,tempfile
from pathlib import Path
import numpy as np
from common import ROOT,load_data,safe_output,write_json
from dual import polynomial_features,gram,validate_gram,solve_dual,recover_bias
from experiment import run,arrays

def test(r):
    checks=[]
    def check(ok,name):
        if not bool(ok):raise RuntimeError('FAILED: '+name)
        checks.append(name)
    def rejects(fn,name):
        try:fn()
        except (ValueError,RuntimeError):check(True,name);return
        raise RuntimeError('DID NOT REJECT: '+name)
    X=np.array([[-1.],[1.]]);y=np.array([-1.,1.]);K=X@X.T
    for C in [.1,.5,1.,3.]:
        d=solve_dual(K,y,C);a=min(C,.5);check(np.allclose(d['alpha'],a,atol=1e-8),'two-point alpha '+str(C));check(abs(d['audit']['dual_value']-(2*a-2*a*a))<1e-8,'two-point dual value '+str(C));check(abs(d['audit']['gap'])<1e-7,'two-point gap '+str(C))
    bound=r['all_bound_two_points'];check(bound['bias_recovery']['free_count']==0,'all bound has no free SV');check(np.allclose([bound['bias_recovery']['lower'],bound['bias_recovery']['upper']],[-.8,.8]),'bound bias interval');check(abs(bound['b'])<1e-8,'bound midpoint b')
    # Non-uniqueness of b is real: both interior choices give the same optimal primal value.
    for b in [-.5,0,.5]:
        g=K@(np.array([.1,.1])*y);P=.5*.2**2+.1*np.maximum(0,1-y*(g+b)).sum();check(np.isclose(P,.18),'bound interval objective '+str(b))
    xx=np.array([[1.,2.],[-.7,.3],[0.,1.]]);phi=polynomial_features(xx);check(np.allclose(gram(xx),phi@phi.T),'explicit Gram identity')
    q=np.array([[2.,-1.],[.4,.8]]);check(np.allclose(gram(q,xx),polynomial_features(q)@phi.T),'rectangular prediction Gram identity')
    rejects(lambda:validate_gram([[1,2],[2,1]]),'indefinite symmetric similarity');rejects(lambda:validate_gram([[1,0],[1,1]]),'nonsymmetric Gram');rejects(lambda:validate_gram([[np.nan]]),'NaN Gram');rejects(lambda:validate_gram(np.ones((2,3))),'nonsquare Gram')
    for C in [0,-1,np.inf,np.nan]:rejects(lambda c=C:solve_dual(K,y,c),'invalid C '+str(C))
    rejects(lambda:solve_dual(K,[0,1],1),'label convention');rejects(lambda:polynomial_features([[1,2,3]]),'map dimension')
    data=load_data();Xt,yt=arrays(data['train']);Kt=gram(Xt);a=np.array(r['dual']['alpha']);ay=a*yt;dd=1.*a.sum()-.5*ay@Kt@ay;check(np.isclose(dd,r['dual']['audit']['dual_value'],atol=1e-10),'independent dual value')
    w=polynomial_features(Xt).T@ay;check(np.allclose(w,r['dual_reconstructed_w']),'w stationarity reconstruction');check(abs(yt@a)<1e-8,'bias stationarity equality')
    m=yt*(Kt@ay+r['dual']['b']);xi=np.maximum(0,1-m);check(np.allclose(xi,r['dual']['slack']),'minimal primal slack')
    for name in ['gap','equality_residual','box_violation','margin_complementarity','slack_complementarity']:check(abs(r['dual']['audit'][name])<1e-7,'KKT '+name)
    check(r['primal_dual_objective_difference']<1e-7,'independently solved primal versus dual');check(r['gram_equivalence_error']<1e-12,'feature map exact')
    for split in ['validation','test']:
        for key in ['max_own_explicit_difference','max_explicit_kernel_difference','max_native_precomputed_difference']:check(r[split][key]<1e-5,'prediction equivalence '+split+' '+key)
    check(not r['test_used_for_selection'],'sealed evaluation declaration')
    with tempfile.TemporaryDirectory() as td:
        dst=Path(td)
        for p in (ROOT/'data').iterdir():
            if p.is_file():(dst/p.name).write_bytes(p.read_bytes())
        q=dst/'train.csv';q.write_text(q.read_text().replace('train-0000','changed-0000'));j=json.loads((dst/'generation.json').read_text());j['sha256']['train.csv']=hashlib.sha256(q.read_bytes()).hexdigest();(dst/'generation.json').write_text(json.dumps(j));rejects(lambda:load_data(dst),'CSV and digest tamper')
        target=dst/'target';target.write_text('safe');link=dst/'link';link.symlink_to(target);rejects(lambda:safe_output(link),'unsafe symlink output')
    fresh=run()
    def compare(a,b,path='report'):
        if isinstance(a,dict):
            check(set(a)==set(b),'keys '+path)
            for k in a:compare(a[k],b[k],path+'.'+k)
        elif isinstance(a,list):
            try:arr=np.asarray(a,dtype=float);other=np.asarray(b,dtype=float)
            except (ValueError,TypeError):arr=None
            if arr is not None:check(np.allclose(arr,other,rtol=1e-7,atol=1e-7),'array replay '+path)
            else:
                check(len(a)==len(b),'length '+path)
                for i,(v,w) in enumerate(zip(a,b)):compare(v,w,path+'.'+str(i))
        elif isinstance(a,(int,float)) and not isinstance(a,bool):check(np.isclose(a,b,rtol=1e-7,atol=1e-7),'scalar replay '+path)
        else:check(a==b,'value replay '+path)
    compare(r,fresh);bad=copy.deepcopy(r);bad['dual']['b']+=.1;rejects(lambda:compare(bad,fresh),'tampered report')
    return {'status':'passed','checks':len(checks),'optimized':not __debug__,'fresh_experiment_replayed':True,'check_names':checks}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',default=str(ROOT/'experiment-result.json'));p.add_argument('--out',default=str(ROOT/'outputs/tests.json'));a=p.parse_args();r=test(json.loads(Path(a.report).read_text()));write_json(safe_output(a.out),r);print({k:v for k,v in r.items() if k!='check_names'})
