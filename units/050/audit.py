"""Independent Fraction, SciPy QR and sklearn checks; no assert dependency."""
from pathlib import Path
import argparse, hashlib, json, tempfile
from fractions import Fraction as F
import numpy as np
from scipy import linalg
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GroupShuffleSplit, TimeSeriesSplit
import experiment as e
ROOT=Path(__file__).resolve().parent

def check(ok,msg):
    if not ok:raise RuntimeError(msg)

def exact_hand():
    X=[[F(v) for v in r] for r in [[1,0,1,0],[1,1,1,0],[1,0,0,1],[1,1,0,1]]];y=list(map(F,[1,3,2,4]));n=4
    grad=[-sum(X[i][j]*y[i] for i in range(n))/n for j in range(4)];theta=[-F(1,5)*g for g in grad]
    nxt=[sum(a*b for a,b in zip(r,theta)) for r in X]
    J=(sum((pred-target)**2 for pred,target in zip(nxt,y))+sum(v*v for v in theta[1:]))/(2*n)
    # Independent exact Gauss-Jordan solution of ridge normal equations.
    aug=[[sum(X[i][j]*X[i][k] for i in range(n))+(1 if j==k and j>0 else 0) for k in range(4)]+[sum(X[i][j]*y[i] for i in range(n))] for j in range(4)]
    for j in range(4):
        pivot=next(i for i in range(j,4) if aug[i][j]);aug[j],aug[pivot]=aug[pivot],aug[j]
        q=aug[j][j];aug[j]=[v/q for v in aug[j]]
        for i in range(4):
            if i!=j:
                q=aug[i][j];aug[i]=[a-q*b for a,b in zip(aug[i],aug[j])]
    return {'gradient':[str(g) for g in grad],'theta_next':[str(v) for v in theta],'next_prediction':[str(v) for v in nxt],'next_objective':str(J),'ridge_optimum':[str(r[-1]) for r in aug]}

def independent_fit(p,tr,te,model,alpha):
    # Encodes groups independently via actual sklearn API, no e.design or e.fit.
    traincols=[p['x'][tr,None]];testcols=[p['x'][te,None]]
    if model!='pooled':
        enc=OneHotEncoder(handle_unknown='ignore',sparse_output=False,dtype=float)
        traincols.append(enc.fit_transform(p['group'][tr,None]));testcols.append(enc.transform(p['group'][te,None]))
    if model=='trend':traincols.append(p['day'][tr,None]/47.);testcols.append(p['day'][te,None]/47.)
    X=np.column_stack(traincols);Xt=np.column_stack(testcols)
    model_sk=Ridge(alpha=alpha,fit_intercept=True,solver='svd').fit(X,p['y'][tr]);pred_sk=model_sk.predict(Xt)
    A=np.column_stack([np.ones(len(tr)),X]);At=np.column_stack([np.ones(len(te)),Xt]);P=np.eye(A.shape[1]);P[0,0]=0
    aug=np.vstack((A,np.sqrt(alpha)*P));target=np.r_[p['y'][tr],np.zeros(A.shape[1])]
    beta=linalg.lstsq(aug,target,lapack_driver='gelsy')[0];pred_qr=At@beta
    return pred_sk,pred_qr

def run(result):
    c,d,s=e.load_data();p=e.panel(c,d,0);refs=[]
    for split,(tr,te) in s.items():
        for model in e.MODELS:
            sk,qr=independent_fit(p,tr,te,model,c['ridge_alpha']);actual=np.array(result['main'][split]['models'][model]['predictions'])
            err_sk=float(np.max(np.abs(sk-actual)));err_qr=float(np.max(np.abs(qr-actual)))
            check(err_sk<2e-10 and err_qr<2e-10,'independent prediction mismatch')
            direct_mse=float(np.mean((sk-p['y'][te])**2));check(abs(direct_mse-result['main'][split]['models'][model]['mse'])<1e-10,'MSE mismatch')
            refs.append({'split':split,'model':model,'sklearn_max_prediction_error':err_sk,'scipy_QR_max_prediction_error':err_qr})
    exact=exact_hand();h=e.hand()
    check(np.allclose([float(F(v)) for v in exact['theta_next']],h['theta_next'],atol=1e-15),'hand update mismatch')
    check(abs(float(F(exact['next_objective']))-h['next_objective'])<1e-14,'hand objective mismatch')
    # Local derivative finite differences on the identical four-row objective.
    X=np.array([[1,0,1,0],[1,1,1,0],[1,0,0,1],[1,1,0,1]],float);y=np.array([1,3,2,4.]);th=np.array([.5,.35,.2,.3]);delta=1e-6
    def J(v):return ((X@v-y)@(X@v-y)+v[1:]@v[1:])/8
    gd=(X.T@(X@th-y)+np.r_[0.,th[1:]])/4
    fd=np.array([(J(th+np.eye(4)[j]*delta)-J(th-np.eye(4)[j]*delta))/(2*delta) for j in range(4)])
    check(np.max(np.abs(gd-fd))<1e-8,'finite difference mismatch')
    # Exact optimum versus numeric implementation, same penalty and columns.
    f=e.fit([0,1,0,1],[0,0,1,1],[0,0,0,0],y,'personalized')
    check(np.allclose(f['beta'],[float(F(v)) for v in exact['ridge_optimum']],atol=1e-12),'exact optimum mismatch')
    # Data never enter training through test targets or oracle means.
    tr,te=s['time'];before=e.fit(p['x'][tr],p['group'][tr],p['day'][tr],p['y'][tr],'trend');changed=p['y'].copy();changed[te]+=10000
    after=e.fit(p['x'][tr],p['group'][tr],p['day'][tr],changed[tr],'trend');check(np.array_equal(before['beta'],after['beta']),'held-out labels affected fit')
    # Actual sklearn split interfaces: independent group exclusion and day-block time gap.
    gt,gq=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=50).split(p['x'],groups=p['group']))
    check(len(np.intersect1d(p['group'][gt],p['group'][gq]))==0,'GroupShuffleSplit overlap')
    ts=[]
    for trd,ted in TimeSeriesSplit(n_splits=3,test_size=8,gap=2).split(np.arange(48)):
        check(trd.max()+2<ted.min(),'time gap failed');ts.append({'train_last_day':int(trd.max()),'test_first_day':int(ted.min()),'test_last_day':int(ted.max())})
    count=0;maxg=0.
    for rep in result['replications']:
        for k in e.SPLITS:
            for m in e.MODELS:
                q=rep['results'][k]['models'][m];check(np.isfinite(q['mse']) and q['mse']>=0,'invalid retained score');check(q['gradient_max']<1e-10,'unconverged normal equation');maxg=max(maxg,q['gradient_max']);count+=1
    check(count==c['independent_repetitions']*12,'missing repeated fits')
    # Same rows plus same group sizes imply micro and macro MSE equality in group/time.
    for k in ['group','time','group_time']:
        for q in result['main'][k]['models'].values():check(abs(q['mse']-q['group_mean_mse'])<1e-12,'balanced weighting mismatch')
    negatives=[]
    def rejects(name,call):
        try:call()
        except (ValueError,TypeError,OSError):negatives.append(name)
        else:raise RuntimeError('expected explicit failure: '+name)
    rejects('NaN feature',lambda:e.fit([0,np.nan],[0,1],[0,1],[1,2],'pooled'))
    rejects('mismatched lengths',lambda:e.fit([0,1],[0],[0,1],[1,2],'pooled'))
    rejects('empty data',lambda:e.fit([],[],[],[],'pooled'))
    rejects('nonintegral group',lambda:e.fit([0,1],[0,.5],[0,1],[1,2],'personalized'))
    rejects('nonpositive alpha',lambda:e.fit([0,1],[0,1],[0,1],[1,2],'pooled',0))
    rejects('source overwrite',lambda:e.atomic_json(ROOT/'lecture.md',{}))
    rejects('same row both splits',lambda:e.validate_split(p,np.array([0,1]),np.array([1,2]),'random',32))
    rejects('same group in group split',lambda:e.validate_split(p,np.array([0]),np.array([1]),'group',32))
    rejects('label delay ignored',lambda:e.validate_split(p,np.array([31]),np.array([33]),'time',32))
    rejects('future trained past evaluation',lambda:e.validate_split(p,np.array([40]),np.array([10]),'time',32))
    # Supported ordinary alternatives: constant x, single row, unknown group, row permutation.
    e.fit([2],[4],[1],[3],'personalized');e.fit([2,2,2],[0,0,1],[0,1,2],[1,2,3],'trend')
    f=e.fit(p['x'][tr],p['group'][tr],p['day'][tr],p['y'][tr],'personalized');unknown=e.predict(f,[1],[999],[40]);check(np.isfinite(unknown).all(),'unknown group fallback failed')
    rev=tr[::-1];fr=e.fit(p['x'][rev],p['group'][rev],p['day'][rev],p['y'][rev],'personalized');check(np.allclose(f['beta'],fr['beta'],atol=1e-10),'row order changed solution')
    with tempfile.TemporaryDirectory(prefix='ml050-audit-',dir=ROOT/'outputs' if (ROOT/'outputs').exists() else None) as td:
        td=Path(td);out=td/'keep.json';out.write_text('KEEP\n')
        rejects('nonfinite JSON preserves existing output',lambda:e.atomic_json(out,{'a':float('nan')}));check(out.read_text()=='KEEP\n','output changed on serialization failure')
        link=td/'alias.json';link.symlink_to(out);rejects('symlink output',lambda:e.atomic_json(link,{}));check(out.read_text()=='KEEP\n','symlink target changed')
        hard=td/'hard.json';hard.hardlink_to(out);rejects('hardlinked output',lambda:e.atomic_json(out,{}));check(out.read_text()=='KEEP\n','hardlink changed')
    # CSV source independently agrees with simulator panel 0, IDs and labels.
    csv=np.genfromtxt(ROOT/'data/panel.csv',delimiter=',',names=True)
    for field,target in [('x','x'),('y','y'),('oracle_mean','mu'),('device','group'),('day','day'),('label_available_day','available')]:check(np.array_equal(csv[field],p[target]),'CSV mismatch: '+field)
    return {'status':'passed','independent_reference_fits':refs,'exact_hand':exact,'gradient_finite_difference_max_error':float(np.max(np.abs(gd-fd))),'all_repeated_fits':count,'max_normal_gradient':maxg,'heldout_target_mutation_test':'passed','actual_sklearn_GroupShuffleSplit':'passed; 45 train groups and 15 test groups','actual_sklearn_TimeSeriesSplit_day_blocks':ts,'negative_tests':negatives,'supported_inputs':['single row','constant x','unknown device','reversed training rows'],'balanced_group_weight_identity':'passed','CSV_and_draws_exact_match':'passed','scope':'author self-check; not a deployment guarantee or independent course review'}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',default=str(ROOT/'experiment-result.json'));p.add_argument('--out',default=str(ROOT/'outputs/audit.json'));a=p.parse_args();e.safe_output(a.out)
    r=run(json.loads(Path(a.report).read_text()));e.atomic_json(a.out,r);print(json.dumps(r,ensure_ascii=False,indent=2))
