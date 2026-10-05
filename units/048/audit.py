"""Author self-check by exact Fraction integration and independent solvers."""
from fractions import Fraction as F
from pathlib import Path
import argparse,itertools,json,tempfile,hashlib
import numpy as np
from scipy.linalg import lstsq as scipy_lstsq
from sklearn.linear_model import LinearRegression
import experiment as e

def require(ok,msg):
    if not ok:raise RuntimeError(msg)

def inner(c):
    return sum((a*b*F(1,i+j+1) if (i+j)%2==0 else F(0)) for i,a in enumerate(c) for j,b in enumerate(c))

def exact():
    rows=[]
    for p in (1,2,3):
        risks=[];tr=[]
        for y in itertools.product([F(-1),F(1)],repeat=3):
            if p==1:c=[sum(y)/3]
            elif p==2:c=[sum(y)/3,(y[2]-y[0])/2]
            else:c=[y[1],(y[2]-y[0])/2,(y[0]+y[2])/2-y[1]]
            risk=1+inner(c);risks.append(risk)
            tr.append(sum((sum(a*F(x)**j for j,a in enumerate(c))-yy)**2 for x,yy in zip([-1,0,1],y))/3)
        rows.append({'p':p,'risk':str(sum(risks)/8),'mean_train':str(sum(tr)/8)})
    require([a['risk'] for a in rows]==['4/3','3/2','9/5'],'exact enumeration')
    require(inner([F(-1),F(0),F(2)])==F(7,15),'quadratic integral')
    return rows

def run(r):
    refs=exact();tests=[]
    for a,b in zip(refs,r['hand']['all_eight_equiprobable_designs']):
        require(abs(float(F(a['risk']))-b['risk'])<1e-13,'exact risk mismatch')
        require(abs(float(F(a['mean_train']))-b['mean_train'])<1e-13,'exact train mismatch')
    tests.append('Fraction all-eight exact enumeration and monomial risk integral')
    cfg,d=e.load_data();beta=np.array(cfg['beta']);q,w=np.polynomial.legendre.leggauss(32);w=w/2
    B=e.basis(q,9);require(np.max(np.abs(B.T@(w[:,None]*B)-np.eye(9)))<2e-13,'orthonormality')
    max_quad=0.;max_solver=0.;max_pred=0.
    for p in (1,2,4,8,9):
        for rep in (0,17,399):
            xx=d['x'][rep,:40];yy=e.basis(xx,9)@beta+d['eps'][rep,:40]
            c,_=e.fit(xx,yy,p);A=e.basis(xx,p)
            alt=scipy_lstsq(A,yy,lapack_driver='gelsy')[0];sk=LinearRegression(fit_intercept=False).fit(A,yy)
            max_solver=max(max_solver,float(np.max(np.abs(c-alt))))
            max_pred=max(max_pred,float(np.max(np.abs(A@c-sk.predict(A)))))
            cp=np.zeros(9);cp[:p]=c
            quad=np.dot(w,(B@(cp-beta))**2)+cfg['sigma']**2;direct=np.sum((cp-beta)**2)+cfg['sigma']**2
            max_quad=max(max_quad,abs(float(quad-direct)))
    require(max_quad<1e-11 and max_solver<1e-10 and max_pred<1e-10,'independent numerical routes')
    tests.append('15 fits against SciPy pivoted QR, sklearn; 32-node integration independent of coefficient risk')
    worst_checks=[]
    for kind in ('complexity','learning','fixed_design'):
        for block in r[kind]:
            ss=block['summary'];p=ss['p'];n=ss['n'];rep=max(block['runs'],key=lambda v:v['risk'])['rep']
            xx=np.linspace(-.95,.95,40) if kind=='fixed_design' else d['x'][rep,:n]
            eps=d['eps_fixed'][rep] if kind=='fixed_design' else d['eps'][rep,:n]
            yy=e.basis(xx,9)@beta+eps;A=e.basis(xx,p)
            C=np.array(block['coefficients'][rep]);alt=scipy_lstsq(A,yy,lapack_driver='gelsy')[0]
            quad=float(np.dot(w,(B@(C-beta))**2)+cfg['sigma']**2);risk=block['runs'][rep]['risk']
            qe=abs(quad-risk);relative=qe/max(1.,abs(risk));pred_rel=float(np.max(np.abs(B[:,:p]@alt-B@C))/max(1.,np.max(np.abs(B@C))))
            require(relative<1e-10 and pred_rel<1e-9,'worst-risk independent-route check')
            worst_checks.append({'kind':kind,'n':n,'p':p,'rep':rep,'risk':risk,'quadrature_abs_difference':qe,'quadrature_scaled_difference':relative,'QR_prediction_scaled_difference':pred_rel})
    tests.append('worst-risk replicate from all 27 blocks checked by function integration and pivoted QR, without excluding any result')
    train=np.array([[v['train'] for v in b['runs']] for b in r['complexity']]);require(np.max(np.diff(train,axis=0))<1e-12,'nested train monotonicity')
    max_res=0.;max_identity=0.
    for block in r['complexity']+r['learning']+r['fixed_design']:
        s=block['summary'];C=np.array(block['coefficients']);R=len(C)
        require(len(block['runs'])==400,'no lost replicate')
        require(np.isfinite(C).all(),'nonfinite coefficient')
        require(abs(s['variance_ddof1']*(R-1)/R-s['variance_ddof0'])<1e-8,'ddof identity')
        require(abs(s['bias2_unbiased_estimate']+s['variance_ddof1']+s['noise']-s['risk_mean'])<1e-8,'bias correction paired identity')
        max_res=max(max_res,s['max_normal_residual']);max_identity=max(max_identity,abs(s['decomposition_residual']))
    require(max_identity<1e-8 and max_res<1e-9,'residual checks')
    tests.append('all 10800 fits retained, normal equations and both finite-repetition identities')
    invalid=[([],[],1),([0,1],[1],1),([0,1],[1,np.inf],1),([0,1],[1,2],True),([0,1],[1,2],0),([0,1],[1,2],10)]
    for x,y,p in invalid:
        try:e.fit(x,y,p)
        except ValueError:pass
        else:raise RuntimeError('invalid accepted')
    tests.append('input guards remain explicit exceptions, independent of assert / -O')
    with tempfile.TemporaryDirectory() as td:
        target=Path(td)/'result.json';e.atomic_json(target,{'ok':True})
        require(json.loads(target.read_text())=={'ok':True},'write failed')
        link=Path(td)/'link.json';link.symlink_to(target)
        try:e.atomic_json(link,{'ok':False})
        except ValueError:pass
        else:raise RuntimeError('symlink accepted')
        try:e.atomic_json(target,{'value':float('nan')})
        except ValueError:pass
        else:raise RuntimeError('NaN serialized')
        require(json.loads(target.read_text())=={'ok':True},'failed serialization overwrote previous result')
    tests.append('atomic strict JSON, symlink rejection, previous output preserved on serialization failure')
    return {'status':'author self-check passed; independent course QA pending','exact_reference':refs,'tests':tests,'max_quadrature_difference':max_quad,'max_scipy_qr_coef_difference':max_solver,'max_sklearn_prediction_difference':max_pred,'max_normal_equation_residual':max_res,'max_finite_repetition_identity_residual':max_identity,'worst_risk_checks':worst_checks,'checked_fits':sum(len(a['runs']) for a in r['complexity']+r['learning']+r['fixed_design'])}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',default=str(e.ROOT/'experiment-result.json'));p.add_argument('--out',default=str(e.ROOT/'outputs/audit.json'));a=p.parse_args();result=run(json.loads(Path(a.report).read_text()));e.atomic_json(a.out,result);print(json.dumps(result,ensure_ascii=False))
