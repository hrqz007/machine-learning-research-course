"""Independent arithmetic, statistical, normalization, isolation and file tests.

Explicit checks remain active under Python -O. No network or data download.
"""
from pathlib import Path
from fractions import Fraction as Q
from decimal import Decimal, localcontext
from unittest.mock import patch
import copy, csv, hashlib, io, json, os, tempfile
import numpy as np
import experiment as e

COUNTS={}
def check(ok,group,message):
    if not bool(ok):raise RuntimeError(group+': '+message)
    COUNTS[group]=COUNTS.get(group,0)+1

def close(a,b,group,tol=1e-10):
    check(np.allclose(a,b,rtol=tol,atol=tol),group,'numeric disagreement')

def rejects(fn,group):
    try:fn()
    except (ValueError,TypeError):COUNTS[group]=COUNTS.get(group,0)+1
    else:raise RuntimeError(group+': invalid input accepted')

def fsolve(A,b):
    det=A[0][0]*A[1][1]-A[0][1]*A[1][0]
    return [(b[0]*A[1][1]-A[0][1]*b[1])/det,(A[0][0]*b[1]-b[0]*A[1][0])/det]

def rational(X,y,lam,rho=Q(0),signs=(1,1)):
    n=len(y);H=[[sum(row[j]*row[k] for row in X)/n for k in range(2)] for j in range(2)]
    c=[sum(row[j]*v for row,v in zip(X,y))/n-lam*rho*signs[j] for j in range(2)]
    return fsolve([[H[j][k]+(lam*(1-rho) if j==k else 0) for k in range(2)] for j in range(2)],c)

def fraction_checks(X,y,c):
    x=[[Q(float(v)) for v in r] for r in X];yy=[Q(float(v)) for v in y];b=[Q(0),Q(0)];ledgers=[]
    for k in range(3):
        got=e.row_ledger(X,y,[float(v) for v in b],.25,0);loss=Q(0);g=[Q(0),Q(0)];rr=[]
        for i,(row,yi) in enumerate(zip(x,yy)):
            pred=sum(v*t for v,t in zip(row,b));r=pred-yi;f=r*r/8;grad=[r*v/4 for v in row]
            for key,val in [('prediction',pred),('residual',r),('loss_contribution',f),('local_dloss_dr',r)]:close(got['rows'][i][key],float(val),'fraction_rows',0)
            close(got['rows'][i]['gradient_contribution'],[float(v) for v in grad],'fraction_rows',0)
            loss+=f;g=[a+v for a,v in zip(g,grad)];rr.append({'prediction':str(pred),'residual':str(r),'loss_contribution':str(f),'gradient_contribution':list(map(str,grad))})
        penalty=sum(v*v for v in b)/8;total=[g[j]+b[j]/4 for j in range(2)]
        close([got['data_loss'],got['l2_penalty'],got['objective']],[float(loss),float(penalty),float(loss+penalty)],'fraction_states',0)
        close(got['smooth_gradient'],list(map(float,total)),'fraction_states',0)
        ledgers.append({'k':k,'beta':list(map(str,b)),'rows':rr,'F':str(loss),'penalty':str(penalty),'J':str(loss+penalty),'total_gradient':list(map(str,total))})
        b=[b[j]-total[j]/2 for j in range(2)]
    check(rational(x,yy,Q(1,4))==[Q(129,134),Q(61,67)],'exact_solutions','Ridge exact')
    check(rational(x,yy,Q(1,2),Q(1,2))==[Q(119,134),Q(49,67)],'exact_solutions','Elastic exact')
    check(ledgers[2]['beta']==['1009/1024','3623/4096'],'fraction_states','two synchronous updates')
    rng=np.random.default_rng(4080)
    for _ in range(120):
        a=rng.integers(-8,9,size=(6,2))/4;z=rng.integers(-8,9,size=6)/4;lam=Q(int(rng.integers(1,17)),16)
        ref=rational([[Q(float(v)) for v in row] for row in a],[Q(float(v)) for v in z],lam)
        close(e.ridge_solve(a,z,float(lam)),list(map(float,ref)),'random_fraction')
        close(e.ridge_augmented(a,z,float(lam)),list(map(float,ref)),'random_fraction')
    return ledgers


def decimal_and_gradient(X,y,c):
    with localcontext() as ctx:
        ctx.prec=110
        for exponent in (2,4,6,8,10):
            eps=10.**(-exponent);a=np.array([[1.,1.],[-1.,-1.],[1.,1.+eps],[-1.,-1.-eps]])
            z=np.array([1.,-1.,1.5,-1.5])
            for lam in (.01,.25,1.):
                A=[[Decimal.from_float(float(v)) for v in row] for row in a];Y=[Decimal.from_float(float(v)) for v in z];L=Decimal.from_float(lam)
                H=[[sum(r[j]*r[k] for r in A)/4+(L if j==k else 0) for k in range(2)] for j in range(2)];b=[sum(r[j]*v for r,v in zip(A,Y))/4 for j in range(2)];ref=fsolve(H,b)
                close(e.ridge_solve(a,z,lam),list(map(float,ref)),'decimal_condition',1e-12)
    rng=np.random.default_rng(4081)
    for _ in range(80):
        b=rng.normal(size=2);lam=float(rng.uniform(.01,2));v=rng.normal(size=2);h=1e-5
        f=lambda t:e.row_ledger(X,y,t,lam,0)['objective']
        numeric=(f(b+h*v)-f(b-h*v))/(2*h);analytic=np.array(e.row_ledger(X,y,b,lam,0)['smooth_gradient'])@v
        close(numeric,analytic,'smooth_finite_difference',1e-8)
        u,s,vt=np.linalg.svd(X,full_matrices=False);ref=vt.T@((s/(s*s+len(y)*lam))*(u.T@y))
        close(e.ridge_solve(X,y,lam),ref,'spectral_reference')


def library_checks(X,y,c):
    outcomes={}
    for fam,lam,rho,truth in [('ridge',.25,0,[129/134,61/67]),('lasso',.5,1,[1.5,0]),('elastic',.5,.5,[119/134,49/67])]:
        a=e.fit_library(X,y,lam,fam,c,rho);b=e.fit_library(np.tile(X,(2,1)),np.tile(y,2),lam,fam,c,rho)
        close(a['beta'],truth,'library_reference',1e-9);close(a['beta'],b['beta'],'sample_duplication',1e-9)
        check(b['library_alpha']==a['library_alpha']*(2 if rho==0 else 1),'sample_duplication','alpha scaling')
        check(a['status']=='converged','library_reference','KKT status');outcomes[fam]=a
    for rho in (0,1):
        a=e.fit_library(X,y,.25,'elastic',c,rho);b=e.fit_library(X,y,.25,'ridge' if rho==0 else 'lasso',c)
        close(a['beta'],b['beta'],'endpoints')
    for fam in e.FAMILIES:close(e.fit_library(X,y,0,fam,c)['beta'],[.25,2],'endpoints')
    for fam in e.FAMILIES:
        rho=0 if fam=='ridge' else 1 if fam=='lasso' else .5
        for lam in c['path_lambdas']:
            chosen,branches=e.enumerate_2d(X,y,lam,rho)
            fit=e.fit_library(X,y,lam,fam,c,rho)
            close(fit['beta'],chosen['beta'],'all_path_branches',2e-7)
            check(fit['kkt_inf']<=1e-7,'all_path_branches','KKT')
    duplicate=np.column_stack(([-1.,-1.,1.,1.],[-1.,-1.,1.,1.]))
    en=e.fit_library(duplicate,y,.5,'elastic',c,.5);la=e.fit_library(duplicate,y,.5,'lasso',c)
    close(en['beta'][0],en['beta'][1],'rank_deficient',2e-9)
    alt=np.array(la['beta'])[::-1];close(duplicate@alt,duplicate@np.array(la['beta']),'rank_deficient')
    close(e.row_ledger(duplicate,y,alt,.5,1)['objective'],e.row_ledger(duplicate,y,la['beta'],.5,1)['objective'],'rank_deficient')
    close(e.ridge_solve(duplicate,y,0),np.linalg.lstsq(duplicate,y,rcond=None)[0],'rank_deficient')
    close(e.ridge_solve(np.zeros((2,2)),[1,2],.25),[0,0],'rank_deficient')
    close(e.ridge_solve([[1,2]],[3],.25),e.ridge_augmented([[1,2]],[3],.25),'rank_deficient')
    short=copy.deepcopy(c);short['solver_max_iter']=1
    fail=e.fit_library(X,y,.001,'elastic',short,.5)
    check(fail['status']=='not_converged' and len(fail['warnings'])>0,'library_failure','keep warning')
    from sklearn.linear_model import ElasticNet
    with patch.object(ElasticNet,'fit',side_effect=ValueError('controlled solver failure')):
        fail2=e.fit_library(X,y,.5,'elastic',c,.5)
    check(fail2['status']=='solver_error' and fail2['beta'] is None,'library_failure','controlled solver error status')
    return {'main':outcomes,'duplicate_elastic':en,'duplicate_lasso':la,'nonconverged':fail,'solver_error':fail2}


def generator_check(splits,c):
    truth=e.read_json(e.ROOT/'data/synthetic_truth.json');seeds=np.random.SeedSequence(c['selection_seed']).spawn(3)
    header=['split','id']+[f'x{i:02}' for i in range(1,21)]+['y'];s=io.StringIO(newline='');w=csv.writer(s,lineterminator='\n');w.writerow(header)
    for name,pre,key,seed in zip(('train','validation','test'),('T','V','E'),('train_rows','validation_rows','test_rows'),seeds):
        rng=np.random.default_rng(seed);n=c[key];Z=rng.normal(size=(n,20));r=c['correlation'];Z[:,1]=r*Z[:,0]+np.sqrt(1-r*r)*Z[:,1];Z[:,3]=r*Z[:,2]+np.sqrt(1-r*r)*Z[:,3]
        X=Z*np.array(truth['scales'])+np.array(truth['offsets']);y=truth['true_intercept']+X@np.array(truth['true_coefficient_raw_units'])+rng.normal(0,c['noise_std'],size=n)
        close(X,splits[name]['X'],'regeneration',0);close(y,splits[name]['y'],'regeneration',0)
        for i,(row,t) in enumerate(zip(X,y),1):w.writerow([name,f'{pre}{i:04}']+list(row)+[t])
    original=(e.ROOT/'data/selection.csv').read_bytes()
    # Exact CSV serializer identity is checked, not just floating closeness.
    check(s.getvalue().encode()==original,'regeneration','CSV bytes differ')


def leakage_checks(splits,c):
    X,y=splits['train']['X'],splits['train']['y'];V,v=splits['validation']['X'],splits['validation']['y'];T,t=splits['test']['X'],splits['test']['y']
    a=e.select_candidate(X,y,V,v,c);before=e.canonical_bytes(a);one=e.final_evaluate(a['chosen'],T,t);two=e.final_evaluate(a['chosen'],T,t+100)
    check(before==e.canonical_bytes(a),'test_isolation','test labels changed locked choice');check(one['mse']!=two['mse'],'test_isolation','test MSE should change')
    b=e.select_candidate(X,y,V+3,v,c);check(a['preprocessing']==b['preprocessing'],'train_scaler','validation changed scaler')
    for aa,bb in zip(a['candidates'],b['candidates']):close(aa['coefficient_raw'],bb['coefficient_raw'],'train_scaler',0)
    close(a['preprocessing']['mean'],X.mean(axis=0),'train_scaler',1e-14)
    close(a['preprocessing']['variance_ddof0'],X.var(axis=0),'train_scaler',1e-14)
    for f in a['candidates']:check(f['prediction_coordinate_max_error']<1e-11,'coordinate_backtransform','prediction mismatch')
    small=copy.deepcopy(c);small['features']=2;small['selection_lambdas']=[.1];small['selection_l1_ratios']=[.5]
    constant=e.select_candidate([[1,0],[1,1],[1,2]],[1,2,3],[[1,3]],[4],small)
    close(constant['preprocessing']['scale'][0],1,'constant_column',0)
    close(constant['chosen']['coefficient_raw'][0],0,'constant_column',0)
    # Exclusion, failed selection and deterministic exact tie rule with real scaler.
    original=e.fit_library
    def fail_regularized(X,y,lam,family,cfg,rho=.5):
        fit=original(X,y,lam,family,cfg,rho)
        if lam>0:fit['status']='not_converged'
        return fit
    with patch.object(e,'fit_library',side_effect=fail_regularized):failed=e.select_candidate(X,y,V,v,c)
    check(failed['status']=='selection_failed' and failed['chosen'] is None,'selection_failures','invalid candidates selected')
    tied=e.select_candidate(np.zeros((2,2)),[1,1],np.zeros((1,2)),[1],small)
    check(tied['chosen']['family']=='ridge','tie_rule','family order')
    return {'selected':{k:a['chosen'][k] for k in ('family','lambda','l1_ratio','validation_mse')},'test_original_mse':one['mse'],'test_mutated_mse':two['mse'],'choice_fit_sha256':a['choice_fit_sha256']}


def raw_guard_checks(X,y,splits,c):
    bad=[True,'1',1+0j,float('nan'),float('inf'),1e-150,np.longdouble('1e-400')]
    def sentinel(*args,**kwargs):raise RuntimeError('computation began before raw validation')
    with patch.object(np.linalg,'solve',side_effect=sentinel),patch.object(np.linalg,'lstsq',side_effect=sentinel),patch.object(np.random,'default_rng',side_effect=sentinel):
        for value in bad:
            b=copy.deepcopy(c);b['monte_carlo_truth'][-1]=value
            rejects(lambda:e.main_report(X,y,splits,b),'precompute_guards')
            a=X.tolist();a[-1][-1]=value
            rejects(lambda:e.ridge_solve(a,y,.25),'precompute_guards')
            yy=y.tolist();yy[-1]=value
            rejects(lambda:e.fit_library(X,yy,.25,'ridge',c),'precompute_guards')
            ss=copy.deepcopy(splits);ss['test']['y']=ss['test']['y'].tolist();ss['test']['y'][-1]=value
            rejects(lambda:e.main_report(X,y,ss,c),'precompute_guards')
            rejects(lambda:e.fit_library(X,y,.25,'elastic',c,value),'precompute_guards')
        for key in ('solver_max_iter','features','monte_carlo_repetitions'):
            b=copy.deepcopy(c);b[key]=True;rejects(lambda:e.main_report(X,y,splits,b),'precompute_guards')
        b=copy.deepcopy(c);b['path_lambdas']=[0,.1,.1];rejects(lambda:e.main_report(X,y,splits,b),'precompute_guards')
        rejects(lambda:e.ridge_solve([[1,2],[3]],[1,2],.1),'precompute_guards')
        rejects(lambda:e.ridge_solve(X,[1,2],.1),'precompute_guards')
        rejects(lambda:e.fit_library(X,y,.1,'alien',c),'precompute_guards')
    with tempfile.TemporaryDirectory() as td:
        d=Path(td);p=d/'bad.json'
        for text in ('{"a":1,"a":2}','{"a":NaN}','{"a":1e-999}','{"a":Infinity}'):
            p.write_text(text);rejects(lambda:e.read_json(p),'json_guards')
        csvp=d/'main.csv';original=(e.ROOT/'data/regression.csv').read_text()
        csvp.write_text(original+original.splitlines()[-1]+'\n');rejects(lambda:e.load_inputs(main_path=csvp),'csv_guards')
        selectp=d/'select.csv';source=(e.ROOT/'data/selection.csv').read_text();lines=source.splitlines()
        for altered in (source+lines[-1]+'\n','\n'.join(lines[:-1]+[lines[-1].rsplit(',',1)[0]+',1e-999'])+'\n',source.replace('test,E0001,','test,T0001,')):
            selectp.write_text(altered);rejects(lambda:e.load_inputs(selection_path=selectp),'csv_guards')


def file_checks():
    with tempfile.TemporaryDirectory() as td:
        d=Path(td);old=d/'result.json';old.write_bytes(b'old');link=d/'linked.json';os.link(old,link)
        rejects(lambda:e.safe_write(link,{'a':1}),'file_guards');check(old.read_bytes()==b'old','file_guards','hardlink changed')
        link.unlink();sym=d/'sym';sym.symlink_to(d,target_is_directory=True);rejects(lambda:e.safe_write(sym/'new.json',{}),'file_guards')
        rejects(lambda:e.safe_write(e.ROOT/'data/regression.csv',{}),'file_guards')
        rejects(lambda:e.safe_write(old,{},[old]),'file_guards')
        for dest in (old,d/'new/nested/result.json'):
            with patch.object(e.os,'replace',side_effect=OSError('controlled atomic failure')):
                try:e.safe_write(dest,{'a':1})
                except OSError:COUNTS['file_guards']+=1
                else:raise RuntimeError('atomic failure not raised')
        check(old.read_bytes()==b'old','file_guards','atomic old bytes');check(not (d/'new').exists(),'file_guards','empty tree not removed')
        check(not list(d.glob('.ml040-*')),'file_guards','temporary retained')
        rejects(lambda:e.safe_write(d/'bad/json',{'x':float('nan')}),'file_guards');check(not (d/'bad').exists(),'file_guards','serialize mutated tree')
        e.safe_write(old,{'a':1});check(json.loads(old.read_text())=={'a':1},'file_guards','normal write')
        # Real byte mutations of every first-cell contract input.
        contract=e.read_json(e.ROOT/'data_integrity.json');base=d/'fixture';base.mkdir()
        (base/'data_integrity.json').write_bytes((e.ROOT/'data_integrity.json').read_bytes())
        for rel in contract['sha256']:
            p=base/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((e.ROOT/rel).read_bytes())
        check(e.verify_fixture(base),'fixture_guards','baseline')
        for rel in contract['sha256']:
            p=base/rel;old=p.read_bytes();p.write_bytes(old+b' ');rejects(lambda:e.verify_fixture(base),'fixture_guards');p.write_bytes(old)


def monte_carlo_checks(X,c):
    a=e.monte_carlo(X,c);noise=np.array(a['paired_noise_same_across_lambda']);truth=np.array(c['monte_carlo_truth']);H=X.T@X/len(X)
    for r in a['records']:
        lam=r['lambda'];B=np.array(r['all_fitted_coefficients']);ref=np.array([e.ridge_augmented(X,X@truth+eps,lam) for eps in noise[:20]])
        close(B[:20],ref,'MC_independent_fit',1e-10)
        errors=B-truth;emp=np.mean(np.sum(errors*errors,axis=1));close(emp,r['empirical_coefficient_mse'],'MC_identity',1e-13)
        bias=np.array(r['analytic_bias']);cov=np.array(r['analytic_covariance']);close(bias@bias+np.trace(cov),r['analytic_coefficient_mse'],'MC_identity',1e-13)
        close(r['analytic_new_label_mse_on_fixed_X']-r['analytic_noiseless_prediction_mse_on_fixed_X'],c['monte_carlo_noise_std']**2,'MC_identity',1e-13)
    return {'max_absolute_mean_deviation':max(float(np.max(np.abs(np.array(r['empirical_mean'])-r['analytic_mean']))) for r in a['records']),
      'note':'finite Monte Carlo discrepancy is reported, not required to be exactly zero'}


def run_audit():
    COUNTS.clear();X,y,s,c=e.load_inputs();e.verify_fixture()
    exact=fraction_checks(X,y,c);decimal_and_gradient(X,y,c);libs=library_checks(X,y,c);generator_check(s,c)
    isolation=leakage_checks(s,c);raw_guard_checks(X,y,s,c);file_checks();mc=monte_carlo_checks(X,c)
    return {'status':'passed','checks':sum(COUNTS.values()),'groups':COUNTS.copy(),'exact_three_states':exact,'library_cases':libs,'leakage':isolation,'monte_carlo':mc}

if __name__=='__main__':print(json.dumps(run_audit(),ensure_ascii=False,indent=2,allow_nan=False))
