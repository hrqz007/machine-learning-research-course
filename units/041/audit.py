"""Independent exact hand arithmetic, library comparisons and adversarial gates."""
from fractions import Fraction as F
from pathlib import Path
from unittest.mock import patch
import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import numpy as np
import scipy
import sklearn
from sklearn.linear_model import Ridge, LinearRegression
import experiment as e


def require(condition,message):
    if not condition: raise RuntimeError(message)


def near(a,b,tol=1e-10):
    if not np.allclose(a,b,atol=tol,rtol=tol): raise RuntimeError('numeric mismatch')


def hand_exact(h,s):
    X=[[F(str(v)) for v in row[1:3]] for row in h];y=[F(str(row[3])) for row in h]
    w=[F(str(v)) for v in s['initial']];lam=F(str(s['hand_lambda']));eta=F(str(s['hand_step'])); states=[]
    for k in range(3):
        rows=[];g=[F(0),F(0)];loss=F(0)
        for i in range(4):
            p=sum(X[i][j]*w[j] for j in range(2));r=p-y[i];ell=r*r/2;loss+=ell/4
            gc=[r*X[i][j]/4 for j in range(2)];g=[g[j]+gc[j] for j in range(2)]
            rows.append({'prediction':p,'residual':r,'half_square':ell,'mean_loss':ell/4,'gradient_contribution':gc})
        pg=[lam*v for v in w];total=[g[j]+pg[j] for j in range(2)];pen=lam*sum(v*v for v in w)/2
        states.append({'w':w[:],'rows':rows,'data_loss':loss,'penalty':pen,'objective':loss+pen,'gradient':total})
        w=[w[j]-eta*total[j] for j in range(2)]
    report=e._hand_report(h,s)
    for exact,actual in zip(states,report['states']):
        for key in ('w','data_loss','penalty','objective','gradient'): near(np.array(exact[key],dtype=float),actual[key],1e-14)
        for er,ar in zip(exact['rows'],actual['rows']):
            for key in er: near(np.array(er[key],dtype=float),ar[key],1e-14)
    H=[[sum(X[i][j]*X[i][k] for i in range(4))/4 for k in range(2)] for j in range(2)]
    c=[sum(X[i][j]*y[i] for i in range(4))/4 for j in range(2)]
    refs={}
    for name,l in [('ols',F(0)),('ridge',lam)]:
        aa=H[0][0]+l;bb=H[0][1];dd=H[1][1]+l;det=aa*dd-bb*bb
        w=[(dd*c[0]-bb*c[1])/det,(aa*c[1]-bb*c[0])/det];near([float(v) for v in w],report['references'][name]['w'])
        refs[name]=[str(v) for v in w]
    def strings(v):
        if isinstance(v,F):return str(v)
        if isinstance(v,list):return [strings(x) for x in v]
        if isinstance(v,dict):return {k:strings(x) for k,x in v.items()}
        return v
    return {'states':strings(states),'references':refs,'method':'Fraction scalar arithmetic; 2x2 elimination independent of numerical solver'}


def mathematical_checks(h,b,d,s):
    X=np.array([r[1:3] for r in h]);y=np.array([r[3] for r in h]);records=[]
    for lam in s['lambdas']:
        ref=e._reference(X,y,lam);lib=LinearRegression(fit_intercept=False).fit(X,y).coef_ if lam==0 else Ridge(alpha=len(y)*lam,fit_intercept=False,solver='svd').fit(X,y).coef_
        near(ref,lib);near(ref,e._reference(np.repeat(X,2,axis=0),np.repeat(y,2),lam))
        records.append({'lambda':lam,'library_coefficient':lib.tolist(),'max_difference':float(np.max(abs(ref-lib)))})
    H=X.T@X/4+s['hand_lambda']*np.eye(2);ref=e._reference(X,y,s['hand_lambda']);fs=e._objective(X,y,ref,s['hand_lambda'])[0]
    for w in [np.array([.2,-.5]),np.array([3.,2.]),np.array([0.,0.])]:
        f,g=e._objective(X,y,w,s['hand_lambda']);diff=w-ref;gap=diff@H@diff/2
        near(g,H@diff);near(f-fs,gap);ev=np.linalg.eigvalsh(H)
        require(np.linalg.norm(g)**2/(2*ev[-1])<=gap+1e-12,'lower gap bound');require(gap<=np.linalg.norm(g)**2/(2*ev[0])+1e-12,'upper gap bound')
    # Rank deficient OLS: coefficients differ along the null space; predictions agree.
    A=np.array([[1.,1.],[2.,2.],[-1.,-1.],[-2.,-2.]]);z=np.array([2.,4.,-2.,-4.]);w=e._reference(A,z,0.);near(w,[1,1]);near(A@(w+[3,-3]),A@w)
    # Independent scale coordinate derivation of a diagonally preconditioned step.
    H=X.T@X/4+.5*np.eye(2);diag=np.diag(H);P=np.diag(1/np.sqrt(diag));u=np.array([.3,-.2]);w=P@u
    _,g=e._objective(X,y,w,.5);L=np.linalg.eigvalsh(P@H@P)[-1];near(P@(u-P@g/L),w-g/diag/L)
    # Adam first step with bias correction and original objective gradient.
    cfg=copy.deepcopy(s);cfg['max_epochs']=2;r=e._solve(X,y,.5,'adam',cfg,s['seed']);g0=-X.T@y/4
    near(r['trace'][1]['w'],-s['adam_step']*g0/(np.abs(g0)+1e-8))
    # Finite differences at multiple fixed random binary-rational data/parameters.
    rng=np.random.default_rng(4109);errors=[]
    for _ in range(32):
        A=rng.integers(-8,9,(7,3))/4;z=rng.integers(-8,9,7)/4;w=rng.integers(-8,9,3)/4;lam=.25
        g=e._objective(A,z,w,lam)[1];fd=[]
        for j in range(3):
            p=w.copy();m=w.copy();p[j]+=1e-5;m[j]-=1e-5;fd.append((e._objective(A,z,p,lam)[0]-e._objective(A,z,m,lam)[0])/2e-5)
        err=float(np.max(abs(g-fd)));require(err<1e-8,'random gradient check');errors.append(err)
    lock=e._selection_report(d,s);changed=copy.deepcopy(d)
    for row in changed:
        if row[1]=='test':row[-1]+=25.
    new=e._selection_report(changed,s);require(lock['lock']==new['lock'],'test leaked into lock');require(lock['lock_sha256']==new['lock_sha256'],'lock digest differs');require(lock['test']['selected_mse']!=new['test']['selected_mse'],'test did not change')
    other=copy.deepcopy(d)
    for row in other:
        if row[1]=='test':row[2]*=9.
    require(e._selection_report(other,s)['lock']==lock['lock'],'test feature leakage')
    return {'library_versions':{'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__},'library_and_replication':records,'random_gradient_trials':32,'max_random_gradient_error':max(errors),'rank_deficiency':'passed','gap_identities_and_bounds':'passed','preconditioned_coordinate_equivalence':'passed','adam_first_step':'passed','test_label_and_feature_isolation':'passed'}


def rejection_checks(h,b,d,s):
    variants=[]
    bads=[True,'0',1+2j,float('nan'),float('inf'),1e-200]
    tiny=np.longdouble('1e-400')
    if tiny!=0: bads.append(tiny)
    require(e.real(np.longdouble(0),'zero')==0.,'valid zero rejected')
    require(e.real(np.float64(.5),'normal')==.5,'valid numpy scalar rejected')
    for bad in bads:
        for kind in ('last_selection_y','last_benchmark_y','last_hand_y','initial_tail','lambda_tail','fd_tail'):
            hh,bb,dd,ss=copy.deepcopy((h,b,d,s))
            if kind=='last_selection_y':dd[-1][-1]=bad
            elif kind=='last_benchmark_y':bb[-1][-1]=bad
            elif kind=='last_hand_y':hh[-1][-1]=bad
            elif kind=='initial_tail':ss['initial'][-1]=bad
            elif kind=='lambda_tail':ss['lambdas'][-1]=bad
            else:ss['fd_steps'][-1]=bad
            variants.append((kind+':'+type(bad).__name__,(hh,bb,dd,ss)))
    for kind in ('last_config','duplicate_id','wrong_row_shape','duplicate_lambda','missing_key','bad_role','column_vector','bad_observer'):
        hh,bb,dd,ss=copy.deepcopy((h,b,d,s))
        if kind=='last_config':ss['coverage_repetitions']=True
        elif kind=='duplicate_id':dd[-1][0]=dd[0][0]
        elif kind=='wrong_row_shape':bb[-1].append(1)
        elif kind=='duplicate_lambda':ss['lambdas'][-1]=ss['lambdas'][0]
        elif kind=='missing_key':del ss['seed']
        elif kind=='bad_role':dd[-1][1]='future'
        elif kind=='column_vector':ss['initial']=[[0],[0]]
        variants.append((kind,(hh,bb,dd,ss)))
    rejects=[]
    def bomb(*args,**kwargs): raise RuntimeError('COMPUTATION HAPPENED BEFORE REJECTION')
    for name,args in variants:
        with patch.object(e,'_hand_report',bomb),patch.object(e,'_optimizer_suite',bomb),patch.object(e,'_selection_report',bomb),patch.object(e,'_coverage',bomb),patch.object(e.np.random,'default_rng',bomb),patch.object(e.np.linalg,'lstsq',bomb):
            try:e.run_experiment(*args,observer=17 if name=='bad_observer' else bomb)
            except (ValueError,TypeError):rejects.append(name)
            else:raise RuntimeError('invalid input accepted')
    json_bad=['{"x":1,"x":2}','{"x":{"v":1,"v":2}}','{"x":NaN}','{"x":Infinity}','{"x":1e999}','{"x":1e-999}','{"x":1e-151}']
    for raw in json_bad:
        try:e.strict_json(raw)
        except ValueError:pass
        else:raise RuntimeError('invalid JSON accepted')
    return {'precompute_rejections':len(rejects),'cases':rejects,'strict_json_rejections':len(json_bad),'sentinel_scope':'first numerical experiment, RNG, lstsq, observer'}


def writing_checks():
    count=0
    with tempfile.TemporaryDirectory() as temp:
        d=Path(temp);p=d/'report.json';p.write_bytes(b'ORIGINAL\n')
        with patch.object(e.os,'replace',side_effect=OSError('injected replacement failure')):
            try:e.atomic_json(p,{'ok':True})
            except OSError:count+=1
            else:raise RuntimeError('replace failure disappeared')
        require(p.read_bytes()==b'ORIGINAL\n','old report damaged');require(not list(d.glob('.unit041-*')),'temporary write remains')
        try:e.atomic_json(p,{'bad':float('nan')})
        except ValueError:count+=1
        require(p.read_bytes()==b'ORIGINAL\n','serialization failure damaged report')
        q=d/'alias.json';q.symlink_to(p)
        hard=d/'hard.json';os.link(p,hard)
        for target in (q,hard,p,d,e.ROOT/'data/hand.csv',e.ROOT/'experiment.py'):
            try:e.check_destination(target)
            except ValueError:count+=1
            else:raise RuntimeError('unsafe destination accepted')
        hard.unlink();alias=d/'alias_dir';alias.symlink_to(d,target_is_directory=True)
        try:e.check_destination(alias/'fresh.json')
        except ValueError:count+=1
        else:raise RuntimeError('symlink parent accepted')
        try:e.check_destination(p,[p])
        except ValueError:count+=1
        else:raise RuntimeError('custom input overwrite accepted')
    return {'checks':count,'replacement_failure_preserves_bytes':True,'temporary_cleanup':True,'symlink_hardlink_asset_and_input_protection':True,'scope':'single-user local files; not a defense against hostile concurrent filesystem races'}


def subprocess_checks():
    hashes=[];rejections=[]
    with tempfile.TemporaryDirectory() as temp:
        base=Path(temp);env=os.environ.copy();env['OPENBLAS_NUM_THREADS']='1';env['PYTHONDONTWRITEBYTECODE']='1'
        for i,(optimized,cwd) in enumerate([(False,e.ROOT),(True,base),(False,base)]):
            out=base/('run'+str(i)+'.json');cmd=[sys.executable]+(['-O'] if optimized else [])+[str(e.ROOT/'experiment.py'),'--output',str(out)]
            result=subprocess.run(cmd,cwd=cwd,env=env,capture_output=True,text=True)
            require(result.returncode==0,'subprocess failed: '+result.stderr);hashes.append(hashlib.sha256(out.read_bytes()).hexdigest())
        require(len(set(hashes))==1,'normal/-O/cwd full JSON mismatch')
        # Invalid LAST fields in optimized fresh processes preserve output bytes.
        for kind in ('last_config','last_selection_y','duplicate_json'):
            folder=base/kind;folder.mkdir()
            for name in ('hand.csv','benchmark.csv','selection.csv','config.json'):(folder/name).write_bytes((e.ROOT/'data'/name).read_bytes())
            if kind=='last_config':
                s=e.strict_json((folder/'config.json').read_text());s['coverage_repetitions']=True;(folder/'config.json').write_text(json.dumps(s))
            elif kind=='last_selection_y':
                lines=(folder/'selection.csv').read_text().splitlines();lines[-1]=lines[-1].rsplit(',',1)[0]+',NaN';(folder/'selection.csv').write_text('\n'.join(lines)+'\n')
            else:
                raw=(folder/'config.json').read_text().rstrip();(folder/'config.json').write_text(raw[:-1]+',"seed":2}')
            out=folder/'unchanged.json';out.write_bytes(b'BEFORE');cmd=[sys.executable,'-O',str(e.ROOT/'experiment.py'),'--data-dir',str(folder),'--output',str(out)]
            r=subprocess.run(cmd,cwd=base,env=env,capture_output=True,text=True);require(r.returncode!=0 and out.read_bytes()==b'BEFORE','bad tail not rejected atomically');rejections.append(kind)
            edge_statuses=[]
        for name,initial in [('one_epoch',[0.,0.]),('stationary_hand',[5/11,9/11])]:
            spec=e.load_inputs()[-1];spec['max_epochs']=1;spec['initial']=initial
            cfg=base/(name+'.json');cfg.write_text(json.dumps(spec))
            edge_hashes=[]
            for opt in (False,True):
                out=base/(name+str(opt)+'.out.json')
                cmd=[sys.executable]+(['-O'] if opt else [])+[str(e.ROOT/'experiment.py'),'--config',str(cfg),'--output',str(out)]
                proc=subprocess.run(cmd,cwd=base,env=env,capture_output=True,text=True)
                require(proc.returncode==0,'edge CLI wrote then crashed: '+proc.stderr)
                obj=e.strict_json(out.read_text());edge_hashes.append(hashlib.sha256(out.read_bytes()).hexdigest())
                require(obj['optimizers']['injected_failure']['status']=='objective_increase_rejected','rejected-step summary missing')
                if name=='stationary_hand':require(np.linalg.norm(obj['hand']['states'][0]['gradient'])<1e-12,'stationary hand initial not stationary')
            require(edge_hashes[0]==edge_hashes[1],'edge normal/-O mismatch');edge_statuses.append({'case':name,'normal_optimized_sha256':edge_hashes[0]})
    return {'full_json_sha256':hashes,'normal_optimized_external_cwd_equal':True,'optimized_bad_tail_rejections':rejections,'short_budget_and_stationary_cli':edge_statuses}


def audit(full=True):
    h,b,d,s=e.load_inputs();r={'unit':'041','exact_hand':hand_exact(h,s),'mathematical':mathematical_checks(h,b,d,s),'input_gates':rejection_checks(h,b,d,s),'output_gates':writing_checks()}
    # Retained failure accounting: reference + initial + rejected candidate.
    X,y=e._arrays(b,'scaled');X=X-X.mean(0);y=y-y.mean();f=e._solve(X,y,.1,'gd_unsafe',s,s['seed'])
    require(f['status']=='objective_increase_rejected','missing real failure');require(f['cost']['attempted_updates']==1 and f['cost']['committed_updates']==0,'wrong failed-step counts');require(f['cost']['objective_rows']==3*len(y),'omitted failed or reference cost')
    r['failure_accounting']={'status':f['status'],'cost':f['cost'],'failed_objective':f['failure']['objective'],'retained_objective':f['final']['objective']}
    if full:r['subprocess']=subprocess_checks()
    return r


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--quick',action='store_true');a=p.parse_args();e.check_destination(a.output)
    result=audit(not a.quick);e.atomic_json(a.output,result);print('PASS: independent math, library, input/output gates'+(' and fresh processes' if not a.quick else ''))
if __name__=='__main__':main()
