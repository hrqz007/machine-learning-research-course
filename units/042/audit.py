"""Independent exact, Decimal, numerical, API and filesystem checks for ML042."""
from decimal import Decimal as D,localcontext
from fractions import Fraction as Q
from pathlib import Path
import argparse,copy,hashlib,json,os,tempfile,unittest.mock as mock
import numpy as np
import experiment as e
COUNT=0

def check(v,message):
    global COUNT
    COUNT+=1
    if not v:raise RuntimeError(message)

def close(a,b,label,atol=3e-14,rtol=3e-13):check(np.allclose(a,b,atol=atol,rtol=rtol),label)

def reject(fn,label):
    try:fn()
    except (ValueError,TypeError,OverflowError):check(True,label);return
    raise RuntimeError('not rejected: '+label)

def dec(v):return D.from_float(float(v))

def decimal_state(x,y,theta,lam=0):
    n=D(len(x));b,w=map(dec,theta);lam=dec(lam);rows=[];g=[D(0),D(0)];H=[[D(0),D(0)],[D(0),D(0)]]
    for xx,yy in zip(x,y):
        xx=dec(xx);yy=int(yy);z=b+w*xx;p=1/(1+(-z).exp());om=1/(1+z.exp());loss=(1+((1-2*yy)*z).exp()).ln();dz=p if yy==0 else -om;v=p*om;A=[D(1),xx]
        gc=[dz*a/n for a in A];hc=[[v*a*c/n for c in A] for a in A]
        g=[a+c for a,c in zip(g,gc)];H=[[H[i][j]+hc[i][j] for j in range(2)] for i in range(2)]
        rows.append({'logit':z,'probability':p,'complement_probability':om,'loss':loss,'mean_loss_contribution':loss/n,'local_dloss_dp':(-1/p if yy else 1/om),'local_dp_dz':v,'stable_dloss_dz':dz,'gradient_contribution':gc,'hessian_contribution':hc})
    F=sum(r['loss'] for r in rows)/n;g[1]+=lam*w;H[1][1]+=lam
    return {'rows':rows,'gradient':g,'hessian':H,'data_loss':F,'objective':F+lam*w*w/2}

def convert(v):
    if isinstance(v,list):return [convert(a) for a in v]
    return float(v)

def audit_science(x,y,c,r):
    A=[[Q(1),Q(int(xx))] for xx in x];dz=[Q(1,2)-Q(int(yy)) for yy in y]
    g=[sum(d*a[j]/4 for d,a in zip(dz,A)) for j in range(2)];H=[[sum(a[i]*a[j]/16 for a in A) for j in range(2)] for i in range(2)]
    check(g==[Q(0),Q(-1,4)] and H==[[Q(1,4),Q(0)],[Q(0),Q(5,8)]],'Fraction initial g/H')
    check([-Q(1,2)*v for v in g]==[Q(0),Q(1,8)],'Fraction synchronous update')
    rng=np.random.default_rng(4217);states=[]
    with localcontext() as ctx:
        ctx.prec=110
        # Every retained row and state of the default GD path, not just final loss.
        for state in r['gd']['trace']:
            ref=decimal_state(x,y,state['theta'])
            for k in ('gradient','hessian','data_loss','objective'):close(state[k],convert(ref[k]),'Decimal full path '+k)
            for actual,wanted in zip(state['rows'],ref['rows']):
                for k,v in wanted.items():close(actual[k],convert(v),'Decimal row '+k)
        # 80 distinct dyadic designs, two labels, 110 digits, positive L2 included.
        for j in range(80):
            xx=rng.integers(-16,17,size=6)/8;yy=np.array([0,1,0,1,0,1]);t=rng.integers(-12,13,size=2)/8;lam=(j%3)/8
            actual=e.row_ledger(xx,yy,t,lam);ref=decimal_state(xx,yy,t,lam)
            for k in ('gradient','hessian','data_loss','objective'):close(actual[k],convert(ref[k]),'random Decimal '+k)
            vv=rng.normal(size=2);H0=np.array(actual['hessian']);close(vv@H0@vv,sum(float(rr['local_dp_dz'])*float(vv@[1.,xi])**2/len(xx) for rr,xi in zip(actual['rows'],xx))+lam*vv[1]**2,'weighted quadratic identity');check(vv@H0@vv>=-1e-14,'PSD')
        # Independent trajectory in high precision, not float inputs each time.
        w=D(0)
        for state in r['gd']['trace']:
            close(state['theta'][1],float(w),'110digit recurrence')
            gw=1/(1+(-2*w).exp())-1+D('.5')/(1+(-w).exp());w-=gw/2
        cert=e.cubic_certificate();t=D(cert['t']);check(abs(t**3-t-2)<D('1e-108'),'cubic residual');close(r['newton']['final']['theta'],[0,float(t.ln())],'Newton independent optimum')
    for fit,lam in zip(r['library'],[0,c['regularization_lambda']]):
        check(fit['classes']==[0.,1.] and not fit['convergence_warning'],'library labels/warnings')
        check(fit['gradient_norm']<1e-9,'library objective residual');close(e.sigmoid(fit['decision_function']),fit['positive_probability'],'probability ordering')
        own=r['newton'] if lam==0 else r['regularized_newton'];close(own['final']['theta'],fit['theta'],'target parity',atol=2e-9)
        check(fit['controlled_zero_score_probe']['library_predict']==0,'actual sklearn tie')
    close(r['duplicated_regularized_library']['theta'],r['library'][1]['theta'],'duplication theta',atol=2e-9)
    close(r['duplicated_regularized_library']['C'],r['library'][1]['C']/2,'duplication C')
    close(r['linear_probability']['theta'],[.5,.1],'linear probability OLS');close(r['linear_probability']['at_x_10'],1.5,'OLS extrapolation')
    check(r['decisions'][1]['accuracy']==.5,'retained half accuracy')
    for point in range(3):
        scan=r['diagnostics']['finite_difference'][point*len(c['fd_steps']):(point+1)*len(c['fd_steps'])]
        check(min(v['gradient_error'] for v in scan)<1e-9,'gradient h scan');check(min(v['hvp_error'] for v in scan)<1e-9,'Hv h scan')
    # Genuine finite positive tails at 40, underflow near 1000. 1100 digits retains e^-1000.
    with localcontext() as ctx:
        ctx.prec=1100
        for z in (-1000.,-740.,-40.,-1.,0.,1.,40.,740.,1000.):
            for yy in (0,1):
                ref=decimal_state([z],[yy],[0,1]);p,l,g,h=e.primitives([z],[yy]);rr=ref['rows'][0]
                for a,k in [(p[0],'probability'),(l[0],'loss'),(g[0],'stable_dloss_dz'),(h[0],'local_dp_dz')]:close(a,float(rr[k]),'1100-digit tail '+k,atol=0,rtol=2e-13)
                ledger=e.row_ledger([z],[yy],[0,1]);check(np.isfinite(ledger['objective']),'extreme ledger')
    from scipy.special import expit,log_expit
    z=np.array(c['extreme_logits']);close(e.sigmoid(z),expit(z),'actual SciPy expit',atol=1e-16);close(e.softplus(-z),-log_expit(z),'actual SciPy log_expit',atol=1e-16)
    ray=r['separation_ray'];check(all(a['loss']>b['loss']>0 for a,b in zip(ray,ray[1:])),'separation positive decreasing');check(ray[-1]['gradient_norm']<1e-10 and ray[-1]['theta_norm']==64,'small gradient divergent ray')
    return {'precision_main':110,'precision_extreme':1100,'default_gd_states':len(r['gd']['trace']),'random_dyadic_designs':80,'exact_initial_gradient':[str(v) for v in g] if isinstance(g,list) else ['0','-1/4'],'cubic':cert,'scipy_checked':True}

def audit_guards(x,y,c):
    bad=[True,'1',1+0j,np.nan,np.inf,-np.inf,1e200,1e-200,np.longdouble('1e-400')];total=0
    # Every failure occurs before internal state or the library fit, not just before output.
    for value in bad:
        cases=[lambda v=value:e.fit([*x[:-1],v],y),lambda v=value:e.fit(x,[*y[:-1],v]),lambda v=value:e.fit(x,y,initial=[0,v]),lambda v=value:e.fit(x,y,lam=v),lambda v=value:e.fit(x,y,max_updates=v),lambda v=value:e.primitives([0,v],[0,1])]
        for f in cases:
            with mock.patch.object(e,'_state',side_effect=RuntimeError('numerics reached')),mock.patch.object(e,'sigmoid',side_effect=RuntimeError('exponential reached')):reject(f,'raw gate')
            total+=1
    for key in c:
        cc=copy.deepcopy(c)
        if isinstance(cc[key],list):cc[key][-1]=True
        elif isinstance(cc[key],str):cc[key]='wrong'
        else:cc[key]=True
        reject(lambda cc=cc:e.validate_config(cc),'every config field')
    for xx,yy in [([1,2],[0,0]),([1,2],[1,1]),([1,2],[0,.5]),([[1],[2]],[0,1]),([],[])]:reject(lambda:e.fit(xx,yy),'shape/singleclass')
    for mode in ('gd','newton'):
        legal=e.fit([1,1],[0,1],initial=[0,0],method=mode);check(legal['status']=='gradient_tolerance','rank deficient legal stationary')
    check(e.fit(x,y,max_updates=1)['updates']==1,'short budget')
    check(e.fit(x,y,initial=[0,float(e.cubic_certificate()['w'])])['updates']==0,'initial stationary')
    # A legal internal state can move outside the +/-100 raw initial range.
    internal=e.fit([-1000,1000],[0,1],learning_rate=1,max_updates=1);check(internal['final']['theta'][1]==500,'do not raw-validate internal iterate')
    reject(lambda:e.fit(x,y,initial=[0,500]),'raw initial box')
    failed=e.fit(x,y,learning_rate=100,max_updates=1);check(failed['updates']==0 and failed['status']=='step_rejected','rejected candidate retained')
    check(failed['cost']['state_attempts']==2 and failed['cost']['state_successes']==2,'rejected state counted')
    orig=e._state;calls=[]
    def postfail(*args):
        result=orig(*args);calls.append(1)
        if len(calls)==2:raise FloatingPointError('injected after computation')
        return result
    with mock.patch.object(e,'_state',side_effect=postfail):post=e.fit(x,y,max_updates=2)
    check(post['status']=='numeric_failure' and post['cost']['state_attempts']==2 and post['cost']['state_successes']==1 and post['updates']==0,'postcompute attempt not omitted')
    with tempfile.TemporaryDirectory() as td:
        path=Path(td)/'bad.json'
        for s in ['{"a":1,"a":2}','{"a":NaN}','{"a":Infinity}','{"a":1e-400}','{"a":1e400}','{"a":-1e-400}']:
            path.write_text(s);reject(lambda:e.read_json(path),'strict JSON')
        path=Path(td)/'data.csv';path.write_text('id,x,y\nB1,1,0\nB1,2,1\n');reject(lambda:e.load_csv(path),'duplicate CSV id')
    for value in (0,np.float32(.5),np.float64(.5)):
        check(e.scalar(value,'legal')==float(value),'legal numeric scalar')
    return {'raw_adversarial_cases':total,'config_fields':len(c),'strict_json_cases':6,'rejected_step':failed['cost'],'postcompute_failure':post['cost']}

def audit_output():
    with tempfile.TemporaryDirectory() as td:
        d=Path(td);out=d/'old.json';out.write_bytes(b'OLD');e.safe_write(out,{'v':1});check(json.loads(out.read_text())=={'v':1},'valid commit');out.write_bytes(b'OLD')
        reject(lambda:e.safe_write(out,{'v':float('nan')}),'serialize before touching');check(out.read_bytes()==b'OLD','serialize preserve')
        reject(lambda:e.safe_write(e.ROOT/'experiment.py',{}),'source guard');reject(lambda:e.safe_write(e.ROOT/'data/model_spec.json',{}),'config guard')
        sy=d/'sy.json';sy.symlink_to(out);reject(lambda:e.safe_write(sy,{}),'symlink');ln=d/'linked.json';os.link(out,ln);reject(lambda:e.safe_write(ln,{}),'hardlink');ln.unlink()
        parent=d/'sym';parent.symlink_to(d,target_is_directory=True);reject(lambda:e.safe_write(parent/'new.json',{}),'symlink parent')
        with mock.patch.object(e.os,'replace',side_effect=OSError('controlled replace failure')):
            try:e.safe_write(out,{'v':2})
            except OSError:pass
            else:raise RuntimeError('replace did not fail')
        check(out.read_bytes()==b'OLD' and not list(d.glob('.ml042-*')),'replace atomicity/temp cleanup')
        with mock.patch.object(e.os,'replace',side_effect=OSError('controlled failure')):
            try:e.safe_write(d/'new/a/file.json',{'v':1})
            except OSError:pass
        check(not (d/'new').exists(),'new empty directory cleanup')
    return {'serialization_preserves':True,'replace_failure_preserves':True,'symlink_hardlink_protected':True,'new_directory_cleanup':True}

def audit():
    global COUNT
    COUNT=0;x,y,ids,sx,sy,sids,c=e.load_inputs();r=e.main_report(x,y,ids,sx,sy,sids,c)
    science=audit_science(x,y,c,r);gates=audit_guards(x,y,c);output=audit_output()
    return {'unit':'042','checks':COUNT,'science':science,'input_gates':gates,'output_gates':output,'core_sha256':hashlib.sha256(e.canonical_bytes(r)).hexdigest()}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();r=audit();e.safe_write(a.out,r);print(json.dumps({'checks':r['checks'],'core_sha256':r['core_sha256']}))
