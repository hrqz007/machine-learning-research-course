"""Author-side reproducible checks; not an independent acceptance decision."""
from pathlib import Path
from fractions import Fraction as Q
from decimal import Decimal as D, localcontext
import copy,hashlib,json,math,os,subprocess,sys,tempfile
import numpy as np
import experiment as e
ROOT=Path(__file__).resolve().parent

def require(test,message):
    if not test:raise RuntimeError(message)
def reject(fn):
    try:fn()
    except (ValueError,TypeError,OverflowError):return
    raise RuntimeError('invalid input accepted')

def rational_forward(rows,th):
    n=Q(len(rows));pr=[sum(row[j]*th[j] for j in range(2)) for row in rows];r=[v-row[2] for v,row in zip(pr,rows)];loss=[z*z/2 for z in r];contrib=[[z*row[j]/n for j in range(2)] for z,row in zip(r,rows)];g=[sum(v[j] for v in contrib) for j in range(2)]
    return pr,r,loss,contrib,g,sum(loss)/n

def reference(data,s,steps=2,decimal=False):
    T=D if decimal else Q
    a=[[T(float(v)) for v in row] for row in data];th=[T(float(v)) for v in s['initial_parameters']];eta=T(float(s['learning_rate']));lam=T(float(s['lambda']));out=[]
    for _ in range(steps):
        n=T(len(a));p=[sum((row[j]*th[j] for j in range(2)),T(0)) for row in a];r=[v-row[2] for v,row in zip(p,a)];ls=[z*z/2 for z in r];gc=[[z*row[j]/n for j in range(2)] for z,row in zip(r,a)];g=[sum((v[j] for v in gc),T(0)) for j in range(2)];z=[v-eta*w for v,w in zip(th,g)];tau=eta*lam;new=[v-tau if v>tau else v+tau if v < -tau else T(0) for v in z]
        out.append({'theta':th[:],'prediction':p,'residual':r,'loss':ls,'contribution':gc,'gradient':g,'z':z,'new':new});th=new
    return out

def primitive_reference(data,theta,s):
    # Emulate each actually-used binary64 product/add/divide through exact rationals.
    def mul(a,b):return float(Q(float(a))*Q(float(b)))
    def add(a,b):return float(Q(float(a))+Q(float(b)))
    def div(a,b):return float(Q(float(a))/Q(float(b)))
    g=[0.,0.];rows=[]
    for row in data:
        products=[mul(row[j],theta[j]) for j in range(2)];p=add(*products);r=add(p,-row[2]);loss=div(mul(r,r),2);gc=[div(mul(r,row[j]),len(data)) for j in range(2)];g=[add(g[j],gc[j]) for j in range(2)];rows.append((p,r,loss,gc))
    z=[add(theta[j],-mul(s['learning_rate'],g[j])) for j in range(2)];tau=mul(s['learning_rate'],s['lambda']);new=[add(v,-tau) if v>tau else add(v,tau) if v < -tau else 0. for v in z]
    return rows,g,z,new

def main():
    data,s=e.load_inputs();report=e.run_experiment(data,s);checks={};scalar=0;max_decimal=0.
    exact=reference(data,s)
    for k,v in enumerate(exact):
        tr=report['paths']['proximal']['trace'][k];fw=tr['old_forward']
        for i,row in enumerate(fw['rows']):
            for key,refkey in [('prediction','prediction'),('residual','residual'),('half_square','loss')]:require(Q(row[key])==v[refkey][i],'exact row '+key);scalar+=1
            for j in range(2):require(Q(row['mean_gradient_contribution'][j])==v['contribution'][i][j],'exact contribution');scalar+=1
            require(row['d_loss_d_residual']==row['residual'] and row['d_residual_d_prediction']==1 and row['d_prediction_d_theta']==data[i,:2].tolist(),'local derivative chain')
        for key,refkey in [('gradient','gradient')]:require([Q(z) for z in fw[key]]==v[refkey],'exact gradient')
        require([Q(z) for z in tr['smooth_trial']]==v['z'],'exact z');require([Q(z) for z in tr['new_forward']['theta']]==v['new'],'exact new')
        pr,rs,ls,gc,gg,ff=rational_forward([[Q(float(z)) for z in row] for row in data],v['new'])
        nf=tr['new_forward']
        for i,row in enumerate(nf['rows']):
            for key,expected in [('prediction',pr[i]),('residual',rs[i]),('half_square',ls[i])]:require(Q(row[key])==expected,'exact new row '+key);scalar+=1
            for j in range(2):require(Q(row['mean_gradient_contribution'][j])==gc[i][j],'exact new contribution');scalar+=1
        require(Q(nf['data_loss'])==ff and [Q(z) for z in nf['gradient']]==gg,'exact new forward summary')
    require(report['paths']['proximal']['trace'][1]['new_forward']['objective']==141/128,'J2');require(report['paths']['proximal']['trace'][1]['new_forward']['theta']==[9/8,0.],'theta2')
    checks['fraction_two_full_rounds']=True
    rng=np.random.default_rng(370037)
    for case in range(32):
        a=rng.integers(-12,13,size=(4,3))/4;ss=copy.deepcopy(s);ss.update(initial_parameters=(rng.integers(-8,9,size=2)/4).tolist(),learning_rate=.125,steps=8,**{'lambda':.375})
        pp=e.run_path(a,ss,'proximal')
        with localcontext() as ctx:
            ctx.prec=120;ref=reference(a,ss,len(pp['trace']),True)
        for t,tr in enumerate(pp['trace']):
            pr,g,z,new=primitive_reference(a,tr['old_forward']['theta'],ss)
            require(g==tr['old_forward']['gradient'] and z==tr['smooth_trial'] and new==tr['new_forward']['theta'],'actual float primitive path')
            for i,row in enumerate(tr['old_forward']['rows']):require(pr[i][0]==row['prediction'] and pr[i][1]==row['residual'] and pr[i][2]==row['half_square'] and pr[i][3]==row['mean_gradient_contribution'],'primitive rows')
            err=max(abs(float(ref[t]['new'][j])-new[j]) for j in range(2));max_decimal=max(max_decimal,err);require(err<=2e-9*(1+max(map(abs,new))),'Decimal reference tolerance');scalar+=4*5+6
    checks['random_dyadic_cases']=32;checks['decimal_precision']=120;checks['max_decimal_parameter_abs_error']=max_decimal;checks['primitive_chain_checked_scalars']=scalar
    # Differentiable f only; central differences do not differentiate the L1 kink.
    theta=np.array([.2,-.7]);g=np.array(e.forward(data,theta,.5)['gradient']);fd=[]
    for j in range(2):
        d=np.eye(2)[j]*1e-6;fd.append((e.forward(data,theta+d,0)['data_loss']-e.forward(data,theta-d,0)['data_loss'])/(2e-6))
    require(np.allclose(g,fd,atol=1e-9,rtol=0),'smooth finite differences');checks['finite_difference_max_error']=float(np.max(np.abs(g-fd)))
    require(e.soft(np.array([-1.,-.25,0.,.25,1.]),.25).tolist()==[-.75,0.,0.,0.,.75],'threshold boundaries')
    for m,p in report['paths'].items():
        require(len(p['trace'])==p['completed_updates'],'trace length');require(p['attempted_updates']>=p['completed_updates'],'attempt count')
        prev=p['initial_forward']
        for tr in p['trace']:
            require(tr['old_forward']==prev,'path continuity');prev=tr['new_forward'];require(abs(prev['objective']-(prev['data_loss']+prev['l1_penalty']))<1e-13,'objective sum')
            if m in ('proximal','coordinate'):require(prev['objective']<=tr['old_forward']['objective']+1e-13,'default objective monotonicity')
        require(prev['theta']==p['final_theta'],'final theta')
        if p['status']=='tolerance_satisfied':require(p['final_diagnostics']['stopping_residual']<=s['tolerance'],'real stop')
    checks['paths_and_statuses']={m:{'status':p['status'],'updates':p['completed_updates']} for m,p in report['paths'].items()}
    # Front gate sentinel: invalid LAST fields must never reach even one solver.
    old=e.run_path;calls=[]
    def bomb(*a,**k):calls.append(1);raise RuntimeError('solver reached')
    e.run_path=bomb
    bads=[]
    for key,value in [('stress_rho',True),('stress_rho','0.5'),('stress_rho',float('nan')),('stress_rho',1e-310),('coordinate_order',[0,True]),('coordinate_order',[0,0]),('initial_parameters',[0.,True]),('steps',False),('lower_bounds',[0,0]),('extra',1)]:
        b=copy.deepcopy(s);b[key]=value
        if key=='lower_bounds':b['upper_bounds']=[0,1]
        bads.append(b)
    for b in bads:reject(lambda:e.run_experiment(data,b))
    for val in [True,'1',float('inf'),1e-310]:
        a=data.tolist();a[-1][-1]=val;reject(lambda:e.run_experiment(a,s))
    for a in [np.array(1),[],[[1,2,3]],[[1,2,3],[1,2]],[[1,2,3,4]]*4]:reject(lambda:e.run_experiment(a,s))
    e.run_path=old;require(not calls,'solver ran before full validation');checks['pre_solver_rejections']=len(bads)+9
    # Boundary configurations remain valid even when some paths stop arithmetically.
    for initial in [[100.,-100.],[-100.,100.]]:
        ss=copy.deepcopy(s);ss['initial_parameters']=initial;rr=e.run_experiment(data,ss);require(rr['paths']['proximal']['initial_forward']['theta']==initial,'custom initial retained')
    tiny=np.full((4,3),1e-100);e.run_experiment(tiny,s)
    huge=np.full((4,3),100.);ss=copy.deepcopy(s);ss.update(initial_parameters=[100.,100.],learning_rate=10.,steps=200);rr=e.run_experiment(huge,ss);require(rr['paths']['proximal']['status'].startswith('arithmetic_range_stop'),'range stop recorded');checks['custom_boundary_and_range_stop']=True
    # No coercion under -O: no tests or guards use Python assert.
    with tempfile.TemporaryDirectory(prefix='unit037-audit-') as td:
        td=Path(td);csvpath=td/'data.csv';config=td/'config.json';csvpath.write_bytes((ROOT/'data/regression.csv').read_bytes());config.write_bytes((ROOT/'data/model_spec.json').read_bytes())
        invalids=[config.read_text().replace('"stress_rho": 0.875','"stress_rho": true'),config.read_text().replace('"stress_rho": 0.875','"stress_rho": 1e-999'),config.read_text().replace('"stress_rho": 0.875','"stress_rho": 0.875, "stress_rho": 0.5')]
        for text in invalids:config.write_text(text);reject(lambda:e.load_inputs(csvpath,config))
        config.write_bytes((ROOT/'data/model_spec.json').read_bytes())
        for text in ['id,x1,x2,y\na,1,2,3\nb,1,2,NaN\n','id,x1,x2,y\na,1,2,3\nb,1,2,1e-999\n','id,x1,x2,y\na,1,2,3\na,1,2,4\n','id,x1,x2,y\na,1,2,3\nb,1,2,4,5\n']:
            csvpath.write_text(text);reject(lambda:e.load_inputs(csvpath,config))
        csvpath.write_bytes((ROOT/'data/regression.csv').read_bytes())
        out=td/'existing';out.mkdir();oldfile=out/'report.json';oldfile.write_bytes(b'old sentinel\n');config.write_text(invalids[0])
        for target in [out,td/'fresh']:
            cp=subprocess.run([sys.executable,*(['-O'] if sys.flags.optimize else []),str(ROOT/'experiment.py'),'--data',str(csvpath),'--config',str(config),'--output-dir',str(target)],capture_output=True)
            require(cp.returncode!=0,'bad CLI accepted');require(oldfile.read_bytes()==b'old sentinel\n','old output changed');require(not (td/'fresh').exists(),'failed input created directory')
        config.write_bytes((ROOT/'data/model_spec.json').read_bytes());before=hashlib.sha256(csvpath.read_bytes()).hexdigest()
        sy=td/'symbol';sy.symlink_to(out,target_is_directory=True);reject(lambda:e.write_report(report,sy/'report.json'))
        hard=td/'hard.json';os.link(csvpath,hard);reject(lambda:e.write_report(report,hard,(csvpath,config)))
        asset=ROOT/'lecture.md';asset_hash=hashlib.sha256(asset.read_bytes()).hexdigest();reject(lambda:e.write_report(report,asset));require(hashlib.sha256(asset.read_bytes()).hexdigest()==asset_hash,'asset mutated')
        reject(lambda:e.write_report(report,csvpath,(csvpath,config)));require(hashlib.sha256(csvpath.read_bytes()).hexdigest()==before,'input mutated')
        invalid_payload={'bad':float('nan')};reject(lambda:e.write_report(invalid_payload,td/'never'/'report.json'));require(not (td/'never').exists(),'serialize failure made dir')
        replace=e.os.replace
        def fail_replace(*args):raise OSError('deliberate atomic replacement failure')
        e.os.replace=fail_replace
        try:
            for target in [oldfile,td/'new'/'nested'/'report.json']:
                try:e.write_report(report,target)
                except OSError:pass
                else:raise RuntimeError('replacement failure ignored')
            require(oldfile.read_bytes()==b'old sentinel\n','write failure changed old bytes')
            require(not (td/'new').exists(),'write failure left new directories')
        finally:e.os.replace=replace
    checks['file_and_cli_protection']=True
    # Normal-cone projection criterion and prox KKT at the analytical points.
    d=e.diagnostics(data,np.array([1.5,0.]),s,'proximal');require(d['lasso_kkt_inf']==0 and d['proximal_mapping_inf']==0 and d['smooth_gradient_inf']==.5,'Lasso optimum')
    d=e.diagnostics(data,np.array([1.,.25]),s,'projected');require(d['projected_mapping_inf']==0 and d['smooth_gradient_inf']==1,'box optimum')
    checks['analytical_optimality']=True
    print(json.dumps({'status':'author checks passed; independent review pending','python':sys.version.split()[0],'numpy':np.__version__,'checks':checks},ensure_ascii=False,sort_keys=True,indent=2))
if __name__=='__main__':main()
