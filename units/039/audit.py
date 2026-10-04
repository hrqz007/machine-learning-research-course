"""Reproducible author audit; input guards use exceptions, never assert."""
from pathlib import Path
from fractions import Fraction as Q
from decimal import Decimal as D, localcontext
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import numpy as np
import experiment as e


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def rejected(fn):
    try:
        fn()
    except (ValueError,TypeError,OverflowError):
        return
    raise RuntimeError('invalid input accepted')


def exact_forward(data, th, T=Q):
    n=T(len(data));out=[];g=[T(0),T(0)];f=T(0)
    for ident,x,y in data:
        x=T(x);y=T(y);p=th[0]+th[1]*x;r=p-y;ell=r*r/2
        gc=[r/n,r*x/n];g=[g[j]+gc[j] for j in range(2)];f+=ell/n
        out.append({'prediction':p,'residual':r,'half_square':ell,'mean_loss_contribution':ell/n,
                    'd_half_square_d_residual':r,'d_residual_d_prediction':T(1),
                    'd_prediction_d_theta':[T(1),x],'mean_gradient_contribution':gc})
    return {'rows':out,'gradient':g,'loss':f}


def run():
    data,s=e.load_inputs();r=e.run_experiment(data,s);checks={};count=0
    theta=[Q(0),Q(0)];exact_states=[]
    for k in range(3):
        ref=exact_forward(data,theta)
        actual=r['paths']['none']['initial'] if k==0 else r['paths']['none']['trace'][k-1]['new']
        for rr,aa in zip(ref['rows'],actual['rows']):
            for key,val in rr.items():
                if isinstance(val,list):
                    require([Q(z) for z in aa[key]]==val,'Fraction '+key);count+=len(val)
                else:require(Q(aa[key])==val,'Fraction '+key);count+=1
        require([Q(v) for v in actual['gradient']]==ref['gradient'] and Q(actual['loss'])==ref['loss'],'Fraction summary')
        exact_states.append({'theta':[str(v) for v in theta],'loss':str(ref['loss']),
                             'gradient':[str(v) for v in ref['gradient']],
                             'rows':[{k:([str(a) for a in v] if isinstance(v,list) else str(v)) for k,v in row.items()} for row in ref['rows']]})
        if k<2:
            theta=[theta[j]-Q(1,2)*ref['gradient'][j] for j in range(2)]
            require([Q(v) for v in r['paths']['none']['trace'][k]['trial_theta']]==theta,'synchronous update')
    require(theta==[Q(9,32),Q(13,32)] and ref['loss']==Q(103,4096),'two-round exact results')
    checks['fraction_two_round_scalars']=count
    checks['fraction_states']=exact_states
    # Independent rational stationary solution, without production linear algebra.
    optimum=exact_forward(data,[Q(3,10),Q(2,5)])
    require(optimum['gradient']==[0,0] and optimum['loss']==Q(1,40),'analytic optimum')
    checks['paired_optimum']={'theta':['3/10','2/5'],'loss':'1/40'}
    # Primitive binary64 arithmetic reference: every arithmetic op rounded once.
    def add(a,b):return float(Q(a)+Q(b))
    def mul(a,b):return float(Q(a)*Q(b))
    def div(a,b):return float(Q(a)/Q(b))
    rng=np.random.default_rng(390039)
    primitive=0;max_decimal=0.
    for case in range(32):
        a=[[float(i+1),float(rng.integers(-8,9))/4,float(rng.integers(0,2))] for i in range(4)]
        ss=copy.deepcopy(s);ss['initial_parameters']=(rng.integers(-4,5,size=2)/4).tolist();ss['learning_rate']=.125;ss['max_steps']=8
        path=e.run_path(a,ss)
        with localcontext() as ctx:
            ctx.prec=120
            th=[D(v) for v in ss['initial_parameters']]
            for tr in path['trace']:
                ff=exact_forward(a,th,D)
                decimal_new=[th[j]-D(ss['learning_rate'])*ff['gradient'][j] for j in range(2)]
                g=[0.,0.];f=0.
                for row,stored in zip(a,tr['old']['rows']):
                    x,y=row[1:];p=add(tr['old']['theta'][0],mul(tr['old']['theta'][1],x));z=add(p,-y);ell=mul(mul(.5,z),z);gc=[div(z,4.),div(mul(z,x),4.)]
                    require(p==stored['prediction'] and z==stored['residual'] and ell==stored['half_square'] and gc==stored['mean_gradient_contribution'],'primitive row')
                    f=add(f,div(ell,4.));g=[add(g[j],gc[j]) for j in range(2)];primitive+=7
                require(g==tr['old']['gradient'] and f==tr['old']['loss'],'primitive aggregate')
                upd=[mul(-ss['learning_rate'],v) for v in g];trial=[add(tr['old']['theta'][j],upd[j]) for j in range(2)]
                require(trial==tr['trial_theta'],'primitive update')
                err=max(abs(float(decimal_new[j])-trial[j]) for j in range(2));max_decimal=max(max_decimal,err)
                require(err<2e-12,'Decimal multistep comparison')
                th=decimal_new
    checks.update(random_dyadic_cases=32,primitive_checked_scalars=primitive,decimal_precision=120,max_decimal_parameter_error=max_decimal)
    # Costs are events, not completed_updates * n.
    for name,path in r['paths'].items():
        c=path['cost'];n=len(data);calls=len(path['trace'])+1
        require(c['objective_calls']==calls and c['gradient_calls']==calls,'event calls')
        require(c['prediction_rows']==calls*n,'prediction count')
        require(c['loss_elements']==calls*(n*n if name=='broadcast' else n),'loss element count')
        require(c['attempted_steps']==len(path['trace']) and c['committed_steps']==sum(t['accepted'] for t in path['trace']),'step counters')
        previous=path['initial']
        for tr in path['trace']:
            require(tr['old']==previous,'continuous trace')
            if tr['accepted']:previous=tr['new']
        require(path['final']==previous,'final committed state')
    for name in ('sign','large_step'):
        p=r['paths'][name];require(p['cost']['committed_steps']==0 and p['cost']['objective_calls']==2 and p['trace'][0]['new']['loss']>p['initial']['loss'],'rejected work counted')
    checks['costs']={k:v['cost'] for k,v in r['paths'].items()}
    checks['statuses']={k:v['status'] for k,v in r['paths'].items()}
    require(checks['statuses']=={'none':'gradient_tolerance','sign':'loss_increase_rejected','broadcast':'gradient_tolerance','tiny_step':'stagnation_gradient_large','large_step':'loss_increase_rejected'},'default statuses')
    table=r['gradient_checks']['table'];mid=next(v for v in table if v['h']==1e-5)
    require(mid['paired_error']<1e-9 and mid['logistic_error']<1e-9 and mid['sign_error']>1.9 and mid['broadcast_vs_intended_error']>.4 and mid['broadcast_self_error']<1e-9,'fault discrimination')
    require(table[0]['logistic_error']/table[1]['logistic_error']>90,'central h squared regime')
    require(table[-1]['logistic_error']>mid['logistic_error']*100,'roundoff regime')
    checks['finite_difference_h1e5']={k:mid[k] for k in ('paired_error','logistic_error','sign_error','broadcast_vs_intended_error','broadcast_self_error')}
    checks['logistic_truncation_ratio']=table[0]['logistic_error']/table[1]['logistic_error']
    # Kink cannot be called differentiable merely because central quotient is 0.
    h=.001;central=(abs(h)-abs(-h))/(2*h);left=(abs(0.)-abs(-h))/h;right=(abs(h)-abs(0.))/h
    require(central==0 and left==-1 and right==1,'kink diagnostic')
    checks['nonsmooth_demo']={'central':central,'left':left,'right':right,'verdict':'not differentiable at zero'}
    # Legal internal states are not subjected to narrow initial-value limits.
    large=e.square_forward(data,[100.,100.],e.Cost());require(large['loss']>0,'legal internal state')
    ss=copy.deepcopy(s);ss['initial_parameters']=[10,10];p=e.run_path(data,ss,'large_step')
    require(p['status']=='loss_increase_rejected' and 'new' in p['trace'][0] and max(map(abs,p['trace'][0]['trial_theta']))>10,'internal candidate beyond initial range evaluated')
    checks['internal_state_beyond_initial_domain']=True
    ss=copy.deepcopy(s);ss['initial_parameters']=[0.3000000001,0.4];ss['gradient_tolerance']=1e-16
    p=e.run_path(data,ss,'tiny_step');tr=p['trace'][0]
    require(tr['update']==[0.,0.] and any(v!=0 for v in tr['proposed_update']) and tr['relative_update']==0 and p['status']=='stagnation_gradient_large','actual versus proposed update quantization')
    checks['quantized_actual_step_distinguished_from_proposal']=True
    # Simulate a failure AFTER actual work; it remains in the cost ledger.
    original=e.square_forward;seen=[0]
    def fail_after_work(*args,**kw):
        out=original(*args,**kw);seen[0]+=1
        if seen[0]==2:raise FloatingPointError('injected after evaluation')
        return out
    e.square_forward=fail_after_work
    try:p=e.run_path(data,s)
    finally:e.square_forward=original
    require(p['status']=='numerical_failure' and p['cost']['objective_calls']==2 and p['cost']['loss_elements']==8 and p['cost']['committed_steps']==0,'failed-work accounting')
    checks['failed_after_work_cost']=p['cost']
    # ALL invalid inputs must be blocked before solver/observer/random/LA/output.
    calls=[]
    def bomb(*args,**kw):calls.append('reached');raise RuntimeError('forbidden work before full validation')
    original_report=e._report;e._report=bomb
    guards=0
    try:
        bad=[]
        for key,value in [('schema_version',True),('schema_version',1.),('max_steps',False),('max_steps',2.),('max_steps',0),('max_steps',1001),('learning_rate','0.5'),('learning_rate',True),('learning_rate',float('nan')),('learning_rate',float('inf')),('learning_rate',1e-310),('learning_rate',0),('learning_rate',1001),('initial_parameters',[0,True]),('initial_parameters',[[0,0]]),('initial_parameters',[0,11]),('gradient_tolerance',1e-310),('step_tolerance',False),('loss_tolerance',float('-inf')),('fd_parameters',[0,'1']),('fd_steps',[.01,.01]),('fd_steps',[.01,False]),('stress_parameters',[[0,0],[0,True]]),('stress_parameters',[[0,0],[0,'1000']]),('stress_parameters',[[0,0],[0,float('nan')]]),('stress_parameters',[[0,0],[0,1e-310]]),('stress_parameters',[[0,0],[0,2001]]),('stress_parameters',[[0,0],[0]]),('extra',1)]:
            q=copy.deepcopy(s);q[key]=value;bad.append(q)
        missing=copy.deepcopy(s);del missing['stress_parameters'];bad.append(missing)
        for q in bad:rejected(lambda q=q:e.run_experiment(data,q,observer=bomb));guards+=1
        for val in [True,'1',float('nan'),float('inf'),1e-310,2.]:
            a=copy.deepcopy(data);a[-1][-1]=val;rejected(lambda a=a:e.run_experiment(a,s,observer=bomb));guards+=1
        for a in [[],[[1,0,0]],[[1,0,0],[2,0]],np.array(1),[[1,0,0,1],[2,0,1,2]],[[1,0,0],[1,0,1]]]:
            rejected(lambda a=a:e.run_experiment(a,s,observer=bomb));guards+=1
        rejected(lambda:e.run_experiment(data,s,observer='not callable'));guards+=1
    finally:e._report=original_report
    require(not calls,'front gate failed');checks['pre_work_raw_input_rejections']=guards
    for raw in ['{"a":1,"a":2}','{"a":1e-999}','{"a":1e999}','{"a":NaN}','{"a":Infinity}','{"a":-Infinity}']:
        rejected(lambda raw=raw:e.strict_json(raw));guards+=1
    checks['strict_json_rejections']=6
    # Actual CSV/JSON LAST-field corruption. No output files may be created.
    cli=[]
    with tempfile.TemporaryDirectory() as temp:
        temp=Path(temp);dp=temp/'rows.csv';cp=temp/'config.json'
        good_csv=(e.ROOT/'data/observations.csv').read_text();good_config=(e.ROOT/'data/experiment_config.json').read_text()
        variants=[('csv-last-underflow',good_csv.replace('4,2,1','4,2,1e-999'),good_config),
                  ('csv-last-nonfinite',good_csv.replace('4,2,1','4,2,NaN'),good_config),
                  ('csv-last-extra',good_csv.replace('4,2,1','4,2,1,0'),good_config),
                  ('json-last-underflow',good_csv,good_config.replace('[0, -1000]','[0, 1e-999]')),
                  ('json-last-bool',good_csv,good_config.replace('[0, -1000]','[0, true]')),
                  ('json-duplicate',good_csv,good_config.replace('"schema_version": 1,','"schema_version": 1, "schema_version": 1,'))]
        for name,ds,cs in variants:
            dp.write_text(ds);cp.write_text(cs);out=temp/(name+'.json')
            proc=subprocess.run([sys.executable,'-O',str(e.ROOT/'experiment.py'),'--data',str(dp),'--config',str(cp),'--output',str(out)],cwd=temp,capture_output=True,text=True)
            require(proc.returncode!=0 and not out.exists() and not proc.stdout,'CLI mutated input not fully rejected')
            cli.append({'case':name,'returncode':proc.returncode,'stdout_bytes':len(proc.stdout),'output_exists':out.exists()})
        # Atomic writer failure preserves an old report and removes temp files.
        old=temp/'existing.json';old.write_bytes(b'OLD');replace=e.os.replace
        def no_replace(*a,**k):raise OSError('injected replace failure')
        e.os.replace=no_replace
        try:
            try:e.atomic_json(old,{'ok':1})
            except OSError:pass
            else:raise RuntimeError('replace failure did not propagate')
        finally:e.os.replace=replace
        require(old.read_bytes()==b'OLD' and not list(temp.glob('.unit039-*')),'atomic writer preservation')
    checks['last_field_cli_tests']=cli;checks['atomic_failure_preserves_old_report']=True
    # A normal output inside the unit may be replaced; it is not a source file.
    with tempfile.TemporaryDirectory(prefix='audit-repeat-',dir=e.ROOT) as temp:
        out=Path(temp)/'report.json';prior=None
        for _ in range(2):
            proc=subprocess.run([sys.executable,str(e.ROOT/'experiment.py'),'--output',str(out)],cwd=temp,capture_output=True,text=True)
            require(proc.returncode==0 and out.is_file(),'repeat output rejected')
            now=out.read_bytes()
            if prior is not None:require(now==prior,'repeat report drift')
            prior=now
    checks['existing_result_inside_unit_can_be_replaced']=True
    # CLI summaries must describe completed updates, even when fewer than two.
    short_cli=[]
    with tempfile.TemporaryDirectory() as td:
        td=Path(td)
        cases=[('one_update',dict(s,max_steps=1),1,'budget_exhausted'),
               ('initial_stationary',dict(s,initial_parameters=[.3,.4]),0,'gradient_tolerance'),
               ('first_rejected',dict(s,learning_rate=10.),0,'loss_increase_rejected')]
        for name,config,updates,status in cases:
            cp=td/(name+'-config.json');cp.write_text(json.dumps(config))
            previous=None
            for flags in ([],['-O']):
                out=td/(name+('-O' if flags else '')+'.json')
                proc=subprocess.run([sys.executable,*flags,str(e.ROOT/'experiment.py'),'--config',str(cp),'--output',str(out)],cwd=td,capture_output=True,text=True)
                require(proc.returncode==0 and out.is_file(),'short CLI failed after write')
                summary=json.loads(proc.stdout);payload=json.loads(out.read_text());path=payload['paths']['none']
                require(summary['completed_updates']==updates==path['cost']['committed_steps'],'short CLI committed count')
                require(summary['statuses']['none']==status==path['status'],'short CLI status')
                require(summary['final_theta']==path['final']['theta'] and summary['final_loss']==path['final']['loss'],'short CLI actual final')
                require(summary['two_round_theta'] is None and summary['two_round_loss'] is None,'short CLI invented second update')
                if previous is not None:require(out.read_bytes()==previous,'short CLI normal/-O mismatch')
                previous=out.read_bytes();short_cli.append({'case':name,'optimized':bool(flags),'completed_updates':updates,'status':status,'returncode':proc.returncode})
    checks['short_trace_cli_summary_cases']=short_cli
    # Preserve source nonzero semantics before binary64 conversion.
    tiny=np.longdouble('1e-400')
    require(tiny!=0,'platform longdouble does not support this boundary probe')
    q=copy.deepcopy(s);q['stress_parameters'][-1][0]=tiny
    reached=[];original_report=e._report
    def extended_bomb(*args,**kw):reached.append(True);raise RuntimeError('extended underflow reached work')
    e._report=extended_bomb
    try:rejected(lambda:e.run_experiment(data,q))
    finally:e._report=original_report
    require(not reached,'extended underflow was not rejected before work')
    require(e.real(np.longdouble(0),'zero')==0 and e.real(np.float64(.25),'float64')==.25 and e.real(np.float32(.25),'float32')==.25,'supported NumPy scalar changed')
    checks['extended_precision_nonzero_conversion_guard']=True
    # Output is fully JSON-finite; nonfinite fault values are explicit strings.
    json.dumps(r,allow_nan=False)
    for case in r['stability']['cases']:
        require(all(math.isfinite(v['loss']) for v in case['stable']['rows']),'stable extreme logits')
    checks['no_nan_averaging']=True
    return {'unit':'039','status':'author checks passed','checks':checks,
            'limits':['Finite tests do not prove all-input correctness.','No browser UI or Anaconda installation tested.']}


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);a=p.parse_args();r=run()
    if a.output:e.atomic_json(a.output,r)
    print(json.dumps({'status':r['status'],'raw_rejections':r['checks']['pre_work_raw_input_rejections'],'primitive_scalars':r['checks']['primitive_checked_scalars']},sort_keys=True))
