"""Independent Fraction/Decimal, spectral, bad-input and fresh-CLI audit."""
from pathlib import Path
from fractions import Fraction as F
from decimal import Decimal, localcontext
import copy, hashlib, json, math, os, random, subprocess, sys, tempfile
import experiment as ex
ROOT=Path(__file__).resolve().parent
count=0

def check(ok,why):
    global count;count+=1
    if not ok:raise RuntimeError(why)

def close(a,b,why,scale=None):
    b=float(b);a=float(a);scale=max(abs(a),abs(b),1e-300) if scale is None else max(scale,abs(a),abs(b),1e-300)
    check(abs(a-b)<=3e-11*scale+16*math.ulp(scale),why+f': {a} vs {b}')

def bad(fn):
    try:fn()
    except (ValueError,TypeError):check(True,'rejected');return
    raise RuntimeError('bad input was accepted')

def exact_forward(data,p):
    pred=[r[0]*p[0]+r[1]*p[1] for r in data];res=[p-r[2] for p,r in zip(pred,data)]
    parts=[r*r/8 for r in res];c=[[r*x[0]/4,r*x[1]/4] for r,x in zip(res,data)]
    g=[sum(v[j] for v in c) for j in range(2)]
    return pred,res,parts,c,g,sum(parts)

def main():
    data,s=ex.load_inputs();report=ex.run_experiment(data,s)
    # Exact hand anchors and every per-row local derivative for both complete rounds.
    expected={'hb':[['1/16','0'],['39/256','-1/2']],'nag':[['1/16','0'],['77/512','0']]}
    for method in ['hb','nag']:
        h=report['hand'][method];check([r['new']['theta'] for r in h]==expected[method],'exact two-round anchors')
        for record in h:
            for key in ['old','evaluation','new']:
                fw=record[key];p=list(map(F,fw['theta']));pred,res,parts,c,g,loss=exact_forward([[F(v) for v in row] for row in data],p)
                check(str(loss)==fw['loss'] and list(map(str,g))==fw['gradient'],'hand totals')
                for i,row in enumerate(fw['rows']):
                    check(F(row['prediction'])==pred[i] and F(row['residual'])==res[i],'exact row forward')
                    check(F(row['loss_contribution'])==parts[i] and F(row['local_dloss_dresidual'])==res[i]/4,'exact local derivative')
                    check(list(map(F,row['gradient_contribution']))==c[i],'exact accumulation')
    rng=random.Random(2035);pathstates=0;roundedstates=0
    for k in range(120):
        scale=rng.choice([1,2,4,8]);dd=[[1,scale,1],[1,-scale,1],[-1,scale,-1],[-1,-scale,-1]]
        method=['gd','hb','nag'][k%3];beta=0 if method=='gd' else rng.choice([.125,.25,.5,.75])
        rate=rng.choice([1/128,1/64,1/32]);initial=[rng.randint(-8,8)/4,rng.randint(-8,8)/4]
        velocity=[0,0] if method=='gd' else [rng.randint(-2,2)/16,rng.randint(-2,2)/16]
        result=ex.run_method(dd,initial,velocity,method,rate,beta,12)
        p=list(map(F,initial));d=list(map(F,velocity));pr=list(map(float,initial));dr=list(map(float,velocity));dq=[[F(v) for v in row] for row in dd]
        for trace in result['trace']:
            evalp=[p[j]+F(beta)*d[j] for j in range(2)] if method=='nag' else list(p)
            preds,res,parts,c,g,loss=exact_forward(dq,evalp);nd=[F(beta)*d[j]-F(rate)*g[j] for j in range(2)];nt=[p[j]+nd[j] for j in range(2)]
            # Rounded independent primitive-order oracle reproduces actual float execution.
            rp=[pr[j]+beta*dr[j] for j in range(2)] if method=='nag' else list(pr)
            rgr=[0.,0.]
            for row in dd:
                residual=(row[0]*rp[0]+row[1]*rp[1])-row[2]
                for j in range(2):rgr[j]+=residual*row[j]/4
            rdr=[beta*dr[j]-rate*rgr[j] for j in range(2)];rpr=[pr[j]+rdr[j] for j in range(2)]
            check(rpr==trace['new']['theta'] and rdr==trace['new_velocity'],'primitive-order full state')
            scale_ref=max(1,*map(abs,evalp),*map(abs,g))
            for j in range(2):
                close(trace['evaluation']['gradient'][j],g[j],'independent Fraction gradient',scale_ref)
                close(trace['new']['theta'][j],nt[j],'independent Fraction parameter',scale_ref)
                close(trace['new_velocity'][j],nd[j],'independent Fraction velocity',scale_ref)
            for i,row in enumerate(trace['evaluation']['rows']):
                close(row['prediction'],preds[i],'row prediction',scale_ref)
                close(row['residual'],res[i],'row residual',scale_ref)
                close(row['loss_contribution'],parts[i],'row loss',scale_ref**2)
                for j in range(2):close(row['gradient_contribution'][j],c[i][j],'row gradient',scale_ref)
            close(trace['evaluation']['loss'],loss,'independent loss',scale_ref**2)
            p,d,pr,dr=nt,nd,rpr,rdr;pathstates+=1;roundedstates+=1
    # Decimal handles long real-rate recursions separately from production row engine.
    decimals=0
    with localcontext() as ctx:
        ctx.prec=100
        for name in ['gd_balanced','hb_quadratic_tuned','nag_strongconvex','nag_stress']:
            r=report['methods'][name];a=Decimal.from_float(r['rate']);b=Decimal.from_float(r['momentum']);p=[Decimal(0),Decimal(1)];d=[Decimal(0),Decimal(0)]
            for trace in r['trace']:
                ep=[p[j]+b*d[j] for j in range(2)] if r['method']=='nag' else p[:]
                g=[ep[0]-1,16*ep[1]];dn=[b*d[j]-a*g[j] for j in range(2)];pn=[p[j]+dn[j] for j in range(2)]
                # Roundoff near theta*=1 has an absolute scale tied to the represented coordinate, not loss.
                for j in range(2):
                    sc=max(1,*[abs(float(v)) for v in ep],*[abs(float(v)) for v in pn]) if name=='nag_stress' else max(1,abs(float(pn[j])))
                    close(trace['new']['theta'][j],pn[j],'100-digit full trajectory',sc)
                p,d=pn,dn;decimals+=1
    # Jury boundaries are mathematical strict inequalities, checked away from rounding borders.
    spectral_cases=0
    for method in ['hb','nag']:
        for beta in [0,.25,.5,.9]:
            for a in [.01,.25,.75,1,1.25,1.49,1.6,1.9,2.01,2.5,3.9]:
                v=ex.spectral(method,a/16,beta,16);upper=2*(1+beta) if method=='hb' else 2*(1+beta)/(1+2*beta)
                if abs(a-upper)<1e-12:continue
                check(v['strict_jury']==(0<a<upper),'Jury simplified exact region')
                check((v['spectral_radius']<1)==v['strict_jury'],'roots vs Jury')
                for re,im in v['roots']:
                    z=complex(re,im);check(abs(z*z-v['A']*z+v['D'])<1e-12,'root polynomial residual')
                spectral_cases+=1
    # Long-state stop and target are separate; no convergence claim from unchanged theta alone.
    check(report['probes']['nearby_floating_stagnation']['status']=='floating_state_stagnation','stagnation status')
    pp=report['probes']['parameter_pause_not_state_stop'];check(pp['trace'][0]['parameter_unchanged'] and not pp['trace'][0]['whole_state_unchanged'],'velocity still changing')
    check(pp['updates']>1,'do not prematurely stop on theta equality')
    check(report['probes']['same_point_with_velocity']['states'][1]['loss']>0,'optimum can be left with nonzero velocity')
    for name,m in report['methods'].items():
        check(m['updates']<=s['steps'] and m['training_sample_gradients']==4*m['training_full_forwards'] and m['training_full_forwards']<=s['steps'],'finite budget')
        for state in m['states']:
            bound=0.0
            for x,row in zip(data,state['rows']):
                eps=8*math.ulp(max(abs(x[0]*state['theta'][0]),abs(x[1]*state['theta'][1]),abs(row['prediction']),abs(x[2])))
                bound+=(2*abs(row['residual'])*eps+eps*eps)/8
            bound+=16*math.ulp(max(state['loss'],state['geometric_gap']))
            check(abs(state['loss']-state['geometric_gap'])<=bound,'forward/geometric loss ULP propagation')
    # Full last-field and raw-type rejection before any arithmetic helper is reached.
    invalid=[]
    for key,vals in {'initial':[[0,True],[0,'1'],[0,1,2]],'initial_velocity':[[0,float('nan')]],'steps':[True,2.,-1,401],
                     'target_gap':[0,'1e-6',1e-300],'hand_rate':[False,0,.4],'hand_momentum':[True,1],
                     'stress_rate':[float('inf')],'stress_momentum':[-.1], 'library_steps':[False,41],
                     'schedule_rates':[[.1,True],[.1,'0.1'],[.1,0],[.1]],'dampening':[True,1,float('nan')]}.items():
        for val in vals:ss=copy.deepcopy(s);ss[key]=val;invalid.append((data,ss))
    for val in [True,'-1',complex(1),F(1),Decimal(1),float('nan'),1e-101]:
        dd=copy.deepcopy(data);dd[-1][-1]=val;invalid.append((dd,s))
    old=ex._run;ex._run=lambda *a:(_ for _ in ()).throw(RuntimeError('computation before full validation'))
    try:
        for dd,ss in invalid:bad(lambda dd=dd,ss=ss:ex.run_experiment(dd,ss))
    finally:ex._run=old
    # Public entry-points do not quietly cast unsupported raw types.
    for value in [True,'0.1',F(1,4),Decimal('.25'),complex(.25)]:bad(lambda v=value:ex.spectral('hb',v,.5,16))
    import numpy as np
    bad(lambda:ex.run_method(data,[0,np.float64(1)],[0,0],'hb',.0625,.5,2))
    # Numerical-range guard is an explicit terminal state under a finite request.
    big=[[1,10,1],[1,-10,1],[-1,10,-1],[-1,-10,-1]]
    bigrun=ex.run_method(big,[0,1],[0,0],'nag',.3,.99,400)
    check(bigrun['status']=='numeric_range_limit' and bigrun['updates']<400,'overflow protected finite trace')
    # Independently count actual calls, including discarded evaluations at numeric stops.
    original_forward=ex._forward
    for method,dd,ii,vv,eta,beta,steps in [
        ('hb',data,[1,1e-100],[0,0],.0625,.5,5),
        ('nag',big,[0,1],[0,0],.3,.99,400),
        ('hb',big,[0,1],[0,0],.3,.99,400),
        ('hb',data,[1,0],[0,0],.0625,.5,5),
        ('hb',data,[0,1],[0,0],.0625,.5,0)]:
        calls=[]
        def counted(data,theta):
            calls.append(tuple(theta));return original_forward(data,theta)
        ex._forward=counted
        try:counted_run=ex.run_method(dd,ii,vv,method,eta,beta,steps)
        finally:ex._forward=original_forward
        check(counted_run['all_forward_calls']==len(calls),'actual forward-call count including stopped step')
        check(counted_run['diagnostic_full_forwards']==1+counted_run['updates'],'actual diagnostic calls')
        check(counted_run['training_full_forwards']==len(calls)-1-counted_run['updates'],'actual gradient evaluations')
        check(counted_run['training_sample_gradients']==4*counted_run['training_full_forwards'],'sample gradients include discarded evaluation')
    cancelled=ex.run_method(data,[1,1e-100],[0,0],'hb',.0625,.5,5)
    check(cancelled['updates']==0 and cancelled['training_full_forwards']==1 and cancelled['all_forward_calls']==2,'zero-update cancellation still consumes one gradient')
    # New processes, normal/-O, independent cwd, malformed LAST CSV/JSON fields, alias protection.
    failures=0
    (ROOT/'results').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT/'results') as td:
        td=Path(td);script=ROOT/'experiment.py';out=td/'old.json';other=td/'optimized.json'
        for mode,dest,cwd in [([],out,td),(['-O'],other,Path('/'))]:
            q=subprocess.run([sys.executable,*mode,str(script),'--output',str(dest)],cwd=cwd,capture_output=True,text=True)
            check(q.returncode==0,'fresh CLI '+q.stderr)
        check(out.read_bytes()==other.read_bytes(),'normal/-O complete JSON exact bytes');baseline=out.read_bytes()
        bad_files=[]
        cfg=td/'bad.json'
        for dd,ss in invalid[:15]:
            try:txt=json.dumps(ss)
            except TypeError:continue
            bad_files.append(('--config',txt))
        bad_files += [('--config',json.dumps(s)[:-1]+',"dampening":0.25}'),
                      ('--config',json.dumps(s).replace('0.25','1e-400')),
                      ('--data','id,x1,x2,y\nS1,1,4,1\nS2,1,-4,1\nS3,-1,4,-1\nS4,-1,-4,NaN\n'),
                      ('--data','id,x1,x2,y\nS1,1,4,1\nS2,1,-4,1\nS3,-1,4,-1\nS4,-1,-4,-1,EXTRA\n')]
        for flag,txt in bad_files:
            cfg.write_text(txt)
            for mode in [[],['-O']]:
                for dest in [out,td/'must-not-exist'/'new.json']:
                    q=subprocess.run([sys.executable,*mode,str(script),flag,str(cfg),'--output',str(dest)],capture_output=True)
                    check(q.returncode!=0,'bad CLI accepted');check(out.read_bytes()==baseline and not (td/'must-not-exist').exists(),'failed CLI changed output or made dir');failures+=1
        sy=td/'symbolic';sy.symlink_to(ROOT/'data/regression.csv');hard=td/'hard';os.link(ROOT/'data/regression.csv',hard)
        for dest in [ROOT/'lecture.md',ROOT/'experiment.py',ROOT/'data/regression.csv',sy,hard]:
            q=subprocess.run([sys.executable,str(script),'--output',str(dest)],capture_output=True)
            check(q.returncode!=0,'protected output allowed')
        digest=hashlib.sha256(baseline).hexdigest()
    result={'status':'passed','checks':count,'independent_Fraction_paths':120,'independent_Fraction_states':pathstates,
            'rounded_primitive_states':roundedstates,'Decimal_precision':100,'Decimal_full_states':decimals,
            'spectral_cases':spectral_cases,'pre_arithmetic_bad_input_cases':len(invalid),
            'bad_CLI_output_preservation_cases':failures,'protected_output_targets':5,'report_sha256':digest}
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
