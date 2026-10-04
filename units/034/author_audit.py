"""Independent exact arithmetic, complete path and failure-protection audit."""
from pathlib import Path
from fractions import Fraction as F
from decimal import Decimal, localcontext
import copy, hashlib, itertools, json, math, subprocess, sys, tempfile
import numpy as np
import experiment as e
ROOT=Path(__file__).resolve().parent
count=0

def check(ok,why):
    global count;count+=1
    if not ok:raise RuntimeError(why)

def close(a,b,why,tol=2e-12):check(abs(float(a)-float(b))<=tol*max(1,abs(float(b))),why)
def bad(fn):
    try:fn()
    except (ValueError,TypeError):check(True,'expected invalid');return
    raise RuntimeError('invalid input accepted')

def main():
    y,s=e.load_inputs();ys=list(map(F,map(int,y)));eta=F(1,4);mu=sum(ys)/4;sigma=sum((v-mu)**2 for v in ys)/4
    hand=e.hand_chain(y,s);check([v['parameter'] for v in hand['states']]==['0','5/4','7/16'],'hand parameters');check([v['loss'] for v in hand['states']]==['15/4','105/32','1745/512'],'hand full losses')
    for st in hand['states']:
        w=F(st['parameter']);close(sum(F(r['half_square']) for r in st['rows'])/4,F(st['loss']),'full loss sum')
        for yi,row in zip(ys,st['rows']):check(F(row['residual'])==w-yi and F(row['full_gradient_contribution'])==(w-yi)/4,'hand forward chain')
    # Complete exact path enumeration, not the same moment recursion as production.
    path_checks=0
    for B,T in [(1,t) for t in range(7)]+[(2,t) for t in range(4)]:
        means=[];squares=[];decay_values=[]
        for flat in itertools.product(range(4),repeat=B*T):
            w=F(0);d=F(0)
            for t in range(T):
                by=sum(ys[i] for i in flat[B*t:B*(t+1)])/B;w=w-eta*(w-by);d=d-F(1,t+4)*(d-by)
            means.append(w-mu);squares.append((w-mu)**2);decay_values.append(d-mu);path_checks+=1
        m=sum(means)/len(means);q=sum(squares)/len(squares);dm=sum(decay_values)/len(decay_values);dq=sum(v*v for v in decay_values)/len(decay_values)
        th=e.theory(T,.25,float(sigma/B),-1);dt=e.theory(T,.25,float(sigma/B),-1,4)
        for a,b in [(th['mean_error'],m),(th['mean_squared_error'],q),(dt['mean_error'],dm),(dt['mean_squared_error'],dq)]:close(a,b,'complete path moment')
    # Every permutation, exact one-epoch formula and actual rounded batch engine.
    perms=[];small=copy.deepcopy(s);small['repetitions']=2;small['sample_gradient_budget']=4
    for order in itertools.permutations(range(4)):
        w=F(0)
        for i in order:w-=eta*(w-ys[i])
        formula=sum(eta*(1-eta)**(3-j)*ys[i] for j,i in enumerate(order));check(w==formula,'order exact formula');perms.append(str(w))
        arr=np.array([[list([i]) for i in order]]*2);states,tr=e.run_batches(y,small,arr)
        close(states[0,-1],w,'permutation solver')
    check(perms[0]=='157/128' and perms[-1]=='43/256','order anchors')
    # Independent random dyadic labels, rates and explicit per-sample operations.
    rng=np.random.default_rng(1997);rounded=0;decimal_refs=0
    for k in range(150):
        yy=rng.integers(-64,65,size=4)/8;ss=copy.deepcopy(s);ss.update(repetitions=2,sample_gradient_budget=16,constant_rate=float(rng.integers(1,8))/8,initial_parameter=float(rng.integers(-10,11))/4)
        arr=rng.integers(0,4,size=(2,8,2));states,tr=e.run_batches(yy,ss,arr,k%2==0)
        for r in range(2):
            wf=float(ss['initial_parameter']);wx=F(wf)
            for t in range(8):
                a=1/(t+4) if k%2==0 else ss['constant_rate'];res=[wf-float(yy[i]) for i in arr[r,t]];g=(res[0]+res[1])/2;wf=wf-a*g
                check(states[r,t+1]==wf,'rounded actual primitive order');rounded+=1
                wx-=F(a)*(wx-sum(F(float(yy[i])) for i in arr[r,t])/2)
            close(wf,wx,'independent exact stored-input trace',1e-10)
        with localcontext() as ctx:
            ctx.prec=100;ae=Decimal.from_float(ss['constant_rate']);v=Decimal(13)/4;rho=(1-ae)**2;t=8;ref=rho**t+ae*v/(2-ae)*(1-rho**t)
            close(e.theory(t,ss['constant_rate'],13/4,-1)['mean_squared_error'],float(ref),'Decimal moment oracle');decimal_refs+=1
    # All invalid configuration cases must fail BEFORE even one generator is made.
    invalid=[]
    for key,vals in {'constant_rate':[True,'0.25',0,-1,2,float('nan'),1e-300],'repetitions':[True,1,2001],'root_seed':[-1,True,2**32],'batch_sizes':[[1,2,True],[1,2,5],[1,1],[],[2]],'decay_offset':[1,False,101],'sample_gradient_budget':[3,True,2001],'hand_indices_one_based':[[4,0],[4,True],[4]],'importance_probabilities':[[.125,.125,.25,0],[.125,.125,.25,True],[.1,.1,.1,.1]]}.items():
        for val in vals:sp=copy.deepcopy(s);sp[key]=val;invalid.append((y,sp))
    for yy in [[-2,0,True,5],[-2,0,'1',5],[-2,0,1,float('inf')],[-2,0,1,1e-300],[[1,2],[3,4]],[],[1]]:invalid.append((yy,s))
    old=e.make_rng;e.make_rng=lambda *a:(_ for _ in ()).throw(RuntimeError('RNG touched before validation'))
    try:
        for yy,sp in invalid:bad(lambda yy=yy,sp=sp:e.run_experiment(yy,sp))
    finally:e.make_rng=old
    # Batch invalid last index and bool arrays are rejected before solver update.
    arr=np.zeros((2,2,2),int);arr[-1,-1,-1]=4;bad(lambda:e.run_batches(y,small,arr));bad(lambda:e.run_batches(y,small,np.zeros((2,2,2),bool)))
    # CLI fresh process and byte-preservation, including malformed final input row.
    cli=0
    with tempfile.TemporaryDirectory() as td:
        td=Path(td);a=td/'a';b=td/'b';script=str(ROOT/'experiment.py')
        for optimized,out,cwd in [(False,a,td),(True,b,Path('/'))]:
            q=subprocess.run([sys.executable]+(['-O'] if optimized else [])+[script,'--output-dir',str(out)],cwd=cwd,capture_output=True,text=True);check(q.returncode==0,'fresh CLI failed')
        check((a/'report.json').read_bytes()==(b/'report.json').read_bytes(),'normal optimized reports differ')
        oldbytes=(a/'report.json').read_bytes()
        for sp in [z[1] for z in invalid[:12]]:
            config=td/'bad.json';config.write_text(json.dumps(sp));new=td/'never'
            for out in [a,new]:
                q=subprocess.run([sys.executable,script,'--config',str(config),'--output-dir',str(out)],capture_output=True);check(q.returncode!=0,'bad config CLI success');check((a/'report.json').read_bytes()==oldbytes and not new.exists(),'bad CLI mutated output');cli+=1
        data=td/'bad.csv';data.write_text('id,y\nS1,-2\nS2,0\nS3,1\nS4,NaN\n')
        for out in [a,td/'new']:
            q=subprocess.run([sys.executable,script,'--data',str(data),'--output-dir',str(out)],capture_output=True);check(q.returncode!=0 and (a/'report.json').read_bytes()==oldbytes and not (td/'new').exists(),'bad late CSV guard');cli+=1
        report=json.loads(oldbytes)
        for name,m in report['methods'].items():
            check(m['sample_gradients_per_run']<=160,'budget overspend');check(m['sample_gradients_per_run']==m['updates']*m['batch_size'],'cost mismatch')
            for r in m['curve']:
                close(r['mean_squared_error'],r['variance_population']+r['mean_error']**2,'moment decomposition');close(r['mean_excess'],r['mean_squared_error']/2,'loss expectation');check(r['excess_q10']<=r['excess_q90'],'quantiles reversed')
            for r in m['first_run_trace']:
                close(sum(r['gradient_contributions']),r['gradient'],'batch accumulation');close(r['new_parameter'],r['old_parameter']-r['rate']*r['gradient'],'actual update');close(r['new_full_loss'],sum(v*v for v in r['new_full_residual'])/8,'next forward')
        digest=hashlib.sha256(oldbytes).hexdigest()
    result={'status':'passed','checks':count,'complete_exact_paths':path_checks,'permutations':24,'random_dyadic_cases':150,'actual_rounded_states':rounded,'Decimal_precision':100,'Decimal_references':decimal_refs,'invalid_pre_rng_cases':len(invalid),'bad_cli_output_protections':cli,'report_sha256':digest}
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
