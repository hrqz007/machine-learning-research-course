"""Same-objective stochastic mean regression; actual batch updates, bounded CPU work.

The CLI needs NumPy only. Theory functions are separate from iterative updates.
All data/config validation precedes the first random generator construction.
"""
from pathlib import Path
from fractions import Fraction as F
from decimal import Decimal, InvalidOperation
import argparse, csv, hashlib, itertools, json, math, os, sys, tempfile
import numpy as np

ROOT=Path(__file__).resolve().parent
FIELDS={'kind','initial_parameter','constant_rate','decay_offset','sample_gradient_budget','repetitions','root_seed','batch_sizes','hand_indices_one_based','importance_probabilities'}

def real(x,name='value',limit=1e4):
    t=type(x)
    if t not in (int,float) and not (t.__module__=='numpy' and t.__name__ in ('int8','int16','int32','int64','uint8','uint16','uint32','uint64','float16','float32','float64')):raise ValueError(name+' must be ordinary finite real')
    try:v=float(x)
    except (OverflowError,ValueError) as e:raise ValueError(name+' conversion') from e
    if not math.isfinite(v) or abs(v)>limit or (v!=0 and abs(v)<1e-8):raise ValueError(name+' outside teaching range')
    if isinstance(x,(int,np.integer)) and int(v)!=int(x):raise ValueError(name+' integer loses precision')
    return v

def integer(x,name,lo,hi):
    if type(x) is not int or not lo<=x<=hi:raise ValueError(name+' must be bounded Python integer')
    return x

def labels_checked(y):
    if not isinstance(y,(list,tuple,np.ndarray)) or isinstance(y,np.ndarray) and y.ndim!=1:raise ValueError('labels must be one dimensional')
    if not 2<=len(y)<=64:raise ValueError('need 2..64 labels')
    return np.array([real(v,'label') for v in y],dtype=float)

def validate(y,spec):
    y=labels_checked(y);n=len(y)
    if type(spec) is not dict or set(spec)!=FIELDS:raise ValueError('incorrect config fields')
    s=dict(spec)
    if s['kind']!='synthetic_stochastic_mean_regression':raise ValueError('incorrect problem kind')
    s['initial_parameter']=real(s['initial_parameter'],'initial parameter')
    s['constant_rate']=real(s['constant_rate'],'rate',2)
    if not 0<s['constant_rate']<2:raise ValueError('rate must be between 0 and 2')
    for k,lo,hi in [('decay_offset',2,100),('sample_gradient_budget',4,2000),('repetitions',2,2000),('root_seed',0,2**32-1)]:s[k]=integer(s[k],k,lo,hi)
    if s['sample_gradient_budget']*s['repetitions']>500000:raise ValueError('CPU budget exceeds teaching limit')
    b=s['batch_sizes']
    if type(b) is not list or not b or len(b)>8:raise ValueError('batch list required')
    b=[integer(v,'batch',1,n) for v in b]
    if len(set(b))!=len(b) or 1 not in b:raise ValueError('unique batch sizes including 1 required')
    if s['sample_gradient_budget']<max(n,max(b)):raise ValueError('budget too small for complete batch')
    s['batch_sizes']=b
    h=s['hand_indices_one_based']
    if type(h) is not list or len(h)!=2:raise ValueError('two hand indices required')
    s['hand_indices_one_based']=[integer(v,'hand index',1,n) for v in h]
    p=s['importance_probabilities']
    if type(p) is not list or len(p)!=n:raise ValueError('probability dimension')
    p=[real(v,'probability',1) for v in p]
    if any(v<=0 for v in p) or abs(math.fsum(p)-1)>1e-12:raise ValueError('positive probabilities summing to one required')
    s['importance_probabilities']=p
    return y,s

def token(x):
    try:d=Decimal(x)
    except (InvalidOperation,TypeError) as e:raise ValueError('invalid numeric token') from e
    if not d.is_finite():raise ValueError('nonfinite token')
    v=float(d)
    if d!=0 and v==0:raise ValueError('token underflow')
    return real(v)

def pairs(items):
    out={}
    for k,v in items:
        if k in out:raise ValueError('duplicate JSON key')
        out[k]=v
    return out

def load_inputs(data=ROOT/'data/labels.csv',config=ROOT/'data/model_spec.json'):
    with Path(data).open(newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=['id','y']:raise ValueError('CSV header must be id,y')
        rows=list(reader)
    if any(set(r)!= {'id','y'} or None in r.values() or not r['id'] or len(r['id'])>40 for r in rows):raise ValueError('invalid CSV row')
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('duplicate sample id')
    y=[token(r['y']) for r in rows]
    s=json.loads(Path(config).read_text(),parse_float=token,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite JSON')),object_pairs_hook=pairs)
    return validate(y,s)

def objective(y,w):
    y=labels_checked(y);w=real(w,'parameter',1e6)
    r=w-y
    return float(np.mean(r*r)/2)

def exact_forward(y,w):
    ys=[F(float(v)) for v in y];w=F(w);n=len(ys)
    rows=[{'id':i+1,'prediction':str(w),'residual':str(w-v),'half_square':str((w-v)**2/2),'local_dloss_dresidual':str(w-v),'dprediction_dw':'1','full_gradient_contribution':str((w-v)/n)} for i,v in enumerate(ys)]
    return {'parameter':str(w),'rows':rows,'loss':str(sum((w-v)**2 for v in ys)/(2*n)),'full_gradient':str(sum(w-v for v in ys)/n)}

def hand_chain(y,s):
    w=F(s['initial_parameter']);eta=F(s['constant_rate']);states=[exact_forward(y,w)];steps=[]
    for i in s['hand_indices_one_based']:
        r=w-F(float(y[i-1]));next_w=w-eta*r
        steps.append({'index_one_based':i,'old_parameter':str(w),'selected_prediction':str(w),'selected_residual':str(r),'selected_half_square':str(r*r/2),'dL_dr':str(r),'dr_dprediction':'1','dprediction_dw':'1','batch_gradient':str(r),'rate':str(eta),'new_parameter':str(next_w)})
        w=next_w;states.append(exact_forward(y,w))
    return {'reference':'exact arithmetic on stored binary64 input values; not substituted into floating solver','states':states,'steps':steps}

def theory(t,eta,var,e0,decay_offset=None):
    if decay_offset is None:
        a=1-eta;mean=a**t*e0;r=a*a;m=r**t*e0*e0+eta*var/(2-eta)*(1-r**t)
    else:
        c=decay_offset-1;mean=c*e0/(t+c);m=(c*c*e0*e0+t*var)/(t+c)**2
    return {'mean_error':mean,'mean_squared_error':m,'variance':max(0.,m-mean*mean),'mean_excess':m/2}

def make_rng(root,group,rep):return np.random.Generator(np.random.PCG64(np.random.SeedSequence([root,group,rep])))

def generate_batches(n,s,b,mechanism,group):
    R=s['repetitions'];T=s['sample_gradient_budget']//b
    if mechanism=='full':return np.broadcast_to(np.arange(n),(R,T,n)).copy()
    if mechanism in ('forward','reverse'):
        a=np.arange(n) if mechanism=='forward' else np.arange(n-1,-1,-1)
        return np.broadcast_to(np.resize(a,T),(R,T)).copy()[:,:,None]
    out=np.empty((R,T,b),dtype=np.int64)
    for rep in range(R):
        rng=make_rng(s['root_seed'],group,rep)
        if mechanism=='replace':out[rep]=rng.integers(0,n,size=(T,b))
        elif mechanism=='subset':
            for t in range(T):out[rep,t]=rng.choice(n,b,replace=False)
        elif mechanism=='reshuffle':
            seq=np.concatenate([rng.permutation(n) for _ in range((T+n-1)//n)])[:T];out[rep,:,0]=seq
        else:raise ValueError('unknown mechanism')
    return out

def run_batches(y,s,batches,decay=False):
    # Complete validation, including the LAST index, before updating any parameter.
    y,s=validate(y,s)
    if not isinstance(batches,np.ndarray) or batches.dtype.kind not in 'iu' or batches.ndim!=3 or batches.shape[0]!=s['repetitions'] or not 1<=batches.shape[2]<=len(y) or batches.shape[1]*batches.shape[2]>s['sample_gradient_budget'] or batches.shape[1]==0 or np.any(batches<0) or np.any(batches>=len(y)):raise ValueError('invalid batch tensor')
    if type(decay) is not bool:raise ValueError('decay flag must be bool')
    R,T,B=batches.shape;w=np.full(R,s['initial_parameter'],dtype=float);states=np.empty((R,T+1));states[:,0]=w
    trace=[]
    for t in range(T):
        old=w.copy();selected=y[batches[:,t,:]];residual=old[:,None]-selected
        gradient=np.mean(residual,axis=1);eta=1/(t+s['decay_offset']) if decay else s['constant_rate']
        w=old-eta*gradient;states[:,t+1]=w
        trace.append({'update':t+1,'cost':(t+1)*B,'indices_one_based':(batches[0,t]+1).tolist(),'old_parameter':float(old[0]),'prediction_per_selected_row':[float(old[0])]*B,'residual_per_selected_row':residual[0].tolist(),'half_square_per_selected_row':(residual[0]**2/2).tolist(),'local_dloss_dresidual':residual[0].tolist(),'dr_dprediction':[1.]*B,'dprediction_dw':[1.]*B,'gradient_contributions':(residual[0]/B).tolist(),'gradient':float(gradient[0]),'rate':eta,'new_parameter':float(w[0]),'new_full_prediction':[float(w[0])]*len(y),'new_full_residual':(w[0]-y).tolist(),'new_full_loss':float(np.mean((w[0]-y)**2)/2)})
    return states,trace

def summarise(states,y,s,B,variance=None,decay=False):
    mu=math.fsum(y)/len(y);e=states-mu;loss=e*e/2;R=states.shape[0];records=[];full_losses=np.mean((states[:,:,None]-y[None,None,:])**2,axis=2)/2
    for t in range(states.shape[1]):
        es=e[:,t];ls=loss[:,t];m=float(np.mean(es));mse=float(np.mean(es*es));v=float(np.mean((es-m)**2));se=float(np.std(ls,ddof=1)/math.sqrt(R));record={'updates':t,'sample_gradients':t*B,'equivalent_passes':t*B/len(y),'mean_error':m,'mean_squared_error':mse,'variance_population':v,'mean_excess':mse/2,'mean_full_loss':float(np.mean(full_losses[:,t])),'excess_of_mean_parameter':m*m/2,'excess_q10':float(np.quantile(ls,.1)),'excess_q90':float(np.quantile(ls,.9)),'mean_excess_standard_error':se,'first_run_parameter':float(states[0,t]),'first_run_excess':float(ls[0])}
        if variance is not None:
            th=theory(t,s['constant_rate'],variance,s['initial_parameter']-mu,s['decay_offset'] if decay else None);record['theory']=th
            record['theory_mean_excess_z']=(record['mean_excess']-th['mean_excess'])/se if se>1e-14 else None
        records.append(record)
    return records

def exact_enumeration(y,w):
    ys=[F(float(v)) for v in y];mu=sum(ys)/len(ys);w=F(w);n=len(ys);sigma=sum((v-mu)**2 for v in ys)/n
    out={}
    for name,combos in [('single',[(i,) for i in range(n)]),('replace_b2',list(itertools.product(range(n),repeat=2))),('subset_b2',list(itertools.combinations(range(n),2)))]:
        gs=[w-sum(ys[i] for i in c)/len(c) for c in combos];mean=sum(gs)/len(gs);v=sum((g-mean)**2 for g in gs)/len(gs)
        out[name]={'cases':len(gs),'gradient_mean':str(mean),'gradient_variance':str(v)}
    out['sigma2']=str(sigma);return out

def run_experiment(y,s):
    y,s=validate(y,s) # MUST precede any make_rng call.
    n=len(y);mu=math.fsum(y)/n;sigma=math.fsum((float(v)-mu)**2 for v in y)/n
    methods={};paired={}
    plans=[('full',n,'full',0,0.)]+[(f'replace_b{b}',b,'replace',100+b,sigma/b) for b in s['batch_sizes']]+[('subset_b2',2,'subset',202,sigma*(n-2)/(2*(n-1))),('reshuffle',1,'reshuffle',300,None),('forward',1,'forward',0,None),('reverse',1,'reverse',0,None)]
    for name,b,mechanism,group,v in plans:
        batches=generate_batches(n,s,b,mechanism,group);states,trace=run_batches(y,s,batches)
        def put(key,arr,tr,decay):
            methods[key]={'mechanism':mechanism,'batch_size':b,'schedule':'decay' if decay else 'constant','updates':len(tr),'sample_gradients_per_run':len(tr)*b,'unused_budget':s['sample_gradient_budget']-len(tr)*b,'diagnostic_full_loss_sample_terms_per_run':n*(len(tr)+1),'diagnostic_first_run_trace_extra_sample_terms':n*len(tr),'final_parameters':arr[:,-1].tolist(),'tail_policy':'stop before an incomplete batch','replications':s['repetitions'],'deterministic_duplicates':mechanism in ('full','forward','reverse'),'stream_group':group if mechanism not in ('full','forward','reverse') else None,'curve':summarise(arr,y,s,b,v,decay),'first_run_trace':tr}
        put(name,states,trace,False)
        if mechanism=='replace' and b in (1,2):
            dstate,dtrace=run_batches(y,s,batches,True);put(f'decay_b{b}',dstate,dtrace,True)
            dif=(dstate[:,-1]-mu)**2/2-(states[:,-1]-mu)**2/2
            paired[f'b{b}']={'comparison':'decay minus constant excess at same final cost, SAME sampled index stream','mean_difference':float(np.mean(dif)),'standard_error_of_paired_difference':float(np.std(dif,ddof=1)/math.sqrt(s['repetitions']))}
    p=s['importance_probabilities'];gs=[s['initial_parameter']-float(v) for v in y];weighted=math.fsum(pi*g for pi,g in zip(p,gs));corrected=math.fsum(pi*g/(n*pi) for pi,g in zip(p,gs))
    return {'unit':'034','numpy_version':np.__version__,'bit_generator':'PCG64','seed_contract':'SeedSequence([root_seed, method_group, repetition]); paired constant/decay reuse identical index arrays','config':s,'labels':y.tolist(),'optimum':mu,'minimum_loss':sigma/2,'single_gradient_noise_variance':sigma,'hand':hand_chain(y,s),'enumeration':exact_enumeration(y,s['initial_parameter']),'importance':{'probabilities':p,'uncorrected_expectation':weighted,'corrected_expectation':corrected,'full_gradient':s['initial_parameter']-mu},'methods':methods,'paired':paired,'limitations':['Synthetic scalar constant regression only; no generalisation or hardware throughput claim.','Quantile bands show run distributions, not mean confidence intervals.','Exact moment theory applies only to fresh independent sampling families.','Diagnostic objective evaluation cost is separate from training sample-gradient budget.','Finite Monte Carlo departures are reported with standard errors, not forced to equal theory.']}

def report_bytes(report):return (json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,default=ROOT/'data/labels.csv');p.add_argument('--config',type=Path,default=ROOT/'data/model_spec.json');p.add_argument('--output-dir',type=Path,default=ROOT/'outputs');a=p.parse_args()
    y,s=load_inputs(a.data,a.config);out=a.output_dir.resolve();target=out/'report.json'
    # Reject input clobbering and file-as-directory before expensive work, without mutation.
    inputs=[a.data.resolve(),a.config.resolve(),Path(__file__).resolve()]
    if target in inputs or out.exists() and not out.is_dir():raise ValueError('unsafe output target')
    report=run_experiment(y,s);payload=report_bytes(report)
    out.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.report-',suffix='.tmp',dir=out)
    try:
        with os.fdopen(fd,'wb') as f:f.write(payload)
        os.replace(tmp,target)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    print(json.dumps({'unit':'034','bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest(),'methods':len(report['methods'])}))
if __name__=='__main__':main()
