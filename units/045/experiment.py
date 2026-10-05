"""ML045: from a supplied class posterior to a minimum conditional-risk action.

No probability model is trained. Known toy posteriors and population weights
permit exact expected-cost evaluation, separately from random observed losses.
"""
from pathlib import Path
from decimal import Decimal
import argparse,csv,hashlib,io,json,math,os,tempfile
import numpy as np
ROOT=Path(__file__).resolve().parent
FIELDS={'kind','false_positive_cost','false_negative_cost','reject_cost','cost_pairs',
 'thresholds','source_prior','target_prior','simulation_seed','samples_per_group',
 'multiclass_probabilities','multiclass_costs','tie_rule'}


def scalar(x,name,lo=0.,hi=10000.,integer=False):
    if isinstance(x,(bool,np.bool_)) or not isinstance(x,(int,float,np.integer,np.floating)):
        raise ValueError(name+' requires a real number, not bool/string/complex')
    if integer and not isinstance(x,(int,np.integer)):raise ValueError(name+' requires integer type')
    try:v=float(x)
    except (ValueError,OverflowError) as ex:raise ValueError(name+' cannot convert') from ex
    if not math.isfinite(v) or not lo<=v<=hi or (x!=0 and (v==0 or abs(v)<1e-100)):
        raise ValueError(name+' outside supported finite raw range')
    return int(x) if integer else v


def array(x,name,ndim,shape=None,lo=0.,hi=10000.):
    if not isinstance(x,(list,tuple,np.ndarray)):raise ValueError(name+' requires list/tuple/array')
    try:a=np.asarray(x,dtype=object)
    except ValueError as ex:raise ValueError(name+' must be rectangular') from ex
    if a.ndim!=ndim or any(v==0 for v in a.shape) or a.size>40000:raise ValueError(name+' bad shape')
    if shape is not None and a.shape!=shape:raise ValueError(name+' shape mismatch')
    return np.array([scalar(v,name,lo,hi) for v in a.flat]).reshape(a.shape)


def distribution(v,name):
    a=array(v,name,1,lo=0,hi=1);total=math.fsum(a)
    if abs(total-1)>1e-12:raise ValueError(name+' must already sum to one within 1e-12')
    return a/total


def probabilities(P,name='probabilities'):
    a=array(P,name,2,lo=0,hi=1)
    if not 1<=len(a)<=2000 or not 2<=a.shape[1]<=20:raise ValueError('P scope n=1..2000 K=2..20')
    totals=np.array([math.fsum(row) for row in a])
    if np.any(abs(totals-1)>1e-12):raise ValueError(name+' rows must sum to one within 1e-12')
    return a/totals[:,None]


def risk_table(P,costs):
    """Cost rows are actions, columns are true classes. All raw values checked first."""
    P=probabilities(P);C=array(costs,'costs',2)
    if not 1<=len(C)<=20 or C.shape[1]!=P.shape[1]:raise ValueError('cost matrix must have 1..20 action rows and K columns')
    contributions=P[:,None,:]*C[None,:,:]
    risks=np.array([[math.fsum(a) for a in row] for row in contributions])
    chosen=np.argmin(risks,axis=1)
    rows=[]
    for i in range(len(P)):
        minimum=float(risks[i,chosen[i]])
        rows.append({'row':i+1,'posterior':P[i].tolist(),'class_loss_contributions':contributions[i].tolist(),
                     'conditional_risks':risks[i].tolist(),'chosen_action':int(chosen[i]),
                     'tied_actions':np.flatnonzero(risks[i]==minimum).tolist(),'minimum_conditional_risk':minimum})
    return {'rows':rows,'costs':C.tolist(),'risks':risks.tolist(),'actions':chosen.tolist(),
            'tie_rule':'first listed action among exactly equal computed risks'}


def expected_cost(P,costs,actions,weights):
    P=probabilities(P);C=array(costs,'costs',2)
    if not 1<=len(C)<=20 or C.shape[1]!=P.shape[1]:raise ValueError('cost shape mismatch')
    a=array(actions,'actions',1,(len(P),),0,len(C)-1)
    if np.any(a!=np.floor(a)):raise ValueError('actions must be integer-valued')
    w=distribution(weights,'weights')
    if len(w)!=len(P):raise ValueError('weights length mismatch')
    a=a.astype(int);per_class=P*C[a];risk=np.array([math.fsum(row) for row in per_class]);contrib=w*risk
    return {'expected_cost':float(math.fsum(contrib)),'per_group_risk':risk.tolist(),
            'weighted_risk_contribution':contrib.tolist(),'actions':a.tolist(),'weights':w.tolist()}


def binary_costs(fp,fn,reject=None):
    fp=scalar(fp,'false_positive');fn=scalar(fn,'false_negative')
    if reject is not None:reject=scalar(reject,'reject_cost')
    return [[0.,fn],[fp,0.]]+([[reject,reject]] if reject is not None else [])


def binary_threshold(fp,fn):
    fp=scalar(fp,'false_positive');fn=scalar(fn,'false_negative')
    return None if fp+fn==0 else fp/(fp+fn)


def reject_bounds(fp,fn,reject):
    fp=scalar(fp,'false_positive');fn=scalar(fn,'false_negative');r=scalar(reject,'reject')
    if fp==0 or fn==0:return {'strict_reject_region':False,'lower':None,'upper':None,'reason':'one error can be avoided at zero cost'}
    lower=r/fn;upper=1-r/fp
    return {'strict_reject_region':bool(lower<upper),'lower':lower,'upper':upper,
            'reason':'reject strictly optimal when lower < p < upper; equality uses action order'}


def threshold_actions(p,threshold):
    p=array(p,'p',1,lo=0,hi=1);t=scalar(threshold,'threshold',0,1)
    return (p>t).astype(int)  # equality chooses action 0, aligned with ordered cost rows


def posterior_from_likelihoods(negative,positive,prior):
    q0=distribution(negative,'P(X|0)');q1=distribution(positive,'P(X|1)');pi=scalar(prior,'prior',0,1)
    if q0.shape!=q1.shape:raise ValueError('class likelihood supports differ')
    negative_mass=(1-pi)*q0;positive_mass=pi*q1;marginal=negative_mass+positive_mass
    if np.any(marginal==0):raise ValueError('some bins have zero marginal probability; conditional undefined')
    posterior=positive_mass/marginal
    return {'prior':pi,'negative_joint_mass':negative_mass.tolist(),'positive_joint_mass':positive_mass.tolist(),
            'marginal':marginal.tolist(),'positive_posterior':posterior.tolist()}


def correct_prior(p,source_prior,target_prior):
    p=array(p,'source posterior',1,lo=0,hi=1)
    source=scalar(source_prior,'source prior',1e-6,1-1e-6);target=scalar(target_prior,'target prior',1e-6,1-1e-6)
    # Avoid infinite odds for p=0 or 1; normalize two nonnegative reweighted masses.
    numerator=(target/source)*p
    other=((1-target)/(1-source))*(1-p)
    return numerator/(numerator+other)


def parse_number(s):
    d=Decimal(s);v=float(d)
    if not math.isfinite(v) or (d!=0 and (v==0 or abs(v)<1e-100)):raise ValueError('numeric token cannot be represented in supported raw range')
    return v


def pairs(items):
    r={}
    for k,v in items:
        if k in r:raise ValueError('duplicate JSON key')
        r[k]=v
    return r


def read_json(path):
    return json.loads(Path(path).read_text(),parse_float=parse_number,object_pairs_hook=pairs,
                     parse_constant=lambda s:(_ for _ in ()).throw(ValueError('nonfinite JSON')))


def validate_config(c):
    if not isinstance(c,dict) or set(c)!=FIELDS or c['kind']!='probability_to_decision_v1':raise ValueError('config fields/kind mismatch')
    q=dict(c)
    for k in ['false_positive_cost','false_negative_cost','reject_cost']:q[k]=scalar(c[k],k)
    q['cost_pairs']=array(c['cost_pairs'],'cost_pairs',2).tolist()
    if any(len(v)!=2 for v in q['cost_pairs']) or len(q['cost_pairs'])>20:raise ValueError('cost pair shape')
    q['thresholds']=array(c['thresholds'],'thresholds',1,lo=0,hi=1).tolist()
    if len(q['thresholds'])>100:raise ValueError('too many thresholds')
    for k in ['source_prior','target_prior']:q[k]=scalar(c[k],k,1e-6,1-1e-6)
    q['simulation_seed']=scalar(c['simulation_seed'],'seed',0,2**32-1,True)
    q['samples_per_group']=scalar(c['samples_per_group'],'samples_per_group',1,5000,True)
    q['multiclass_probabilities']=probabilities(c['multiclass_probabilities']).tolist()
    C=array(c['multiclass_costs'],'multiclass costs',2)
    if not 1<=len(C)<=20 or C.shape[1]!=len(q['multiclass_probabilities'][0]):raise ValueError('multiclass costs shape')
    q['multiclass_costs']=C.tolist()
    if c['tie_rule']!='first listed action among exactly equal computed risks':raise ValueError('tie rule mismatch')
    return q


def load_csv(path,columns):
    reader=csv.DictReader(io.StringIO(Path(path).read_text()))
    if reader.fieldnames!=columns:raise ValueError('CSV columns mismatch')
    out=[];seen=set()
    for row in reader:
        if set(row)!=set(columns) or any(v is None for v in row.values()):raise ValueError('malformed CSV')
        name=row['id']
        if not name or len(name)>40 or not name.isascii() or not all(v.isalnum() or v in '_-' for v in name) or name in seen:raise ValueError('bad/duplicate ID')
        seen.add(name);out.append([name,*[parse_number(row[k]) for k in columns[1:]]])
    if not out:raise ValueError('CSV empty')
    return out


def load_inputs(data=None,shift=None,config=None):
    c=validate_config(read_json(config or ROOT/'data/decision_spec.json'))
    rows=load_csv(data or ROOT/'data/posteriors.csv',['id','p_positive','weight'])
    p=array([v[1] for v in rows],'positive posterior',1,lo=0,hi=1);w=distribution([v[2] for v in rows],'population weights')
    sr=load_csv(shift or ROOT/'data/label_shift.csv',['id','p_x_given_negative','p_x_given_positive'])
    q0=distribution([v[1] for v in sr],'negative class likelihood');q1=distribution([v[2] for v in sr],'positive class likelihood')
    return p,w,[v[0] for v in rows],q0,q1,[v[0] for v in sr],c


def _simulate(p,w,policies,costs,seed,per_group):
    # Inputs here were validated by main_report before any RNG use.
    rng=np.random.default_rng(seed);labels=(rng.random((len(p),per_group))<p[:,None]).astype(int);reports={}
    for name,actions in policies.items():
        C=np.array(costs[name]);a=np.array(actions,dtype=int)
        observed=C[a[:,None],labels];mean_by_group=observed.mean(axis=1)
        true_risk=(1-p)*C[a,0]+p*C[a,1]
        variance=p*(1-p)*(C[a,1]-C[a,0])**2
        reports[name]={'observed_mean_cost':float(w@mean_by_group),'expected_cost':float(w@true_risk),
         'conditional_cost_variance':variance.tolist(),'stratified_mean_standard_error':float(np.sqrt(np.sum(w*w*variance/per_group))),
         'observed_group_mean_cost':mean_by_group.tolist(),'observed_positive_counts':labels.sum(axis=1).tolist()}
    from sklearn.metrics import confusion_matrix
    for name in ['cost_optimal','accuracy_optimal']:
        a=np.repeat(np.array(policies[name],dtype=int),per_group)
        cm=confusion_matrix(labels.ravel(),a,labels=[0,1],sample_weight=np.repeat(w/per_group,per_group))
        C=np.array(costs[name]);reports[name]['sklearn_weighted_confusion']=cm.tolist()
        reports[name]['sklearn_recomputed_cost']=float(np.sum(cm*C.T))
    return {'sampling':'independent Bernoulli labels within each fixed stratum, fixed per-group sample count',
            'seed':seed,'per_group':per_group,'total_labels':int(labels.size),'results':reports}


def main_report(p,w,ids,q0,q1,shift_ids,c):
    c=validate_config(c);p=array(p,'positive posterior',1,lo=0,hi=1);w=distribution(w,'weights')
    if len(w)!=len(p) or not isinstance(ids,(list,tuple)) or len(ids)!=len(p) or any(not isinstance(v,str) or not v or len(v)>40 for v in ids) or len(set(ids))!=len(ids):raise ValueError('main lengths/IDs')
    q0=distribution(q0,'P(X|0)');q1=distribution(q1,'P(X|1)')
    if len(q0)!=len(q1) or not isinstance(shift_ids,(list,tuple)) or len(shift_ids)!=len(q0) or any(not isinstance(v,str) or not v or len(v)>40 for v in shift_ids) or len(set(shift_ids))!=len(shift_ids):raise ValueError('shift shapes/IDs')
    if len(p)>2000:raise ValueError('too many rows')
    P=np.column_stack([1-p,p]);C=binary_costs(c['false_positive_cost'],c['false_negative_cost']);CR=binary_costs(c['false_positive_cost'],c['false_negative_cost'],c['reject_cost'])
    decision=risk_table(P,C);reject=risk_table(P,CR);accuracy=risk_table(P,[[0,1],[1,0]])
    policies={'cost_optimal':decision['actions'],'accuracy_optimal':accuracy['actions'],'with_reject':reject['actions']}
    evals={name:expected_cost(P,CR if name=='with_reject' else C,a,w) for name,a in policies.items()}
    for name in ['cost_optimal','accuracy_optimal']:
        aa=np.array(policies[name]);evals[name]['expected_error_rate']=float(w@np.where(aa==0,p,1-p))
    evals['with_reject']['reject_mass']=float(w@(np.array(reject['actions'])==2))
    sweep=[]
    for t in c['thresholds']:
        a=threshold_actions(p,t);sweep.append({'threshold':t,**expected_cost(P,C,a,w)})
    costs=[]
    for fp,fn in c['cost_pairs']:
        CC=binary_costs(fp,fn);d=risk_table(P,CC)
        costs.append({'FP':fp,'FN':fn,'threshold':binary_threshold(fp,fn),'actions':d['actions'],'expected_cost':expected_cost(P,CC,d['actions'],w)['expected_cost']})
    source=posterior_from_likelihoods(q0,q1,c['source_prior']);target=posterior_from_likelihoods(q0,q1,c['target_prior'])
    corrected=correct_prior(source['positive_posterior'],c['source_prior'],c['target_prior']);SP=np.column_stack([1-np.array(source['positive_posterior']),source['positive_posterior']]);TP=np.column_stack([1-np.array(target['positive_posterior']),target['positive_posterior']])
    stale=risk_table(SP,C);new=risk_table(TP,C)
    shift={'source':source,'target':target,'corrected_positive_posterior':corrected.tolist(),'stale_actions':stale['actions'],'corrected_actions':new['actions'],
       'stale_expected_target_cost':expected_cost(TP,C,stale['actions'],target['marginal']),
       'corrected_expected_target_cost':expected_cost(TP,C,new['actions'],target['marginal']),
       'assumption':'P_target(X|Y)=P_source(X|Y); class priors known in this toy example'}
    multi=risk_table(c['multiclass_probabilities'],c['multiclass_costs']);multi['maximum_probability_actions']=np.argmax(c['multiclass_probabilities'],axis=1).tolist()
    simulation=_simulate(p,w,policies,{'cost_optimal':C,'accuracy_optimal':C,'with_reject':CR},c['simulation_seed'],c['samples_per_group'])
    return {'unit':'045','config':c,'data':{'ids':ids,'positive_posterior':p.tolist(),'weights':w.tolist(),'shift_ids':shift_ids,'class_likelihood_negative':q0.tolist(),'class_likelihood_positive':q1.tolist()},
      'decision':decision,'rejection':reject,'accuracy_decision':accuracy,'evaluation':evals,
      'binary_threshold':binary_threshold(c['false_positive_cost'],c['false_negative_cost']),
      'rejection_bounds':reject_bounds(c['false_positive_cost'],c['false_negative_cost'],c['reject_cost']),
      'threshold_sweep':sweep,'cost_scenarios':costs,'label_shift':shift,'multiclass':multi,'simulation':simulation,
      'risk_derivatives_wrt_positive_probability':[c['false_negative_cost'],-c['false_positive_cost'],0.],
      'versions':{'numpy':np.__version__,'sklearn':__import__('sklearn').__version__},
      'limitations':['supplied oracle toy probabilities, no trained/calibrated model claim','population costs known only because toy distribution specified','no test-label threshold selection; threshold sweep is a predeclared population mechanism check','constant reject cost abstracts review cost and any residual errors','prior correction assumes invariant class conditionals and known priors','exact ties compared on computed float risks; near-boundary rounding may change action on arbitrary inputs']}


def canonical_bytes(obj):return (json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def safe_write(path,obj,protected=()):
    raw=canonical_bytes(obj);p=Path(path).absolute()
    for v in (p,*p.parents):
        if v.is_symlink():raise ValueError('symlink output component')
    if p.suffix!='.json' or (p.exists() and (not p.is_file() or p.stat().st_nlink>1)):raise ValueError('unsafe JSON output')
    assets=[v for v in ROOT.rglob('*') if v.is_file() and 'outputs' not in v.relative_to(ROOT).parts]+[Path(v) for v in protected]
    if any(p.resolve()==v.resolve() or (p.exists() and v.exists() and os.path.samefile(p,v)) for v in assets):raise ValueError('cannot overwrite input or course asset')
    p.parent.mkdir(parents=True,exist_ok=True);temp=None
    try:
        fd,temp=tempfile.mkstemp(prefix='.ml045-',dir=p.parent)
        with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        os.replace(temp,p);temp=None
    finally:
        if temp is not None:Path(temp).unlink(missing_ok=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data');ap.add_argument('--shift');ap.add_argument('--config');ap.add_argument('--out',default=str(ROOT/'outputs/result.json'));a=ap.parse_args()
    if a.data is None and a.shift is None and a.config is None:
        hashes=read_json(ROOT/'data_integrity.json')['sha256']
        for rel,h in hashes.items():
            if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=h:raise ValueError('teaching input changed: '+rel)
    r=main_report(*load_inputs(a.data,a.shift,a.config));safe_write(a.out,r,[v for v in [a.data,a.shift,a.config] if v]);print(json.dumps({'expected_costs':{k:v['expected_cost'] for k,v in r['evaluation'].items()},'core_sha256':hashlib.sha256(canonical_bytes(r)).hexdigest()}))
if __name__=='__main__':main()
