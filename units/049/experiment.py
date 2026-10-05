"""ML049: exact finite-class guarantees, actual gaps and selection misuse."""
from pathlib import Path
import argparse,csv,io,json,hashlib,math
from fractions import Fraction
import numpy as np
from numeric import ROOT,scalar,read_json,safe_write,canonical_bytes
from protocol import validate
from finite_class import validate_population,validate_sample,ledger,exact_enumeration
from bounds import radius,tail,agnostic_sample_requirement,realizable_sample_requirement
from selection import choose,analytic_minimum,vc_patterns

def csv_read(path,header):
    reader=csv.DictReader(io.StringIO(Path(path).read_text()))
    if reader.fieldnames!=header:raise ValueError('CSV header mismatch')
    rows=[];seen=set()
    for row in reader:
        if set(row)!=set(header) or any(v is None for v in row.values()):raise ValueError('malformed CSV row')
        ident=row['id']
        if not ident or len(ident)>40 or not ident.isascii() or not all(v.isalnum() or v in '_-' for v in ident) or ident in seen:raise ValueError('invalid/duplicate ID')
        seen.add(ident);out={'id':ident}
        for key in header[1:]:
            token=row[key]
            if len(token)>12 or not token.lstrip('-').isdigit():raise ValueError('CSV finite integer token required')
            out[key]=int(token)
        rows.append(out)
    return rows

def load_inputs(data_directory=None,config=None):
    base=Path(data_directory) if data_directory else ROOT/'data';c=validate(read_json(config or base/'protocol.json'))
    pop=csv_read(base/'population.csv',['id','x','y','mass_numerator','mass_denominator']);hs=csv_read(base/'hypotheses.csv',['id','predict_x_minus2','predict_x_minus1','predict_x_plus1','predict_x_plus2']);hyp=[{'id':v['id'],'predictions':[v[k] for k in ['predict_x_minus2','predict_x_minus1','predict_x_plus1','predict_x_plus2']]} for v in hs]
    sample=csv_read(base/'hand_sample.csv',['id','x','y']);validate_population(pop,hyp);validate_sample(sample)
    return pop,hyp,sample,c

def summarize(values):
    a=np.asarray(values,dtype=float);sd=float(a.std(ddof=1));return {'mean':float(a.mean()),'sd':sd,'mc_standard_error':sd/math.sqrt(len(a)),'min':float(a.min()),'max':float(a.max()),'p05':float(np.quantile(a,.05)),'p50':float(np.quantile(a,.5)),'p95':float(np.quantile(a,.95))}

def event_summary(values):
    a=np.asarray(values,dtype=bool);count=int(a.sum());n=len(a);p=count/n
    return {'count':count,'repetitions':n,'frequency':p,'binomial_mc_standard_error':math.sqrt(p*(1-p)/n),'zero_events_one_sided_95_upper':-math.expm1(math.log(.05)/n) if count==0 else None,'zero_note':'zero observed violations do not establish a theorem; upper bound assumes independent repeated experiments'}

def main_trials(pop,hyp,c):
    o=validate_population(pop,hyp);rng=np.random.default_rng(c['main_seed']);R=c['main_repetitions'];risk=o['risk'];minimum=float(risk.min());M=len(hyp);out=[]
    for n in c['main_sample_sizes']:
        counts=rng.multinomial(n,o['mass_numerators']/o['denominator'],size=R);errors=counts@o['loss'];chosen=np.argmin(errors,axis=1);train=errors[np.arange(R),chosen]/n;population=risk[chosen];sup=np.max(np.abs(errors/n-risk[None,:]),axis=1);optimism=population-train;excess=population-minimum;eps=radius(n,c['delta'],M);fixed=radius(n,c['delta']);hstar=int(np.argmin(risk));fixed_gap=np.abs(errors[:,hstar]/n-risk[hstar]);records=[]
        for i in range(R):records.append({'repetition':i,'category_counts':counts[i].tolist(),'chosen_index':int(chosen[i]),'selected_training_risk':float(train[i]),'selected_population_risk':float(population[i]),'signed_optimism':float(optimism[i]),'excess_risk':float(excess[i]),'uniform_absolute_gap':float(sup[i]),'predeclared_best_absolute_gap':float(fixed_gap[i])})
        out.append({'n':n,'M':M,'delta':c['delta'],'uniform_radius':eps,'fixed_radius':fixed,'ERM_excess_bound_raw':2*eps,'ERM_excess_bound_using_risk_range':min(1-minimum,2*eps),'uniform_radius_vacuous':bool(eps>=1),'records':records,
         'summary':{k:summarize(v) for k,v in [('selected_training_risk',train),('selected_population_risk',population),('signed_optimism',optimism),('excess_risk',excess),('uniform_absolute_gap',sup),('fixed_absolute_gap',fixed_gap)]},
         'violations':{'uniform':event_summary(sup>eps),'selected_using_uniform':event_summary(abs(optimism)>eps),'selected_misusing_fixed':event_summary(abs(optimism)>fixed),'fixed_predeclared':event_summary(fixed_gap>fixed),'ERM_excess':event_summary(excess>2*eps)},'sampling':'exact multinomial counts of IID draws from the eight-category population; order does not affect empirical risks'})
    return out

def noise_trials(c):
    rng=np.random.default_rng(c['noise_seed']);R=c['noise_repetitions'];maxM=max(c['noise_candidate_counts']);out=[]
    for n in c['noise_sample_sizes']:
        counts=rng.binomial(n,.5,size=(R,maxM)).astype(np.int64);scenarios=[]
        # Full validation occurs once here; each selection call receives only training counts.
        for M in c['noise_candidate_counts']:
            selected=choose(counts[:,:M],n,M);index=np.array(selected['selected_indices']);minimum=np.array(selected['selected_error_counts']);gap=abs(minimum/n-.5);single=radius(n,c['delta']);uniform=radius(n,c['delta'],M);hist=np.bincount(minimum,minlength=n+1)
            scenarios.append({'M':M,'fixed_radius':single,'uniform_radius':uniform,'selection':selected,'selected_error_count_histogram':hist.tolist(),'mean_training_risk':float((minimum/n).mean()),'mean_signed_optimism':float((.5-minimum/n).mean()),'fixed_radius_violations':event_summary(gap>single),'uniform_radius_violations':event_summary(gap>uniform),'analytic':analytic_minimum(n,M,c['delta'])})
        # Only after all training choices are fixed, draw independent holdout errors for maxM's chosen rule.
        frozen_choice=scenarios[-1]['selection'];choice_hash=hashlib.sha256(canonical_bytes(frozen_choice)).hexdigest();nh=c['holdout_count'];holdout=rng.binomial(nh,.5,size=R);eps=radius(nh,c['delta']);holdout_info={'n':nh,'selected_from_M':maxM,'training_choice_sha256':choice_hash,'error_counts':holdout.tolist(),'fixed_radius':eps,'violations':event_summary(abs(holdout/nh-.5)>eps),'true_risk':.5,'scope':'conditional on training-selected index, fresh independent holdout errors are Binomial(n_holdout,1/2); no holdout feedback to selection'}
        out.append({'n':n,'scenarios':scenarios,'independent_holdout':holdout_info,'first_training_error_count_row':counts[0].tolist(),'sampling':'For fixed h_j(X)=X_j, Y=0 and independent fair feature bits, counts K_j are independent Binomial(n,1/2). Sampling these counts is exactly the loss-count distribution; no neural network training claimed.'})
    return out

def hand_report(pop,hyp,sample,c):
    first=ledger(sample,pop,hyp);added=[{'id':'ADDED'+str(i+1),'x':x,'y':y} for i,(x,y) in enumerate(c['hand_added_batch'])];second=ledger(sample+added,pop,hyp)
    return {'initial':first,'added_training_batch':added,'after_batch':second,'next_forward_on_all_four_inputs':{'x':[-2,-1,1,2],'initial_rule_predictions':first['selected_prediction_vector'],'updated_rule_predictions':second['selected_prediction_vector']},'scope':'added batch is explicitly additional training data, not a sealed evaluation set; selection happens after full batch losses are summed'}

def main_report(pop,hyp,sample,c):
    c=validate(c);o=validate_population(pop,hyp);sample=validate_sample(sample)
    if len(sample)+len(c['hand_added_batch'])>1000:raise ValueError('total hand sample including added batch exceeds1000')
    if any('ADDED'+str(i+1) in {r['id'] for r in sample} for i in range(len(c['hand_added_batch']))):raise ValueError('hand appended IDs collide')
    hand=hand_report(pop,hyp,sample,c);exact=[]
    for n in c['exact_sample_sizes']:
        result=exact_enumeration(pop,hyp,n,c['exact_epsilon_numerators'],c['exact_epsilon_denominator'])
        for row in result['tails']:
            eps=float(Fraction(row['epsilon']));row['fixed_Hoeffding']=tail(n,eps);row['uniform_Hoeffding_union']=tail(n,eps,len(hyp))
        exact.append(result)
    repeated=main_trials(pop,hyp,c);noise=noise_trials(c);n=c['dependent_repeat_count'];eps=radius(n,c['delta'])
    dependent={'nominal_n':n,'actual_independent_bits':1,'marginal_true_risk':.5,'possible_empirical_risks':[0.,1.],'outcome_probabilities':[.5,.5],'absolute_gap_always':.5,'misused_IID_radius':eps,'misused_IID_violation_probability':float(.5>eps),'reason':'all n loss values are exact copies of one fair Bernoulli bit; identical marginals without independence'}
    table=[{'epsilon':ep,'M':M,'agnostic':agnostic_sample_requirement(ep,c['delta'],M),'realizable':realizable_sample_requirement(ep,c['delta'],M)} for ep in [.05,.1,.2] for M in [1,len(hyp),512]]
    import scipy,sklearn
    return {'unit':'049','protocol':c,'data':{'population':pop,'hypotheses':hyp,'hand_sample':sample},'population':{'outcome_predictions':o['predictions'].tolist(),'outcome_losses':o['loss'].tolist(),'risk_numerators':o['risk_numerators'].tolist(),'risk_denominator':o['denominator'],'risks':o['risk'].tolist(),'minimum_risk':float(o['risk'].min()),'hypothesis_count':len(hyp)},'hand':hand,'exact':exact,'main_repetitions':repeated,'selection_noise':noise,'dependence_counterexample':dependent,'sample_complexity_table':table,'vc':vc_patterns(c['vc_max_points']),'versions':{'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__},'limitations':['fixed finite H and IID [0,1] losses are substantive assumptions','uniform event permits data-dependent selection within the predeclared H, not arbitrary post-hoc H','Hoeffding uses independent observations, not independent hypotheses','noisy default population is agnostic, not realizable','large or zero observed failure rates are reported unchanged','VC examples introduce shattering, not a proof of arbitrary infinite-class guarantees','finite lookup ERM has no gradient; all per-sample forward/loss/selection stages shown']}

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-directory');p.add_argument('--config');p.add_argument('--out',default=str(ROOT/'outputs/result.json'));a=p.parse_args()
    if a.data_directory is None and a.config is None:
        for rel,expected in read_json(ROOT/'data_integrity.json')['sha256'].items():
            if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=expected:raise ValueError('teaching data changed: '+rel)
    inputs=load_inputs(a.data_directory,a.config);result=main_report(*inputs);protected=[a.config] if a.config else []
    if a.data_directory:protected.extend(v for v in Path(a.data_directory).glob('*') if v.is_file())
    safe_write(a.out,result,protected);print(json.dumps({'risks':result['population']['risks'],'hand_initial':result['hand']['initial']['selected_id'],'hand_after_batch':result['hand']['after_batch']['selected_id'],'noise_first_sample_size':[{ 'M':v['M'],'empirical_misused_failure':v['fixed_radius_violations']['frequency'],'analytic_misused_failure':v['analytic']['selected_fixed_radius_violation_probability']} for v in result['selection_noise'][0]['scenarios']],'core_sha256':hashlib.sha256(canonical_bytes(result)).hexdigest()}))
if __name__=='__main__':main()
