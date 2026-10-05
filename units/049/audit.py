"""Independent exact enumeration, binomial references and input/selection checks."""
import argparse,copy,hashlib,itertools,json,math,tempfile
from pathlib import Path
from fractions import Fraction
from decimal import Decimal,localcontext
from unittest.mock import patch
import numpy as np
from scipy.stats import binom
import experiment as e
import finite_class as fc
import selection as sel
import bounds
from numeric import canonical_bytes,safe_write

def check(value,message):
    if not value:raise RuntimeError(message)
def close(a,b,message,atol=2e-13):
    if not np.allclose(a,b,rtol=2e-13,atol=atol):raise RuntimeError(message)
def reject(fn,message):
    try:fn()
    except ValueError:return
    raise RuntimeError('invalid input accepted: '+message)

def exact_reference(pop,hyp,r):
    # Do not call the experiment's composition iterator or loss-matrix builder.
    weights=[Fraction(v['mass_numerator'],v['mass_denominator']) for v in pop];H=[v['predictions'] for v in hyp];xs=[-2,-1,1,2]
    losses=[[int(h[xs.index(v['x'])]!=v['y']) for h in H] for v in pop]
    risks=[sum(p*losses[i][j] for i,p in enumerate(weights)) for j in range(len(H))]
    close([float(v) for v in risks],r['population']['risks'],'direct Fraction population risk')
    reference={};n=4;den=Fraction(0);training=Fraction(0);selected_risk=Fraction(0);uniformgap=Fraction(0);selection=[Fraction(0)]*len(H);tails={Fraction(v,r['protocol']['exact_epsilon_denominator']):[Fraction(0),Fraction(0)] for v in r['protocol']['exact_epsilon_numerators']};best=min(range(len(H)),key=lambda j:risks[j])
    for sequence in itertools.product(range(8),repeat=n):
        probability=math.prod(weights[i] for i in sequence);den+=probability;emp=[Fraction(sum(losses[i][j] for i in sequence),n) for j in range(len(H))];chosen=min(range(len(H)),key=lambda j:emp[j]);selection[chosen]+=probability;training+=probability*emp[chosen];selected_risk+=probability*risks[chosen];gaps=[abs(x-y) for x,y in zip(emp,risks)];uniformgap+=probability*max(gaps)
        for ep,counts in tails.items():
            if gaps[best]>ep:counts[0]+=probability
            if max(gaps)>ep:counts[1]+=probability
    check(den==1,'ordered sequence probability normalization')
    target=next(v for v in r['exact'] if v['n']==4)
    for key,expected in [('expected_training_risk_fraction',training),('expected_selected_risk_fraction',selected_risk),('expected_uniform_gap_fraction',uniformgap)]:check(Fraction(target[key])==expected,'4096 ordered-sequence reference '+key)
    check([Fraction(v) for v in target['selection_probability_fractions']]==selection,'ordered sequence selection probabilities')
    for row in target['tails']:
        x=tails[Fraction(row['epsilon'])];check(Fraction(row['fixed_tail_fraction'])==x[0] and Fraction(row['uniform_tail_fraction'])==x[1],'ordered sequence tail')
    # Every fixed rule's error is Bernoulli with its own exact population risk.
    for result in r['exact']:
        n=result['n']
        for row in result['tails']:
            epsilon=Fraction(row['epsilon']);individual=[]
            for prob in risks:individual.append(sum(Fraction(math.comb(n,k))*prob**k*(1-prob)**(n-k) for k in range(n+1) if abs(Fraction(k,n)-prob)>epsilon))
            check(individual[best]==Fraction(row['fixed_tail_fraction']),'exact fixed binomial tail')
            check(Fraction(row['uniform_tail_fraction'])<=sum(individual),'exact event union bound')
            check(float(Fraction(row['uniform_tail_fraction']))<=row['uniform_Hoeffding_union']['probability_bound']+1e-14,'uniform Hoeffding bound')
    return {'population_risk_fractions':[str(v) for v in risks],'ordered_sequences_checked':4096,'exact_composition_n':[v['n'] for v in r['exact']],'all_fixed_Binomial_tails_and_event_union_checked':True}

def hand_checks(pop,hyp,sample,r):
    for key,ss in [('initial',sample),('after_batch',sample+r['hand']['added_training_batch'])]:
        out=r['hand'][key];totals=[]
        for j,h in enumerate(hyp):
            errors=[int(h['predictions'][[-2,-1,1,2].index(v['x'])]!=v['y']) for v in ss];totals.append(sum(errors))
            check([row['losses'][j] for row in out['rows']]==errors,'hand per-row losses')
            close(sum(row['mean_loss_contributions'][j] for row in out['rows']),sum(errors)/len(ss),'mean contributions')
        check(out['loss_counts']==totals,'loss count sum');check(out['selected_index']==min(range(len(hyp)),key=lambda j:totals[j]),'first tie exact ERM')
    check(r['hand']['initial']['selected_id']=='H1' and r['hand']['after_batch']['selected_id']=='H2','declared hand transition')
    check(r['hand']['initial']['tie_indices']==[1,2],'declared hand tie set')
    return {'full_forward_loss_chain_stages':2,'initial_tie_set':[1,2],'selected_transition':['H1','H2'],'no_gradient_claim':True}

def bounds_checks():
    examples=[]
    with localcontext() as ctx:
        ctx.prec=80
        for n,M,delta in [(1,1,.05),(20,512,.05),(1000,5,1e-12),(1000000,1048576,.2)]:
            expected=((Decimal(2*M)/Decimal(str(delta))).ln()/(2*n)).sqrt();v=bounds.radius(n,delta,M);close(v,float(expected),'Decimal radius',atol=2e-15);examples.append({'n':n,'M':M,'delta':delta,'radius':v})
    for n in [1,20,100,1000]:
        check(bounds.radius(n,.05,1)<=bounds.radius(n,.05,512),'radius monotone H')
    for epsilon in [.05,.1,.2]:
        s=bounds.agnostic_sample_requirement(epsilon,.05,5)['sufficient_sample_size'];check(2*bounds.radius(s,.05,5)<=epsilon+1e-14,'agnostic sample inversion');check(2*bounds.radius(s-1,.05,5)>epsilon,'minimal integer in displayed sufficient formula')
    return {'Decimal80_radius_examples':examples,'sample_requirement_inversions_checked':True}

def main_trial_checks(pop,hyp,c,r):
    rng=np.random.default_rng(c['main_seed']);prob=np.array([v['mass_numerator']/v['mass_denominator'] for v in pop]);risk=np.array(r['population']['risks']);H=np.array([v['predictions'] for v in hyp]);Ys=np.array([v['y'] for v in pop]);loss=(H[:,np.repeat(np.arange(4),2)].T!=Ys[:,None]).astype(int);total=0
    for block in r['main_repetitions']:
        n=block['n'];counts=rng.multinomial(n,prob,size=c['main_repetitions']);stored=np.array([v['category_counts'] for v in block['records']]);check(np.array_equal(counts,stored),'all multinomial draws reproduced');errs=counts@loss;emp=errs/n;selected=np.argmin(errs,axis=1);rows=block['records'];total+=len(rows)
        check([v['chosen_index'] for v in rows]==selected.tolist(),'all empirical selectors')
        close([v['uniform_absolute_gap'] for v in rows],np.abs(emp-risk).max(axis=1),'all uniform gaps');close([v['selected_population_risk'] for v in rows],risk[selected],'all selected true risk')
        close([v['signed_optimism'] for v in rows],risk[selected]-emp[np.arange(len(rows)),selected],'all signed gaps');close([v['excess_risk'] for v in rows],risk[selected]-risk.min(),'all ERM excess')
        eps=block['uniform_radius'];check(np.all((np.abs(emp-risk).max(axis=1)>eps)|(risk[selected]-risk.min()<=2*eps+1e-14)),'pointwise ERM 2epsilon implication')
        for key in ['selected_training_risk','selected_population_risk','signed_optimism','excess_risk','uniform_absolute_gap']:
            a=np.array([v[key] for v in rows]);close(block['summary'][key]['mean'],a.mean(),'all repeated means');close(block['summary'][key]['mc_standard_error'],a.std(ddof=1)/math.sqrt(len(a)),'repeat MCSE')
    return {'all_repetitions_reconstructed':total,'all_forward_counts_selected_risks_gaps_checked':True,'pointwise_ERM_2epsilon_implication_checked':True}

def noise_checks(c,r):
    rng=np.random.default_rng(c['noise_seed']);diagnostics=[];max_error=0.;max_naive_error=0.
    for block in r['selection_noise']:
        n=block['n'];counts=rng.binomial(n,.5,size=(c['noise_repetitions'],max(c['noise_candidate_counts'])));check(block['first_training_error_count_row']==counts[0].tolist(),'noise matrix first row')
        for scenario in block['scenarios']:
            M=scenario['M'];chosen=np.argmin(counts[:,:M],axis=1);minimum=counts[np.arange(len(counts)),chosen];check(scenario['selection']['selected_indices']==chosen.tolist(),'noise selected indices');check(scenario['selection']['selected_error_counts']==minimum.tolist(),'noise min counts')
            k=np.arange(n+1);survival=binom.sf(k,n,.5)**M;naive=np.r_[1.,survival[:-1]]-survival;reference=scenario['analytic'];max_naive_error=max(max_naive_error,float(np.max(np.abs(naive-reference['minimum_pmf']))))
            # Near-one survival subtraction loses digits after raising to M; use
            # complementary CDFs plus log1p/expm1, not a looser test tolerance.
            cdf=binom.cdf(k,n,.5);sf=binom.sf(k,n,.5)
            with np.errstate(divide='ignore',invalid='ignore'):
                log_survival=M*np.where(cdf<=.5,np.log1p(-cdf),np.log(sf));prior=np.r_[0.,log_survival[:-1]]
                pmf=np.exp(prior)*(-np.expm1(log_survival-prior));pmf[np.isneginf(prior)]=0.
            max_error=max(max_error,float(np.max(np.abs(pmf-reference['minimum_pmf']))));close(pmf,reference['minimum_pmf'],'actual SciPy stable binomial minimum PMF',atol=3e-14)
            for mode,key in [('fixed','selected_fixed_radius_violation_probability'),('uniform','selected_uniform_radius_violation_probability')]:
                epsilon=scenario[mode+'_radius'];prob=float(pmf[abs(k/n-.5)>epsilon].sum());close(prob,reference[key],'analytic violation via SciPy distribution',atol=5e-14)
            exact=reference['selected_fixed_radius_violation_probability'];obs=scenario['fixed_radius_violations']['frequency'];se=math.sqrt(exact*(1-exact)/c['noise_repetitions']);diagnostics.append({'n':n,'M':M,'analytic_misused_bound_failure':exact,'observed_failure':obs,'MC_z_difference':None if se==0 else (obs-exact)/se})
        ho=block['independent_holdout'];holdout=rng.binomial(c['holdout_count'],.5,size=c['noise_repetitions']);check(ho['error_counts']==holdout.tolist(),'holdout draws after selection');check(ho['training_choice_sha256']==hashlib.sha256(canonical_bytes(block['scenarios'][-1]['selection'])).hexdigest(),'holdout choice freeze')
        # Selection receives no holdout data, and caller-side holdout mutation cannot affect it.
        saved=canonical_bytes(sel.choose(counts,n,max(c['noise_candidate_counts'])));holdout[:]=0
        check(saved==canonical_bytes(sel.choose(counts,n,max(c['noise_candidate_counts']))),'holdout mutation changed selection')
    return {'actual_SciPy_stable_binomial_PMF_max_error':max_error,'naive_survival_subtraction_max_error':max_naive_error,'all_noise_and_holdout_draws_reproduced':True,'holdout_mutation_selection_invariant':True,'MC_diagnostics_not_acceptance_thresholds':diagnostics}

def vc_checks(r):
    rows=r['vc']['ordered_distinct_point_patterns']
    for row in rows:
        m=row['points'];check(row['threshold_count']==m+1,'threshold growth count');check(row['interval_count']==m*(m+1)//2+1,'interval growth count')
    check(rows[0]['threshold_count']==2 and [1,0] in rows[1]['threshold_missing'],'threshold shatters1 not2')
    check(rows[1]['interval_count']==4 and [1,0,1] in rows[2]['interval_missing'],'interval shatters2 not3')
    return {'threshold_VC':1,'interval_VC':2,'all_ordered_pattern_counts_verified':True}

def guards(pop,hyp,sample,c):
    for key in c:
        bad=copy.deepcopy(c)
        if isinstance(bad[key],list):
            if isinstance(bad[key][-1],list):bad[key][-1][-1]=True
            else:bad[key][-1]=True
        else:bad[key]=True
        with patch.object(e,'hand_report',side_effect=RuntimeError('computation reached')):reject(lambda:e.main_report(pop,hyp,sample,bad),'final protocol value '+key)
    for domain in ['population','hypotheses','sample']:
        pp,hh,ss=copy.deepcopy(pop),copy.deepcopy(hyp),copy.deepcopy(sample)
        if domain=='population':pp[-1]['mass_numerator']=True
        elif domain=='hypotheses':hh[-1]['predictions'][-1]=True
        else:ss[-1]['y']=True
        with patch.object(e,'hand_report',side_effect=RuntimeError('computation reached')):reject(lambda:e.main_report(pp,hh,ss,c),'last '+domain+' value')
    long_sample=[{'id':'L'+str(i),'x':-2,'y':0} for i in range(1000)]
    with patch.object(e,'hand_report',side_effect=RuntimeError('computation reached')):reject(lambda:e.main_report(pop,hyp,long_sample,c),'combined hand resource bound')
    reject(lambda:fc.exact_enumeration(pop,hyp,4,[],8),'empty epsilon list')
    for value in [True,'2',2+0j,float('nan'),float('inf'),1e100,1e-200]:
        reject(lambda:bounds.radius(value,.05),'sample size type');reject(lambda:bounds.radius(20,value),'delta type');reject(lambda:sel.choose([[0,value]],20,1),'last unused candidate entry type')
    reject(lambda:sel.choose(np.array([[0,21]],dtype=np.uint64),20,1),'unsigned count range')
    reject(lambda:sel.choose(np.array([[False,True]],dtype=bool),20,1),'bool matrix')
    check(sel.choose(np.array([[0,0]],dtype=np.int64),20,2)['selected_indices']==[0],'deterministic ties')
    with tempfile.TemporaryDirectory() as td:
        path=Path(td)/'out.json';path.write_bytes(b'OLD');reject(lambda:safe_write(path,{'bad':float('nan')}),'serialize before mutation');check(path.read_bytes()==b'OLD','invalid JSON preserves old')
        with patch('numeric.os.replace',side_effect=OSError('injected failure')):
            try:safe_write(path,{'ok':1})
            except OSError:pass
            else:raise RuntimeError('replace failure absent')
        check(path.read_bytes()==b'OLD' and not list(Path(td).glob('.ml049-*')),'atomic failure cleanup');link=Path(td)/'alias.json';link.symlink_to(path);reject(lambda:safe_write(link,{}),'symlink output');reject(lambda:safe_write(e.ROOT/'data/protocol.json',{}),'input protection')
    return {'all_config_fields_and_loaded_last_values_precompute_checked':True,'numeric_integer_bool_complex_domain_checked':True,'atomic_output_and_aliases_checked':True}

def audit():
    pop,hyp,sample,c=e.load_inputs();r=e.main_report(pop,hyp,sample,c)
    return {'unit':'049','exact':exact_reference(pop,hyp,r),'hand':hand_checks(pop,hyp,sample,r),'bounds':bounds_checks(),'main_repetitions':main_trial_checks(pop,hyp,c,r),'noise':noise_checks(c,r),'VC':vc_checks(r),'guards':guards(pop,hyp,sample,c),'dependence_failure_probability':r['dependence_counterexample']['misused_IID_violation_probability'],'core_sha256':hashlib.sha256(canonical_bytes(r)).hexdigest()}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();r=audit();safe_write(a.out,r);print(json.dumps(r,ensure_ascii=False))
