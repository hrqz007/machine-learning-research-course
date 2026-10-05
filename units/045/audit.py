"""Exact finite-policy and probability references for the decision lesson."""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import argparse,copy,hashlib,json,os,tempfile
from unittest.mock import patch
import numpy as np
import experiment as e


def require(b,message):
    if not b:raise RuntimeError(message)


def near(a,b,message,atol=3e-13,rtol=3e-13):require(np.allclose(np.asarray(a,dtype=float),np.asarray(b,dtype=float),atol=atol,rtol=rtol),message)


def reject(fn,message):
    try:fn()
    except (ValueError,TypeError,OverflowError):return
    raise RuntimeError('invalid input accepted: '+message)


def frac(v):return F.from_float(float(v))


def exact_risks(P,C):
    return [[sum(frac(pk)*frac(ck) for pk,ck in zip(p,c)) for c in C] for p in P]


def science(p,w,ids,q0,q1,sids,c,r):
    P=[[1-float(v),float(v)] for v in p];C=[[0,3],[1,0]];CR=C+[[F(3,8),F(3,8)]]
    R=exact_risks(P,C);RR=exact_risks(P,[[float(x) for x in row] for row in CR])
    for named,cost,risk in [('decision',C,R),('rejection',CR,RR)]:
        rows=r[named]['rows'];choices=[]
        for i,row in enumerate(rows):
            expected=[float(v) for v in risk[i]];near(row['conditional_risks'],expected,'exact conditional risk')
            action=min(range(len(cost)),key=lambda a:risk[i][a]);choices.append(action)
            require(row['chosen_action']==action,'exact minimum and first-action tie')
            expected_ties=[a for a in range(len(cost)) if risk[i][a]==risk[i][action]]
            require(row['tied_actions']==expected_ties,'all exact ties retained')
            contributions=[[frac(P[i][k])*frac(cost[a][k]) for k in range(2)] for a in range(len(cost))]
            near(row['class_loss_contributions'],[[float(v) for v in q] for q in contributions],'per-class expected-loss chain')
        near(r['evaluation']['cost_optimal' if named=='decision' else 'with_reject']['expected_cost'],sum(frac(wi)*row[a] for wi,row,a in zip(w,risk,choices)),'exact population cost')
    # Enumerate all deterministic policies. This checks pointwise minimization independently.
    enumerated=[]
    for risk,A in [(R,2),(RR,3)]:
        minimum=None;ties=0
        for policy in product(range(A),repeat=len(p)):
            value=sum(frac(wi)*row[a] for wi,row,a in zip(w,risk,policy))
            if minimum is None or value<minimum:minimum=value;ties=1
            elif value==minimum:ties+=1
        enumerated.append({'actions_per_bin':A,'policies':A**len(p),'minimum':str(minimum),'optimal_policy_count':ties})
    require(enumerated[0]['minimum']=='11/32' and enumerated[1]['minimum']=='1/4','enumerated minima')
    require(enumerated[0]['optimal_policy_count']==2 and enumerated[1]['optimal_policy_count']==4,'nonunique optimal policies at exact boundaries')
    near(r['evaluation']['accuracy_optimal']['expected_cost'],F(17,32),'accuracy policy cost');near(r['evaluation']['accuracy_optimal']['expected_error_rate'],F(7,32),'minimum error');near(r['evaluation']['cost_optimal']['expected_error_rate'],F(1,4),'cost policy error')
    near(r['evaluation']['with_reject']['reject_mass'],F(3,8),'reject population mass')
    require(r['decision']['actions']==[0,0,0,1,1,1,1,1],'binary actions')
    require(r['rejection']['actions']==[0,0,2,2,2,1,1,1],'rejection actions')
    for scenario in r['cost_scenarios']:
        fp=frac(scenario['FP']);fn=frac(scenario['FN'])
        if fp+fn==0:require(scenario['threshold'] is None and scenario['actions']==[0]*len(p),'both zero costs')
        else:near(scenario['threshold'],fp/(fp+fn),'threshold ratio')
    shift=r['label_shift'];near(shift['source']['positive_posterior'],[F(1,13),F(1,4),F(4,7)],'source posterior');near(shift['target']['positive_posterior'],[F(3,7),F(3,4),F(12,13)],'target posterior')
    near(shift['corrected_positive_posterior'],shift['target']['positive_posterior'],'odds correction versus Bayes enumeration')
    near(shift['stale_expected_target_cost']['expected_cost'],F(37,32),'stale target cost');near(shift['corrected_expected_target_cost']['expected_cost'],F(1,4),'corrected target cost')
    near(e.correct_prior([0,1,.25],.25,.75),[0,1,.75],'posterior endpoints and odds factor9')
    for prior in [.125,.5,.875]:
        reference=[]
        for a,b in zip(q0,q1):
            top=frac(prior)*frac(b);bottom=top+(1-frac(prior))*frac(a);reference.append(top/bottom)
        near(e.posterior_from_likelihoods(q0,q1,prior)['positive_posterior'],reference,'independent exact prior grid')
    require(r['multiclass']['actions']==[1,1] and r['multiclass']['maximum_probability_actions']==[0,2],'multiclass MAP differs from cost optimum')
    near(r['multiclass']['risks'],[[2,.75,1.25],[3,.75,1]],'multiclass conditional risks')
    # General rectangular action-by-class matrices; one action is allowed.
    cases=[([[.5,.25,.25],[.125,.375,.5]],[[1,2,3],[3,2,1]]),([[.25]*4],[[0,1,2,3]]),([[0,1],[1,0]],[[2,3],[0,4],[4,0]])]
    for P0,C0 in cases:
        table=e.risk_table(P0,C0);ref=exact_risks(P0,C0);near(table['risks'],ref,'rectangular exact risks')
        require(table['actions']==[min(range(len(C0)),key=lambda a:row[a]) for row in ref],'rectangular actions')
    for name,sim in r['simulation']['results'].items():
        a=r['evaluation'][name]['actions'];cost=CR if name=='with_reject' else C
        variance=[];observed=[]
        for i,action in enumerate(a):
            pp=frac(p[i]);c0=frac(cost[action][0]);c1=frac(cost[action][1]);variance.append(pp*(1-pp)*(c1-c0)**2)
            count=sim['observed_positive_counts'][i];n=c['samples_per_group'];observed.append((F(n-count,n)*c0+F(count,n)*c1))
        near(sim['conditional_cost_variance'],variance,'exact conditional cost variance')
        near(sim['observed_mean_cost'],sum(frac(wi)*v for wi,v in zip(w,observed)),'observed count-to-cost ledger')
        near(sim['stratified_mean_standard_error']**2,sum(frac(wi)**2*v/c['samples_per_group'] for wi,v in zip(w,variance)),'stratified sampling variance')
        if name!='with_reject':near(sim['sklearn_recomputed_cost'],sim['observed_mean_cost'],'actual sklearn weighted confusion cost')
    return {'all_deterministic_policies':enumerated,'expected_cost_exact':{'cost_optimal':'11/32','accuracy_optimal':'17/32','reject':'1/4'},'oracle_shift_exact':['1/13','1/4','4/7','3/7','3/4','12/13'],'rectangular_cases':len(cases),'sklearn_weighted_confusion_verified':True}


def guards(p,w,ids,q0,q1,sids,c):
    bad=[True,'1',1+0j,np.nan,np.inf,-1,1e200,1e-200,np.longdouble('1e-400')]
    for value in bad:
        P=[[.5,.5],[.5,value]];C=[[0,3],[1,value]]
        reject(lambda:e.risk_table(P,[[0,3],[1,0]]),'last posterior');reject(lambda:e.risk_table([[.5,.5]],C),'last cost')
    for key in c:
        badc=copy.deepcopy(c)
        if isinstance(badc[key],list):badc[key][-1]=True
        else:badc[key]=True
        with patch.object(e,'_simulate',side_effect=RuntimeError('RNG reached early')):reject(lambda:e.main_report(p,w,ids,q0,q1,sids,badc),'last config '+key)
    for P,C in [([],[[0,1],[1,0]]),([[.5,.5]],[[0,1,2]]),([[.2,.2]],[[0,1],[1,0]]),([[.5,.5]],[]),([[.5,.5],[.2]],[[0,1],[1,0]])]:reject(lambda:e.risk_table(P,C),'shape/normalization')
    reject(lambda:e.correct_prior([.5],0,.5),'source prior zero');reject(lambda:e.correct_prior([.5],.5,1),'target prior one');reject(lambda:e.posterior_from_likelihoods([1,0],[1,0],.5),'undefined zero-mass conditional')
    near(e.posterior_from_likelihoods([.5,.5],[.5,.5],0)['positive_posterior'],[0,0],'Bayes supports prior0 when marginal exists')
    require(not e.reject_bounds(0,3,0)['strict_reject_region'],'zero FP no strict reject advantage')
    require(not e.reject_bounds(1,3,.75)['strict_reject_region'],'expensive reject not strictly better')
    require(e.risk_table([[.25,.75]],[[0,0],[0,0]])['actions']==[0],'all equal tie first')
    with tempfile.TemporaryDirectory() as d:
        pth=Path(d);out=pth/'result.json';out.write_bytes(b'OLD')
        reject(lambda:e.safe_write(out,{'bad':float('nan')}),'invalid output JSON');require(out.read_bytes()==b'OLD','serialize before write')
        with patch.object(e.os,'replace',side_effect=OSError('simulated failure')):
            try:e.safe_write(out,{'ok':1})
            except OSError:pass
            else:raise RuntimeError('replace failure lost')
        require(out.read_bytes()==b'OLD' and not list(pth.glob('.ml045-*')),'atomic output failure')
        sy=pth/'sym.json';sy.symlink_to(out);reject(lambda:e.safe_write(sy,{}),'symbolic alias');ha=pth/'hard.json';os.link(out,ha);reject(lambda:e.safe_write(ha,{}),'hardlink alias')
        reject(lambda:e.safe_write(e.ROOT/'data/decision_spec.json',{}),'protect teaching input')
        js=pth/'bad.json'
        for txt in ['{"x":1,"x":2}','{"x":NaN}','{"x":1e-400}']:
            js.write_text(txt);reject(lambda:e.read_json(js),'strict JSON')
    return {'raw_posterior_and_cost_tail_cases':len(bad)*2,'all_config_fields_pre_RNG_checked':True,'zero_cost_and_reject_boundaries_checked':True,'atomic_and_alias_protection_checked':True}


def audit():
    inputs=e.load_inputs();r=e.main_report(*inputs)
    return {'unit':'045','science':science(*inputs,r),'guards':guards(*inputs),'core_sha256':hashlib.sha256(e.canonical_bytes(r)).hexdigest()}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args();r=audit();e.safe_write(args.out,r);print(json.dumps(r,ensure_ascii=False))
