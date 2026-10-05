"""Independent calculus, score, rank and repeated-split checks for ML047."""
from pathlib import Path
from decimal import Decimal as D,localcontext
from fractions import Fraction as F
import argparse,copy,hashlib,json,math,tempfile,os
from unittest.mock import patch
import numpy as np
import experiment as e
import calibration as cb
import conformal as cf
from numeric import safe_write,canonical_bytes

def check(v,message):
    if not v:raise RuntimeError(message)

def close(a,b,message,atol=3e-12,rtol=3e-12):check(np.allclose(np.asarray(a,dtype=float),np.asarray(b,dtype=float),atol=atol,rtol=rtol),message)

def reject(f,label):
    try:f()
    except (ValueError,TypeError,OverflowError):return
    raise RuntimeError('invalid input accepted: '+label)

def decimal_temperature(z,y,a):
    with localcontext() as ctx:
        ctx.prec=100;aa=D.from_float(float(a));n=D(len(z));rows=[]
        for raw,yy in zip(z,y):
            raw=D.from_float(float(raw));s=aa*raw;p=1/(1+(-s).exp());loss=-(p.ln() if yy else (1-p).ln());delta=p-D(int(yy));h=p*(1-p)
            rows.append({'probability':float(p),'loss':float(loss),'gradient_contribution':float(raw*delta/n),'hessian_contribution':float(raw*raw*h/n)})
        return rows

def hand_checks(r,c):
    hand=r['hand'];z=hand['raw_logits'];y=hand['labels']
    for state in hand['states']:
        rows=decimal_temperature(z,y,state['inverse_temperature'])
        for row,ref in zip(state['rows'],rows):
            for key,value in ref.items():close(row[key],value,'independent100-digit row '+key)
            close(row['local_dloss_dp']*row['local_dp_dscaled_logit']*row['local_dscaled_logit_da']/4,row['gradient_contribution'],'explicit local chain')
        close(sum(row['gradient_contribution'] for row in rows),state['gradient'],'independent gradient sum')
    m=2*math.log(3);close(hand['states'][0]['gradient'],m*.15,'known first gradient');close(hand['states'][1]['inverse_temperature'],1-.5*m*.15,'first synchronous update')
    optimum=hand['fitted']['final'];close(optimum['inverse_temperature'],.5,'analytic hand inverse temperature',atol=2e-10)
    h=hand['optimum_metrics'];close(h['binary_Brier'],F(3,16),'exact Brier optimum');close(h['positive_class_reliability']['ece'],F(1,4),'positive-class ECE remains quarter');close(h['top_label_reliability']['ece'],0,'top-label ECE zero',atol=2e-10)
    close(hand['initial_metrics']['binary_Brier'],F(21,100),'initial Brier');close(hand['initial_metrics']['positive_class_reliability']['ece'],F(1,4),'initial positive ECE');close(hand['initial_metrics']['top_label_reliability']['ece'],F(3,20),'initial confidence ECE');check(h['accuracy']==.75,'same binary decision accuracy')
    errors=[]
    for a in [.3,.8,1.2]:
        state=cb.temperature_state(z,y,a)
        best=1
        for step in [1e-4,1e-5,1e-6]:
            plus=cb.temperature_state(z,y,a+step);minus=cb.temperature_state(z,y,a-step)
            dg=(plus['mean_log_loss']-minus['mean_log_loss'])/(2*step);dh=(plus['gradient']-minus['gradient'])/(2*step)
            best=min(best,max(abs(dg-state['gradient']),abs(dh-state['hessian'])))
        check(best<1e-9,'nonstationary gradient/Hessian difference');errors.append(best)
    check(cb.fit_temperature([0,0],[0,1])['status']=='lower_boundary','flat objective boundary explicit')
    check(cb.fit_temperature([1,2],[0,0])['final']['infinite_temperature'],'infinite-temperature boundary')
    check(cb.fit_temperature([1,2],[1,1])['status']=='upper_boundary','restricted upper boundary explicit')
    edges=np.linspace(0,1,6);rel=cb.reliability(edges,[0,1,0,1,0,1],5)
    check([b['count'] for b in rel['bins']]==[1,1,1,1,2],'returned floating bin edges assigned to right, final1 retained')
    sparse=cb.reliability([0,1],[0,1],5);check(sparse['bins'][1]['mean_score'] is None and sum(v['count'] for v in sparse['bins'])==2,'empty bins null, no lost endpoints')
    return {'Decimal_precision':100,'full_hand_states':3,'finite_difference_best_errors':errors,'distinct_ECE_definitions_checked':True,'temperature_boundary_states_checked':True}

def rank_checks():
    cases=[]
    for n,alpha,k in [(9,.1,9),(9,.05,10),(9,.29,8),(99,.29,71),(19,.05,19),(1,.1,2)]:
        q=cf.finite_sample_quantile(np.arange(1,n+1),alpha);check(q['rank']==k,'exact nominal rank')
        check(q['infinite']==(k>n),'infinity rank condition')
        if k<=n:check(q['quantile']==k,'direct order statistic')
        cases.append({'n':n,'alpha':alpha,'rank':k,'infinite':q['infinite']})
    exact_covered=naive_covered=0;records=[]
    scores=np.arange(10,dtype=float)
    for held_out in range(10):
        cal=np.delete(scores,held_out);q=cf.finite_sample_quantile(cal,.1);naive=float(np.quantile(cal,.9,method='linear'))
        exact_covered+=int(scores[held_out]<=q['quantile']);naive_covered+=int(scores[held_out]<=naive)
        records.append({'held_out_rank':held_out+1,'test_score':float(scores[held_out]),'correct_quantile':q['quantile'],'naive_quantile':naive})
    check(exact_covered==9 and naive_covered==8,'finite rank enumeration vs naive interpolation')
    tied=cf.finite_sample_quantile([0]*9,.1);covered=cf.interval_report([0,1],[0,1],tied)
    check(covered['coverage']==1 and covered['width']==0,'closed zero-score ties')
    infinite=cf.finite_sample_quantile([1,2],.1);check(cf.interval_report([0],[100],infinite)['infinite_width'],'infinite interval representation')
    correct=cf.finite_sample_quantile(np.arange(1,11),.2);naive_higher=float(np.quantile(np.arange(1,11),correct['rank']/10,method='higher'))
    check(correct['quantile']==9 and naive_higher==10,'library higher recipe not identical order statistic')
    return {'rank_cases':cases,'ten_exchangeable_rank_cases':records,'correct_coverage':'9/10','naive_interpolation_coverage':'8/10','higher_recipe_example':{'direct_kth':9,'quantile_higher':10},'ties_and_infinity_checked':True}

def simple_line(x,y):
    mx=math.fsum(x)/len(x);my=math.fsum(y)/len(y);num=math.fsum((xx-mx)*(yy-my) for xx,yy in zip(x,y));den=math.fsum((xx-mx)**2 for xx in x);w=num/den;return [my-w*mx,w]

def regression_checks(reg,c,r):
    main=r['conformal']['main'];b,w=simple_line(reg['train']['x'],reg['train']['y']);close(main['fit']['theta'],[b,w],'independent scalar OLS')
    scores=sorted(abs(yy-b-w*xx) for xx,yy in zip(reg['calibration']['x'],reg['calibration']['y']));k=main['quantile']['rank'];q=scores[k-1];close(main['quantile']['quantile'],q,'independent sorted residual order statistic')
    close(main['baseline']['coverage'],np.mean([abs(yy-b-w*xx)<=q for xx,yy in zip(reg['evaluation']['x'],reg['evaluation']['y'])]),'independent main interval coverage')
    from sklearn.neighbors import KNeighborsRegressor
    model=KNeighborsRegressor(n_neighbors=1);model.fit(reg['calibration']['x'][:,None],reg['calibration']['y']);pred=model.predict(reg['evaluation']['x'][:,None]);same=model.predict(reg['calibration']['x'][:,None]);close(same,reg['calibration']['y'],'actual1NN calibration memorization');check(main['invalid_calibration_reuse']['quantile']['quantile']==0,'invalid reused calibration zero width')
    close(np.mean(pred==reg['evaluation']['y']),main['invalid_calibration_reuse']['evaluation']['coverage'],'actual1NN invalid zero-width coverage')
    # Reconstruct the complete repeated draw stream, then use scalar OLS/order sorting independently.
    rng=np.random.default_rng(c['regression_seed']+1);maximum_error=0.
    def draw(n):
        x=rng.uniform(c['regression_feature_low'],c['regression_feature_high'],n);mu=c['regression_true_intercept']+c['regression_true_slope']*x;sd=c['regression_noise_intercept']+c['regression_noise_abs_slope']*abs(x);noise=rng.normal(size=n)
        return x,mu+sd*noise,mu+c['regression_noise_shift_factor']*sd*noise
    for saved in r['conformal']['repetitions']:
        tx,ty,_=draw(c['regression_train_count']);cx,cy,_=draw(c['regression_calibration_count']);xx,yy,shift=draw(c['regression_evaluation_count']);bb,ww=simple_line(tx,ty);maximum_error=max(maximum_error,max(abs(np.array([bb,ww])-saved['theta'])));ss=sorted(abs(y0-bb-ww*x0) for x0,y0 in zip(cx,cy));threshold=ss[saved['rank']-1]
        close(saved['quantile'],threshold,'all repeated quantiles');baseline=np.abs(yy-bb-ww*xx)<=threshold;shifted=np.abs(shift-bb-ww*xx)<=threshold
        close(saved['coverage'],baseline.mean(),'all repeated baseline coverage');close(saved['shifted_coverage'],shifted.mean(),'all repeated shifted coverage')
        for name,mask in [('low_abs_x',abs(xx)<=c['regression_group_abs_boundary']),('high_abs_x',abs(xx)>c['regression_group_abs_boundary'])]:
            check(saved['groups'][name]['n']==int(mask.sum()),'group sample count');close(saved['groups'][name]['coverage'],baseline[mask].mean(),'group coverage')
    for key in ['coverage','shifted_coverage','width','invalid_reuse_coverage']:
        values=np.array([v[key] for v in r['conformal']['repetitions']]);summary=r['conformal']['summary'][key];close(summary['mean'],values.mean(),'MC mean');close(summary['mc_standard_error'],values.std(ddof=1)/math.sqrt(len(values)),'MCSE is across repetitions')
    return {'independent_scalar_OLS_max_coefficient_error':maximum_error,'all_repeated_splits_checked':len(r['conformal']['repetitions']),'sorted_residual_quantiles_checked':True,'actual_sklearn_invalid_1NN_counterexample_checked':True}

def temperature_checks(classification,c,r):
    t=r['temperature'];fit=t['frozen_choice']['fit'];reference=t['scipy_reference'];check(reference['success'],'SciPy bounded scalar fit did not succeed');close(fit['final']['inverse_temperature'],reference['inverse_temperature'],'independent scalar optimizer',atol=2e-7)
    check(fit['projected_gradient_residual']<1e-9,'temperature projected optimality')
    for split in ['calibration','evaluation']:
        for timing in ['before','after']:
            m=t['metrics'][split][timing];close(m['binary_Brier'],m['library_binary_Brier'],'actual sklearn Brier convention');close(m['mean_log_loss'],m['library_log_loss'],'actual sklearn ordinary log score');check(sum(v['count'] for v in m['positive_class_reliability']['bins'])==m['n'],'bin counts')
    changed=copy.deepcopy(classification);changed['evaluation']['y']=1-changed['evaluation']['y'];changed['evaluation']['true_probability']=1-changed['evaluation']['true_probability'];other=e.temperature_report(changed,c)
    check(other['choice_sha256']==t['choice_sha256'] and canonical_bytes(other['frozen_choice'])==canonical_bytes(t['frozen_choice']),'evaluation labels/oracle cannot change fitted temperature')
    return {'SciPy_fit_verified':True,'sklearn_Brier_log_loss_verified':True,'evaluation_label_mutation_leaves_choice_byte_identical':True}

def guards(classification,regression,c):
    bad=[True,'1',1+0j,np.nan,np.inf,1e200,1e-200,np.longdouble('1e-400')]
    for value in bad:
        reject(lambda:cb.temperature_state([0,value],[0,1],1),'bad final logit');reject(lambda:cb.temperature_state([0,1],[0,value],1),'bad final label');reject(lambda:cf.finite_sample_quantile([0,value],.1),'bad final score')
    for key in c:
        changed=copy.deepcopy(c)
        if isinstance(changed[key],list):changed[key][-1]=True
        else:changed[key]=True
        with patch.object(e,'hand_report',side_effect=RuntimeError('numerics reached')):reject(lambda:e.main_report(classification,regression,changed),'bad protocol field '+key)
    for domain in ['classification','regression']:
        cc=copy.deepcopy(classification);rr=copy.deepcopy(regression)
        if domain=='classification':cc['evaluation']['y']=cc['evaluation']['y'].astype(object);cc['evaluation']['y'][-1]=True
        else:rr['evaluation']['shifted_y']=rr['evaluation']['shifted_y'].astype(object);rr['evaluation']['shifted_y'][-1]=True
        with patch.object(e,'hand_report',side_effect=RuntimeError('numerics reached')):reject(lambda:e.main_report(cc,rr,c),'bad loaded final array')
    for score,alpha in [([], .1),([1,2],0),([1,2],1),([-1,2],.1)]:reject(lambda:cf.finite_sample_quantile(score,alpha),'quantile domain')
    reject(lambda:cf.fit_line([1,1],[1,2]),'rank-deficient line')
    with tempfile.TemporaryDirectory() as td:
        out=Path(td)/'out.json';out.write_bytes(b'OLD');reject(lambda:safe_write(out,{'bad':float('nan')}),'strict JSON');check(out.read_bytes()==b'OLD','serialize before output')
        with patch('numeric.os.replace',side_effect=OSError('injected failure')):
            try:safe_write(out,{'ok':1})
            except OSError:pass
            else:raise RuntimeError('injected failure absent')
        check(out.read_bytes()==b'OLD' and not list(Path(td).glob('.ml047-*')),'atomic failure cleanup')
        sy=Path(td)/'alias.json';sy.symlink_to(out);reject(lambda:safe_write(sy,{}),'symlink output')
        reject(lambda:safe_write(e.ROOT/'data/protocol.json',{}),'source protection')
    return {'all_protocol_fields_precompute_checked':True,'loaded_last_fields_precompute_checked':True,'rank_and_quantile_domain_checked':True,'atomic_output_checked':True}

def audit():
    classification,regression,c=e.load_inputs();r=e.main_report(classification,regression,c)
    return {'unit':'047','hand':hand_checks(r,c),'rank':rank_checks(),'temperature':temperature_checks(classification,c,r),'regression':regression_checks(regression,c,r),'guards':guards(classification,regression,c),'core_sha256':hashlib.sha256(canonical_bytes(r)).hexdigest()}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();r=audit();safe_write(a.out,r);print(json.dumps(r,ensure_ascii=False))
