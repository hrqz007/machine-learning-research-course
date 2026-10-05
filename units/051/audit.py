"""Independent mathematical, data-role, and ordinary-input checks for ML051."""
from pathlib import Path
from fractions import Fraction as F
import argparse,copy,hashlib,json,math,tempfile
from unittest.mock import patch
import numpy as np
from numpy.polynomial.legendre import leggauss
from numeric import read_json,canonical_bytes,safe_write
import experiment as e
import cross_validation as cv
import models,hand_example
from folds import validate_folds,make_folds
from generate_data import generate
from reference import basis,qr_fit,ReferenceRidge,actual_grid_search

def check(v,msg):
    if not v:raise RuntimeError(msg)
def close(a,b,msg,atol=2e-10):
    if not np.allclose(a,b,rtol=2e-9,atol=atol):raise RuntimeError(msg)
def reject(fn,msg):
    try:fn()
    except ValueError:return
    raise RuntimeError('invalid input accepted: '+msg)

def hand_check(c,r):
    x=list(map(lambda v:F(str(v)),c['hand_training_x']));y=list(map(lambda v:F(str(v)),c['hand_training_y']));out=r['hand'];scores=[]
    for j,lam in enumerate(c['hand_candidate_penalties']):
        lam=F(str(lam));sse=F(0)
        for k,va in enumerate(c['hand_inner_validation_indices']):
            tr=[i for i in range(4) if i not in va];w=sum(x[i]*y[i] for i in tr)/(sum(x[i]**2 for i in tr)+len(tr)*lam);v=out['candidates'][j]['folds'][k];close(v['fit']['weight'],float(w),'Fraction hand inner coefficient')
            for i,row in zip(va,v['validation_rows']):
                loss=(w*x[i]-y[i])**2;sse+=loss;close(row['prediction'],float(w*x[i]),'hand forward');close(row['squared_loss'],float(loss),'hand validation loss')
        scores.append(sse/4);close(out['candidates'][j]['pooled_CV_MSE'],float(sse/4),'hand pooled CV')
    check(out['selected_index']==min(range(len(scores)),key=lambda j:scores[j]),'hand selection')
    w=F(str(c['hand_initial_weight']));lam=F(str(c['hand_trace_penalty']));rate=F(str(c['hand_learning_rate']))
    for stage in out['gradient_trace']:
        g=F(0);loss=F(0)
        for xx,yy,row in zip(x,y,stage['rows']):
            p=w*xx;res=p-yy;g+=res*xx/4;loss+=res*res/8;close(row['prediction'],float(p),'trace forward');close(row['local_dloss_dprediction'],float(res),'trace local derivative');close(row['mean_gradient_contribution'],float(res*xx/4),'trace gradient contribution')
        close(stage['total_objective'],float(loss+lam*w*w/2),'trace objective');close(stage['total_gradient'],float(g+lam*w),'trace total gradient');w=w-rate*(g+lam*w)
    return {'Fraction_inner_scores':[str(v) for v in scores],'three_forward_stages_all_rows_checked':True,'two_synchronous_updates_checked':True}

def one_grid(x,y,selection,candidates,maximum):
    for fold in selection['folds']:
        tr=fold['training_indices_local'];va=fold['validation_indices_local'];row=[]
        for j,c in enumerate(candidates):
            theta=qr_fit(x[tr],y[tr],c);pred=basis(x[va],c['degree'])@theta;sse=float(np.sum((pred-y[va])**2));row.append(sse)
            delta=abs(sse-fold['candidate_SSE'][j]);maximum[0]=max(maximum[0],delta);maximum[1]=max(maximum[1],delta/(1+abs(sse)));close(sse,fold['candidate_SSE'][j],'independent QR all candidate SSE')
        close(np.array(row)/len(va),fold['candidate_MSE'],'candidate MSE denominator')
    pooled=np.sum([f['candidate_SSE'] for f in selection['folds']],axis=0)/len(x);close(pooled,selection['candidate_scores'],'pooled weights');check(selection['selected_index']==int(np.argmin(pooled)),'candidate argmin')

def risk_reference(model,c):
    # 24-point Gauss-Legendre integrates squared degree<=9 error exactly in exact arithmetic.
    nodes,weights=leggauss(24);truth=basis(nodes,len(c['true_coefficients'])-1)@np.array(c['true_coefficients']);fit=basis(nodes,model['candidate']['degree'])@np.array(model['coefficients']);return float(np.dot(weights,(fit-truth)**2)/2+c['noise_sd']**2)

def dataset_checks(d,record,c,maximum,sklearn_grid=False):
    x=np.asarray(d['x']);y=np.asarray(d['y']);n=len(x);validate_folds(record['outer_splits'],n);outer_total=0.;nested_risks=0.
    for outer in record['nested']['folds']:
        tr=outer['training_indices'];va=outer['validation_indices'];choice=outer['choice'];s=choice['selection'];check(set(tr).isdisjoint(va),'outer disjoint')
        for inner in s['folds']:
            check(set(inner['training_indices_global']).issubset(tr) and set(inner['validation_indices_global']).issubset(tr),'inner indices subset of outer training');check(set(inner['training_indices_global']).isdisjoint(va) and set(inner['validation_indices_global']).isdisjoint(va),'outer labels excluded')
        one_grid(x[tr],y[tr],s,c['candidates'],maximum);check(hashlib.sha256(canonical_bytes(choice)).hexdigest()==outer['choice_sha256'],'choice frozen before outer labels')
        model=choice['refitted_model'];ref=ReferenceRidge(**{'degree':model['candidate']['degree'],'penalty':model['candidate']['penalty']}).fit(x[tr,None],y[tr]);close(ref.coefficients_,model['coefficients'],'actual sklearn selected outer model')
        pred=ref.predict(x[va,None]);sse=float(np.sum((pred-y[va])**2));close(pred,outer['validation_predictions'],'outer prediction');close(sse,outer['validation_SSE'],'outer SSE');outer_total+=sse;rr=risk_reference(model,c);close(rr,outer['oracle_risk_same_fitted_model']['total_MSE'],'independent quadrature outer risk');nested_risks+=len(va)*rr
        if sklearn_grid:
            folds=[{'training_indices':v['training_indices_local'],'validation_indices':v['validation_indices_local']} for v in s['folds']];g=actual_grid_search(x[tr],y[tr],folds,c['candidates']);check(g['selected_index']==s['selected_index'],'actual GridSearchCV selection');close(g['pooled_candidate_MSE'],s['candidate_scores'],'actual GridSearchCV candidate scores')
    close(outer_total/n,record['nested']['pooled_outer_MSE'],'pooled outer score');close(nested_risks/n,record['nested']['mean_oracle_risk_same_outer_models'],'same nested target')
    flat=record['flat'];one_grid(x,y,flat['selection'],c['candidates'],maximum);oracle=[]
    for k,model in enumerate(flat['selected_fold_models']):
        tr=record['outer_splits'][k]['training_indices'];ref=ReferenceRidge(degree=model['candidate']['degree'],penalty=model['candidate']['penalty']).fit(x[tr,None],y[tr]);close(ref.coefficients_,model['coefficients'],'actual sklearn selected flat fold model');risk=risk_reference(model,c);close(risk,flat['oracle_risks_same_selected_fold_models'][k]['total_MSE'],'flat quadrature');oracle.append(len(record['outer_splits'][k]['validation_indices'])*risk)
    close(sum(oracle)/n,flat['mean_oracle_risk_same_selected_fold_models'],'flat same-model target')
    final=record['final_choice'];check(hashlib.sha256(canonical_bytes(final)).hexdigest()==record['final_choice_sha256'],'final choice hash');ref=ReferenceRidge(**final['selected_candidate']).fit(x[:,None],y);close(ref.coefficients_,final['model']['coefficients'],'final full-data reference')
    if sklearn_grid:
        g=actual_grid_search(x,y,record['outer_splits'],c['candidates']);check(g['selected_index']==final['selected_index'],'full GridSearchCV index');close(g['pooled_candidate_MSE'],final['candidate_CV_scores'],'full GridSearchCV scores')

def independence_checks(main,c,r):
    outer=r['primary']['nested']['folds'][0];tr=outer['training_indices'];va=outer['validation_indices'];before=cv.select_and_refit(main['x'][tr],main['y'][tr],c['candidates'],c['inner_folds'],c['split_seed']+1,tr,True);y=main['y'].copy();y[va]=999
    after=cv.select_and_refit(main['x'][tr],y[tr],c['candidates'],c['inner_folds'],c['split_seed']+1,tr,True);check(canonical_bytes(before)==canonical_bytes(after),'outer labels changed its inner choice')
    # Changing the diagnostic oracle cannot change any training choice.
    changed=copy.deepcopy(c);changed['true_coefficients']=[0.,0.];changed['noise_sd']=1.1;other=cv.run_dataset(main['x'],main['y'],changed,c['split_seed'],False)
    check(other['final_choice_sha256']==r['primary']['final_choice_sha256'],'oracle influenced final selection')
    for i,row in enumerate(other['nested']['folds']):
        left=row['choice'];right=r['primary']['nested']['folds'][i]['choice'];a=copy.deepcopy(right)
        for fold in a['selection']['folds']:fold.pop('candidate_fits',None)
        check(canonical_bytes(left)==canonical_bytes(a),'oracle influenced nested choice')
    chosen=r['primary']['final_choice'];test=r['final_test'];check(test['choice_sha256_before_test']==r['primary']['final_choice_sha256'],'final test ordering fingerprint');p=models.predict(chosen['model'],test['test_data']['x']);loss=(p-np.array(test['test_data']['y']))**2;close(loss,test['squared_losses'],'final test full losses');close(loss.mean(),test['MSE'],'final test score')
    return {'outer_label_mutation_leaves_its_choice_byte_identical':True,'oracle_change_leaves_all_choices_identical':True,'final_test_not_passed_to_selection_API':True}

def guard_checks(main,repeated,c):
    bad=copy.deepcopy(c);bad['split_demo_gap']=True
    with patch.object(e,'run_dataset',side_effect=RuntimeError('fit reached')):reject(lambda:e.main_report(main,repeated,bad),'last config')
    rr=copy.deepcopy(repeated);rr[-1]['y']=rr[-1]['y'].astype(object);rr[-1]['y'][-1]=True
    with patch.object(e,'run_dataset',side_effect=RuntimeError('fit reached')):reject(lambda:e.main_report(main,rr,c),'last repeated label')
    for value in [True,'1',1+0j,np.nan,np.inf]:reject(lambda:models.fit([-1,0,1],[0,1,value],{'degree':1,'penalty':.1}),'bad final training value')
    folds=make_folds(7,3,1);bad=copy.deepcopy(folds);bad[-1]['validation_indices'][0]=bad[-1]['training_indices'][0];reject(lambda:validate_folds(bad,7),'overlapping fold')
    # Unequal-fold score must weight observations, not average fold MSE blindly.
    xx=np.linspace(-1,1,7);yy=np.array([0,1,0,2,1,3,2]);out,_=cv.evaluate_grid(xx,yy,folds,[{'degree':0,'penalty':0}]);close(out['candidate_scores'][0],sum(f['candidate_SSE'][0] for f in out['folds'])/7,'unequal fold weights')
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/'out.json';p.write_bytes(b'OLD');reject(lambda:safe_write(p,{'bad':np.nan}),'nonfinite output');check(p.read_bytes()==b'OLD','preserve existing JSON')
        alias=Path(td)/'alias.json';alias.symlink_to(p);reject(lambda:safe_write(alias,{}),'symlink output');reject(lambda:safe_write(e.ROOT/'data/protocol.json',{}),'source overwrite')
    return {'ordinary_invalid_input_and_precompute_last_field_checks':True,'unequal_fold_weighting_checked':True,'JSON_preservation_and_alias_checks':True}

def audit(report=None):
    main,repeated,c=e.load_inputs();r=e.main_report(main,repeated,c) if report is None else report
    check(r['unit']=='051' and r['protocol']==c,'report protocol')
    for expected,actual in [(main,r['data']['development'])]+list(zip(repeated,r['data']['repeated_development'])):
        check(expected['ids']==actual['ids'],'report IDs');close(expected['x'],actual['x'],'report x');close(expected['y'],actual['y'],'report y')
    check(len(r['repeated'])==len(repeated),'repeat count');maximum=[0.,0.];dataset_checks(main,r['primary'],c,maximum,True)
    for d,row in zip(repeated,r['repeated']):dataset_checks(d,row['result'],c,maximum)
    for filename,raw in generate(c).items():check((e.ROOT/'data'/filename).read_bytes()==raw,'source CSV generation')
    flat=np.array([v['result']['flat']['oracle_minus_CV'] for v in r['repeated']]);nested=np.array([v['result']['nested']['oracle_minus_CV'] for v in r['repeated']])
    for name,vals in [('flat_optimism',flat),('nested_optimism',nested)]:close(r['repeated_summary'][name]['mean'],vals.mean(),'repeat mean');close(r['repeated_summary'][name]['MCSE_mean'],vals.std(ddof=1)/math.sqrt(len(vals)),'MCSE across datasets')
    demo=r['splitter_demo']
    for f in demo['splitters']['GroupKFold']:check(set(f['training_groups']).isdisjoint(f['validation_groups']),'GroupKFold groups')
    for f in demo['splitters']['TimeSeriesSplit']:check(f['training_max_time']+demo['gap']<f['validation_min_time'],'time order/gap')
    return {'unit':'051','hand':hand_check(c,r),'independent_all_grid_SSE_max_absolute_error':maximum[0],'independent_all_grid_SSE_max_scaled_error':maximum[1],'all_81_datasets_inner_and_outer_grid_scores_checked':True,'actual_sklearn_selected_models_checked':True,'actual_GridSearchCV_primary_five_inner_and_final_checked':True,'selected_predictor_population_risks_checked_by_quadrature':True,'independence':independence_checks(main,c,r),'guards':guard_checks(main,repeated,c),'data_CSV_byte_reproduced':True,'splitter_group_time_constraints_checked':True,'core_sha256':hashlib.sha256(canonical_bytes(r)).hexdigest()}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report');p.add_argument('--out',required=True);a=p.parse_args();r=audit(read_json(a.report) if a.report else None);safe_write(a.out,r);print(json.dumps(r,ensure_ascii=False))
