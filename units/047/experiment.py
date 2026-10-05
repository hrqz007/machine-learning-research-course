"""ML047: independent probability calibration and split-conformal experiments."""
from pathlib import Path
import argparse,csv,hashlib,io,json,math
import numpy as np
from numeric import ROOT,scalar,vector,labels,read_json,token,canonical_bytes,safe_write
from protocol import validate
import calibration as cb
import conformal as cf
from generate_data import regression_draw


def read_csv(path,columns):
    reader=csv.DictReader(io.StringIO(Path(path).read_text()))
    if reader.fieldnames!=columns:raise ValueError('CSV columns mismatch: '+Path(path).name)
    out=[];seen=set()
    for row in reader:
        if set(row)!=set(columns) or any(v is None for v in row.values()):raise ValueError('malformed CSV row')
        sid=row['id']
        if not sid or len(sid)>40 or not sid.isascii() or not all(v.isalnum() or v in '_-' for v in sid) or sid in seen:raise ValueError('invalid/duplicate ID')
        seen.add(sid);d={'id':sid}
        for key in columns[1:]:d[key]=row[key] if key=='split' else token(row[key])
        out.append(d)
    if not out:raise ValueError('empty CSV')
    return out


def load_inputs(directory=None,config=None):
    base=Path(directory) if directory else ROOT/'data';c=validate(read_json(config or base/'protocol.json'))
    rows=read_csv(base/'classification.csv',['id','split','x','raw_logit','y','true_probability']);classification={}
    if any(r['split'] not in ('calibration','evaluation') for r in rows):raise ValueError('unknown classification split')
    for role,key in [('calibration','classification_calibration_count'),('evaluation','classification_evaluation_count')]:
        group=[r for r in rows if r['split']==role]
        if len(group)!=c[key]:raise ValueError('classification split count does not match protocol')
        classification[role]={'ids':[r['id'] for r in group],'x':vector([r['x'] for r in group],'classification x',-100,100),'z':vector([r['raw_logit'] for r in group],'classification logits'),
          'y':labels([r['y'] for r in group]),'true_probability':vector([r['true_probability'] for r in group],'teaching true probability',0,1)}
    regression={}
    for role,key in [('train','regression_train_count'),('calibration','regression_calibration_count'),('evaluation','regression_evaluation_count')]:
        columns=['id','x','y','true_mean','noise_sd']+(['shifted_y'] if role=='evaluation' else []);rr=read_csv(base/('regression_'+role+'.csv'),columns)
        if len(rr)!=c[key]:raise ValueError('regression split count mismatch')
        r={'ids':[v['id'] for v in rr]}
        for name in columns[1:]:r[name]=vector([v[name] for v in rr],role+' '+name,0 if name=='noise_sd' else -1000,1000)
        regression[role]=r
    return classification,regression,c


def validate_loaded(classification,regression,c):
    if not isinstance(classification,dict) or set(classification)!={'calibration','evaluation'}:raise ValueError('classification split dictionary mismatch')
    if not isinstance(regression,dict) or set(regression)!={'train','calibration','evaluation'}:raise ValueError('regression split dictionary mismatch')
    def ids(value,n):
        if not isinstance(value,(list,tuple)) or len(value)!=n or any(not isinstance(v,str) or not v or len(v)>40 for v in value) or len(set(value))!=n:raise ValueError('invalid loaded IDs')
        return list(value)
    cc={}
    for role,key in [('calibration','classification_calibration_count'),('evaluation','classification_evaluation_count')]:
        d=classification[role];n=c[key]
        if not isinstance(d,dict) or set(d)!={'ids','x','z','y','true_probability'}:raise ValueError('classification fields mismatch')
        cc[role]={'ids':ids(d['ids'],n),'x':vector(d['x'],'classification x',-100,100,n),'z':vector(d['z'],'classification logits',-1000,1000,n),'y':labels(d['y'],n),'true_probability':vector(d['true_probability'],'oracle probability',0,1,n)}
    rr={}
    for role,key in [('train','regression_train_count'),('calibration','regression_calibration_count'),('evaluation','regression_evaluation_count')]:
        d=regression[role];n=c[key];keys={'ids','x','y','true_mean','noise_sd'}|({'shifted_y'} if role=='evaluation' else set())
        if not isinstance(d,dict) or set(d)!=keys:raise ValueError('regression fields mismatch')
        out={'ids':ids(d['ids'],n)}
        for name in keys-{'ids'}:out[name]=vector(d[name],role+' '+name,0 if name=='noise_sd' else -100 if name=='x' else -1000,100 if name=='x' else 1000,n)
        if np.any(out['noise_sd']<=0):raise ValueError('documented noise standard deviations must be positive')
        rr[role]=out
    return cc,rr


def hand_report(c):
    magnitude=2*math.log(3);z=magnitude*np.array(c['hand_signs']);y=np.array(c['hand_labels']);a=c['hand_initial_inverse_temperature'];states=[]
    for k in range(3):
        state=cb.temperature_state(z,y,a);state['stage']=k;states.append(state)
        if k<2:a=float(np.clip(a-c['hand_learning_rate']*state['gradient'],0,4))
    fit=cb.fit_temperature(z,y,c['temperature_inverse_bounds'],c['temperature_tolerance'])
    default=c['hand_labels']==[0,1,1,1] and c['hand_signs']==[-1.,-1.,1.,1.]
    return {'raw_logits':z.tolist(),'labels':y.tolist(),'states':states,'fitted':fit,
      'initial_metrics':cb.evaluate(z,y,c['hand_initial_inverse_temperature'],c['reliability_bins']),
      'optimum_metrics':cb.evaluate(z,y,fit['final']['inverse_temperature'],c['reliability_bins']),
      'analytic_certificate':{'inverse_temperature':.5,'temperature':2.,'positive_probabilities':[.25,.25,.75,.75],'binary_Brier':.1875,'positive_class_ECE':.25,'top_label_ECE':0.,'accuracy':.75,'scope':'declared symmetric four-logit example only'} if default else None}


def temperature_report(classification,c):
    cal=classification['calibration'];evaluation=classification['evaluation']
    fitted=cb.fit_temperature(cal['z'],cal['y'],c['temperature_inverse_bounds'],c['temperature_tolerance'])
    # Choice record is frozen before any evaluation metrics are computed.
    choice={'fit':fitted,'calibration_count':len(cal['y']),'calibration_ids':cal['ids'],'bins':c['reliability_bins']}
    choice_hash=hashlib.sha256(canonical_bytes(choice)).hexdigest();a=fitted['final']['inverse_temperature']
    from scipy.optimize import minimize_scalar
    lo,hi=c['temperature_inverse_bounds'];reference=minimize_scalar(lambda s:cb.temperature_state(cal['z'],cal['y'],s,rows=False)['mean_log_loss'],bounds=(lo,hi),method='bounded',options={'xatol':1e-12,'maxiter':500})
    metrics={}
    for name,d in [('calibration',cal),('evaluation',evaluation)]:metrics[name]={'before':cb.evaluate(d['z'],d['y'],1,c['reliability_bins']),'after':cb.evaluate(d['z'],d['y'],a,c['reliability_bins'])}
    profile=[{'a':float(s),'calibration_log_loss':cb.temperature_state(cal['z'],cal['y'],float(s),rows=False)['mean_log_loss']} for s in np.linspace(lo,hi,81)]
    return {'frozen_choice':choice,'choice_sha256':choice_hash,'metrics':metrics,'calibration_profile':profile,
            'scipy_reference':{'success':bool(reference.success),'message':str(reference.message),'inverse_temperature':float(reference.x),'objective':float(reference.fun),'function_evaluations':int(reference.nfev)},
            'true_probability_scope':'oracle columns are present only for mechanism documentation and not passed to fit_temperature'}


def regression_report(regression,c):
    train=regression['train'];cal=regression['calibration'];test=regression['evaluation']
    main=cf.trial(train['x'],train['y'],cal['x'],cal['y'],test['x'],test['y'],test['shifted_y'],c['miscoverage'],c['regression_group_abs_boundary'])
    rng=np.random.default_rng(c['regression_seed']+1);records=[]
    for rep in range(c['regression_repetitions']):
        tx,ty,_,_,_=regression_draw(rng,c['regression_train_count'],c);cx,cy,_,_,_=regression_draw(rng,c['regression_calibration_count'],c);ex,ey,shifted,_,_=regression_draw(rng,c['regression_evaluation_count'],c)
        result=cf.trial(tx,ty,cx,cy,ex,ey,shifted,c['miscoverage'],c['regression_group_abs_boundary'])
        records.append({'repetition':rep,'theta':result['fit']['theta'],'rank':result['quantile']['rank'],'quantile':result['quantile']['quantile'],'infinite':result['quantile']['infinite'],
           'coverage':result['baseline']['coverage'],'shifted_coverage':result['noise_shift']['coverage'],'width':result['baseline']['width'],'groups':result['groups'],'invalid_reuse_coverage':result['invalid_calibration_reuse']['evaluation']['coverage']})
    def summarize(values):
        observed=[float(v) for v in values if v is not None]
        if not observed:return {'available_repetitions':0,'mean':None,'sd':None,'mc_standard_error':None}
        vals=np.array(observed);return {'available_repetitions':len(vals),'mean':float(vals.mean()),'sd':float(vals.std(ddof=1)) if len(vals)>1 else None,'mc_standard_error':float(vals.std(ddof=1)/math.sqrt(len(vals))) if len(vals)>1 else None}
    summary={key:summarize([r[key] for r in records]) for key in ['coverage','shifted_coverage','width','invalid_reuse_coverage']}
    summary['low_abs_x']=summarize([r['groups']['low_abs_x']['coverage'] for r in records]);summary['high_abs_x']=summarize([r['groups']['high_abs_x']['coverage'] for r in records])
    grid=np.linspace(c['regression_feature_low'],c['regression_feature_high'],61);T=np.array([r['theta'] for r in records]);pred=T[:,0,None]+T[:,1,None]*grid[None,:]
    uncertainty={'x':grid.tolist(),'known_conditional_noise_variance':((c['regression_noise_intercept']+c['regression_noise_abs_slope']*abs(grid))**2).tolist(),
       'repeated_training_prediction_variance_ddof1':pred.var(axis=0,ddof=1).tolist(),'mean_prediction':pred.mean(axis=0).tolist(),
       'scope':'sampling variability of this correctly specified line estimator; not a complete Bayesian posterior or all model misspecification uncertainty'}
    boundaries=[cf.finite_sample_quantile(np.arange(1,10,dtype=float),a) for a in c['miscoverage_boundary_examples']]
    return {'main':main,'repetitions':records,'summary':summary,'monte_carlo_seed':c['regression_seed']+1,
       'uncertainty_components':uncertainty,'small_n_quantile_examples':boundaries,
       'mc_scope':'summaries across independent training/calibration/evaluation repetitions; test points sharing calibration are not counted as independent replications'}


def main_report(classification,regression,c):
    c=validate(c)
    classification,regression=validate_loaded(classification,regression,c)
    hand=hand_report(c);temperature=temperature_report(classification,c);conformal=regression_report(regression,c)
    data={'classification':{k:{name:v.tolist() if isinstance(v,np.ndarray) else v for name,v in d.items()} for k,d in classification.items()},
          'regression':{k:{name:v.tolist() if isinstance(v,np.ndarray) else v for name,v in d.items()} for k,d in regression.items()}}
    import scipy,sklearn
    return {'unit':'047','protocol':c,'hand':hand,'temperature':temperature,'conformal':conformal,'data':data,'versions':{'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__},
      'limitations':['two separate calibration and regression experiments; no trained neural-network result claimed','positive-class reliability and top-label confidence reliability are explicitly different objects','split conformal guarantees marginal coverage under exchangeability, not per-group or per-individual coverage','miscoverage decimal semantics and finite-sample infinity case explicit','noise shift and calibration reuse intentionally violate conditions','Monte Carlo is a diagnostic, not the rank proof itself']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-directory');p.add_argument('--config');p.add_argument('--out',default=str(ROOT/'outputs/result.json'));a=p.parse_args()
    if a.data_directory is None and a.config is None:
        for rel,h in read_json(ROOT/'data_integrity.json')['sha256'].items():
            if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=h:raise ValueError('teaching data changed: '+rel)
    inputs=load_inputs(a.data_directory,a.config);r=main_report(*inputs);protected=[a.config] if a.config else []
    if a.data_directory:protected.extend(p for p in Path(a.data_directory).glob('*') if p.is_file())
    safe_write(a.out,r,protected);print(json.dumps({'temperature':r['temperature']['frozen_choice']['fit']['final']['temperature'],'coverage_summary':r['conformal']['summary'],'core_sha256':hashlib.sha256(canonical_bytes(r)).hexdigest()}))
if __name__=='__main__':main()
