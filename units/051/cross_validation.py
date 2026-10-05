"""Inner parameter search and outer evaluation with explicit data roles."""
import hashlib,math
import numpy as np
from numeric import vector,scalar,canonical_bytes
from folds import make_folds,validate_folds
import models

def evaluate_grid(x,y,folds,candidates,source_indices=None,detail=False):
    x=vector(x,'CV x',-1,1);y=vector(y,'CV y',-1000,1000,len(x));folds=validate_folds(folds,len(x))
    if not isinstance(candidates,list) or not 1<=len(candidates)<=30:raise ValueError('candidate list')
    candidates=[models.candidate(c) for c in candidates]
    if len({(c['degree'],c['penalty']) for c in candidates})!=len(candidates):raise ValueError('duplicate candidate')
    if any(len(f['training_indices'])<max(c['degree'] for c in candidates)+2 for f in folds):raise ValueError('CV training fold too small for candidates')
    if source_indices is None:source_indices=list(range(len(x)))
    if not isinstance(source_indices,list) or len(source_indices)!=len(x):raise ValueError('global source indices length')
    source_indices=[scalar(v,'global source index',0,10000,True) for v in source_indices]
    if len(set(source_indices))!=len(source_indices):raise ValueError('duplicate global source index')
    sse=np.zeros((len(folds),len(candidates)));fitted=[];records=[]
    for k,fold in enumerate(folds):
        tr=fold['training_indices'];va=fold['validation_indices'];models_here=[];prediction_rows=[]
        for j,c in enumerate(candidates):
            model=models.fit(x[tr],y[tr],c);prediction=models.predict(model,x[va]);errors=(prediction-y[va])**2;sse[k,j]=math.fsum(errors);models_here.append(model)
            if detail:prediction_rows.append({'candidate_index':j,'model':model,'validation_predictions':prediction.tolist(),'validation_squared_errors':errors.tolist()})
        fitted.append(models_here)
        record={'fold':k,'training_indices_local':tr,'validation_indices_local':va,'training_indices_global':[source_indices[i] for i in tr],'validation_indices_global':[source_indices[i] for i in va],'n_validation':len(va),'candidate_SSE':sse[k].tolist(),'candidate_MSE':(sse[k]/len(va)).tolist()}
        if detail:record['candidate_fits']=prediction_rows
        records.append(record)
    scores=sse.sum(axis=0)/len(x);chosen=int(np.argmin(scores));choice={'candidate_scores':scores.tolist(),'selected_index':chosen,'selected_candidate':candidates[chosen],'tie_indices':np.flatnonzero(scores==scores[chosen]).tolist(),'aggregation':'sum validation SSE over all folds divided by number of validation observations; weights unequal folds by sample count','source_indices':source_indices,'folds':records}
    return choice,fitted

def select_and_refit(x_training,y_training,candidates,inner_folds,seed,source_indices=None,detail=False):
    # There is deliberately no outer-validation label or oracle argument.
    x=vector(x_training,'selection training x',-1,1);y=vector(y_training,'selection training y',-1000,1000,len(x));folds=make_folds(len(x),inner_folds,seed)
    choice,_=evaluate_grid(x,y,folds,candidates,source_indices,detail);model=models.fit(x,y,choice['selected_candidate'])
    record={'selection':choice,'refitted_model':model};return record

def run_dataset(x,y,c,split_seed,detail=False):
    x=vector(x,'dataset x',-1,1);y=vector(y,'dataset y',-1000,1000,len(x));outer=make_folds(len(x),c['outer_folds'],split_seed);nested=[];oof=np.empty(len(x));sse_total=0.;risk_weighted=0.
    for k,fold in enumerate(outer):
        tr=fold['training_indices'];va=fold['validation_indices'];choice=select_and_refit(x[tr],y[tr],c['candidates'],c['inner_folds'],split_seed+1+k,tr,detail)
        # Serialize the full choice before touching this outer fold's outcomes.
        fingerprint=hashlib.sha256(canonical_bytes(choice)).hexdigest();prediction=models.predict(choice['refitted_model'],x[va]);squared=(prediction-y[va])**2;sse=float(math.fsum(squared));risk=models.population_risk(choice['refitted_model'],c['true_coefficients'],c['noise_sd']);oof[va]=prediction;sse_total+=sse;risk_weighted+=len(va)*risk['total_MSE']
        nested.append({'outer_fold':k,'training_indices':tr,'validation_indices':va,'choice':choice,'choice_sha256':fingerprint,'validation_predictions':prediction.tolist(),'validation_squared_errors':squared.tolist(),'validation_SSE':sse,'validation_MSE':sse/len(va),'oracle_risk_same_fitted_model':risk})
    nested_score=sse_total/len(x);nested_oracle=risk_weighted/len(x)
    flat,models_by_fold=evaluate_grid(x,y,outer,c['candidates'],list(range(len(x))),detail);j=flat['selected_index'];flat_models=[row[j] for row in models_by_fold];flat_risks=[models.population_risk(model,c['true_coefficients'],c['noise_sd']) for model in flat_models];flat_oracle=math.fsum(len(outer[k]['validation_indices'])*v['total_MSE'] for k,v in enumerate(flat_risks))/len(x);flat_score=flat['candidate_scores'][j]
    final_model=models.fit(x,y,flat['selected_candidate']);final_choice={'selected_index':j,'selected_candidate':flat['selected_candidate'],'candidate_CV_scores':flat['candidate_scores'],'development_count':len(x),'model':final_model};final_hash=hashlib.sha256(canonical_bytes(final_choice)).hexdigest()
    return {'outer_splits':outer,'nested':{'folds':nested,'OOF_predictions':oof.tolist(),'pooled_outer_MSE':nested_score,'mean_oracle_risk_same_outer_models':nested_oracle,'oracle_minus_CV':nested_oracle-nested_score,'target':'selection-and-refit algorithm trained on outer-training rows; not the final all-data fitted model'},'flat':{'selection':flat,'selected_fold_models':flat_models,'oracle_risks_same_selected_fold_models':flat_risks,'minimum_reused_CV_MSE':flat_score,'mean_oracle_risk_same_selected_fold_models':flat_oracle,'oracle_minus_CV':flat_oracle-flat_score,'target':'same validation folds were used to choose the hyperparameter and to report its minimum score'},'final_choice':final_choice,'final_choice_sha256':final_hash,'final_model_oracle_risk':models.population_risk(final_model,c['true_coefficients'],c['noise_sd']),'different_training_size_note':'outer models use n minus validation-fold size; final model uses all n rows, so their risk targets differ'}
