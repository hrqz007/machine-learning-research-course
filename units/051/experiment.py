"""ML051 nested versus reused CV, paired to the same predictors' oracle risks."""
from pathlib import Path
import argparse,csv,io,json,hashlib,math
import numpy as np
from numeric import ROOT,read_json,token,scalar,vector,canonical_bytes,safe_write
from protocol import validate
from generate_data import final_test
from cross_validation import run_dataset
from folds import splitter_demo
import models,hand_example

def read_csv(path,header):
    reader=csv.DictReader(io.StringIO(Path(path).read_text()))
    if reader.fieldnames!=header:raise ValueError('CSV schema mismatch')
    rows=[];seen=set()
    for row in reader:
        if set(row)!=set(header) or any(v is None for v in row.values()):raise ValueError('malformed CSV')
        ident=row['id']
        if not ident or len(ident)>40 or not ident.isascii() or not all(v.isalnum() or v in '_-' for v in ident) or ident in seen:raise ValueError('invalid/duplicate ID')
        seen.add(ident);d={'id':ident}
        for k in header[1:]:
            if k=='repetition':
                if not row[k].isdigit():raise ValueError('integer repetition required')
                d[k]=int(row[k])
            else:d[k]=token(row[k])
        rows.append(d)
    return rows

def rows_to_dataset(rows,n):
    if len(rows)!=n:raise ValueError('dataset row count mismatch')
    return {'ids':[v['id'] for v in rows],'x':vector([v['x'] for v in rows],'development x',-1,1,n),'y':vector([v['y'] for v in rows],'development y',-1000,1000,n),'true_mean':vector([v['true_mean'] for v in rows],'oracle true mean',-100,100,n)}

def load_inputs(directory=None,config=None):
    base=Path(directory) if directory else ROOT/'data';c=validate(read_json(config or base/'protocol.json'));main=rows_to_dataset(read_csv(base/'development.csv',['id','x','y','true_mean']),c['sample_count']);rr=read_csv(base/'repeated_development.csv',['id','repetition','x','y','true_mean'])
    if len(rr)!=c['sample_count']*c['repetitions'] or any(v['repetition'] not in range(c['repetitions']) for v in rr):raise ValueError('repeated rows/count/index mismatch')
    repeated=[rows_to_dataset([v for v in rr if v['repetition']==j],c['sample_count']) for j in range(c['repetitions'])]
    return main,repeated,c

def validate_dataset(d,n):
    if not isinstance(d,dict) or set(d)!={'ids','x','y','true_mean'}:raise ValueError('dataset fields')
    if not isinstance(d['ids'],list) or len(d['ids'])!=n or len(set(d['ids']))!=n or any(not isinstance(v,str) or not v for v in d['ids']):raise ValueError('dataset IDs')
    return {'ids':list(d['ids']),'x':vector(d['x'],'dataset x',-1,1,n),'y':vector(d['y'],'dataset y',-1000,1000,n),'true_mean':vector(d['true_mean'],'oracle true mean',-100,100,n)}

def summarize(a):
    a=np.asarray(a,dtype=float);return {'count':len(a),'mean':float(a.mean()),'SD_across_independent_datasets':float(a.std(ddof=1)),'MCSE_mean':float(a.std(ddof=1)/math.sqrt(len(a))),'min':float(a.min()),'max':float(a.max()),'negative_count':int((a<0).sum())}

def main_report(main,repeated,c):
    c=validate(c);main=validate_dataset(main,c['sample_count'])
    if not isinstance(repeated,list) or len(repeated)!=c['repetitions']:raise ValueError('repeated dataset list count')
    repeated=[validate_dataset(d,c['sample_count']) for d in repeated]
    hand=hand_example.report(c);primary=run_dataset(main['x'],main['y'],c,c['split_seed'],detail=True)
    # The final test draw occurs only after this full-data choice has been frozen.
    choice_hash=primary['final_choice_sha256'];test=final_test(c);pred=models.predict(primary['final_choice']['model'],test['x']);loss=(pred-test['y'])**2
    test_report={'choice_sha256_before_test':choice_hash,'test_data':{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in test.items()},'predictions':pred.tolist(),'squared_losses':loss.tolist(),'MSE':float(loss.mean()),'conditional_test_mean_loss_SE':float(loss.std(ddof=1)/math.sqrt(len(loss))),'scope':'fresh test after all-data selection; conditional test sampling uncertainty, not across-training-dataset uncertainty'}
    records=[]
    for j,d in enumerate(repeated):
        result=run_dataset(d['x'],d['y'],c,c['split_seed']+100*(j+1),detail=False);records.append({'repetition':j,'result':result})
    keys=['flat_CV','flat_same_models_oracle','flat_optimism','nested_CV','nested_same_models_oracle','nested_optimism','final_all_data_oracle','flat_minus_nested_CV'];values={k:[] for k in keys}
    for row in records:
        a=row['result'];flat=a['flat'];nested=a['nested'];vals=[flat['minimum_reused_CV_MSE'],flat['mean_oracle_risk_same_selected_fold_models'],flat['oracle_minus_CV'],nested['pooled_outer_MSE'],nested['mean_oracle_risk_same_outer_models'],nested['oracle_minus_CV'],a['final_model_oracle_risk']['total_MSE'],flat['minimum_reused_CV_MSE']-nested['pooled_outer_MSE']]
        for k,v in zip(keys,vals):values[k].append(v)
    import scipy,sklearn
    return {'unit':'051','protocol':c,'data':{'development':{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in main.items()},'repeated_development':[{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in d.items()} for d in repeated]},'hand':hand,'primary':primary,'final_test':test_report,'repeated':records,'repeated_summary':{k:summarize(v) for k,v in values.items()},'splitter_demo':splitter_demo(c['split_demo_gap']),'versions':{'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__},'limitations':['synthetic IID regression only; group/time splitters are index-design demonstrations','nested CV evaluates the whole inner selection and refit procedure on outer-training sample size','flat and nested optimism paired to SAME corresponding fold-trained models, not confused with full-data training size','outer folds overlap in training data and are not independent replications','MCSE calculated across independent datasets, no simple t test on five outer folds','oracle risks never enter candidate selection','one run can show reversed differences; fixed protocol and all results retained']}

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-directory');p.add_argument('--config');p.add_argument('--out',default=str(ROOT/'outputs/result.json'));a=p.parse_args()
    if a.data_directory is None and a.config is None:
        for rel,h in read_json(ROOT/'data_integrity.json')['sha256'].items():
            if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=h:raise ValueError('teaching data changed '+rel)
    r=main_report(*load_inputs(a.data_directory,a.config));protected=[a.config] if a.config else []
    if a.data_directory:protected.extend(v for v in Path(a.data_directory).glob('*') if v.is_file())
    safe_write(a.out,r,protected);print(json.dumps({'primary_flat':r['primary']['flat']['minimum_reused_CV_MSE'],'primary_nested':r['primary']['nested']['pooled_outer_MSE'],'repeated_summary':r['repeated_summary'],'core_sha256':hashlib.sha256(canonical_bytes(r)).hexdigest()}))
if __name__=='__main__':main()
