"""Validate the complete nested-CV protocol before any fit or draw."""
from numeric import scalar,vector
FIELDS={'kind','main_seed','repeated_seed','split_seed','test_seed','sample_count','repetitions','test_count','outer_folds','inner_folds','true_coefficients','noise_sd','candidates','tie_rule','hand_training_x','hand_training_y','hand_test_x','hand_test_y','hand_inner_validation_indices','hand_candidate_penalties','hand_trace_penalty','hand_initial_weight','hand_learning_rate','split_demo_gap'}
def validate(c):
    if not isinstance(c,dict) or set(c)!=FIELDS or c['kind']!='nested_cross_validation_legendre_v1' or c['tie_rule']!='first_candidate_in_declared_order':raise ValueError('protocol fields/conventions mismatch')
    q=dict(c)
    ranges={'main_seed':(0,2**32-1),'repeated_seed':(0,2**32-1),'split_seed':(0,2**32-1),'test_seed':(0,2**32-1),'sample_count':(30,300),'repetitions':(2,200),'test_count':(20,10000),'outer_folds':(2,10),'inner_folds':(2,10),'split_demo_gap':(0,2)}
    for k,(lo,hi) in ranges.items():q[k]=scalar(c[k],k,lo,hi,True)
    q['noise_sd']=scalar(c['noise_sd'],'noise sd',.05,3);q['true_coefficients']=vector(c['true_coefficients'],'true Legendre coefficients',-10,10).tolist()
    if len(q['true_coefficients'])>10:raise ValueError('true degree exceeds9')
    if not isinstance(c['candidates'],list) or not 1<=len(c['candidates'])<=30:raise ValueError('candidate list requires1..30')
    q['candidates']=[];seen=set()
    for row in c['candidates']:
        if not isinstance(row,dict) or set(row)!={'degree','penalty'}:raise ValueError('candidate fields')
        degree=scalar(row['degree'],'candidate degree',0,9,True);penalty=scalar(row['penalty'],'candidate penalty',0,10)
        if degree==0 and penalty!=0:raise ValueError('constant candidate has no penalized coefficient; require penalty0')
        if (degree,penalty) in seen:raise ValueError('duplicate candidate')
        seen.add((degree,penalty));q['candidates'].append({'degree':degree,'penalty':penalty})
    outer_min=q['sample_count']-(q['sample_count']+q['outer_folds']-1)//q['outer_folds'];inner_min=outer_min-(outer_min+q['inner_folds']-1)//q['inner_folds']
    if inner_min<max(v['degree'] for v in q['candidates'])+2:raise ValueError('smallest inner training fold too small')
    if q['repetitions']*q['outer_folds']*q['inner_folds']*len(q['candidates'])>250000:raise ValueError('teaching fit-count resource limit')
    for key,n in [('hand_training_x',4),('hand_training_y',4),('hand_test_x',2),('hand_test_y',2)]:q[key]=vector(c[key],key,-5 if key.endswith('_x') else -20,5 if key.endswith('_x') else 20,n).tolist()
    if not isinstance(c['hand_inner_validation_indices'],list) or len(c['hand_inner_validation_indices'])!=2:raise ValueError('two hand folds required')
    rows=[]
    for row in c['hand_inner_validation_indices']:
        if not isinstance(row,list) or len(row)!=2:raise ValueError('hand folds size2')
        rows.append([scalar(v,'hand fold index',0,3,True) for v in row])
    if sorted(sum(rows,[]))!=[0,1,2,3]:raise ValueError('hand folds must partition four training rows')
    q['hand_inner_validation_indices']=rows;q['hand_candidate_penalties']=vector(c['hand_candidate_penalties'],'hand candidate penalties',0,10).tolist()
    if len(q['hand_candidate_penalties'])>5 or len(set(q['hand_candidate_penalties']))!=len(q['hand_candidate_penalties']):raise ValueError('hand penalty list')
    q['hand_trace_penalty']=scalar(c['hand_trace_penalty'],'hand trace penalty',0,10);q['hand_initial_weight']=scalar(c['hand_initial_weight'],'hand initial weight',-10,10);q['hand_learning_rate']=scalar(c['hand_learning_rate'],'hand learning rate',1e-6,1)
    for val in rows:
        train=[i for i in range(4) if i not in val]
        if sum(q['hand_training_x'][i]**2 for i in train)==0 and 0 in q['hand_candidate_penalties']:raise ValueError('unpenalized hand slope is unidentifiable')
    return q
