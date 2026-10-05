"""Full protocol validation before random draws or selection."""
from numeric import scalar,vector
FIELDS={'kind','main_seed','main_sample_sizes','main_repetitions','delta','exact_sample_sizes','exact_epsilon_numerators','exact_epsilon_denominator','noise_seed','noise_sample_sizes','noise_candidate_counts','noise_repetitions','holdout_count','dependent_repeat_count','tie_rule','loss','hand_added_batch','vc_max_points'}
def integers(x,name,lo,hi,maximum=20):
    if not isinstance(x,list) or not 1<=len(x)<=maximum:raise ValueError(name+' must be a bounded list')
    v=[scalar(y,name,lo,hi,True) for y in x]
    if len(set(v))!=len(v) or v!=sorted(v):raise ValueError(name+' must be strictly increasing')
    return v

def validate(c):
    if not isinstance(c,dict) or set(c)!=FIELDS or c['kind']!='finite_class_learning_guarantees_v1' or c['tie_rule']!='first_declared_hypothesis' or c['loss']!='zero_one':raise ValueError('protocol fields or conventions mismatch')
    q=dict(c)
    for key in ['main_seed','noise_seed']:q[key]=scalar(c[key],key,0,2**32-1,True)
    for key in ['main_repetitions','noise_repetitions']:q[key]=scalar(c[key],key,2,10000,True)
    q['delta']=scalar(c['delta'],'delta',1e-12,1-1e-12)
    q['main_sample_sizes']=integers(c['main_sample_sizes'],'main sample sizes',1,10000,12)
    q['exact_sample_sizes']=integers(c['exact_sample_sizes'],'exact sample sizes',1,8,8)
    q['exact_epsilon_denominator']=scalar(c['exact_epsilon_denominator'],'epsilon denominator',1,100,True)
    q['exact_epsilon_numerators']=integers(c['exact_epsilon_numerators'],'epsilon numerators',1,q['exact_epsilon_denominator'],12)
    q['noise_sample_sizes']=integers(c['noise_sample_sizes'],'noise sample sizes',2,200,8)
    q['noise_candidate_counts']=integers(c['noise_candidate_counts'],'candidate counts',1,2048,10)
    q['holdout_count']=scalar(c['holdout_count'],'holdout count',2,2000,True);q['dependent_repeat_count']=scalar(c['dependent_repeat_count'],'dependent repeat count',1,10000,True);q['vc_max_points']=scalar(c['vc_max_points'],'VC demonstration max points',3,8,True)
    if q['noise_repetitions']*max(q['noise_candidate_counts'])>5000000:raise ValueError('simulation matrix exceeds teaching resource limit')
    if not isinstance(c['hand_added_batch'],list) or not 1<=len(c['hand_added_batch'])<=10:raise ValueError('hand added batch requires1..10 rows')
    rows=[]
    for row in c['hand_added_batch']:
        v=vector(row,'hand added row',-2,2,2)
        if v[0] not in [-2,-1,1,2] or v[1] not in [0,1]:raise ValueError('hand added row x/y outside finite domain')
        rows.append([int(v[0]),int(v[1])])
    q['hand_added_batch']=rows
    return q
