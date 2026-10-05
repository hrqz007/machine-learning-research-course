"""Fixed teaching protocol and complete validation before data generation or fitting."""
from numeric import scalar,vector,labels
FIELDS={'kind','hand_base_logit_magnitude','hand_labels','hand_signs','hand_initial_inverse_temperature','hand_learning_rate','temperature_inverse_bounds','temperature_tolerance','classification_seed','classification_calibration_count','classification_evaluation_count','classification_raw_logit_factor','classification_true_slope','classification_true_intercept','reliability_bins','regression_seed','regression_train_count','regression_calibration_count','regression_evaluation_count','regression_repetitions','regression_true_intercept','regression_true_slope','regression_noise_intercept','regression_noise_abs_slope','regression_noise_shift_factor','regression_feature_low','regression_feature_high','regression_group_abs_boundary','miscoverage','miscoverage_boundary_examples','interval_display_count'}

def validate(c):
    if not isinstance(c,dict) or set(c)!=FIELDS or c['kind']!='calibration_and_split_conformal_v1':raise ValueError('protocol fields/kind mismatch')
    q=dict(c)
    if c['hand_base_logit_magnitude']!='2*log(3)':raise ValueError('hand magnitude is the declared expression 2*log(3), never eval arbitrary strings')
    q['hand_labels']=labels(c['hand_labels'],4).tolist();s=vector(c['hand_signs'],'hand signs',-1,1,4)
    if any(v not in [-1,1] for v in s):raise ValueError('hand signs must be -1 or 1')
    q['hand_signs']=s.tolist()
    bounds=vector(c['temperature_inverse_bounds'],'temperature bounds',0,4,2)
    if bounds[0]>=bounds[1]:raise ValueError('temperature bounds must increase')
    q['temperature_inverse_bounds']=bounds.tolist()
    ranges={'hand_initial_inverse_temperature':(0,4),'hand_learning_rate':(1e-6,1),'temperature_tolerance':(1e-13,1e-4),'classification_raw_logit_factor':(.1,5),'classification_true_slope':(-3,3),'classification_true_intercept':(-3,3),'regression_true_intercept':(-10,10),'regression_true_slope':(-10,10),'regression_noise_intercept':(.05,2),'regression_noise_abs_slope':(0,2),'regression_noise_shift_factor':(.1,5),'regression_feature_low':(-5,5),'regression_feature_high':(-5,5),'regression_group_abs_boundary':(.1,5),'miscoverage':(1e-6,1-1e-6)}
    for k,(lo,hi) in ranges.items():q[k]=scalar(c[k],k,lo,hi)
    integers={'classification_seed':(0,2**32-2),'classification_calibration_count':(20,5000),'classification_evaluation_count':(20,10000),'reliability_bins':(2,20),'regression_seed':(0,2**32-2),'regression_train_count':(3,1000),'regression_calibration_count':(2,1000),'regression_evaluation_count':(20,2000),'regression_repetitions':(2,500),'interval_display_count':(1,40)}
    for k,(lo,hi) in integers.items():q[k]=scalar(c[k],k,lo,hi,True)
    q['miscoverage_boundary_examples']=vector(c['miscoverage_boundary_examples'],'miscoverage examples',1e-6,1-1e-6).tolist()
    if len(q['miscoverage_boundary_examples'])>12:raise ValueError('too many boundary examples')
    if q['regression_feature_high']-q['regression_feature_low']<.5:raise ValueError('feature range too narrow or reversed')
    if q['interval_display_count']>q['regression_evaluation_count']:raise ValueError('display count exceeds test rows')
    return q
