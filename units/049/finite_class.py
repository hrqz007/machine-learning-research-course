"""Explicit fixed-hypothesis forward/loss ledgers and exact multinomial risks."""
import math
from fractions import Fraction
import numpy as np
from numeric import scalar
X=(-2,-1,1,2)

def validate_population(pop,hyp):
    if not isinstance(pop,list) or len(pop)!=8 or not isinstance(hyp,list) or not 1<=len(hyp)<=16:raise ValueError('eight outcomes and1..16 hypotheses required')
    numbers=[];den=None
    for i,row in enumerate(pop):
        if not isinstance(row,dict) or set(row)!={'id','x','y','mass_numerator','mass_denominator'}:raise ValueError('population row schema')
        xx=scalar(row['x'],'population x',-2,2,True);yy=scalar(row['y'],'population y',0,1,True)
        if xx!=X[i//2] or yy!=i%2:raise ValueError('population outcomes/order mismatch')
        num=scalar(row['mass_numerator'],'mass numerator',0,10000,True);dd=scalar(row['mass_denominator'],'mass denominator',1,10000,True)
        if den is None:den=dd
        elif den!=dd:raise ValueError('common mass denominator required')
        if not isinstance(row['id'],str) or not row['id']:raise ValueError('population ID')
        numbers.append(num)
    if sum(numbers)!=den or len({v['id'] for v in pop})!=8:raise ValueError('population mass or IDs')
    seen=set();ids=set();matrix=[]
    for row in hyp:
        if not isinstance(row,dict) or set(row)!={'id','predictions'}:raise ValueError('hypothesis schema')
        if not isinstance(row['id'],str) or not row['id'] or row['id'] in ids:raise ValueError('hypothesis ID')
        if not isinstance(row['predictions'],list) or len(row['predictions'])!=4:raise ValueError('hypothesis forward vector length')
        v=[scalar(z,'hypothesis prediction',0,1,True) for z in row['predictions']]
        if tuple(v) in seen:raise ValueError('duplicate hypothesis')
        seen.add(tuple(v));ids.add(row['id']);matrix.append(v)
    H=np.array(matrix,dtype=np.int64);Y=np.array([v['y'] for v in pop]);P=H[:,np.arange(8)//2].T;loss=(P!=Y[:,None]).astype(np.int64);nums=np.array(numbers,dtype=np.int64);risk_numerator=nums@loss
    return {'hypothesis_ids':[v['id'] for v in hyp],'predictions':P,'loss':loss,'mass_numerators':nums,'denominator':den,'risk_numerators':risk_numerator,'risk':risk_numerator/den,'H':H}

def validate_sample(sample):
    if not isinstance(sample,list) or not 1<=len(sample)<=1000:raise ValueError('sample rows length')
    out=[];seen=set()
    for row in sample:
        if not isinstance(row,dict) or set(row)!={'id','x','y'}:raise ValueError('sample row schema')
        if not isinstance(row['id'],str) or not row['id'] or row['id'] in seen:raise ValueError('sample ID')
        x=scalar(row['x'],'sample x',-2,2,True);y=scalar(row['y'],'sample y',0,1,True)
        if x not in X:raise ValueError('sample x outside four-point domain')
        seen.add(row['id']);out.append({'id':row['id'],'x':x,'y':y})
    return out

def ledger(sample,pop,hyp):
    obj=validate_population(pop,hyp);sample=validate_sample(sample);rows=[];total=np.zeros(len(hyp),dtype=np.int64)
    for row in sample:
        pred=obj['H'][:,X.index(row['x'])];loss=(pred!=row['y']).astype(np.int64);total+=loss
        rows.append({**row,'predictions':pred.tolist(),'losses':loss.tolist(),'mean_loss_contributions':(loss/len(sample)).tolist()})
    chosen=int(np.argmin(total));ties=np.flatnonzero(total==total.min()).tolist()
    return {'rows':rows,'sample_size':len(sample),'loss_counts':total.tolist(),'empirical_risks':(total/len(sample)).tolist(),'population_risks':obj['risk'].tolist(),'population_risk_fractions':[str(Fraction(int(v),obj['denominator'])) for v in obj['risk_numerators']],'tie_indices':ties,'selected_index':chosen,'selected_id':hyp[chosen]['id'],'selected_population_risk':float(obj['risk'][chosen]),'selected_prediction_vector':hyp[chosen]['predictions'],'optimization':'finite exact search after all losses are summed; no gradient or parameter-gradient update applies to this discrete hypothesis list'}

def compositions(n,k):
    if k==1:yield (n,);return
    for first in range(n+1):
        for rest in compositions(n-first,k-1):yield (first,)+rest

def exact_enumeration(pop,hyp,n,epsilon_numerators,epsilon_denominator):
    o=validate_population(pop,hyp);n=scalar(n,'exact sample size',1,8,True);ed=scalar(epsilon_denominator,'epsilon denominator',1,100,True)
    if not isinstance(epsilon_numerators,(list,tuple)) or not 1<=len(epsilon_numerators)<=12:raise ValueError('epsilon numerators require1..12 entries')
    en=[scalar(v,'epsilon numerator',1,ed,True) for v in epsilon_numerators]
    if en!=sorted(set(en)):raise ValueError('epsilon numerators must be strictly increasing')
    D=o['denominator'];nums=[int(v) for v in o['mass_numerators']];rn=[int(v) for v in o['risk_numerators']];hstar=int(np.argmin(rn));den=D**n;mass=0;states=0;weighted_train=weighted_population=weighted_sup=0;hist=[0]*len(hyp);tails=[{'epsilon':str(Fraction(v,ed)),'fixed_weight':0,'uniform_weight':0} for v in en]
    fac=[math.factorial(i) for i in range(n+1)];powers=[[v**j for j in range(n+1)] for v in nums]
    for counts in compositions(n,8):
        states+=1;w=fac[n]//math.prod(fac[j] for j in counts)*math.prod(powers[k][v] for k,v in enumerate(counts));mass+=w
        errors=np.array(counts,dtype=np.int64)@o['loss'];chosen=int(np.argmin(errors));hist[chosen]+=w;weighted_train+=w*int(errors[chosen]);weighted_population+=w*rn[chosen]
        gaps=[abs(int(errors[j])*D-rn[j]*n) for j in range(len(hyp))];maximum=max(gaps);weighted_sup+=w*maximum
        for eps,v in zip(en,tails):
            if gaps[hstar]*ed>eps*n*D:v['fixed_weight']+=w
            if maximum*ed>eps*n*D:v['uniform_weight']+=w
    if mass!=den:raise RuntimeError('exact enumerated probabilities do not sum to one')
    for v in tails:
        for key in ['fixed','uniform']:
            f=Fraction(v.pop(key+'_weight'),den);v[key+'_tail_fraction']=str(f);v[key+'_tail_probability']=float(f)
    er=Fraction(weighted_population,den*D);tr=Fraction(weighted_train,den*n);sup=Fraction(weighted_sup,den*n*D)
    return {'n':n,'composition_states':states,'ordered_sequence_count':8**n,'total_probability_fraction':'1','expected_selected_risk_fraction':str(er),'expected_training_risk_fraction':str(tr),'expected_optimism_fraction':str(er-tr),'expected_excess_risk_fraction':str(er-Fraction(min(rn),D)),'expected_uniform_gap_fraction':str(sup),'expected_selected_risk':float(er),'expected_training_risk':float(tr),'expected_uniform_gap':float(sup),'selection_probability_fractions':[str(Fraction(v,den)) for v in hist],'tails':tails}
