"""Four training rows, inner two-fold search and explicit gradient steps."""
import math
from numeric import scalar,vector

def slope_fit(x,y,penalty):
    x=vector(x,'hand x',-5,5);y=vector(y,'hand y',-20,20,len(x));lam=scalar(penalty,'hand penalty',0,10);xy=math.fsum(a*b for a,b in zip(x,y));xx=math.fsum(a*a for a in x);den=xx+len(x)*lam
    if den<=0:raise ValueError('unidentifiable slope')
    return {'sum_xy':xy,'sum_x_squared':xx,'n':len(x),'penalty':lam,'denominator':den,'weight':xy/den}

def state(x,y,w,penalty):
    x=vector(x,'hand x',-5,5);y=vector(y,'hand y',-20,20,len(x));w=scalar(w,'trace weight',-1e6,1e6);lam=scalar(penalty,'hand penalty',0,10);rows=[]
    for i,(xx,yy) in enumerate(zip(x,y)):
        prediction=w*xx;residual=prediction-yy;rows.append({'row':i,'x':float(xx),'y':float(yy),'prediction':float(prediction),'residual':float(residual),'half_squared_loss':float(residual**2/2),'mean_data_loss_contribution':float(residual**2/(2*len(x))),'local_dloss_dprediction':float(residual),'local_dprediction_dw':float(xx),'mean_gradient_contribution':float(residual*xx/len(x))})
    data_loss=math.fsum(v['mean_data_loss_contribution'] for v in rows);g=math.fsum(v['mean_gradient_contribution'] for v in rows);return {'weight':w,'rows':rows,'data_half_MSE':data_loss,'penalty_value':lam*w*w/2,'total_objective':data_loss+lam*w*w/2,'data_gradient':g,'penalty_gradient':lam*w,'total_gradient':g+lam*w,'Hessian':math.fsum(v*v for v in x)/len(x)+lam}

def report(c):
    x=c['hand_training_x'];y=c['hand_training_y'];candidates=[]
    for lam in c['hand_candidate_penalties']:
        splits=[];total=0.
        for k,va in enumerate(c['hand_inner_validation_indices']):
            tr=[i for i in range(4) if i not in va];fit=slope_fit([x[i] for i in tr],[y[i] for i in tr],lam);rows=[]
            for i in va:
                pred=fit['weight']*x[i];loss=(pred-y[i])**2;total+=loss;rows.append({'index':i,'x':x[i],'y':y[i],'prediction':pred,'squared_loss':loss})
            splits.append({'fold':k,'training_indices':tr,'validation_indices':va,'fit':fit,'validation_rows':rows,'validation_MSE':math.fsum(v['squared_loss'] for v in rows)/len(rows)})
        candidates.append({'penalty':lam,'folds':splits,'pooled_CV_MSE':total/4})
    selected=min(range(len(candidates)),key=lambda j:candidates[j]['pooled_CV_MSE']);chosen=candidates[selected]['penalty'];refit=slope_fit(x,y,chosen);test=[]
    for xx,yy in zip(c['hand_test_x'],c['hand_test_y']):test.append({'x':xx,'y':yy,'prediction':refit['weight']*xx,'squared_loss':(refit['weight']*xx-yy)**2})
    trace=[];w=c['hand_initial_weight']
    for stage in range(3):
        v=state(x,y,w,c['hand_trace_penalty']);v['stage']=stage;trace.append(v)
        if stage<2:w=w-c['hand_learning_rate']*v['total_gradient']
    return {'candidates':candidates,'selected_index':selected,'selected_penalty':chosen,'outer_training_refit':refit,'outer_test_rows':test,'outer_test_MSE':math.fsum(v['squared_loss'] for v in test)/len(test),'gradient_trace':trace,'trace_closed_form_optimum':slope_fit(x,y,c['hand_trace_penalty']),'scope':'slope-only hand example, separate from Legendre experiment; inner selection never uses two outer-test labels; trace penalty is fixed in advance and is not a new tuned candidate'}
