"""Independent exact algebra, manual preprocessing/cross-fitting, and SciPy QR."""
from pathlib import Path
from fractions import Fraction as F
import argparse,json,math,tempfile
import numpy as np
from scipy.linalg import lstsq
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.exceptions import NotFittedError
import experiment as e
ROOT=Path(__file__).resolve().parent

def need(ok,msg):
    if not ok:raise RuntimeError(msg)

def quantile(values,q):
    a=sorted(map(float,values));at=(len(a)-1)*q;i=int(math.floor(at));j=int(math.ceil(at));return a[i]+(at-i)*(a[j]-a[i])

def design_reference(train,test,lo=.05,hi=.95):
    tr=np.asarray(train[:,:2],float);te=np.asarray(test[:,:2],float);med=[]
    for j in range(2):
        vals=tr[:,j][~np.isnan(tr[:,j])];med.append(quantile(vals,.5) if len(vals) else 0.)
    a=np.where(np.isnan(tr),med,tr);b=np.where(np.isnan(te),med,te);lower=np.array([quantile(a[:,j],lo) for j in range(2)]);upper=np.array([quantile(a[:,j],hi) for j in range(2)]);a=np.clip(a,lower,upper);b=np.clip(b,lower,upper)
    mean=np.array([sum(a[:,j])/len(a) for j in range(2)]);var=np.array([sum((a[:,j]-mean[j])**2)/len(a) for j in range(2)]);scale=np.where(var==0,1,np.sqrt(var));cats=sorted(set(train[:,2]));hot=lambda x:np.array([[float(v==c) for c in cats] for v in x[:,2]])
    return np.column_stack([(a-mean)/scale,np.isnan(tr).astype(float),hot(train)]),np.column_stack([(b-mean)/scale,np.isnan(te).astype(float),hot(test)]),{'median':med,'lower':lower,'upper':upper,'mean':mean,'var':var}

def ridge_qr(A,y,B,alpha):
    A=np.column_stack([np.ones(len(A)),A]);B=np.column_stack([np.ones(len(B)),B]);P=np.eye(A.shape[1]);P[0,0]=0;beta=lstsq(np.vstack([A,np.sqrt(alpha)*P]),np.r_[y,np.zeros(len(P))],lapack_driver='gelsy')[0];return B@beta

def manual_folds(n,k,seed=None):
    order=np.arange(n)
    if seed is not None:np.random.RandomState(seed).shuffle(order)
    sizes=np.full(k,n//k,dtype=int);sizes[:n%k]+=1;start=0;allidx=np.arange(n);out=[]
    for size in sizes:
        va=np.sort(order[start:start+size]);tr=allidx[~np.isin(allidx,va)];out.append((tr,va));start+=size
    return out

def manual_encode(train_cat,y,test_cat,smooth):
    mean=float(sum(y)/len(y));stats={}
    for cat,value in zip(train_cat,y):
        n,sy=stats.get(cat,(0,0.));stats[cat]=(n+1,sy+float(value))
    out=[]
    for cat in test_cat:
        if cat in stats:n,sy=stats[cat];out.append((sy+smooth*mean)/(n+smooth))
        else:out.append(mean)
    return np.array(out)

def manual_crossfit(X,y,k,smooth,seed):
    out=np.empty(len(y))
    for tr,va in manual_folds(len(y),k,seed):out[va]=manual_encode(X[tr],y[tr],X[va],smooth)
    return out

def exact_hand():
    q=list(map(F,[-1,0,0,1]));y=list(map(F,[1,2,2,5]));A=[list(map(F,r)) for r in [[1,1,0,0],[1,1,0,1],[1,0,1,0],[1,0,1,0]]]
    grad=[-sum(A[i][j]*y[i] for i in range(4))/4 for j in range(4)];gw=-sum(q[i]*y[i] for i in range(4))/4;theta=[-F(1,5)*v for v in grad];wcoef=-F(1,5)*gw
    pred=[sum(a*b for a,b in zip(row,theta))+2*wcoef*q[i] for i,row in enumerate(A)];loss=sum((pred[i]-y[i])**2 for i in range(4))/8
    return {'gradient_non_scaled':[str(v) for v in grad],'gradient_w_div_sqrt2':str(gw),'theta_non_scaled':[str(v) for v in theta],'theta_w_div_sqrt2':str(wcoef),'next_predictions':[str(v) for v in pred],'next_mean_half_loss':str(loss),'validation_next':['9/10','19/20']}

def run(result):
    c,d,s=e.load_data();max_group=max_selection=max_encoding=max_enc_pred=0.;fit_ids_checked=0
    for rep in result['replications']:
        rid=rep['rep'];X=e.group_X(d,rid);y=d['group_y'][rid]
        for fold,b in enumerate(rep['group']):
            tr=s[f'group_{fold}_train'];va=s[f'group_{fold}_valid'];A,B,st=design_reference(X[tr],X[va],*c['group_experiment']['clip_quantiles']);p=ridge_qr(A,y[tr],B,c['group_experiment']['ridge_alpha']);max_group=max(max_group,float(np.max(np.abs(p-b['correct_prediction']))))
            need(set(b['preprocessing_state']['fit_row_ids'])==set(tr),'preprocessor fit IDs changed');need(not set(tr)&set(va),'outer overlap');need(not set(d['group'][tr])&set(d['group'][va]),'group overlap');fit_ids_checked+=len(tr)
            for key,ref in [('medians',st['median']),('clip_lower',st['lower']),('clip_upper',st['upper']),('scale_mean',st['mean']),('scale_variance_ddof0',st['var'])]:need(np.allclose(b['preprocessing_state'][key],ref,rtol=1e-12,atol=1e-12),'state mismatch '+key)
        SX=d['selection_X'][rid];SY=d['selection_y'][rid]
        for fold,b in enumerate(rep['selection']):
            tr=s[f'selection_{fold}_train'];va=s[f'selection_{fold}_valid'];A=SX[tr];Y=SY[tr];C=A-A.mean(0);yc=Y-Y.mean();corr=(C.T@yc)/np.sqrt(np.sum(C*C,axis=0)*sum(yc*yc));top=np.sort(np.argsort(abs(corr))[-c['selection_experiment']['selected_features']:]);need(top.tolist()==b['correct_features'],'manual Pearson top-k mismatch');pr=ridge_qr(A[:,top],Y,SX[va][:,top],1.);max_selection=max(max_selection,abs(e.mse(SY[va],pr)-b['correct_mse']))
        CX=d['category'][rid];CY=d['category_y'][rid];cfg=c['encoding_experiment']
        for fold,b in enumerate(rep['encoding']):
            tr=s[f'encoding_{fold}_train'];va=s[f'encoding_{fold}_valid'];oof=manual_crossfit(CX[tr],CY[tr],cfg['inner_folds'],cfg['smooth'],c['encoder_seed']);valid=manual_encode(CX[tr],CY[tr],CX[va],cfg['smooth']);full=manual_encode(CX[tr],CY[tr],CX[tr],cfg['smooth']);max_encoding=max(max_encoding,float(np.max(abs(oof-b['crossfit_train_encoding']))),float(np.max(abs(valid-b['valid_encoding_from_full_train']))),float(np.max(abs(full-b['full_train_encoding']))))
            pp=ridge_qr(oof[:,None],CY[tr],valid[:,None],cfg['ridge_alpha']);max_enc_pred=max(max_enc_pred,float(np.max(abs(pp-b['valid_predictions']))));seen=[]
            for prov in b['inner_provenance']:
                need(not set(prov['fit_row_ids'])&set(prov['encoded_row_ids']),'inner label contamination');need(set(prov['fit_row_ids'])|set(prov['encoded_row_ids'])==set(tr),'inner union incorrect');need(not set(va)&set(prov['fit_row_ids']),'outer label in inner fit');seen+=prov['encoded_row_ids']
            need(sorted(seen)==tr.tolist(),'each training row must be encoded once')
    need(max_group<2e-10 and max_selection<2e-10 and max_encoding<2e-12 and max_enc_pred<2e-10,'independent reference mismatch')
    ex=exact_hand();h=result['hand'];need(np.allclose([float(F(v)) for v in ex['next_predictions']],[v['next_prediction'] for v in h['rows']],atol=1e-14),'exact hand mismatch');need(abs(float(F(ex['next_mean_half_loss']))-h['next_half_loss'])<1e-14,'exact loss mismatch')
    A=np.array([v['design'] for v in h['rows']]);yy=np.array([1,2,2,5.]);th=np.array(h['theta_next']);gd=A.T@(A@th-yy)/4;delta=1e-6;J=lambda t:sum((A@t-yy)**2)/8;fd=np.array([(J(th+np.eye(5)[j]*delta)-J(th-np.eye(5)[j]*delta))/(2*delta) for j in range(5)]);need(max(abs(fd-gd))<1e-8,'finite difference mismatch')
    eh=result['encoder_hand'];need(np.allclose(manual_crossfit(np.array(eh['categories']),np.array(eh['y']),2,2,None),eh['crossfit_training']),'encoder small example mismatch')
    tr=s['encoding_0_train'];va=s['encoding_0_valid'];X=d['category'][0,:,None];y=d['category_y'][0];p=Pipeline([('encode',e.make_encoder(c)),('model',Ridge(alpha=1,solver='svd'))]).fit(X[tr],y[tr]);need(np.allclose(p.predict(X[va]),result['replications'][0]['encoding'][0]['valid_predictions']),'actual TargetEncoder pipeline mismatch')
    y2=y.copy();y2[va]+=50;p2=Pipeline([('encode',e.make_encoder(c)),('model',Ridge(alpha=1,solver='svd'))]).fit(X[tr],y2[tr]);need(np.array_equal(p.predict(X[va]),p2.predict(X[va])),'heldout labels affected fit');bad=e.make_encoder(c).fit(X,y).transform(X[va]);bad2=e.make_encoder(c).fit(X,y2).transform(X[va]);bd=float(np.max(abs(bad-bad2)));need(bd>1,'negative control failed')
    y3=y[tr].copy();y3[0]+=10;enc1=e.make_encoder(c).fit_transform(X[tr],y[tr]);enc2=e.make_encoder(c).fit_transform(X[tr],y3);need(enc1[0,0]==enc2[0,0],'own label entered crossfit feature')
    gx=e.group_X(d,0);tg=s['group_0_train'];vg=s['group_0_valid'];pp=e.make_pipeline().fit(gx[tg],d['group_y'][0,tg]);before=e.state(pp['pre'],tg);gx2=gx.copy();gx2[vg,:2]=np.asarray(gx2[vg,:2],float)+100;pp2=e.make_pipeline().fit(gx2[tg],d['group_y'][0,tg]);need(before==e.state(pp2['pre'],tg),'heldout X affected fit')
    failures=[]
    def rejects(name,call):
        try:call()
        except (ValueError,TypeError,NotFittedError,OSError) as err:failures.append({'case':name,'error_type':type(err).__name__,'message':str(err).splitlines()[0]})
        else:raise RuntimeError('expected rejection: '+name)
    rejects('transform before fit',lambda:e.QuantileClipper().transform([[1,2]]));rejects('invalid quantile order',lambda:e.QuantileClipper(.8,.2).fit([[1,2]]));rejects('nonfinite numeric',lambda:e.QuantileClipper().fit([[1,np.inf]]));rejects('feature count changed',lambda:e.QuantileClipper().fit([[1,2]]).transform([[1]]));rejects('empty fit',lambda:e.QuantileClipper().fit(np.empty((0,2))));rejects('nonpositive ridge alpha',lambda:e.make_pipeline(0));rejects('source overwrite',lambda:e.atomic_json(ROOT/'lecture.md',{}))
    am=np.array([[np.nan,0,'A'],[np.nan,0,'A'],[np.nan,0,'B']],object);f=e.make_pipeline().fit(am,[1,2,3]);Z=f['pre'].transform(np.array([[7,0,'Z'],[np.nan,0,'B']],object));need(Z.shape==(2,6) and np.isfinite(Z).all(),'all-missing shape failed');need(np.all(Z[:,:2]==0),'constant clip behavior');need(np.all(Z[0,4:]==0),'unknown category fallback');one=e.make_pipeline().fit(np.array([[1.,2.,'A']],object),[3.]);need(np.isfinite(one.predict(np.array([[2.,3.,'B']],object))).all(),'single row failed')
    with tempfile.TemporaryDirectory(prefix='ml052-audit-') as td:
        td=Path(td);out=td/'keep.json';out.write_text('KEEP\n');rejects('nonfinite JSON preserves output',lambda:e.atomic_json(out,{'x':float('nan')}));need(out.read_text()=='KEEP\n','serialization damaged output');link=td/'alias';link.symlink_to(out);rejects('symlink output',lambda:e.atomic_json(link,{}));hard=td/'hard';hard.hardlink_to(out);rejects('hardlink output',lambda:e.atomic_json(out,{}));need(out.read_text()=='KEEP\n','path rejection damaged output')
    need(result['affine_control']['max_prediction_difference']<1e-12,'OLS affine control changed predictions')
    return {'status':'passed','independent_reference_scope':'all 12 datasets and 4 folds in each of 3 mechanisms','group_max_prediction_error_vs_manual_preprocessing_and_SciPy_QR':max_group,'selection_max_MSE_error_vs_manual_Pearson_and_QR':max_selection,'TargetEncoder_max_encoding_error_vs_count_sum_manual_KFold':max_encoding,'TargetEncoder_max_prediction_error_vs_manual_encoding_and_QR':max_enc_pred,'fit_row_memberships_checked':fit_ids_checked,'exact_hand':ex,'hand_gradient_difference_max':float(max(abs(fd-gd))),'actual_TargetEncoder_Pipeline':'passed','heldout_label_mutation':'correct unchanged; forbidden global encoding changes','global_encoding_mutation_max_change':bd,'own_label_crossfit_invariance':'passed','heldout_X_train_statistics_invariance':'passed','ordinary_support':{'one_row':'passed','constant_column':'passed','all_missing_training_column':'retained and imputed zero; fitted clip range collapses to zero','unknown_category':'all-zero one-hot; no new column'},'failures_preserved':failures,'scope_limit':'author self-check; independent course review and real-world validation are distinct'}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',default=str(ROOT/'experiment-result.json'));p.add_argument('--out',default=str(ROOT/'outputs/audit.json'));a=p.parse_args();e.safe_output(a.out);r=run(json.loads(Path(a.report).read_text()));e.atomic_json(a.out,r);print(json.dumps(r,ensure_ascii=False,indent=2))
