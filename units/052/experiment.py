"""ML052: complete fit provenance and three separately interpretable mechanisms."""
from pathlib import Path
import argparse,hashlib,json,os,tempfile
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer,MissingIndicator
from sklearn.preprocessing import StandardScaler,OneHotEncoder,TargetEncoder
from sklearn.linear_model import Ridge,LinearRegression
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_array,check_is_fitted
ROOT=Path(__file__).resolve().parent

def require(ok,message):
    if not ok:raise ValueError(message)

def safe_output(path):
    p=Path(os.path.abspath(path))
    for q in (p,*p.parents):require(not q.is_symlink(),'symlink output is not supported')
    if p.resolve().is_relative_to(ROOT):require(p.resolve().is_relative_to(ROOT/'outputs'),'output must not overwrite course files')
    if p.exists():require(p.is_file() and p.stat().st_nlink==1,'output must be a single regular file')
    return p

def atomic_json(path,value):
    p=safe_output(path);payload=json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n';p.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=p.parent,delete=False,prefix='.ml052-') as f:f.write(payload);temp=f.name
    try:os.replace(temp,p)
    finally:
        if os.path.exists(temp):os.unlink(temp)

class QuantileClipper(TransformerMixin,BaseEstimator):
    """Fit linear quantiles of already imputed training rows; never delete rows."""
    def __init__(self,lower=.05,upper=.95):self.lower=lower;self.upper=upper
    def fit(self,X,y=None):
        require(not isinstance(self.lower,bool) and not isinstance(self.upper,bool),'quantiles must be real scalars')
        require(np.isscalar(self.lower) and np.isscalar(self.upper) and np.isfinite(self.lower) and np.isfinite(self.upper) and 0<=self.lower<self.upper<=1,'require 0 <= lower < upper <= 1')
        a=check_array(X,dtype=float);self.n_features_in_=a.shape[1];self.n_samples_seen_=len(a);self.lower_=np.quantile(a,self.lower,axis=0,method='linear');self.upper_=np.quantile(a,self.upper,axis=0,method='linear');return self
    def transform(self,X):
        check_is_fitted(self,['lower_','upper_']);a=check_array(X,dtype=float);require(a.shape[1]==self.n_features_in_,'feature count changed');return np.clip(a,self.lower_,self.upper_)
    def get_feature_names_out(self,input_features=None):
        check_is_fitted(self,'lower_');return np.asarray(input_features if input_features is not None else ['x'+str(i) for i in range(self.n_features_in_)],dtype=object)

def load_data():
    for rel,h in json.loads((ROOT/'data_integrity.json').read_text()).items():require(hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h,'data hash mismatch: '+rel)
    c=json.loads((ROOT/'data/protocol.json').read_text())
    with np.load(ROOT/'data/draws.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
    with np.load(ROOT/'data/splits.npz',allow_pickle=False) as z:s={k:z[k] for k in z.files}
    return c,d,s

def group_X(d,rep):
    x=np.empty((len(d['group']),3),dtype=object);x[:,:2]=d['numeric'][rep];x[:,2]=['S'+str(v) for v in d['site'][rep]];return x

def make_preprocessor(lo=.05,hi=.95,scale=True):
    num=[('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('clip',QuantileClipper(lo,hi))]
    if scale:num.append(('scale',StandardScaler()))
    return ColumnTransformer([('num',Pipeline(num),[0,1]),('missing',MissingIndicator(features='all'),[0,1]),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),[2])],remainder='drop',sparse_threshold=0)

def make_pipeline(alpha=15.,lo=.05,hi=.95):
    require(np.isscalar(alpha) and not isinstance(alpha,bool) and np.isfinite(alpha) and alpha>0,'positive finite alpha required')
    return Pipeline([('pre',make_preprocessor(lo,hi)),('model',Ridge(alpha=alpha,solver='svd'))])

def state(pre,fit_ids):
    num=pre.named_transformers_['num'];o={'fit_row_ids':list(map(int,fit_ids)),'medians':num['impute'].statistics_.tolist(),'clip_lower':num['clip'].lower_.tolist(),'clip_upper':num['clip'].upper_.tolist(),'clip_n_fit':int(num['clip'].n_samples_seen_),'category_vocabulary':pre.named_transformers_['cat'].categories_[0].tolist(),'missing_columns':[0,1]}
    if 'scale' in num.named_steps:o.update({'scale_mean':num['scale'].mean_.tolist(),'scale_variance_ddof0':num['scale'].var_.tolist(),'scale_scale':num['scale'].scale_.tolist(),'scale_n_fit':int(num['scale'].n_samples_seen_)})
    return o

def mse(y,p):return float(np.mean((np.asarray(y)-np.asarray(p))**2))

def group_run(c,d,s,rep):
    X=group_X(d,rep);y=d['group_y'][rep];gc=c['group_experiment'];rows=[]
    for fold in range(c['outer_folds']):
        tr=s[f'group_{fold}_train'];va=s[f'group_{fold}_valid'];require(not np.intersect1d(d['group'][tr],d['group'][va]).size,'outer group overlap')
        model=make_pipeline(gc['ridge_alpha'],*gc['clip_quantiles']).fit(X[tr],y[tr]);pred=model.predict(X[va]);pre=model['pre']
        before=make_preprocessor(*gc['clip_quantiles'],scale=False).fit(X[tr],y[tr]);all_processed=before.transform(X).astype(float)
        # Only the StandardScaler fitted scope changes; all preceding steps stay train-only.
        forbidden=StandardScaler().fit(all_processed[:,:2]);badX=all_processed.copy();badX[:,:2]=forbidden.transform(all_processed[:,:2]);bad=Ridge(alpha=gc['ridge_alpha'],solver='svd').fit(badX[tr],y[tr]);bp=bad.predict(badX[va])
        rows.append({'fold':fold,'n_train':len(tr),'n_valid':len(va),'train_groups':np.unique(d['group'][tr]).tolist(),'valid_groups':np.unique(d['group'][va]).tolist(),'valid_row_ids':va.tolist(),'correct_mse':mse(y[va],pred),'global_scale_mse':mse(y[va],bp),'correct_prediction':pred.tolist(),'global_scale_prediction':bp.tolist(),'preprocessing_state':state(pre,tr),'global_scale_state':{'fit_row_ids':list(range(len(X))),'mean':forbidden.mean_.tolist(),'variance_ddof0':forbidden.var_.tolist()},'coefficients':model['model'].coef_.tolist(),'intercept':float(model['model'].intercept_)})
    return rows

def selection_run(c,d,s,rep):
    X=d['selection_X'][rep];y=d['selection_y'][rep];q=c['selection_experiment'];glob=SelectKBest(f_regression,k=q['selected_features']).fit(X,y);badX=glob.transform(X);rows=[]
    for fold in range(c['outer_folds']):
        tr=s[f'selection_{fold}_train'];va=s[f'selection_{fold}_valid'];p=Pipeline([('select',SelectKBest(f_regression,k=q['selected_features'])),('model',Ridge(alpha=q['ridge_alpha'],solver='svd'))]).fit(X[tr],y[tr]);bad=Ridge(alpha=q['ridge_alpha'],solver='svd').fit(badX[tr],y[tr])
        rows.append({'fold':fold,'correct_mse':mse(y[va],p.predict(X[va])),'global_selection_mse':mse(y[va],bad.predict(badX[va])),'correct_features':p['select'].get_support(indices=True).tolist(),'global_features':glob.get_support(indices=True).tolist(),'train_mean_baseline_mse':mse(y[va],np.full(len(va),y[tr].mean()))})
    return rows

def make_encoder(c):
    q=c['encoding_experiment'];return TargetEncoder(target_type='continuous',smooth=q['smooth'],cv=q['inner_folds'],shuffle=True,random_state=c['encoder_seed'])

def encoding_run(c,d,s,rep):
    X=d['category'][rep,:,None];y=d['category_y'][rep];q=c['encoding_experiment'];be=make_encoder(c).fit(X,y);badX=be.transform(X);rows=[]
    for fold in range(c['outer_folds']):
        tr=s[f'encoding_{fold}_train'];va=s[f'encoding_{fold}_valid'];enc=make_encoder(c);zt=enc.fit_transform(X[tr],y[tr]);zv=enc.transform(X[va]);model=Ridge(alpha=q['ridge_alpha'],solver='svd').fit(zt,y[tr]);p=model.predict(zv)
        se=make_encoder(c).fit(X[tr],y[tr]);selftr=se.transform(X[tr]);selfva=se.transform(X[va]);sm=Ridge(alpha=q['ridge_alpha'],solver='svd').fit(selftr,y[tr]);selfp=sm.predict(selfva);bm=Ridge(alpha=q['ridge_alpha'],solver='svd').fit(badX[tr],y[tr]);badp=bm.predict(badX[va])
        inner=[{'fit_row_ids':tr[it].tolist(),'encoded_row_ids':tr[iv].tolist()} for it,iv in KFold(q['inner_folds'],shuffle=True,random_state=c['encoder_seed']).split(tr)]
        rows.append({'fold':fold,'correct_mse':mse(y[va],p),'self_encoded_mse':mse(y[va],selfp),'global_target_mse':mse(y[va],badp),'train_mean_baseline_mse':mse(y[va],np.full(len(va),y[tr].mean())),'correct_training_mse':mse(y[tr],model.predict(zt)),'self_encoded_training_mse':mse(y[tr],sm.predict(selftr)),'train_row_ids':tr.tolist(),'valid_row_ids':va.tolist(),'inner_provenance':inner,'crossfit_train_encoding':zt.ravel().tolist(),'full_train_encoding':selftr.ravel().tolist(),'valid_encoding_from_full_train':zv.ravel().tolist(),'valid_predictions':p.tolist(),'full_train_target_mean':float(enc.target_mean_),'coefficient':float(model.coef_[0]),'intercept':float(model.intercept_)})
    return rows

def hand():
    X=np.array([[1,'A'],[np.nan,'A'],[3,'B'],[5,'B']],dtype=object);x=X[:,0].astype(float);im=np.where(np.isnan(x),np.nanmedian(x),x);lo,hi=np.quantile(im,[0,1]);cl=np.clip(im,lo,hi);mu=cl.mean();var=cl.var();z=(cl-mu)/np.sqrt(var);A=np.column_stack([np.ones(4),z,X[:,1]=='A',X[:,1]=='B',np.isnan(x)]).astype(float);y=np.array([1,2,2,5.]);pred=A@np.zeros(5);r=pred-y;loc=r[:,None]*A;grad=loc.mean(0);new=-.2*grad;nxt=A@new
    rows=[{'id':'H'+str(i+1),'raw_x':None if np.isnan(x[i]) else float(x[i]),'category':str(X[i,1]),'imputed':float(im[i]),'clipped':float(cl[i]),'scaled':float(z[i]),'design':A[i].tolist(),'y':float(y[i]),'prediction':float(pred[i]),'half_loss':float(.5*r[i]**2),'local_derivatives':loc[i].tolist(),'next_prediction':float(nxt[i]),'next_half_loss':float(.5*(nxt[i]-y[i])**2)} for i in range(4)]
    B=np.array([[1,(np.clip(7,lo,hi)-mu)/np.sqrt(var),0,0,0],[1,0,0,1,1]])
    return {'statistics':{'median':float(np.nanmedian(x)),'clip_range':[float(lo),float(hi)],'mean':float(mu),'variance_ddof0':float(var),'scale':float(np.sqrt(var)),'category_vocabulary':['A','B'],'fit_rows':['H1','H2','H3','H4']},'rows':rows,'gradient_mean':grad.tolist(),'theta_next':new.tolist(),'learning_rate':.2,'initial_half_loss':float(np.mean(.5*r*r)),'next_half_loss':float(np.mean(.5*(nxt-y)**2)),'valid_design':B.tolist(),'valid_next_prediction':(B@new).tolist()}

def encoder_hand():
    X=np.array(['A','B','A','C','A','B','D','C'])[:,None];y=np.arange(1,9,dtype=float);enc=TargetEncoder(target_type='continuous',smooth=2,cv=2,shuffle=False);oof=enc.fit_transform(X,y);full=enc.transform(X)
    return {'categories':X.ravel().tolist(),'y':y.tolist(),'smooth':2,'inner_folds':[{'train':[4,5,6,7],'valid':[0,1,2,3]},{'train':[0,1,2,3],'valid':[4,5,6,7]}],'crossfit_training':oof.ravel().tolist(),'full_training_transform':full.ravel().tolist(),'full_target_mean':float(enc.target_mean_),'valid_categories':['A','C','Z'],'valid_transform':enc.transform(np.array(['A','C','Z'])[:,None]).ravel().tolist()}

def affine_control():
    x=np.array([-2.,-1,0,1])[:,None];y=np.array([1.,2,1,4]);v=np.array([2.,4])[:,None];pred=[];states=[]
    for pool in [x,np.vstack([x,v])]:
        s=StandardScaler().fit(pool);m=LinearRegression().fit(s.transform(x),y);pred.append(m.predict(s.transform(v)).tolist());states.append({'mean':s.mean_.tolist(),'scale':s.scale_.tolist(),'fitted_n':len(pool)})
    return {'train_x':x.ravel().tolist(),'train_y':y.tolist(),'valid_x':v.ravel().tolist(),'predictions':pred,'scaler_states':states,'max_prediction_difference':float(np.max(np.abs(np.array(pred[0])-pred[1]))),'scope':'full-rank unpenalized least squares with intercept; invertible affine rescaling only'}

def summarize(reps,name,fields):
    out={}
    for key in fields:
        v=np.array([np.mean([f[key] for f in r[name]]) for r in reps]);out[key]={'per_dataset':v.tolist(),'mean':float(v.mean()),'sd_across_independent_datasets':float(v.std(ddof=1)),'min':float(v.min()),'max':float(v.max())}
    first=fields[0];out['paired_differences']={key:{'mean_bad_minus_correct':float(np.mean(np.array(out[key]['per_dataset'])-out[first]['per_dataset'])),'bad_worse_count':int(np.sum(np.array(out[key]['per_dataset'])>out[first]['per_dataset'])),'bad_better_count':int(np.sum(np.array(out[key]['per_dataset'])<out[first]['per_dataset']))} for key in fields[1:] if 'baseline' not in key};return out

def run():
    c,d,s=load_data();reps=[{'rep':r,'group':group_run(c,d,s,r),'selection':selection_run(c,d,s,r),'encoding':encoding_run(c,d,s,r)} for r in range(c['independent_repetitions'])]
    return {'protocol':c,'hand':hand(),'encoder_hand':encoder_hand(),'affine_control':affine_control(),'replications':reps,'summary':{'group':summarize(reps,'group',['correct_mse','global_scale_mse']),'selection':summarize(reps,'selection',['correct_mse','global_selection_mse','train_mean_baseline_mse']),'encoding':summarize(reps,'encoding',['correct_mse','self_encoded_mse','global_target_mse','train_mean_baseline_mse'])},'limitations':['Separate mechanisms; cross-mechanism score differences are not a single causal effect','No final untouched test set or deployment claim; all 12 fixed synthetic datasets are teaching evaluations','Group experiment predicts new devices in the same regime, not future-time observations','TargetEncoder row cross-fitting is valid here only because labels/rows have no hidden group effect or chronological target']}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(ROOT/'outputs/result.json'));a=p.parse_args();safe_output(a.out);r=run();atomic_json(a.out,r);print(json.dumps({'out':str(a.out),'summary':r['summary']},ensure_ascii=False,indent=2))
