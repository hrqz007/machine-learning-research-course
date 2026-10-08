"""固定搜索预算的经典方法比较；测试标签从不进入选参函数。"""
import argparse,importlib.metadata,json,pickle,time
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import log_loss,brier_score_loss,accuracy_score,roc_auc_score
from common import ROOT,require,load_data,safe_output,write_json,wilson

def xy(data):return np.column_stack([data[f'x{i}'] for i in range(6)]),data['y'].astype(int)

def make_model(family,param):
    if family=='logistic':return make_pipeline(StandardScaler(),LogisticRegression(C=float(param),max_iter=1000,solver='lbfgs'))
    if family=='forest':return RandomForestClassifier(n_estimators=80,max_depth=param,min_samples_leaf=5,random_state=68,n_jobs=1)
    if family=='knn':return make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=int(param)))
    raise ValueError('unknown model family')

def point_losses(y,p):
    y=np.asarray(y);p=np.asarray(p,dtype=float)
    require(y.shape==p.shape and y.ndim==1 and len(y)>0,'matching nonempty vectors required')
    require(np.isin(y,[0,1]).all() and np.isfinite(p).all() and ((p>=0)&(p<=1)).all(),'binary labels and valid probabilities required')
    q=np.clip(p,np.finfo(float).eps,1-np.finfo(float).eps)
    return -(y*np.log(q)+(1-y)*np.log1p(-q))

def calibration(y,p,bins=5):
    require(isinstance(bins,int) and bins>0,'positive bin count required')
    point_losses(y,p);rows=[]
    # [0,.2),...,[.8,1]。最后一格显式包含1，空格不作0%事件率。
    ix=np.minimum((p*bins).astype(int),bins-1)
    for b in range(bins):
        use=ix==b;n=int(use.sum());rows.append({'bin':b,'n':n,'mean_probability':float(p[use].mean()) if n else None,'event_rate':float(y[use].mean()) if n else None})
    return rows

def metrics(y,p):
    loss=point_losses(y,p);pred=p>=.5;correct=int((pred==y).sum());cal=calibration(y,p)
    return {'n':len(y),'log_loss':float(loss.mean()),'brier':float(np.mean((p-y)**2)),'accuracy':float(correct/len(y)),'accuracy_wilson_95':wilson(correct,len(y)),'roc_auc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,'ece_5':float(sum(z['n']*abs(z['mean_probability']-z['event_rate']) for z in cal if z['n'])/len(y)),'calibration':cal}

def paired_bootstrap(y,p,q,repeats=1200,seed=68099):
    require(isinstance(repeats,int) and repeats>=20,'at least20 bootstrap draws')
    delta=point_losses(y,p)-point_losses(y,q);rng=np.random.default_rng(seed);draw=rng.integers(0,len(y),size=(repeats,len(y)));means=delta[draw].mean(axis=1)
    return {'mean_delta':float(delta.mean()),'percentile_95':np.quantile(means,[.025,.975]).tolist(),'repeats':repeats,'seed':seed}

def select(X,y,V,v):
    # 唯一选参输入只有训练与验证。每个方法族正好3个配置。
    settings={'logistic':[.1,1,10],'forest':[3,6,None],'knn':[5,15,35]};rows=[];models={};finalists={}
    for family,params in settings.items():
        family_rows=[]
        for index,param in enumerate(params):
            model=make_model(family,param);start=time.perf_counter();model.fit(X,y);seconds=time.perf_counter()-start
            p=model.predict_proba(V)[:,1];name=family+'_'+str(index);models[name]=model
            row={'name':name,'family':family,'parameter':param,'validation_log_loss':float(log_loss(v,p)),'fit_seconds':seconds};rows.append(row);family_rows.append(row)
        finalists[family]=min(family_rows,key=lambda r:(r['validation_log_loss'],r['name']))['name']
    winner=min(rows,key=lambda r:(r['validation_log_loss'],r['name']))['name']
    return rows,models,finalists,winner

def run():
    d=load_data();X,y=xy(d['train']);V,v=xy(d['validation']);T,t=xy(d['test'])
    rows,models,finalists,winner=select(X,y,V,v)
    # 完成选参后才读取测试预测。为了分离选参和重训影响，保留训练集拟合参数，不做train+val重训。
    base=DummyClassifier(strategy='prior').fit(X,y);chosen={'prior':base,**{k:models[val] for k,val in finalists.items()}};res={}
    for family,m in chosen.items():
        p=m.predict_proba(T)[:,1];m.predict_proba(T);times=[]
        for _ in range(9):
            start=time.perf_counter();m.predict_proba(T);times.append(time.perf_counter()-start)
        slices={}
        for label,mask in {'abs_x0_below_0.5':abs(T[:,0])<.5,'abs_x0_at_least_0.5':abs(T[:,0])>=.5}.items():slices[label]=metrics(t[mask],p[mask])
        losses=point_losses(t,p);worst=np.argsort(-losses,kind='stable')[:12]
        res[family]={'probabilities':p.tolist(),'metrics':metrics(t,p),'slices':slices,'median_batch_seconds':float(np.median(times)),'batch_size':len(T),'serialized_model_bytes':len(pickle.dumps(m,protocol=5)),'failures':[{'id':str(d['test']['id'][i]),'y':int(t[i]),'p':float(p[i]),'p_true':float(d['test']['p_true'][i]),'log_loss':float(losses[i])} for i in worst]}
    comparisons={k:paired_bootstrap(t,np.array(z['probabilities']),np.array(res['logistic']['probabilities'])) for k,z in res.items() if k!='logistic'}
    selected_family=next(r['family'] for r in rows if r['name']==winner)
    # 解释需求：标准化线性系数有方向，但不是因果效应。
    coeff=models[finalists['logistic']].named_steps['logisticregression'].coef_[0].tolist()
    return {'unit':'068','seed':68021,'versions':{p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn']},'split_ids':{k:z['id'].tolist() for k,z in d.items()},'selection_metric':'validation log_loss; exact ties by name','threshold':.5,'candidate_rows':rows,'finalists':finalists,'winner':winner,'selected_family':selected_family,'test':res,'paired_vs_logistic':comparisons,'standardized_logistic_coefficients':coeff,'hand':{'y':[1,0],'p':[.8,.3],'log_loss':float((-np.log(.8)-np.log(.7))/2),'brier':(.2**2+.3**2)/2},'cost_note':'fit and median9 warmed225-row batches are machine-dependent; pickle bytes are serialized size, not RAM','test_note':'single held-out synthetic split; intervals conditional on fitted models, not full selection uncertainty'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(ROOT/'outputs/result.json'));a=p.parse_args();r=run();write_json(safe_output(a.out),r);print(json.dumps({'winner':r['winner'],'test_log_loss':{k:round(v['metrics']['log_loss'],6) for k,v in r['test'].items()}},ensure_ascii=False))
if __name__=='__main__':main()
