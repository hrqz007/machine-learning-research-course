"""Four-candidate searches; separate stopping, tuning, calibration and final test sets."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,inspect,json,platform,time
from pathlib import Path
import numpy as np,sklearn
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from common import ROOT,load_data,safe_output,write_json
from boosting import derivatives,leaf_weight,split_gain,metrics,reliability
CONFIGS=[{'learning_rate':.05,'max_leaf_nodes':7,'l2_regularization':0.},
         {'learning_rate':.05,'max_leaf_nodes':15,'l2_regularization':1.},
         {'learning_rate':.1,'max_leaf_nodes':7,'l2_regularization':1.},
         {'learning_rate':.1,'max_leaf_nodes':15,'l2_regularization':4.}]

def arrays(d):return np.column_stack([d['x1'],d['x2'],d['category']]),d['y'].astype(int)

def run(directory=None):
    data={s:arrays(d) for s,d in load_data(directory).items()}
    X,y=data['train'];Xs,ys=data['stop'];Xv,yv=data['tune'];Xc,yc=data['calibration']
    models=[];candidates=[]
    with threadpool_limits(limits=1):
        for i,cfg in enumerate(CONFIGS):
            model=HistGradientBoostingClassifier(**cfg,max_iter=150,min_samples_leaf=12,max_bins=63,
                categorical_features=[False,False,True],early_stopping=True,scoring='loss',n_iter_no_change=12,
                tol=1e-5,validation_fraction=None,random_state=63)
            start=time.perf_counter();model.fit(X,y,X_val=Xs,y_val=ys);elapsed=time.perf_counter()-start
            candidates.append({'index':i,'config':cfg,'iterations':int(model.n_iter_),
                'tune_log_loss':metrics(yv,model.predict_proba(Xv)[:,1])['log_loss'],
                'fit_seconds':elapsed,'train_loss':(-model.train_score_).tolist(),
                'stop_loss':(-model.validation_score_).tolist()});models.append(model)
        chosen=min(range(4),key=lambda i:(candidates[i]['tune_log_loss'],i));model=models[chosen]
        linear=[];lmodels=[]
        for C in [.01,.1,1.,10.]:
            # All imputation, scaling, and category fitting see training rows only.
            pre=ColumnTransformer([('numeric',make_pipeline(SimpleImputer(add_indicator=True),StandardScaler()),[0,1]),
                                   ('category',OneHotEncoder(handle_unknown='ignore',sparse_output=False),[2])])
            lm=make_pipeline(pre,LogisticRegression(C=C,max_iter=1000,random_state=63))
            start=time.perf_counter();lm.fit(X,y);elapsed=time.perf_counter()-start
            linear.append({'C':C,'tune_log_loss':metrics(yv,lm.predict_proba(Xv)[:,1])['log_loss'],'fit_seconds':elapsed});lmodels.append(lm)
        li=min(range(4),key=lambda i:(linear[i]['tune_log_loss'],i))
        # This fixed sigmoid calibration recipe is fitted only after model selection.
        calibrator=LogisticRegression(C=1e6,solver='lbfgs',max_iter=1000)
        calibrator.fit(model.decision_function(Xc).reshape(-1,1),yc)
        # Only now is the sealed test data used to estimate final performance.
        Xt,yt=data['test'];raw=model.predict_proba(Xt)[:,1]
        cal=calibrator.predict_proba(model.decision_function(Xt).reshape(-1,1))[:,1]
        baseline=lmodels[li].predict_proba(Xt)[:,1]
        unknown=np.array([[0.,np.nan,9.],[0.,np.nan,np.nan]])
        unknown_p=model.predict_proba(unknown)[:,1]
    g,h=derivatives(np.array([0,0,1,1]),np.zeros(4))
    default=HistGradientBoostingClassifier().get_params()
    keep=['learning_rate','max_iter','max_leaf_nodes','max_depth','min_samples_leaf','l2_regularization','max_bins','categorical_features','early_stopping','validation_fraction','n_iter_no_change','tol']
    result={'lesson':'063','versions':{'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__},
        'split_sizes':{s:len(v[1]) for s,v in data.items()},'default_audit':{k:default[k] for k in keep},
        'hist_candidates':candidates,'linear_candidates':linear,'selected_hist_index':chosen,'selected_linear_index':li,
        'calibrator':{'coefficient':float(calibrator.coef_[0,0]),'intercept':float(calibrator.intercept_[0]),'fit_split':'calibration'},
        'test':{'hist_raw':metrics(yt,raw),'hist_sigmoid':metrics(yt,cal),'linear':metrics(yt,baseline)},
        'test_predictions':{'y':yt.tolist(),'raw':raw.tolist(),'calibrated':cal.tolist(),'linear':baseline.tolist()},
        'reliability':{'raw':reliability(yt,raw),'calibrated':reliability(yt,cal)},
        'unknown_category_probabilities':unknown_p.tolist(),
        'hand':{'gradients':g.tolist(),'hessians':h.tolist(),'left_weight_l2_0':leaf_weight(1.,.5,0.),
                'left_weight_l2_1':leaf_weight(1.,.5,1.),'gain_l2_0':split_gain(1.,.5,-1.,.5),
                'gain_l2_1':split_gain(1.,.5,-1.,.5,1.)},
        'budget':{'hist_candidates':4,'linear_candidates':4,'hist_max_iterations_per_candidate':150,'hist_actual_iterations':sum(v['iterations'] for v in candidates),'equal_compute_claim':False},
        'test_used_for_selection':False,'calibration_recipe_fixed_before_test':True}
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(ROOT/'outputs/result.json'));p.add_argument('--data-directory');a=p.parse_args();r=run(a.data_directory);write_json(safe_output(a.out),r);print(json.dumps({'selected':r['selected_hist_index'],'test':r['test']},ensure_ascii=False))
if __name__=='__main__':main()
