"""Soft-margin C sweep, independent primal solve, outlier and unit-scaling probes."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import argparse,json,platform,time
import numpy as np,sklearn
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from common import ROOT,load_data,write_json,safe_output
from margin import solve_primal,objective,geometry
CS=[.03,.3,3.,30.]
def arrays(d):return np.column_stack([d['x1'],d['x2']]),d['y']
def model_record(model,X,y,C):
    w=model.coef_[0];b=float(model.intercept_[0]);return {'C':C,'w':w.tolist(),'b':b,'support_indices':model.support_.tolist(),'n_support':int(model.n_support_.sum()),'objective':objective(X,y,w,b,C),**geometry(X,y,w,b)}
def run(directory=None):
    data={s:arrays(v) for s,v in load_data(directory).items()};X,y=data['train'];Xv,yv=data['validation'];Xt,yt=data['test']
    candidates=[];models=[]
    for C in CS:
        m=SVC(kernel='linear',C=C,tol=1e-9,shrinking=False);start=time.perf_counter();m.fit(X,y);elapsed=time.perf_counter()-start
        r=model_record(m,X,y,C);r.update(validation_accuracy=float(accuracy_score(yv,m.predict(Xv))),fit_seconds=elapsed);candidates.append(r);models.append(m)
    selected=max(range(4),key=lambda i:(candidates[i]['validation_accuracy'],-i));chosen=models[selected];C=CS[selected]
    primal=solve_primal(X,y,C)
    # A controlled diagnostic: same fixed C, only the designated outlier is removed.
    clean=SVC(kernel='linear',C=C,tol=1e-9,shrinking=False).fit(X[1:],y[1:]);outlier_probe={'with':model_record(chosen,X,y,C),'without':model_record(clean,X[1:],y[1:],C)}
    # Change only the unit of feature 2. Scaling is fit to training rows, not test.
    unit=np.array([1.,100.]);Xu=X*unit;Xvu=Xv*unit;Xtu=Xt*unit
    raw=SVC(kernel='linear',C=C,tol=1e-8,shrinking=False).fit(Xu,y)
    sc=StandardScaler().fit(Xu);scaled=SVC(kernel='linear',C=C,tol=1e-9,shrinking=False).fit(sc.transform(Xu),y)
    wr=scaled.coef_[0]/sc.scale_;br=float(scaled.intercept_[0]-wr@sc.mean_)
    # Store boundaries back in the original units to compare their positions honestly.
    scaling={'unit_multiplier':unit.tolist(),'raw_units_w':(raw.coef_[0]*unit).tolist(),'raw_units_b':float(raw.intercept_[0]),
        'standardized_original_w':(wr*unit).tolist(),'standardized_original_b':br,'train_mean_changed_units':sc.mean_.tolist(),'train_scale_changed_units':sc.scale_.tolist(),
        'validation_accuracy_raw_units':float(accuracy_score(yv,raw.predict(Xvu))),
        'validation_accuracy_standardized':float(accuracy_score(yv,scaled.predict(sc.transform(Xvu))))}
    hardX=np.array([[-1,-1],[-1,1],[1,-1],[1,1]],float);hardy=np.array([-1,-1,1,1])
    hard=SVC(kernel='linear',C=100,tol=1e-10).fit(hardX,hardy)
    return {'lesson':'064','versions':{'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__},
        'split_sizes':{s:len(v[1]) for s,v in data.items()},'candidates':candidates,'selected_index':selected,
        'primal':primal,'primal_vs_svc':{'objective_difference':abs(primal['objective']-candidates[selected]['objective']),
            'max_score_difference':float(np.max(np.abs(X@np.array(primal['w'])+primal['b']-chosen.decision_function(X))))},
        'outlier_probe':outlier_probe,'scaling_probe':scaling,
        'hard_example':{'X':hardX.tolist(),'y':hardy.tolist(),'w':hard.coef_[0].tolist(),'b':float(hard.intercept_[0]),**geometry(hardX,hardy,hard.coef_[0],float(hard.intercept_[0]))},
        'test':{'accuracy':float(accuracy_score(yt,chosen.predict(Xt))),'y':yt.tolist(),'scores':chosen.decision_function(Xt).tolist()},
        'test_used_for_selection':False,'diagnostics_used_for_selection':False,'C_convention':'0.5 norm(w)^2 + C sum hinge; intercept unpenalized'}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(ROOT/'outputs/result.json'));p.add_argument('--data-directory');a=p.parse_args();r=run(a.data_directory);write_json(safe_output(a.out),r);print(json.dumps({'selected':r['selected_index'],'primal_difference':r['primal_vs_svc'],'test_accuracy':r['test']['accuracy']}))
