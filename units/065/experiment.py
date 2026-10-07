"""Small dual/primal checks and exact polynomial feature versus kernel equivalence."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,json,platform
from pathlib import Path
import numpy as np,sklearn
from sklearn.svm import SVC
from common import ROOT,load_data,safe_output,write_json
from dual import gram,polynomial_features,solve_dual
from margin import solve_primal

def arrays(d):return np.column_stack([d['x1'],d['x2']]),d['y']
def run(directory=None):
    Xh=np.array([[-1.],[1.]]);yh=np.array([-1.,1.]);hand=solve_dual(Xh@Xh.T,yh,1.)
    bounded=solve_dual(Xh@Xh.T,yh,.1)
    data={s:arrays(v) for s,v in load_data(directory).items()};X,y=data['train'];Xv,yv=data['validation'];Xt,yt=data['test'];C=2.
    Phi=polynomial_features(X);K=gram(X);dual=solve_dual(K,y,C);primal=solve_primal(Phi,y,C)
    explicit=SVC(kernel='linear',C=C,tol=1e-10,shrinking=False).fit(Phi,y)
    precomputed=SVC(kernel='precomputed',C=C,tol=1e-10,shrinking=False).fit(K,y)
    native=SVC(kernel='poly',degree=2,gamma=1,coef0=0,C=C,tol=1e-10,shrinking=False).fit(X,y)
    ay=np.array(dual['alpha'])*y;dw=Phi.T@ay
    def compare(Z):
        own=gram(Z,X)@ay+dual['b'];ep=explicit.decision_function(polynomial_features(Z));kp=precomputed.decision_function(gram(Z,X));np_=native.decision_function(Z)
        return {'own_dual':own.tolist(),'explicit':ep.tolist(),'precomputed':kp.tolist(),'native_poly':np_.tolist(),
                'max_own_explicit_difference':float(np.max(np.abs(own-ep))),
                'max_explicit_kernel_difference':float(np.max(np.abs(ep-kp))),
                'max_native_precomputed_difference':float(np.max(np.abs(np_-kp)))}
    test=compare(Xt);val=compare(Xv);wpr=np.array(primal['w'])
    xx,yy=np.meshgrid(np.linspace(-1.7,1.7,51),np.linspace(-1.7,1.7,51));grid=np.c_[xx.ravel(),yy.ravel()];grid_score=gram(grid,X)@ay+dual['b']
    return {'lesson':'065','versions':{'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__},
        'C':C,'protocol':'fixed C and homogeneous degree2 kernel, no hyperparameter selection',
        'split_sizes':{s:len(v[1]) for s,v in data.items()},'hand_two_points':hand,'all_bound_two_points':bounded,
        'dual':dual,'primal':primal,'dual_reconstructed_w':dw.tolist(),
        'gram_equivalence_error':float(np.max(np.abs(K-Phi@Phi.T))),
        'gram_eigenvalues':np.linalg.eigvalsh(K).tolist(),'invalid_similarity_eigenvalues':np.linalg.eigvalsh([[1,2],[2,1]]).tolist(),
        'primal_dual_objective_difference':float(abs(primal['objective']-dual['audit']['dual_value'])),
        'primal_dual_max_train_score_difference':float(np.max(np.abs(Phi@wpr+primal['b']-np.array(dual['scores'])))),
        'validation':val,'test':{'y':yt.tolist(),**test,'accuracy':float(np.mean(np.where(np.array(test['own_dual'])>=0,1,-1)==yt))},
        'grid':{'axis':np.linspace(-1.7,1.7,51).tolist(),'scores':grid_score.reshape(51,51).tolist()},
        'test_used_for_selection':False,'support_indices':np.flatnonzero(np.array(dual['alpha'])>1e-6).tolist()}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(ROOT/'outputs/result.json'));p.add_argument('--data-directory');a=p.parse_args();r=run(a.data_directory);write_json(safe_output(a.out),r);print(json.dumps({'gap':r['dual']['audit']['gap'],'primal_dual':r['primal_dual_objective_difference'],'kernel_equivalence':r['test']['max_explicit_kernel_difference'],'test_accuracy':r['test']['accuracy']}))
