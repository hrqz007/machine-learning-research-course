"""Run exact hand example, all-parameter gradient audit and offline training."""
from pathlib import Path
import argparse, json, platform
import numpy as np
from numpy_core import *

def hand_example():
    X=np.array([[1.,2.],[-1.,1.]]); y=np.array([0,1])
    p=dict(W1=np.array([[1.,-1.],[.5,1.]]),b1=np.array([0.,.5]),
           W2=np.array([[.2,-.1],[-.3,.4]]),b2=np.array([.1,-.2]))
    loss,g,c=loss_and_grad(X,y,p,activation='relu')
    return dict(loss=loss, parameters={k:v.tolist() for k,v in p.items()},
                gradients={k:v.tolist() for k,v in g.items()},
                cache={k:v.tolist() for k,v in c.items()})

def gradient_audit():
    X=np.array([[.3,-.7],[1.2,.4],[-.5,.9]]); y=np.array([0,1,0])
    p=initialize(seed=8502,hidden=3); _,g,_=loss_and_grad(X,y,p)
    sweep=[]
    for eps in [1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8]:
        num=finite_difference(X,y,p,eps)
        sweep.append(dict(epsilon=eps, max_abs=max(float(np.max(np.abs(g[k]-num[k]))) for k in g)))
    num=finite_difference(X,y,p)
    layers={k:dict(count=int(g[k].size),max_abs=float(np.max(np.abs(g[k]-num[k]))),
                  scaled_error=float(np.linalg.norm(g[k]-num[k])/max(1.,np.linalg.norm(g[k]),np.linalg.norm(num[k])))) for k in g}
    _,sumg,_=loss_and_grad(X,y,p,reduction='sum')
    _,dupg,_=loss_and_grad(np.repeat(X,2,axis=0),np.repeat(y,2),p)
    return dict(total_parameters=sum(v.size for v in p.values()),per_parameter_block=layers,epsilon_sweep=sweep,
        sum_mean_scaling_max_abs=max(float(np.max(np.abs(sumg[k]-3*g[k]))) for k in g),
        duplicated_batch_mean_max_abs=max(float(np.max(np.abs(dupg[k]-g[k]))) for k in g),
        missing_batch_division_expected_ratio=3,
        missing_batch_division_detected=max(float(np.max(np.abs(sumg[k]-num[k]))) for k in g)>1e-3)

def run():
    data=load_data(); X,y=data['train']; p=initialize(); history=[]
    initial=metrics(X,y,p)
    for step in range(1201):
        if step%40==0:
            history.append(dict(step=step,train=metrics(X,y,p),validation=metrics(*data['validation'],p)))
        if step<1200:
            _,g,_=loss_and_grad(X,y,p); p=sgd_step(p,g,.15)
    return dict(unit='085',environment=dict(python=platform.python_version(),numpy=np.__version__,dtype='float64',device='cpu'),
                hand_example=hand_example(), gradient_audit=gradient_audit(),
                training=dict(seed=8501,hidden=8,activation='tanh',learning_rate=.15,updates=1200,
                              batch_size=len(y),initial_train=initial,final_train=metrics(X,y,p),
                              final_validation=metrics(*data['validation'],p),final_test=metrics(*data['test'],p),history=history),
                final_parameters={k:v.tolist() for k,v in p.items()},
                interpretation='Implementation audit and one synthetic fixed-seed demonstration; no real-world generalization claim.')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='outputs/result.json');args=ap.parse_args()
    result=run(); out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('gradient_audit',)},indent=2))
    print(json.dumps({k:v for k,v in result['training'].items() if k!='history'},indent=2))
