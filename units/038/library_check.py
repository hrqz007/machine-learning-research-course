"""Run actual SciPy 1.17 optimizer, update-strategy, and Wolfe comparisons."""
import argparse, json, sys
from pathlib import Path
import numpy as np
import scipy
from scipy.optimize import minimize, BFGS, line_search
from experiment import load_inputs, RowObjective, optimize, inverse_update, serialized, safe_write

def run_library():
    x,y,c=load_inputs();results=[]
    for model,key in [('linear','linear_initial_parameters'),('square','nonlinear_initial_parameters')]:
        for method in ('BFGS','L-BFGS-B'):
            obj=RowObjective(x,y,model)
            fun=lambda z:obj.evaluate(z,False)['f']
            jac=lambda z:obj.evaluate(z,True)['g']
            opts={'gtol':1e-10,'maxiter':100}
            if method=='BFGS':opts.update(norm=2,c1=1e-4,c2=.9)
            else:opts.update(ftol=0.,maxls=40,maxcor=5)
            result=minimize(fun,c[key],jac=jac,method=method,options=opts)
            final=obj.evaluate(result.x,True,True)
            custom=optimize(RowObjective(x,y,model),c[key],c,method.lower().replace('l-bfgs-b','lbfgs'),trace=False)
            results.append({'model':model,'method':method,'success':bool(result.success),'status':int(result.status),
                'message':str(result.message),'iterations':int(result.nit),'nfev':int(result.nfev),'njev':int(result.njev),
                'theta':result.x.tolist(),'f':final['f'],'gradient':final['g'].tolist(),'gradient_norm_2':float(np.linalg.norm(final['g'])),
                'gradient_target_2_met':bool(np.linalg.norm(final['g'])<=1e-10),
                'hessian_eigenvalues':np.linalg.eigvalsh(final['H']).tolist(),
                'custom_status':custom['status'],'custom_f':custom['f'],'custom_gradient_norm':custom['gradient_norm'],
                'comparison_note':'different line searches/stopping norms: no iterate equality assertion'})
    s=np.array([.5,2.]);dy=np.array([.5,8.]);own,info=inverse_update(np.eye(2),s,dy,1e-12)
    strategy=BFGS(exception_strategy='skip_update',min_curvature=1e-12,init_scale=np.eye(2));strategy.initialize(2,'inv_hess');strategy.update(s,dy)
    error=float(np.max(np.abs(strategy.get_matrix()-own)))
    if error>1e-14:raise RuntimeError('actual BFGS strategy update mismatch')
    obj=RowObjective(x,y,'linear');point=np.zeros(2);f=lambda z:obj.evaluate(z,False)['f'];g=lambda z:obj.evaluate(z)['g'];p=-g(point)
    ls=line_search(f,g,point,p,c1=1e-4,c2=.9,maxiter=40)
    alpha=ls[0]
    if alpha is None:raise RuntimeError('Wolfe line search unexpectedly failed')
    old_slope=float(g(point)@p);new_slope=float(g(point+alpha*p)@p)
    armijo=f(point+alpha*p)<=f(point)+1e-4*alpha*old_slope
    strong=abs(new_slope)<=.9*abs(old_slope)
    if not (armijo and strong):raise RuntimeError('returned step failed independent strong Wolfe test')
    return {'python':sys.version.split()[0],'numpy':np.__version__,'scipy':scipy.__version__,
        'optimizer_results':results,'strategy':{'C':strategy.get_matrix().tolist(),'max_absolute_difference':error,'initial_scale':'explicit identity','exception_strategy':'skip_update'},
        'actual_line_search':{'alpha':float(alpha),'function_calls_reported':int(ls[1]),'gradient_calls_reported':int(ls[2]),
            'old_slope':old_slope,'new_slope':new_slope,'armijo':bool(armijo),'strong_wolfe':bool(strong)},
        'limits':'SciPy BFGS strategy, minimize BFGS and L-BFGS-B are distinct APIs; no bounds supplied to L-BFGS-B in this test.'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,default=Path('outputs'));args=ap.parse_args()
    r=run_library();safe_write(args.output_dir,'library-result.json',serialized(r));print(json.dumps(r,ensure_ascii=False))
if __name__=='__main__':main()
