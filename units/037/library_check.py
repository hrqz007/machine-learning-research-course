"""Optional actual scikit-learn 1.8.0 comparison with explicit objective scaling."""
from pathlib import Path
import argparse,json,sys
import numpy as np
from experiment import load_inputs,run_path,forward,diagnostics,write_report

def check():
    # Validate complete inputs before importing or calling an optional solver.
    data,s=load_inputs()
    import scipy,sklearn
    from sklearn.linear_model import Lasso
    if sklearn.__version__!='1.8.0':raise RuntimeError('This recorded comparison requires scikit-learn==1.8.0')
    result=[]
    for label,a in [('main',data),('correlated',np.column_stack((data[:,0],s['stress_rho']*data[:,0]+(1-s['stress_rho'])*data[:,1],data[:,2])))]:
        model=Lasso(alpha=s['lambda'],fit_intercept=False,selection='cyclic',tol=1e-12,max_iter=10000,precompute=False,positive=False)
        model.fit(a[:,:2],a[:,2])
        custom=run_path(a,s,'coordinate');theta=np.array(custom['final_theta'])
        delta=float(np.max(np.abs(theta-model.coef_)))
        if delta>2e-9:raise RuntimeError('coordinate result differs from library beyond absolute tolerance')
        result.append({'design':label,'coef':model.coef_.tolist(),'intercept':float(model.intercept_),'n_iter':int(model.n_iter_),'dual_gap':float(model.dual_gap_),'our_theta':theta.tolist(),'our_status':custom['status'],'max_absolute_difference':delta,'our_kkt':diagnostics(a,theta,s,'coordinate')['lasso_kkt_inf'],'library_kkt':diagnostics(a,model.coef_,s,'coordinate')['lasso_kkt_inf'],'library_objective':forward(a,model.coef_,s['lambda'])['objective']})
    return {'python':sys.version.split()[0],'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'settings':{'alpha':s['lambda'],'fit_intercept':False,'selection':'cyclic','tol':1e-12,'max_iter':10000,'positive':False,'precompute':False},'objective':'||X theta-y||^2/(2n) + alpha ||theta||_1; alpha=lambda','absolute_tolerance':2e-9,'comparisons':result,'status':'author library check passed'}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=Path('library_outputs'));a=p.parse_args();r=check();write_report(r,a.output_dir/'library-result.json');print(json.dumps(r,ensure_ascii=False,indent=2))
