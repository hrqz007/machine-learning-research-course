"""显式检查，不依赖 Python assert；-O 下仍执行同一批检查。"""
from pathlib import Path
import argparse,copy,hashlib,json,sys,tempfile
import numpy as np
from common import ROOT,require,load_data,safe_output,write_json
from experiment import run
CHECKS=[]
def check(ok,name):
    if not bool(ok):raise ValueError('CHECK FAILED: '+name)
    CHECKS.append(name)
def close(a,b,name,tol=1e-8):check(np.allclose(a,b,atol=tol,rtol=tol),name)
def rejects(fun,name):
    try:fun()
    except (ValueError,np.linalg.LinAlgError):CHECKS.append(name);return
    raise ValueError('Expected rejection: '+name)
def compare(actual,expected,path='report'):
    # 计时无法逐次相等。其内部一致性另行检查，不能把正常机器抖动当算法错误。
    if isinstance(expected,dict):
        require(set(actual)==set(expected),'unexpected keys '+path)
        for k in expected:
            if k in ['fit_seconds','final_fit_seconds','single_latency','batch_latency']:continue
            compare(actual[k],expected[k],path+'.'+k)
    elif isinstance(expected,list):
        require(len(actual)==len(expected),'list length '+path)
        for i,(a,b) in enumerate(zip(actual,expected)):compare(a,b,path+'['+str(i)+']')
    elif isinstance(expected,(float,int)) and not isinstance(expected,bool):
        require(np.isfinite(actual) and np.isclose(actual,expected,atol=1e-8,rtol=1e-8),'numeric mismatch '+path)
    else:require(actual==expected,'value mismatch '+path)

def common_checks(report):
    d=load_data();check(len(d)==3,'three verified CSV splits')
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp);copydir=p/'data';copydir.mkdir()
        for f in (ROOT/'data').iterdir():
            if f.is_file():(copydir/f.name).write_bytes(f.read_bytes())
        csv=copydir/'train.csv';lines=csv.read_text().splitlines();fields=lines[1].split(',');fields[1]=str(float(fields[1])+.1);lines[1]=','.join(fields);csv.write_text('\n'.join(lines)+'\n')
        rejects(lambda:load_data(copydir),'changed data rejected')
        g=json.loads((copydir/'generation.json').read_text());g['sha256']['train.csv']=hashlib.sha256(csv.read_bytes()).hexdigest();(copydir/'generation.json').write_text(json.dumps(g))
        rejects(lambda:load_data(copydir),'changed data plus matching digest rejected')
        target=p/'target';target.write_text('keep');link=p/'link';link.symlink_to(target)
        rejects(lambda:safe_output(link),'symlink output rejected')
        import os
        hard=p/'hard';os.link(target,hard);rejects(lambda:safe_output(hard),'hardlink output rejected')
    expected=run();compare(report,expected);check(True,'full deterministic report recomputed from source')
    return expected

def specific(report,expected):
    from kernels import rbf,polynomial,krr_fit,krr_predict
    from sklearn.metrics.pairwise import rbf_kernel,polynomial_kernel
    from experiment import candidates,xy
    from sklearn.preprocessing import StandardScaler
    X=np.array([[-1.,0.],[0.,1.],[1.,2.]])
    for gamma in [.1,1.,10.]:
        K=rbf(X,X,gamma);close(K,K.T,'RBF symmetry '+str(gamma));close(np.diag(K),np.ones(3),'RBF unit diagonal '+str(gamma));check(np.min(np.linalg.eigvalsh(K))>-1e-10,'RBF PSD '+str(gamma));close(K,rbf_kernel(X,gamma=gamma),'RBF sklearn '+str(gamma))
    for degree in [1,2,3,4]:
        K=polynomial(X,X,.5,1.,degree);close(K,polynomial_kernel(X,gamma=.5,coef0=1.,degree=degree),'polynomial sklearn '+str(degree));check(np.min(np.linalg.eigvalsh(K))>-1e-9,'polynomial PSD '+str(degree))
    close(report['hand_krr']['coefficient'],[1,-1],'hand KRR coefficients');close(report['hand_krr']['prediction'],[.5,0,-.5],'hand KRR predictions')
    rejects(lambda:rbf([[1]],[[2]],0),'zero gamma rejected');rejects(lambda:rbf([[np.nan]],[[2]]),'nan kernel input rejected');rejects(lambda:rbf([[1,2]],[[1]]),'dimension mismatch rejected');rejects(lambda:polynomial(X,X,degree=2.5),'noninteger degree rejected');rejects(lambda:polynomial(X,X,coef0=-1),'negative coef0 rejected');rejects(lambda:krr_fit(X,[1,2],1),'bad target size rejected');rejects(lambda:krr_fit(X,[1,2,3],0),'zero regularization rejected')
    d=load_data();Xt,yt=xy(d['train']);Xv,yv=xy(d['validation']);Xs,ys=xy(d['test'])
    for family,row in report['models'].items():
        check(len(candidates(family))==9 and len(row['trials'])==9,'equal candidate budget '+family)
        values=[t['validation_accuracy'] for t in row['trials']];check(row['best_trial']==max(range(9),key=lambda i:values[i]),'validation-only first-tie selection '+family)
        for i,t in enumerate(row['trials']):
            close(t['scaler_mean'],Xt.mean(axis=0),'training-only standardization '+family+str(i));close(t['validation_accuracy'],np.mean(np.array(t['validation_prediction'])==yv),'validation metric '+family+str(i));check(t['fit_seconds']>=0,'finite fit duration '+family+str(i))
        close(row['refit_scaler_mean'],np.vstack([Xt,Xv]).mean(axis=0),'refit scaler uses train plus validation '+family);close(row['test_accuracy'],np.mean(np.array(row['test_prediction'])==ys),'test metric '+family)
        check(row['selected_prediction_arrays_bytes']>0 and row['serialized_pipeline_bytes']>0,'storage positive '+family)
        for name in ['single_latency','batch_latency']:
            t=row[name];v=np.array(t['milliseconds']);check(len(v)==15 and np.isfinite(v).all() and (v>0).all(),'timing observations '+family+name);close(t['median_ms'],np.median(v),'timing median '+family+name);close([t['q25_ms'],t['q75_ms']],np.quantile(v,[.25,.75]),'timing quartiles '+family+name)
        check(row['support_vectors'] is None if family=='linear' else 0<row['support_vectors']<=360,'support vector semantics '+family)
    for row in report['storage']:check(row['dense_float64_gram_bytes']==8*row['n']**2,'quadratic storage '+str(row['n']))
    for row in report['approximation']:
        close(row['mean_error'],np.mean(row['relative_frobenius_errors']),'approximation mean '+str(row['components'])+row['method']);close(row['std_error'],np.std(row['relative_frobenius_errors'],ddof=1),'approximation spread '+str(row['components'])+row['method'])
    altered=copy.deepcopy(report);altered['models']['rbf']['test_accuracy']-=.1;rejects(lambda:compare(altered,expected),'changed reported accuracy rejected')
    altered=copy.deepcopy(report);altered['hand_krr']['coefficient'][0]+=.1;rejects(lambda:compare(altered,expected),'changed hand coefficient rejected')
    altered=copy.deepcopy(report);altered['models']['poly']['best_trial']=8;rejects(lambda:compare(altered,expected),'changed selection rejected')

def main():
    p=argparse.ArgumentParser();p.add_argument('--report',default=str(ROOT/'experiment-result.json'));p.add_argument('--out',default=str(ROOT/'outputs/test-result.json'));a=p.parse_args();report=json.loads(Path(a.report).read_text());expected=common_checks(report);specific(report,expected);result={'status':'passed','checks':len(CHECKS),'python_optimized':not __debug__,'details':CHECKS};write_json(safe_output(a.out),result);print(json.dumps({'status':'passed','checks':len(CHECKS),'python_optimized':not __debug__}))
if __name__=='__main__':main()
