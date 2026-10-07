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
    from gaussian_process import kernel,fit,predict
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF
    h=report['hand'];a=2**(-.25)
    close(h['C'],[[1.25,.5],[.5,1.25]],'hand training covariance');close(h['coefficient'],[4/3,-4/3],'hand solve');close(h['mean'],0,'hand midpoint mean');close(h['latent_variance'],1-2*a*a/1.75,'hand latent variance');close(h['observation_variance'],1-2*a*a/1.75+.25,'hand observation variance');close(h['lml'],-4/3-.5*np.log(21/16)-np.log(2*np.pi),'independent scalar LML')
    d=load_data();X=d['train']['x'][:,None];y=d['train']['y'];Z=np.array([[-2.],[0.],[4.]])
    for ell in [.25,1,3]:
        K=kernel(X,X,ell);close(K,K.T,'kernel symmetric '+str(ell));check(np.min(np.linalg.eigvalsh(K))>-1e-10,'kernel PSD '+str(ell))
        m=fit(X,y,ell);mu,cov=predict(m,Z);mo,co=predict(m,Z,observation=True)
        close(mu,mo,'observation and latent mean '+str(ell));close(co-cov,np.eye(3)*.0324,'only physical noise added to predictive covariance '+str(ell));check(np.min(np.linalg.eigvalsh(cov))>-1e-10,'posterior PSD '+str(ell));check((np.diag(cov)<=1+1e-10).all(),'posterior does not exceed unit prior variance '+str(ell));close(m['L']@m['L'].T,m['C'],'Cholesky reconstruction '+str(ell));close(m['C']@m['a'],y,'linear solve '+str(ell))
        # 独立直接 solve 作小矩阵验算；生产算法仍用 Cholesky。
        cross=kernel(X,Z,ell);direct=kernel(Z,Z,ell)-cross.T@np.linalg.solve(m['C'],cross);close(cov,direct,'direct conditional covariance '+str(ell))
    for key,error in report['library_max_errors'].items():check(error<1e-8,'library numerical agreement '+key)
    rejects(lambda:fit(X,y,noise_variance=-1),'negative noise rejected');rejects(lambda:fit(X,y,jitter=-1),'negative jitter rejected');rejects(lambda:fit(X,y,length_scale=0),'zero length scale rejected');rejects(lambda:fit(X,y[:-1]),'target dimension rejected');rejects(lambda:kernel([[np.nan]],[[1]]),'nonfinite input rejected');rejects(lambda:kernel([[1,2]],[[1]]),'dimension mismatch rejected')
    check(report['jitter_check']['singular_without_jitter_failed'],'duplicate inputs fail with no noise or jitter');check(report['jitter_check']['finite_coefficients'],'small explicit jitter produces finite solve');close(report['noise_api']['max_mean_difference'],0,'alpha white equal mean');close([report['noise_api']['min_variance_difference'],report['noise_api']['max_variance_difference']],[.0324,.0324],'WhiteKernel includes test noise')
    for row in report['variants']:
        for name in ['interpolation','extrapolation']:
            s=row[name];m=np.array(s['mean']);v=np.array(s['latent_variance']);truth=d[name]['f_true'];obs=d[name]['y']
            close(s['latent_rmse'],np.sqrt(np.mean((m-truth)**2)),'recomputed RMSE '+row['label']+name);close(s['latent_pointwise_95_fraction'],np.mean(abs(m-truth)<=1.96*np.sqrt(v)),'latent grid fraction '+row['label']+name);close(s['observed_pointwise_95_fraction'],np.mean(abs(m-obs)<=1.96*np.sqrt(v+row['noise_variance'])),'observation grid fraction '+row['label']+name)
    p=report['lml_grid'];parts=np.array(p['fixed_noise_parts']);check(parts.shape==(46,3),'LML decomposition shape');close(parts[:,2],np.full(46,-len(X)/2*np.log(2*np.pi)),'LML normalization constant')
    for n in report['cost']:check(n['one_dense_matrix_bytes']==8*n['n']**2,'storage '+str(n['n']));close(n['cholesky_leading_flops'],n['n']**3/3,'cubic cost '+str(n['n']))
    altered=copy.deepcopy(report);altered['hand']['latent_variance']+=.25;rejects(lambda:compare(altered,expected),'noise double count rejected')
    altered=copy.deepcopy(report);altered['variants'][0]['extrapolation']['latent_pointwise_95_fraction']=.95;rejects(lambda:compare(altered,expected),'invented coverage rejected')
    altered=copy.deepcopy(report);altered['lml_grid']['values'][0][0]+=1;rejects(lambda:compare(altered,expected),'changed marginal likelihood rejected')

def main():
    p=argparse.ArgumentParser();p.add_argument('--report',default=str(ROOT/'experiment-result.json'));p.add_argument('--out',default=str(ROOT/'outputs/test-result.json'));a=p.parse_args();report=json.loads(Path(a.report).read_text());expected=common_checks(report);specific(report,expected);result={'status':'passed','checks':len(CHECKS),'python_optimized':not __debug__,'details':CHECKS};write_json(safe_output(a.out),result);print(json.dumps({'status':'passed','checks':len(CHECKS),'python_optimized':not __debug__}))
if __name__=='__main__':main()
