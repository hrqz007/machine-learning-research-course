"""普通与优化模式均执行显式检查；重算报告、独立公式与破坏性负例。"""
from pathlib import Path
import argparse,copy,hashlib,json,os,tempfile
import numpy as np
from sklearn.decomposition import PCA
from common import ROOT,require,load_data,safe_output,write_json
from experiment import run,matrix
from pca import fit,transform,inverse_transform,reconstruct
CHECKS=[]
def check(ok,name):
    if not bool(ok):raise ValueError('CHECK FAILED: '+name)
    CHECKS.append(name)
def close(a,b,name):check(np.allclose(a,b,atol=1e-9,rtol=1e-9),name)
def rejects(fun,name):
    try:fun()
    except (ValueError,TypeError):CHECKS.append(name);return
    raise ValueError('Expected rejection: '+name)
def compare(a,b,path='report'):
    if isinstance(b,dict):
        require(set(a)==set(b),'keys differ '+path)
        for k in b:
            if k in ['fit_seconds','median_batch_seconds']:continue
            compare(a[k],b[k],path+'.'+k)
    elif isinstance(b,list):
        require(len(a)==len(b),'length differs '+path)
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+str(i))
    elif isinstance(b,(float,int)) and not isinstance(b,bool):require(np.isfinite(a) and np.isclose(a,b,atol=1e-9,rtol=1e-9),'number differs '+path)
    else:require(a==b,'value differs '+path)
def main():
    p=argparse.ArgumentParser();p.add_argument('--report',default=str(ROOT/'experiment-result.json'));p.add_argument('--out',default=str(ROOT/'outputs/test-result.json'));a=p.parse_args();r=json.loads(Path(a.report).read_text());expected=run();compare(r,expected);check(True,'complete deterministic report recomputed');d=load_data();X=matrix(d['train']);T=matrix(d['test']);h=r['hand'];close(h['covariance'],[[10/3,8/3],[8/3,10/3]],'manual covariance');close(h['model']['eigenvalues'],[6,2/3],'manual eigenvalues');close(h['scores'],[[3/np.sqrt(2)],[3/np.sqrt(2)],[-3/np.sqrt(2)],[-3/np.sqrt(2)]],'manual scores');close(h['sse'],2,'manual residual SSE');close(h['reconstruction'],[[1.5,1.5],[1.5,1.5],[-1.5,-1.5],[-1.5,-1.5]],'manual reconstruction')
    for method in ['svd','eigh']:
        for k in range(1,7):
            m=fit(X,k,method);W=m['components'];Z=transform(m,X);R=reconstruct(m,X)
            close(W@W.T,np.eye(k),f'{method} k{k} orthonormal axes');close(Z.mean(0),np.zeros(k),f'{method} k{k} scores centered');close(np.sum((X-R)**2),(len(X)-1)*m['eigenvalues'][k:].sum(),f'{method} k{k} discarded trace identity');close((X-R)@W.T,np.zeros((len(X),k)),f'{method} k{k} orthogonal residual');close(R.mean(0),X.mean(0),f'{method} k{k} reconstructed mean');close(reconstruct(m,T),PCA(k,svd_solver='full').fit(X).inverse_transform(PCA(k,svd_solver='full').fit(X).transform(T)),f'{method} k{k} library reconstruction')
    m=fit(X,2);translated=fit(X+3,2);close(m['components'].T@m['components'],translated['components'].T@translated['components'],'translation preserves subspace');close(transform(m,X),transform(translated,X+3),'translation preserves scores');neg=copy.deepcopy(m);neg['components']*=-1;close(reconstruct(m,T),reconstruct(neg,T),'sign flip preserves reconstruction');close(reconstruct(fit(X,6),T),T,'full dimensional exact reconstruction')
    const=fit(np.ones((5,3)),2);check(np.isfinite(const['explained_variance_ratio']).all(),'constant data finite ratio');close(const['explained_variance_ratio'],[0,0,0],'constant data zero variance ratio');close(reconstruct(const,np.ones((2,3))),np.ones((2,3)),'constant data reconstruction')
    for field,error in r['library_errors'].items():check(error<1e-9,'library agreement '+field)
    curve=r['reconstruction_curve'];check(all(curve[i]['train_sse']>=curve[i+1]['train_sse']-1e-9 for i in range(5)),'training SSE nonincreasing');check(r['prediction_counterexample']['variance_ratio_pc1']>.99,'counterexample high explained variance');check(r['prediction_counterexample']['full_accuracy']>.95 and r['prediction_counterexample']['pc1_accuracy']<.6,'counterexample predictive loss');close(r['centering']['incorrect_self_centered_mean'],[0,0],'self centering hides mean shift');check(np.linalg.norm(r['centering']['shifted_test_score_mean'])>.1,'correct transformation retains shift')
    for args,label in [(([[1,2]],1),'one row'),(([[1,np.nan],[2,3]],1),'nonfinite'),(([[1,2],[2,3]],0),'zero k'),(([[1,2],[2,3]],2),'k above centered rank bound'),(([[1,2],[2,3]],True),'boolean k'),(([[1,2],[2,3]],1,'bad'),'unknown method')]:rejects(lambda args=args:fit(*args),label+' rejected')
    rejects(lambda:transform(m,[[1,2]]),'transform feature mismatch');rejects(lambda:inverse_transform(m,[[1]]),'inverse score mismatch')
    with tempfile.TemporaryDirectory() as tmp:
        q=Path(tmp);c=q/'data';c.mkdir()
        for f in (ROOT/'data').glob('*'):
            if f.is_file():(c/f.name).write_bytes(f.read_bytes())
        f=c/'train.csv';lines=f.read_text().splitlines();v=lines[1].split(',');v[1]=str(float(v[1])+.1);lines[1]=','.join(v);f.write_text('\n'.join(lines)+'\n');rejects(lambda:load_data(c),'CSV edit rejected');g=json.loads((c/'generation.json').read_text());g['sha256']['train.csv']=hashlib.sha256(f.read_bytes()).hexdigest();(c/'generation.json').write_text(json.dumps(g));rejects(lambda:load_data(c),'CSV plus digest edit rejected');target=q/'target';target.write_text('keep');link=q/'link';link.symlink_to(target);rejects(lambda:safe_output(link),'symlink rejected');hard=q/'hard';os.link(target,hard);rejects(lambda:safe_output(hard),'hardlink rejected')
    for field in ['hand','reconstruction_curve','prediction_counterexample']:
        z=copy.deepcopy(r)
        if field=='hand':z[field]['sse']+=1
        elif field=='reconstruction_curve':z[field][0]['train_sse']+=1
        else:z[field]['pc1_accuracy']=.99
        rejects(lambda:compare(z,expected),'changed '+field+' rejected')
    report={'status':'passed','checks':len(CHECKS),'python_optimized':not __debug__,'details':CHECKS};write_json(safe_output(a.out),report);print(json.dumps({k:report[k] for k in ['status','checks','python_optimized']}))
if __name__=='__main__':main()
