"""显式验证，普通 Python 与 python -O 运行同样的检查。"""
from pathlib import Path
import argparse,copy,hashlib,json,os,tempfile
import numpy as np
from common import ROOT,load_data,require,safe_output,write_json
from experiment import run
CHECKS=[]
def check(ok,name):
    if not bool(ok):raise ValueError('CHECK FAILED: '+name)
    CHECKS.append(name)
def close(a,b,name,tol=1e-8):check(np.allclose(a,b,atol=tol,rtol=tol),name)
def rejects(fn,name):
    try:fn()
    except (ValueError,TypeError):CHECKS.append(name);return
    raise ValueError('Expected rejection: '+name)
def compare(a,b,path='report'):
    if isinstance(b,dict):
        require(set(a)==set(b),'keys differ: '+path)
        for k in b:compare(a[k],b[k],path+'.'+k)
    elif isinstance(b,list):
        require(len(a)==len(b),'length differs: '+path)
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+str(i))
    elif isinstance(b,(float,int)) and not isinstance(b,bool):require(np.isfinite(a) and np.isclose(a,b,atol=1e-8,rtol=1e-8),'number differs: '+path)
    else:require(a==b,'value differs: '+path)
def common_checks(report):
    data=load_data();check(len(data)==3,'three data files verified against generator')
    with tempfile.TemporaryDirectory() as td:
        d=Path(td)/'data';d.mkdir()
        for f in (ROOT/'data').iterdir():
            if f.is_file():(d/f.name).write_bytes(f.read_bytes())
        p=next(d.glob('*.csv'));rows=p.read_text().splitlines();cols=rows[1].split(',');cols[1]=str(float(cols[1])+.1);rows[1]=','.join(cols);p.write_text('\n'.join(rows)+'\n')
        rejects(lambda:load_data(d),'changed data rejected')
        m=json.loads((d/'generation.json').read_text());m['sha256'][p.name]=hashlib.sha256(p.read_bytes()).hexdigest();(d/'generation.json').write_text(json.dumps(m));rejects(lambda:load_data(d),'changed data even with repaired digest rejected')
        target=Path(td)/'target';target.write_text('keep');link=Path(td)/'link';link.symlink_to(target);rejects(lambda:safe_output(link),'symlink output rejected');hard=Path(td)/'hard';os.link(target,hard);rejects(lambda:safe_output(hard),'hardlink output rejected')
    expected=run();compare(report,expected);check(True,'full deterministic report recomputed')
    altered=copy.deepcopy(report);altered['seed']+=1;rejects(lambda:compare(altered,expected),'tampered report rejected')
    return data
def specific(r,data):
    from kmeans import fit,lloyd,initialize,predict,squared_distances
    h=r['hand'];close(h['centers'],[[5/3],[10.5]],'manual centers');close(h['inertia'],31/6,'manual final sum');close(h['history'][0]['before'],114,'manual initial objective');close(h['history'][0]['after_update'],65,'manual first update');close(h['history'][0]['after_assign'],45.5,'manual second assignment');close(h['history'][1]['after_update'],31/6,'manual second update')
    check(h['history'][0]['labels_before']==[0,1,1,1,1],'manual first labels');check(h['labels']==[0,0,0,1,1],'manual final labels')
    rX=np.c_[data['restarts']['x1'],data['restarts']['x2']];sets=[(rX,r['random']),(rX,r['kmeans_plus_plus']),(np.array([[0],[2],[3],[10],[11]]),h),(np.array([[0],[0],[1],[10]]),r['empty']),(np.ones((4,2)),r['duplicate'])]
    for idx,(X,m) in enumerate(sets):
        for j,row in enumerate(m['history']):
            check(row['after_update']<=row['before']+1e-8,f'model {idx} update {j} nonincrease');check(row['after_assign']<=row['after_update']+1e-8,f'model {idx} assign {j} nonincrease')
            C=np.array(row['centers_after']);z=np.array(row['labels_before']);close(np.sum((X-C[z])**2),row['after_update'],f'model {idx} independently reconstructed update {j}')
        C=np.array(m['centers']);z=np.array(m['labels']);close(np.sum((X-C[z])**2),m['inertia'],f'model {idx} returned objective');check(np.array_equal(z,predict(m,X)),f'model {idx} final labels nearest center');check(m['converged'],f'model {idx} converged')
        for k in np.unique(z):close(C[k],X[z==k].mean(axis=0),f'model {idx} center {k} is cluster mean')
    check(r['empty']['history'][0]['empty_clusters']==[1],'duplicate seeds create empty cluster');check(r['empty']['effective_k']==3,'empty cluster recovered');close(r['empty']['inertia'],0,'empty rescue objective');check(r['duplicate']['effective_k']==1,'identical points honestly report effective K=1');close(r['duplicate']['inertia'],0,'identical points finite zero cost')
    for init in ['random','k-means++']:
        short=fit(rX,2,init=init,n_init=3);long=fit(rX,2,init=init,n_init=15);close(short['restart_inertias'],long['restart_inertias'][:3],init+' candidate prefix unchanged');check(long['inertia']<=short['inertia']+1e-9,init+' increased search budget cannot lose previous best')
        a=initialize(rX,4,np.random.default_rng(8),init);b=initialize(rX,4,np.random.default_rng(8),init);close(a,b,init+' seeded reproducibility')
    check(max(r['random']['restart_inertias'])>3*min(r['random']['restart_inertias']),'experiment actually exhibits local minima');check(r['library_inertia_difference']<1e-8,'same initialization library agreement')
    limited=fit([[0],[2],[3],[10],[11]],2,init=[[0],[3]],n_init=1,max_iter=1);check(not limited['converged'],'iteration cap reports not converged');check(np.array_equal(limited['labels'],predict(limited,[[0],[2],[3],[10],[11]])),'iteration cap keeps prediction and labels consistent')
    a=fit(rX,2,n_init=10);b=fit(rX*7+np.array([2,-8]),2,n_init=10);close(b['inertia'],49*a['inertia'],'uniform scale squares objective');check(np.array_equal(a['labels'][:,None]==a['labels'],b['labels'][:,None]==b['labels']),'translation uniform scaling preserves partition')
    k1=fit(rX,1);close(k1['centers'][0],rX.mean(axis=0),'K=1 equals mean');kn=fit([[0],[1],[2]],3);close(kn['inertia'],0,'K=n distinct points zero distortion')
    for kw in [{'k':0},{'k':201},{'k':True},{'k':2,'n_init':0},{'k':2,'max_iter':0},{'k':2,'tol':-1},{'k':2,'tol':np.nan},{'k':2,'init':'typo'},{'k':2,'init':[[0,0]],'n_init':1}]:rejects(lambda kw=kw:fit(rX,**kw),'bad parameter '+str(kw))
    for X in [[],[1,2],[[np.nan]],[[np.inf]],[[]]]:rejects(lambda X=X:fit(X,1),'bad matrix '+str(X))
    rejects(lambda:squared_distances([[1]],[[1,2]]),'feature mismatch');rejects(lambda:squared_distances([[1e308]],[[-1e308]]),'overflow rejected')
    check(np.array_equal(predict({'centers':[[0],[2]]},[[1]]),[0]),'ties choose first center')
    altered=copy.deepcopy(r);altered['hand']['history'][0]['after_update']=115;rejects(lambda:compare(altered,r),'increasing objective tamper rejected')
    for row in r['storage_examples']:check(row['broadcast_bytes']==8*row['n']*row['k']*row['d'],'broadcast temporary storage formula')
def main():
    p=argparse.ArgumentParser();p.add_argument('--report',default=str(ROOT/'experiment-result.json'));p.add_argument('--out',default=str(ROOT/'outputs/test-result.json'));a=p.parse_args();r=json.loads(Path(a.report).read_text());data=common_checks(r);specific(r,data);out={'status':'passed','checks':len(CHECKS),'python_optimized':not __debug__,'details':CHECKS};write_json(safe_output(a.out),out);print(json.dumps({k:v for k,v in out.items() if k!='details'}))
if __name__=='__main__':main()
