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
def same_partition(a,b):
    a=np.asarray(a);b=np.asarray(b);return np.array_equal(a[:,None]==a,b[:,None]==b)
def specific(r,data):
    from density_hierarchy import dbscan,agglomerative,cut_k,distances
    from sklearn.cluster import DBSCAN
    from scipy.cluster.hierarchy import linkage
    h=r['hand_density'];check(h['neighbor_counts']==[4,4,4,5,2,1],'hand neighbor counts include self');check(h['core']==[True,True,True,True,False,False],'hand core mask');check(h['labels']==[0,0,0,0,0,-1],'hand border and noise');close(h['noise_fraction'],1/6,'hand noise fraction')
    X=np.array(h['X']);D=distances(X);close(D,D.T,'distance symmetry');close(np.diag(D),0,'zero distance diagonal')
    m=dbscan([[0],[1]],1,2);check(m['core'].all(),'closed eps boundary included');check(m['clusters']==1,'eps equals distance creates edge');m=dbscan([[0]],.01,1);check(m['core'][0] and m['labels'][0]==0,'self counts even for singleton')
    m=dbscan([[0],[10],[20]],.1,2);check(np.all(m['labels']==-1),'all noise case');m=dbscan([[0],[0],[0]],.1,3);check(m['core'].all() and m['clusters']==1,'duplicates count as observations')
    # 一个非核心边界点同时触及左右核心团，但不能把两团接起来。
    amb=np.array([[-1.3],[-1.2],[-1.1],[-1],[1],[1.1],[1.2],[1.3],[0]])
    a=dbscan(amb,1.05,4);check(a['clusters']==2,'border cannot bridge core components');check(a['ambiguous_border_indices']==[8],'ambiguous shared border logged');check(not a['core'][8],'shared border is not core');check(a['labels'][8]==0,'documented border tie rule')
    for s in r['scenarios']:
        X=np.c_[data[s['name']]['x1'],data[s['name']]['x2']];truth=data[s['name']]['generated_group'].astype(int)
        for name,m in s['methods'].items():
            z=np.asarray(m['labels']);check(len(z)==len(X),'label length '+s['name']+name);close(m['noise_fraction'],np.mean(z<0),'noise fraction '+s['name']+name);check(sum(m['sizes'])+np.sum(z<0)==len(X),'all observations accounted '+s['name']+name);close(np.asarray(m['generated_contingency']).sum(),len(X),'contingency covers all observations '+s['name']+name)
            if name.startswith('dbscan'):
                lib=DBSCAN(eps=m['eps'],min_samples=6,algorithm='brute').fit(X);check(np.array_equal(np.flatnonzero(m['core']),lib.core_sample_indices_),'library core mask '+s['name']+name);check(same_partition(z,lib.labels_),'library partition '+s['name']+name);check(np.array_equal(z<0,lib.labels_<0),'library noise set '+s['name']+name)
                counts=(distances(X)<=m['eps']).sum(axis=1);close(counts,m['neighbor_counts'],'neighbor reconstruction '+s['name']+name);check(np.all(counts[np.asarray(m['core'])]>=6),'core threshold '+s['name']+name)
    ward=linkage(np.array([[0.],[1.],[4.],[7.]]),method='ward');close(ward[-1,2],np.sqrt(50),'Ward height sqrt twice SSE increment');close(ward[-1,2]**2/2,25,'Ward merge SSE increment')
    more=dbscan(np.array(h['X']),.3,5);check(more['core'].tolist()==[False,False,False,True,False,False],'m=5 only one core');check(more['labels'].tolist()==h['labels'],'m=5 membership unchanged but core changed')
    rev=dbscan(amb[::-1],1.05,4);check(rev['clusters']==2,'reorder preserves two core components');check(same_partition(a['labels'][:8],rev['labels'][::-1][:8]),'reorder preserves core partition')
    # 使用无距离平局的小数据，对照独立库的每一步链接值。
    tiny=np.array([[0,0],[.7,.1],[2.9,.4],[7.3,-.2],[11.5,.8]])
    for method in ['single','complete','average']:
        Z=agglomerative(tiny,method);lib=linkage(tiny,method=method);close(Z,lib,'hierarchy matches scipy '+method);check(np.all(np.diff(Z[:,2])>=-1e-12),'monotone merge heights '+method)
        for k in range(1,len(tiny)+1):check(len(np.unique(cut_k(Z,k)))==k,'exact cut count '+method+str(k))
    hand=r['hand_hierarchy']['merges'];close(hand['single'][-1][2],3,'hand single last height');close(hand['complete'][-1][2],7,'hand complete last height');close(hand['average'][-1][2],5,'hand average last height')
    rings=r['scenarios'][0]['methods'];check(rings['dbscan_0.24']['sizes']==[150,150],'ring recovery at chosen scale');check(rings['dbscan_0.1']['noise_fraction']==.5,'outer ring rejected at small scale');check(rings['single']['sizes']==[150,150],'single linkage ring recovery')
    unequal=r['scenarios'][1]['methods'];check(unequal['dbscan_0.13']['clusters']==4,'sparse group fragments');check(unequal['dbscan_0.4']['clusters']==1,'large radius merges density groups')
    bridge=r['scenarios'][2]['methods'];check(max(bridge['single']['sizes'])==196,'single linkage bridge chain');check(bridge['dbscan_0.24']['clusters']==2,'DBSCAN keeps core groups separated')
    for eps in [0,-1,np.nan,np.inf]:rejects(lambda eps=eps:dbscan([[0]],eps,1),'bad eps '+str(eps))
    for m in [0,-1,1.5,True]:rejects(lambda m=m:dbscan([[0]],.1,m),'bad min_samples '+str(m))
    for X in [[],[1,2],[[np.nan]],[[np.inf]]]:rejects(lambda X=X:dbscan(X),'invalid data '+str(X))
    rejects(lambda:agglomerative([[0],[1]],'typo'),'unknown linkage');rejects(lambda:cut_k(np.array([[0,1,1,2]]),3),'cut k too large');rejects(lambda:cut_k([[0,1,1,9]],1),'bad linkage member count');rejects(lambda:cut_k([[0,0,1,2]],1),'duplicate merge IDs');rejects(lambda:distances([[1e308],[-1e308]]),'distance overflow')
    malformed=[[0,1,1,2],[2,3,3,2],[99,99,-1,99]]
    rejects(lambda:cut_k(malformed,2),'invalid linkage tail after selected prefix rejected');rejects(lambda:cut_k(malformed,4),'invalid linkage tail also rejected for singleton cut')
    check(agglomerative([[2]],'single').shape==(0,4),'singleton hierarchy');check(cut_k(np.empty((0,4)),1).tolist()==[0],'singleton cut')
    for c in r['complexity']:check(c['dense_float64_bytes']==8*c['n']**2,'dense memory formula '+str(c['n']))
def main():
    p=argparse.ArgumentParser();p.add_argument('--report',default=str(ROOT/'experiment-result.json'));p.add_argument('--out',default=str(ROOT/'outputs/test-result.json'));a=p.parse_args();r=json.loads(Path(a.report).read_text());data=common_checks(r);specific(r,data);out={'status':'passed','checks':len(CHECKS),'python_optimized':not __debug__,'details':CHECKS};write_json(safe_output(a.out),out);print(json.dumps({k:v for k,v in out.items() if k!='details'}))
if __name__=='__main__':main()
