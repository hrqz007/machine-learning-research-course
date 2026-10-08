"""普通与优化模式均执行显式检查；重算报告、独立公式与破坏性负例。"""
from pathlib import Path
import argparse,copy,hashlib,json,os,tempfile
import numpy as np
from sklearn.manifold import trustworthiness
from common import ROOT,require,load_data,safe_output,write_json
from experiment import run,matrix,summarize
from neighborhoods import trust,recall,ordering,distance_rank_correlation
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
    p=argparse.ArgumentParser();p.add_argument('--report',default=str(ROOT/'experiment-result.json'));p.add_argument('--out',default=str(ROOT/'outputs/test-result.json'));a=p.parse_args();r=json.loads(Path(a.report).read_text());expected=run();compare(r,expected);check(True,'complete seeded embeddings and metrics recomputed');d=load_data();X=matrix(d['roll']);G=matrix(d['gaussian']);intr=np.column_stack([d['roll']['arc_length'],d['roll']['height']]);check(len(r['records'])==14,'all14 presets retained');check(len(r['null_control'])==2,'both null seeds retained');check(r['n_roll']==320 and r['n_null']==240,'data sizes');check(len(set(d['roll']['id'])&set(d['gaussian']['id']))==0,'unique dataset IDs')
    H=np.array(r['hand']['X']);Z=np.array(r['hand']['Z']);close(trust(H,Z,1),.5,'manual rank penalty4 trust0.5');close(recall(H,Z,1),0,'manual no shared nearest neighbors');close(trust(H,H,1),1,'identity trust1');close(recall(H,H,1),1,'identity recall1');close(trust(H,Z,1),trustworthiness(H,Z,n_neighbors=1),'manual library trust')
    for i,row in enumerate(r['records']):
        z=np.array(row['embedding']);check(z.shape==(320,2) and np.isfinite(z).all(),str(i)+' finite320x2 embedding');m=summarize(X,z,intr)
        for k,v in m.items():close(v,row['metrics'][k],str(i)+' independently recomputed '+k)
        check(0<=m['trustworthiness_10']<=1 and 0<=m['neighbor_recall_10']<=1,str(i)+' local metrics bounded');check(m['library_trust_error']<1e-12,str(i)+' library exact trust');check(-1<=m['global_distance_spearman']<=1,str(i)+' rank correlation bounded');check(row['fit_seconds']>0,str(i)+' actual nonzero runtime');close(trust(X,z+100,10),m['trustworthiness_10'],str(i)+' translation invariant metric');close(recall(X,z*3,10),m['neighbor_recall_10'],str(i)+' scale invariant neighbor recall')
    for row in r['null_control']:
        z=np.array(row['embedding']);check(z.shape==(240,2),'null shape '+str(row['seed']));close(summarize(G,z)['trustworthiness_10'],row['metrics']['trustworthiness_10'],'null independently checked'+str(row['seed']))
    # 弧长导数sqrt(1+t²)与参数曲线的速度一致。
    t=np.array([5.,8.,12.]);f=lambda x:.5*(x*np.sqrt(1+x*x)+np.arcsinh(x));derivative=(f(t+1e-5)-f(t-1e-5))/(2e-5);check(np.allclose(derivative,np.sqrt(1+t*t),atol=1e-8),'intrinsic arc length derivative');close(np.sqrt(d['roll']['x0']**2+d['roll']['x2']**2),d['roll']['t'],'roll radial coordinate');close(d['roll']['x1'],d['roll']['height'],'roll height retained')
    for args,label in [((H,Z,0),'zero k'),((H,Z,2),'k at half n'),((H,Z,True),'boolean k'),((H,Z[:3],1),'row mismatch'),(([[0],[np.nan],[2]],[[0],[1],[2]],1),'nonfinite')]:rejects(lambda args=args:trust(*args),label+' rejected')
    rejects(lambda:distance_rank_correlation(np.ones((4,2)),np.ones((4,2))),'constant distance correlation rejected')
    with tempfile.TemporaryDirectory() as tmp:
        q=Path(tmp);c=q/'data';c.mkdir()
        for f in (ROOT/'data').glob('*'):
            if f.is_file():(c/f.name).write_bytes(f.read_bytes())
        f=c/'roll.csv';lines=f.read_text().splitlines();v=lines[1].split(',');v[1]=str(float(v[1])+.1);lines[1]=','.join(v);f.write_text('\n'.join(lines)+'\n');rejects(lambda:load_data(c),'CSV edit rejected');g=json.loads((c/'generation.json').read_text());g['sha256']['roll.csv']=hashlib.sha256(f.read_bytes()).hexdigest();(c/'generation.json').write_text(json.dumps(g));rejects(lambda:load_data(c),'CSV plus digest edit rejected');target=q/'target';target.write_text('keep');link=q/'link';link.symlink_to(target);rejects(lambda:safe_output(link),'symlink rejected');hard=q/'hard';os.link(target,hard);rejects(lambda:safe_output(hard),'hardlink rejected')
    for field in ['hand','records','null_control']:
        z=copy.deepcopy(r)
        if field=='hand':z[field]['trustworthiness']=1
        elif field=='records':z[field][0]['embedding'][0][0]+=1
        else:z[field][0]['metrics']['trustworthiness_10']=1
        rejects(lambda:compare(z,expected),'changed '+field+' rejected')
    report={'status':'passed','checks':len(CHECKS),'python_optimized':not __debug__,'details':CHECKS};write_json(safe_output(a.out),report);print(json.dumps({k:report[k] for k in ['status','checks','python_optimized']}))
if __name__=='__main__':main()
