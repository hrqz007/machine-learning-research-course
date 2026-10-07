"""Analytic geometry, constrained optimization, replay and anti-tamper checks."""
import argparse,copy,hashlib,json,tempfile
from pathlib import Path
import numpy as np
from common import ROOT,load_data,write_json,safe_output
from margin import objective,geometry,solve_primal
from experiment import run,arrays

def test(report):
    checks=[]
    def check(ok,name):
        if not bool(ok):raise RuntimeError('FAILED: '+name)
        checks.append(name)
    def rejects(fn,name):
        try:fn()
        except (ValueError,RuntimeError):check(True,name);return
        raise RuntimeError('DID NOT REJECT '+name)
    X=np.array([[-1,-1],[-1,1],[1,-1],[1,1]],float);y=np.array([-1,-1,1,1]);w=np.array([1.,0.]);g=geometry(X,y,w,0)
    check(np.allclose(g['functional_margin'],1),'hard canonical functional margin');check(g['canonical_full_width']==2,'hard width2');check(objective(X,y,w,0,1)==.5,'hard primal value')
    gg=geometry(X,y,3*w,0);check(np.allclose(g['signed_distance'],gg['signed_distance']),'geometric distance scaling invariant');check(np.allclose(gg['functional_margin'],3),'functional margin scaling changes')
    for m,expected in [(-.4,1.4),(0.,1.),(.4,.6),(1.,0.),(2.,0.)]:check(np.isclose(max(0,1-m),expected),'hinge region '+str(m))
    solved=solve_primal(X,y,100);check(np.allclose(solved['w'],w,atol=1e-7),'analytic w matched');check(abs(solved['b'])<1e-7,'analytic b matched')
    check(report['primal_vs_svc']['objective_difference']<1e-7,'independent objective match');check(report['primal_vs_svc']['max_score_difference']<1e-5,'independent scores match')
    check(report['primal']['max_constraint_violation']<1e-7,'slack feasibility');check(report['primal']['hinge_slack_difference']<1e-6,'minimal slacks equal hinge')
    check(report['selected_index']==max(range(4),key=lambda i:(report['candidates'][i]['validation_accuracy'],-i)),'validation tie break rule')
    check(not report['test_used_for_selection'] and not report['diagnostics_used_for_selection'],'sealed test and diagnostics')
    rows=report['candidates'];check(np.all(np.diff([v['norm_w'] for v in rows])>=-1e-5),'norm monotonic with C');check(np.all(np.diff([sum(v['slack']) for v in rows])<=1e-5),'hinge sum monotonic with C')
    data=load_data();Xt,yt=arrays(data['train'])
    for row in rows:
        C=row['C'];ww=np.array(row['w']);b=row['b'];check(np.isclose(objective(Xt,yt,ww,b,C),row['objective']),'objective reconstruction '+str(C))
        m=yt*(Xt@ww+b);check(np.allclose(np.maximum(0,1-m),row['slack']),'slack reconstruction '+str(C));check(np.all(m[np.array(row['support_indices'])]<=1+1e-5),'support within band '+str(C))
    pred=np.where(np.array(report['test']['scores'])>=0,1,-1);check(np.isclose(np.mean(pred==report['test']['y']),report['test']['accuracy']),'test accuracy reconstruction')
    for C in [0,-1,np.nan,np.inf]:rejects(lambda c=C:solve_primal(X,y,c),'invalid C '+str(C))
    rejects(lambda:solve_primal([[1,np.nan]],[-1],1),'NaN data');rejects(lambda:solve_primal(X,[0,0,1,1],1),'invalid labels');rejects(lambda:geometry(X,y,[0,0],0),'zero normal distance')
    with tempfile.TemporaryDirectory() as td:
        dst=Path(td)
        for src in (ROOT/'data').iterdir():
            if src.is_file():(dst/src.name).write_bytes(src.read_bytes())
        q=dst/'train.csv';q.write_text(q.read_text().replace('train-0000','edited-0000'));meta=json.loads((dst/'generation.json').read_text());meta['sha256']['train.csv']=hashlib.sha256(q.read_bytes()).hexdigest();(dst/'generation.json').write_text(json.dumps(meta));rejects(lambda:load_data(dst),'CSV and digest tampering')
        target=dst/'target';target.write_text('preserve');link=dst/'link';link.symlink_to(target);rejects(lambda:safe_output(link),'unsafe output link');check(target.read_text()=='preserve','target preserved')
    fresh=run()
    def compare(a,b,path='report'):
        if isinstance(a,dict):
            check(set(a)==set(b),'keys '+path)
            for k in a:
                if k!='fit_seconds':compare(a[k],b[k],path+'.'+k)
        elif isinstance(a,list):
            if a and all(isinstance(v,(int,float)) for v in a):check(np.allclose(a,b,rtol=1e-7,atol=1e-7),'array replay '+path)
            else:
                check(len(a)==len(b),'length '+path)
                for i,(v,z) in enumerate(zip(a,b)):compare(v,z,path+'.'+str(i))
        elif isinstance(a,(int,float)) and not isinstance(a,bool):check(np.isclose(a,b,rtol=1e-7,atol=1e-7),'scalar replay '+path)
        else:check(a==b,'value replay '+path)
    compare(report,fresh);bad=copy.deepcopy(report);bad['test']['accuracy']=0.;rejects(lambda:compare(bad,fresh),'tampered report')
    return {'status':'passed','checks':len(checks),'optimized':not __debug__,'fresh_experiment_replayed':True,'check_names':checks}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',default=str(ROOT/'experiment-result.json'));p.add_argument('--out',default=str(ROOT/'outputs/tests.json'));a=p.parse_args();r=test(json.loads(Path(a.report).read_text()));write_json(safe_output(a.out),r);print({k:v for k,v in r.items() if k!='check_names'})
