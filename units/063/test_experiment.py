"""Independent identities, finite differences, leakage boundaries and replay checks."""
import argparse,copy,hashlib,json,tempfile,sys
from pathlib import Path
import numpy as np
from scipy.special import expit
from common import ROOT,load_data,safe_output,write_json
from boosting import derivatives,leaf_weight,split_gain,metrics
from experiment import run

def test(report):
    checks=[]
    def check(ok,name):
        if not bool(ok):raise RuntimeError('FAILED: '+name)
        checks.append(name)
    def rejects(fn,name):
        try:fn()
        except (ValueError,RuntimeError):check(True,name);return
        raise RuntimeError('DID NOT REJECT: '+name)
    g,h=derivatives([0,0,1,1],[0,0,0,0]);check(np.array_equal(g,[.5,.5,-.5,-.5]),'hand gradients');check(np.array_equal(h,[.25]*4),'hand Hessians')
    check(np.isclose(leaf_weight(1,.5,0),-2),'unregularized leaf');check(np.isclose(leaf_weight(1,.5,1),-2/3),'regularized leaf')
    check(np.isclose(split_gain(1,.5,-1,.5),2),'unregularized split gain');check(np.isclose(split_gain(1,.5,-1,.5,1),2/3),'regularized split gain')
    for y in [0,1]:
        for f in [-3.,0.,2.]:
            eps=1e-4;loss=lambda s:np.logaddexp(0,s)-y*s
            gg,hh=derivatives([y],[f]);check(np.isclose(gg[0],(loss(f+eps)-loss(f-eps))/(2*eps),atol=1e-8),f'gradient finite difference {y} {f}')
            check(np.isclose(hh[0],(loss(f+eps)-2*loss(f)+loss(f-eps))/eps**2,atol=1e-6),f'Hessian finite difference {y} {f}')
    for args in [(1,-1,0),(1,0,0),(1,.5,-1),(np.nan,1,1)]:rejects(lambda a=args:leaf_weight(*a),'invalid leaf '+str(args))
    for args in [([2],[0]),([0],[np.inf]),([] ,[]),([0,1],[0])]:rejects(lambda a=args:derivatives(*a),'invalid derivative '+str(args))
    check(report['selected_hist_index']==min(range(4),key=lambda i:report['hist_candidates'][i]['tune_log_loss']),'tune chooses histogram candidate')
    check(report['selected_linear_index']==min(range(4),key=lambda i:report['linear_candidates'][i]['tune_log_loss']),'tune chooses linear candidate')
    check(report['test_used_for_selection'] is False,'sealed test declaration');check(report['calibrator']['fit_split']=='calibration','separate calibration')
    check(report['budget']['hist_actual_iterations']<=600,'bounded tree iterations');check(not report['budget']['equal_compute_claim'],'no false compute equivalence')
    for row in report['hist_candidates']:
        check(len(row['train_loss'])==row['iterations']+1,'initial train loss recorded '+str(row['index']));check(len(row['stop_loss'])==row['iterations']+1,'initial stop loss recorded '+str(row['index']))
    # Independently recompute metrics from saved probabilities, not just ranges.
    y=report['test_predictions']['y']
    for key,pkey in [('hist_raw','raw'),('hist_sigmoid','calibrated'),('linear','linear')]:
        prob=np.array(report['test_predictions'][pkey]);check(np.isfinite(prob).all() and ((prob>0)&(prob<1)).all(),'probability bounds '+key)
        for metric,value in metrics(y,prob).items():check(np.isclose(value,report['test'][key][metric],atol=1e-12),'metric recalculation '+key+' '+metric)
    check(np.allclose(report['unknown_category_probabilities'][0],report['unknown_category_probabilities'][1]),'unseen category routes as missing')
    d=load_data();check(sum(len(v['id']) for v in d.values())==1160,'sample inventory')
    # Corrupt both CSV and its digest: deterministic source must still reject it.
    with tempfile.TemporaryDirectory() as td:
        dst=Path(td)
        for src in (ROOT/'data').glob('*'):
            if src.is_file():(dst/src.name).write_bytes(src.read_bytes())
        q=dst/'train.csv';raw=q.read_text().replace('train-0000','tampered-0',1);q.write_text(raw)
        m=json.loads((dst/'generation.json').read_text());m['sha256']['train.csv']=hashlib.sha256(q.read_bytes()).hexdigest();(dst/'generation.json').write_text(json.dumps(m))
        rejects(lambda:load_data(dst),'CSV and digest simultaneous tamper')
        target=dst/'target';target.write_text('preserve');link=dst/'link';link.symlink_to(target)
        rejects(lambda:safe_output(link),'symlink output rejection');check(target.read_text()=='preserve','symlink target preserved')
    fresh=run()
    def compare(a,b,path='report'):
        if isinstance(a,dict):
            check(set(a)==set(b),'keys '+path)
            for k in a:
                if k=='fit_seconds':continue
                compare(a[k],b[k],path+'.'+k)
        elif isinstance(a,list):
            if a and all(isinstance(v,(int,float)) for v in a):check(np.allclose(a,b,rtol=1e-9,atol=1e-10),'replay array '+path)
            else:
                check(len(a)==len(b),'length '+path)
                for i,(v,w) in enumerate(zip(a,b)):compare(v,w,path+'.'+str(i))
        elif isinstance(a,(int,float)) and not isinstance(a,bool):check(np.isclose(a,b,rtol=1e-9,atol=1e-10),'replay scalar '+path)
        else:check(a==b,'replay value '+path)
    compare(report,fresh)
    bad=copy.deepcopy(report);bad['selected_hist_index']=(bad['selected_hist_index']+1)%4
    rejects(lambda:compare(bad,fresh),'tampered report replay rejection')
    return {'status':'passed','checks':len(checks),'optimized':not __debug__,'check_names':checks,'fresh_experiment_replayed':True}

def main():
    p=argparse.ArgumentParser();p.add_argument('--report',default=str(ROOT/'experiment-result.json'));p.add_argument('--out',default=str(ROOT/'outputs/tests.json'));a=p.parse_args();r=test(json.loads(Path(a.report).read_text()));write_json(safe_output(a.out),r);print(json.dumps({k:v for k,v in r.items() if k!='check_names'}))
if __name__=='__main__':main()
