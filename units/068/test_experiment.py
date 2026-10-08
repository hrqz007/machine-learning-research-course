"""普通与优化模式均执行显式检查；重算报告、独立公式与破坏性负例。"""
from pathlib import Path
import argparse,copy,hashlib,json,os,tempfile
import numpy as np
from sklearn.metrics import log_loss,brier_score_loss
from common import ROOT,require,load_data,safe_output,write_json
from experiment import run,point_losses,calibration,metrics,paired_bootstrap,xy,select
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
    p=argparse.ArgumentParser();p.add_argument('--report',default=str(ROOT/'experiment-result.json'));p.add_argument('--out',default=str(ROOT/'outputs/test-result.json'));a=p.parse_args();r=json.loads(Path(a.report).read_text());expected=run();compare(r,expected);check(True,'complete deterministic report recomputed')
    d=load_data();ids=[set(z['id']) for z in d.values()];check(sum(map(len,ids))==900,'all900 unique IDs');check(all(not(ids[i]&ids[j]) for i in range(3) for j in range(i)),'disjoint splits');check(len(r['candidate_rows'])==9,'exactly9 candidate fits');check(all(sum(z['family']==f for z in r['candidate_rows'])==3 for f in r['finalists']),'three configurations each')
    best=min(r['candidate_rows'],key=lambda z:(z['validation_log_loss'],z['name']));check(r['winner']==best['name'],'winner uses validation metric only');check(r['threshold']==.5,'threshold fixed before test')
    close(point_losses([1,0],[.8,.3]),[-np.log(.8),-np.log(.7)],'hand per-row natural log');close(r['hand']['log_loss'],(-np.log(.8)-np.log(.7))/2,'hand mean logloss');close(r['hand']['brier'],.065,'hand Brier')
    T,y=xy(d['test'])
    for family,item in r['test'].items():
        q=np.array(item['probabilities']);m=item['metrics'];close(m['log_loss'],log_loss(y,q),family+' sklearn logloss');close(m['brier'],brier_score_loss(y,q),family+' sklearn Brier');close(m['accuracy'],np.mean((q>=.5)==y),family+' accuracy');check(sum(z['n'] for z in m['calibration'])==len(y),family+' calibration counts');check(m['accuracy_wilson_95'][0]<=m['accuracy']<=m['accuracy_wilson_95'][1],family+' Wilson encloses point');check(item['serialized_model_bytes']>0 and item['median_batch_seconds']>0,family+' measured costs positive');check(sum(z['n'] for z in item['slices'].values())==225,family+' slices partition');check(len(item['failures'])==12 and all(item['failures'][i]['log_loss']>=item['failures'][i+1]['log_loss'] for i in range(11)),family+' failures sorted');check(all(z['id'] in ids[2] for z in item['failures']),family+' failures only test rows')
    zero=paired_bootstrap(y,np.array(r['test']['logistic']['probabilities']),np.array(r['test']['logistic']['probabilities']));close(zero['percentile_95'],[0,0],'paired identical predictions zero interval')
    c=calibration(np.array([0,1]),np.array([0.,1.]));check(c[0]['n']==1 and c[-1]['n']==1,'endpoint1 in last bin');check(c[2]['event_rate'] is None,'empty bin explicit None');check(metrics(np.zeros(3),np.array([.1,.2,.3]))['roc_auc'] is None,'single-class slice no fabricated AUC')
    for y0,p0,label in [([0],[1.1],'probability above1'),([0],[float('nan')],'NaN'),([2],[.4],'nonbinary label'),([0,1],[.4],'length mismatch'),([],[],'empty arrays')]:rejects(lambda y0=y0,p0=p0:point_losses(y0,p0),label+' rejected')
    rejects(lambda:paired_bootstrap(y,np.ones(len(y))*.5,np.ones(len(y))*.4,repeats=2),'insufficient bootstrap count')
    with tempfile.TemporaryDirectory() as tmp:
        q=Path(tmp);c=q/'data';c.mkdir()
        for f in (ROOT/'data').glob('*'):
            if f.is_file():(c/f.name).write_bytes(f.read_bytes())
        f=c/'train.csv';lines=f.read_text().splitlines();v=lines[1].split(',');v[1]=str(float(v[1])+.1);lines[1]=','.join(v);f.write_text('\n'.join(lines)+'\n');rejects(lambda:load_data(c),'CSV edit rejected');g=json.loads((c/'generation.json').read_text());g['sha256']['train.csv']=hashlib.sha256(f.read_bytes()).hexdigest();(c/'generation.json').write_text(json.dumps(g));rejects(lambda:load_data(c),'CSV plus digest edit rejected')
        target=q/'target';target.write_text('keep');link=q/'link';link.symlink_to(target);rejects(lambda:safe_output(link),'symlink rejected');hard=q/'hard';os.link(target,hard);rejects(lambda:safe_output(hard),'hardlink rejected')
    for field in ['winner','test','paired_vs_logistic']:
        altered=copy.deepcopy(r)
        if field=='winner':altered[field]='invented'
        elif field=='test':altered[field]['forest']['metrics']['log_loss']+=.1
        else:altered[field]['forest']['percentile_95'][0]+=.1
        rejects(lambda:compare(altered,expected),'changed '+field+' rejected')
    report={'status':'passed','checks':len(CHECKS),'python_optimized':not __debug__,'details':CHECKS};write_json(safe_output(a.out),report);print(json.dumps({k:report[k] for k in ['status','checks','python_optimized']}))
if __name__=='__main__':main()
