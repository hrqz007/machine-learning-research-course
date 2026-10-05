"""Deterministic synthetic panel builder. Writes only to a requested new directory."""
from pathlib import Path
import argparse, csv, hashlib, io, json, zipfile
import numpy as np
ROOT=Path(__file__).resolve().parent

def deterministic_npz(path, **arrays):
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for k,v in arrays.items():
            buffer=io.BytesIO();np.lib.format.write_array(buffer,np.asarray(v),allow_pickle=False)
            info=zipfile.ZipInfo(k+'.npy',date_time=(2020,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,buffer.getvalue())

def make(config, directory):
    directory=Path(directory).resolve()
    if directory==ROOT/'data':raise ValueError('preserve shipped data; generate in a new directory')
    directory.mkdir(parents=True,exist_ok=True)
    if any(directory.iterdir()):raise ValueError('generation directory must be empty')
    rng=np.random.default_rng(config['data_seed']);R=config['independent_repetitions']+1
    G,T=config['n_devices'],config['n_days']
    a=rng.normal(0,config['device_sd'],(R,G));xnoise=rng.normal(size=(R,G,T));eps=rng.normal(0,config['noise_sd'],(R,G,T))
    deterministic_npz(directory/'draws.npz',a=a,xnoise=xnoise,eps=eps)
    group=np.repeat(np.arange(G),T);day=np.tile(np.arange(T),G);n=len(day)
    splitrng=np.random.default_rng(config['split_seed']);permutation=splitrng.permutation(n);k=int(n*config['test_fraction'])
    testgroups=np.sort(splitrng.permutation(G)[:int(G*config['test_fraction'])]);ingroup=np.isin(group,testgroups)
    past=day+config['label_delay_days']<=config['fit_cutoff_day'];future=day>=config['future_first_day']
    splits={'random':(np.sort(permutation[k:]),np.sort(permutation[:k])),
      'group':(np.flatnonzero(~ingroup),np.flatnonzero(ingroup)),
      'time':(np.flatnonzero(past),np.flatnonzero(future)),
      'group_time':(np.flatnonzero(past&~ingroup),np.flatnonzero(future&ingroup))}
    deterministic_npz(directory/'splits.npz',**{f'{name}_{role}':idx for name,(tr,te) in splits.items() for role,idx in [('train',tr),('test',te)]})
    with (directory/'split_membership.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['row_id','device','day',*splits])
        for i in range(n):w.writerow([i,int(group[i]),int(day[i]),*[('train' if i in tr else 'test' if i in te else 'unused') for tr,te in splits.values()]])
    x=xnoise[0].ravel()+config['x_drift_per_day']*day
    mu=config['intercept']+config['signal_slope']*x+a[0,group]+config['response_drift_per_day']*day;y=mu+eps[0].ravel()
    with (directory/'panel.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['row_id','device','day','label_available_day','x','y','oracle_mean'])
        for i in range(n):w.writerow([i,int(group[i]),int(day[i]),int(day[i]+config['label_delay_days']),format(x[i],'.17g'),format(y[i],'.17g'),format(mu[i],'.17g')])
    (directory/'hand.csv').write_text('row_id,device,x,y,role\nH1,A,0,1,train\nH2,A,1,3,train\nH3,B,0,2,train\nH4,B,1,4,train\nH5,A,2,5,test\nH6,C,2,7,test\n')
    (directory/'protocol.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
    checks={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.iterdir()) if p.is_file()}
    (directory/'integrity.json').write_text(json.dumps(checks,indent=2)+'\n')
    return checks
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);args=p.parse_args()
    print(json.dumps(make(json.loads((ROOT/'data/protocol.json').read_text()),args.directory),indent=2))
