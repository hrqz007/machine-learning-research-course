"""原创低秩相关数据与高方差低预测信息反例。"""
from pathlib import Path
import argparse,csv,hashlib,io,json
import numpy as np
SEED=69021

def generated():
    rng=np.random.default_rng(SEED);A=np.array([[2,1,.5,0,1,-1],[0,1,2,-1,.5,1.]])
    latent=rng.normal(size=(480,2))*[2,.7];X=latent@A+.15*rng.normal(size=(480,6))+[10,-3,2,0,5,-2]
    signal=rng.normal(size=(600,2))*[10,.4];y=(signal[:,1]>0).astype(int);result={}
    for name,ids,Z,labels in [('train',np.arange(320),X[:320],None),('test',np.arange(320,480),X[320:],None),('signal_train',np.arange(400),signal[:400],y[:400]),('signal_test',np.arange(400,600),signal[400:],y[400:])]:
        s=io.StringIO();w=csv.writer(s,lineterminator='\n');w.writerow(['id']+[f'x{i}' for i in range(Z.shape[1])]+(['y'] if labels is not None else []))
        for j,(i,row) in enumerate(zip(ids,Z)):w.writerow([('signal_' if name.startswith('signal') else 'main_')+str(i)]+[f'{v:.12f}' for v in row]+([int(labels[j])] if labels is not None else []))
        result[name+'.csv']=s.getvalue().encode()
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',default='outputs/generated-data');a=p.parse_args();out=Path(a.directory);out.mkdir(parents=True,exist_ok=True);raw=generated()
    for name,data in raw.items():(out/name).write_bytes(data)
    (out/'generation.json').write_text(json.dumps({'seed':SEED,'kind':'original synthetic low-rank data and variance-prediction counterexample','sha256':{k:hashlib.sha256(v).hexdigest() for k,v in raw.items()}},indent=2)+'\n')
if __name__=='__main__':main()
