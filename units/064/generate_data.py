"""Original nonpersonal, two-dimensional binary data with one labeled outlier."""
from pathlib import Path
import argparse,csv,hashlib,io,json
import numpy as np
ROOT=Path(__file__).resolve().parent
SIZES={'train':64,'validation':80,'test':160}
def generated():
    rng=np.random.default_rng(64021);result={}
    for split,n in SIZES.items():
        y=np.tile([-1,1],n//2);rng.shuffle(y)
        X=rng.normal(size=(n,2))*[.65,1.0];X[:,0]+=1.1*y
        if split=='train':X[0]=[1.8,.8];y[0]=-1
        s=io.StringIO(newline='');w=csv.writer(s,lineterminator='\n');w.writerow(['id','x1','x2','y'])
        for i in range(n):w.writerow([f'{split}-{i:04d}',f'{X[i,0]:.10f}',f'{X[i,1]:.10f}',int(y[i])])
        result[split+'.csv']=s.getvalue().encode()
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',default=str(ROOT/'outputs/regenerated-data'));a=p.parse_args();d=Path(a.directory);d.mkdir(parents=True,exist_ok=True);files=generated()
    for name,raw in files.items():(d/name).write_bytes(raw)
    (d/'generation.json').write_text(json.dumps({'seed':64021,'kind':'original synthetic nonpersonal data','sizes':SIZES,'outlier':'train-0000 fixed x=(1.8,0.8), y=-1','sha256':{n:hashlib.sha256(b).hexdigest() for n,b in files.items()}},indent=2)+'\n')
