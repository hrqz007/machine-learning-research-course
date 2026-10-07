"""Original two-dimensional XOR-like data; independent fixed splits."""
from pathlib import Path
import argparse,csv,hashlib,io,json
import numpy as np
ROOT=Path(__file__).resolve().parent
SIZES={'train':16,'validation':24,'test':80}
def generated():
    rng=np.random.default_rng(65021);result={}
    for split,n in SIZES.items():
        signs=np.tile([[-1,-1],[-1,1],[1,-1],[1,1]],(n//4,1));rng.shuffle(signs)
        X=signs*rng.uniform(.55,1.45,size=(n,2));y=-np.prod(signs,axis=1)
        s=io.StringIO(newline='');w=csv.writer(s,lineterminator='\n');w.writerow(['id','x1','x2','y'])
        for i in range(n):w.writerow([f'{split}-{i:04d}',f'{X[i,0]:.10f}',f'{X[i,1]:.10f}',int(y[i])])
        result[split+'.csv']=s.getvalue().encode()
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',default=str(ROOT/'outputs/regenerated-data'));a=p.parse_args();d=Path(a.directory);d.mkdir(parents=True,exist_ok=True);files=generated()
    for name,raw in files.items():(d/name).write_bytes(raw)
    (d/'generation.json').write_text(json.dumps({'seed':65021,'kind':'original synthetic nonpersonal data','sizes':SIZES,'sha256':{n:hashlib.sha256(b).hexdigest() for n,b in files.items()}},indent=2)+'\n')
