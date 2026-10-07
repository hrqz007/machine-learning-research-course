"""Original nonpersonal tabular benchmark; every split is generated independently."""
from pathlib import Path
import argparse,csv,hashlib,io,json
import numpy as np
ROOT=Path(__file__).resolve().parent
SIZES={'train':480,'stop':120,'tune':160,'calibration':160,'test':240}
def generated():
    rng=np.random.default_rng(63021);result={}
    for split,n in SIZES.items():
        x1=rng.normal(size=n);x2=rng.uniform(-2,2,n);category=rng.integers(0,3,n)
        missing=rng.random(n)<.18
        # The missingness flag is observable and informative in this toy population.
        logit=1.3*np.sin(1.8*x1)+1.0*(x2>0)+np.array([-.9,.1,.8])[category]+.7*missing-.6
        y=rng.binomial(1,1/(1+np.exp(-logit)))
        s=io.StringIO(newline='');w=csv.writer(s,lineterminator='\n');w.writerow(['id','x1','x2','category','y'])
        for i in range(n):w.writerow([f'{split}-{i:04d}',f'{x1[i]:.10f}','nan' if missing[i] else f'{x2[i]:.10f}',int(category[i]),int(y[i])])
        result[split+'.csv']=s.getvalue().encode()
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',default=str(ROOT/'outputs/regenerated-data'));a=p.parse_args();d=Path(a.directory);d.mkdir(parents=True,exist_ok=True)
    files=generated()
    for name,raw in files.items():(d/name).write_bytes(raw)
    (d/'generation.json').write_text(json.dumps({'seed':63021,'kind':'original synthetic nonpersonal data','sizes':SIZES,'sha256':{n:hashlib.sha256(b).hexdigest() for n,b in files.items()}},indent=2)+'\n')
if __name__=='__main__':main()
