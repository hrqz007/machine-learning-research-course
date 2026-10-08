"""固定随机种子并冻结 CSV 字节；labels 只用于解释生成机制，不用于拟合。"""
from pathlib import Path
import argparse,csv,hashlib,io,json
import numpy as np
from common import ROOT,safe_output,write_json
SEED=7101

def generated():
    rng=np.random.default_rng(SEED);sets={}
    centers=np.array([[-3,-1],[-3,1],[3,-1],[3,1]])
    X=np.vstack([c+rng.normal(0,.18,(50,2)) for c in centers]);sets['restarts']=(X,np.repeat(np.arange(4),50))
    y=np.repeat([0,1],120);X=np.c_[np.where(y==0,-2,2)+rng.normal(0,.35,240),rng.normal(0,1.0,240)];sets['scale']=(X,y)
    t=np.linspace(0,2*np.pi,180,endpoint=False);X=np.vstack([np.c_[r*np.cos(t),r*np.sin(t)]+rng.normal(0,.025,(180,2)) for r in [.6,1.5]]);sets['rings']=(X,np.repeat([0,1],180))
    output={}
    for name,(X,y) in sets.items():
        s=io.StringIO(newline='');w=csv.writer(s,lineterminator='\n');w.writerow(['id','x1','x2','generated_group'])
        for i,(x,g) in enumerate(zip(X,y)):w.writerow([name+'-'+str(i),f'{x[0]:.10f}',f'{x[1]:.10f}',int(g)])
        output[name+'.csv']=s.getvalue().encode()
    return output

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',default=str(ROOT/'outputs/generated-data'));a=p.parse_args();d=Path(a.directory);files=generated()
    for name,raw in files.items():safe_output(d/name).write_bytes(raw)
    write_json(safe_output(d/'generation.json'),{'seed':SEED,'columns':['id','x1','x2','generated_group'],'purpose':'synthetic teaching only; generated_group is never a fit input','sha256':{n:hashlib.sha256(v).hexdigest() for n,v in files.items()}})
if __name__=='__main__':main()
