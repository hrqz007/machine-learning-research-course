"""环、异密度和桥接噪声三类完全可重建数据。"""
from pathlib import Path
import argparse,csv,hashlib,io
import numpy as np
from common import ROOT,safe_output,write_json
SEED=7201

def generated():
    rng=np.random.default_rng(SEED);sets={};t=np.linspace(0,2*np.pi,150,endpoint=False)
    X=np.vstack([np.c_[r*np.cos(t),r*np.sin(t)]+rng.normal(0,.018,(150,2)) for r in [.55,1.35]]);sets['rings']=(X,np.repeat([0,1],150))
    dense=rng.normal([-1.3,0],[.085,.085],(100,2));sparse=rng.normal([.6,0],[.48,.48],(130,2));noise=rng.uniform([-2,-1.3],[1.9,1.3],(35,2));sets['unequal']=(np.vstack([dense,sparse,noise]),np.r_[np.zeros(100),np.ones(130),np.full(35,-1)])
    left=rng.normal([-1.5,0],[.17,.17],(90,2));right=rng.normal([1.5,0],[.17,.17],(90,2));bridge=np.c_[np.linspace(-1.1,1.1,13),np.zeros(13)];out=np.array([[0,1.5],[0,-1.5],[2.6,1.5],[-2.6,-1.5]]);sets['bridge']=(np.vstack([left,right,bridge,out]),np.r_[np.zeros(90),np.ones(90),np.full(17,-1)])
    output={}
    for name,(X,y) in sets.items():
        s=io.StringIO(newline='');w=csv.writer(s,lineterminator='\n');w.writerow(['id','x1','x2','generated_group'])
        for i,(x,g) in enumerate(zip(X,y)):w.writerow([name+'-'+str(i),f'{x[0]:.10f}',f'{x[1]:.10f}',int(g)])
        output[name+'.csv']=s.getvalue().encode()
    return output

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',default=str(ROOT/'outputs/generated-data'));a=p.parse_args();d=Path(a.directory);files=generated()
    for name,raw in files.items():safe_output(d/name).write_bytes(raw)
    write_json(safe_output(d/'generation.json'),{'seed':SEED,'columns':['id','x1','x2','generated_group'],'purpose':'synthetic teaching; generated_group withheld from all algorithms','sha256':{n:hashlib.sha256(v).hexdigest() for n,v in files.items()}})
if __name__=='__main__':main()
