"""已知三个一维来源的原创混合样本。"""
from pathlib import Path
import argparse,csv,io,hashlib
import numpy as np
from common import ROOT,write_json

def generated():
    rng=np.random.default_rng(7401);result={}
    for name,n in [('train',360),('audit',240)]:
        source=rng.choice(3,n,p=[.35,.4,.25])  # 未观察来源的真实抽样概率。
        x=np.array([-3.,0.,3.])[source]+np.array([.45,.8,.45])[source]*rng.normal(size=n)
        out=io.StringIO();w=csv.writer(out,lineterminator='\n');w.writerow(['id','x','source'])
        for i in range(n):w.writerow([f'{name}-{i:03d}',f'{x[i]:.12f}',int(source[i])])
        result[name+'.csv']=out.getvalue().encode()
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',default=str(ROOT/'outputs/generated-data'));a=p.parse_args();d=Path(a.directory);d.mkdir(parents=True,exist_ok=True);files=generated()
    for n,b in files.items():(d/n).write_bytes(b)
    write_json(d/'generation.json',{'seed':7401,'origin':'original synthetic; no download','sha256':{n:hashlib.sha256(b).hexdigest() for n,b in files.items()}})
if __name__=='__main__':main()
