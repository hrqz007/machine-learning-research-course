"""四个观测、两个真实潜因子的原创线性Gaussian模型。"""
from pathlib import Path
import argparse,csv,io,hashlib
import numpy as np
from common import ROOT,write_json
TRUE_W=np.array([[1.4,.2],[.9,-.6],[.1,1.2],[-.7,.8]])  # 行是观测，列是潜因子。

def generated():
    rng=np.random.default_rng(7501);result={}
    for name,n,noise in [('train',600,[.25]*4),('audit',300,[.25]*4),('hetero_train',600,[.1,.4,.8,1.2]),('hetero_audit',300,[.1,.4,.8,1.2])]:
        z=rng.normal(size=(n,2));x=z@TRUE_W.T+rng.normal(size=(n,4))*np.array(noise)  # 标准差逐列乘噪声。
        out=io.StringIO();w=csv.writer(out,lineterminator='\n');w.writerow(['id','x0','x1','x2','x3','z0','z1'])
        for i in range(n):w.writerow([f'{name}-{i:03d}',*[f'{v:.12f}' for v in x[i]],*[f'{v:.12f}' for v in z[i]]])
        result[name+'.csv']=out.getvalue().encode()
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',default=str(ROOT/'outputs/generated-data'));a=p.parse_args();d=Path(a.directory);d.mkdir(parents=True,exist_ok=True);files=generated()
    for n,b in files.items():(d/n).write_bytes(b)
    write_json(d/'generation.json',{'seed':7501,'origin':'original synthetic; no download','true_W':TRUE_W.tolist(),'isotropic_noise_variance':.0625,'sha256':{n:hashlib.sha256(b).hexdigest() for n,b in files.items()}})
if __name__=='__main__':main()
