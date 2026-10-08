"""连续Swiss roll与无预设簇的Gaussian对照；颜色变量绝不进入算法输入。"""
from pathlib import Path
import argparse,csv,hashlib,io,json
import numpy as np
SEED=70021

def generated():
    rng=np.random.default_rng(SEED);n=320;t=rng.uniform(1.5*np.pi,4.5*np.pi,n);h=rng.uniform(0,21,n);X=np.column_stack([t*np.cos(t),h,t*np.sin(t)]);s=.5*(t*np.sqrt(1+t*t)+np.arcsinh(t));G=rng.normal(size=(240,6));result={}
    for name,Z,extra in [('roll',X,np.column_stack([t,h,s])),('gaussian',G,None)]:
        out=io.StringIO();w=csv.writer(out,lineterminator='\n');w.writerow(['id']+[f'x{i}' for i in range(Z.shape[1])]+(['t','height','arc_length'] if extra is not None else []))
        for i,row in enumerate(Z):w.writerow([name+'_'+str(i)]+[f'{a:.12f}' for a in row]+([f'{a:.12f}' for a in extra[i]] if extra is not None else []))
        result[name+'.csv']=out.getvalue().encode()
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',default='outputs/generated-data');a=p.parse_args();out=Path(a.directory);out.mkdir(parents=True,exist_ok=True);raw=generated()
    for name,data in raw.items():(out/name).write_bytes(data)
    (out/'generation.json').write_text(json.dumps({'seed':SEED,'kind':'original continuous Swiss roll and single standard Gaussian null control','sha256':{k:hashlib.sha256(v).hexdigest() for k,v in raw.items()}},indent=2)+'\n')
if __name__=='__main__':main()
