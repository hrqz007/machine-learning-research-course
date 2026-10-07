"""原创带噪回归教学数据。真函数用于教学评估，绝不输入 GP 的拟合过程。"""
from pathlib import Path
import argparse,hashlib,io
import numpy as np
ROOT=Path(__file__).resolve().parent

def truth(x):return np.sin(1.35*x)+.18*x

def generated():
    rng=np.random.default_rng(67021);out={};cursor=0
    sets=[('train',np.array([-3.,-2.6,-2.1,-1.7,-1.2,-.8,.7,1.1,1.6,2.,2.4,3.])),('interpolation',np.linspace(-3,3,121)),('extrapolation',np.linspace(3.2,6.,121))]
    for name,x in sets:
        f=truth(x);y=f+rng.normal(0,.18,len(x));buf=io.StringIO();buf.write('id,x,f_true,y\n')
        for j in range(len(x)):buf.write(f'{cursor+j},{x[j]:.12f},{f[j]:.12f},{y[j]:.12f}\n')
        cursor+=len(x);out[name+'.csv']=buf.getvalue().encode()
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',default=str(ROOT/'outputs/generated-data'));a=p.parse_args()
    from common import safe_output,write_json
    hashes={}
    for name,raw in generated().items():safe_output(Path(a.directory)/name).write_bytes(raw);hashes[name]=hashlib.sha256(raw).hexdigest()
    write_json(safe_output(Path(a.directory)/'generation.json'),{'seed':67021,'noise_std':.18,'kind':'original synthetic sinusoid plus trend','sha256':hashes})
if __name__=='__main__':main()
