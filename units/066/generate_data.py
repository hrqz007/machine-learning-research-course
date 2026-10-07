"""原创双环数据。固定生成器是来源真值，不从互联网下载任何数据。"""
from pathlib import Path
import argparse, hashlib, io, json
import numpy as np
ROOT=Path(__file__).resolve().parent

def generated():
    rng=np.random.default_rng(66021);out={};cursor=0
    for split,n in [('train',240),('validation',120),('test',120)]:
        y=np.tile([0,1],n//2);rng.shuffle(y)
        angle=rng.uniform(0,2*np.pi,n);radius=np.where(y==0,1.,.43)+rng.normal(0,.09,n)
        # 人为改变两维单位；标准化必须只用训练输入拟合。
        X=np.column_stack([3*radius*np.cos(angle),.25*radius*np.sin(angle)])
        buf=io.StringIO();buf.write('id,x1,x2,y\n')
        for j in range(n):buf.write(f'{cursor+j},{X[j,0]:.12f},{X[j,1]:.12f},{y[j]}\n')
        cursor+=n;out[split+'.csv']=buf.getvalue().encode()
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',default=str(ROOT/'outputs/generated-data'));a=p.parse_args()
    from common import safe_output,write_json
    hashes={}
    for name,raw in generated().items():safe_output(Path(a.directory)/name).write_bytes(raw);hashes[name]=hashlib.sha256(raw).hexdigest()
    write_json(safe_output(Path(a.directory)/'generation.json'),{'seed':66021,'kind':'original synthetic concentric circles','sha256':hashes})
if __name__=='__main__':main()
