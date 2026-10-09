"""原创合成数据；固定种子，不访问网络。"""
from pathlib import Path
import argparse, csv, io, hashlib
import numpy as np
from common import ROOT, write_json

def generated():
    rng=np.random.default_rng(7301)  # 独立生成器，不改变其他程序的随机状态。
    result={}
    for name in ['development','audit','null']:
        n=240  # 每份数据含240个彼此不同的人工对象。
        z=rng.integers(0,3,n)  # 生成来源，仅用于外部核验。
        centers=np.array([[-3.,0.],[0.,3.],[3.,0.]])
        X=rng.normal(size=(n,2))*0.65+centers[z]
        if name=='null': X=rng.normal(size=(n,2))  # 单个圆形高斯，不预设混合来源。
        y=2.*X[:,0]-X[:,1]+rng.normal(size=n)*0.5  # 下游连续响应，不输入聚类。
        out=io.StringIO();writer=csv.writer(out,lineterminator='\n')
        writer.writerow(['id','x0','x1','source','outcome'])
        for i in range(n):writer.writerow([f'{name}-{i:03d}',*map(lambda v:f'{v:.12f}',X[i]),int(z[i]),f'{y[i]:.12f}'])
        result[name+'.csv']=out.getvalue().encode()
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',default=str(ROOT/'outputs/generated-data'));a=parser.parse_args()
    directory=Path(a.directory);directory.mkdir(parents=True,exist_ok=True);files=generated()
    for name,raw in files.items():(directory/name).write_bytes(raw)
    write_json(directory/'generation.json',{'seed':7301,'origin':'original synthetic; no download','sha256':{k:hashlib.sha256(v).hexdigest() for k,v in files.items()}})
if __name__=='__main__':main()
