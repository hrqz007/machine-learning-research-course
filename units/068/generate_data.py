"""原创合成质检任务；确定性CSV为实验唯一数据来源，不下载数据。"""
from pathlib import Path
import argparse,csv,hashlib,io,json
import numpy as np
SEED=68021

def generated():
    rng=np.random.default_rng(SEED);n=900
    X=rng.normal(size=(n,6))
    # 前两维影响非线性事件概率；第三维有线性作用；其余为无关传感器。
    score=1.6*X[:,0]*X[:,1]+.9*X[:,2]-.6
    p=1/(1+np.exp(-score));y=(rng.random(n)<p).astype(int)
    # 划分只由预先固定的随机排列决定，没有按模型结果挑选样本。
    order=rng.permutation(n);result={}
    for split,ids in zip(['train','validation','test'],[order[:450],order[450:675],order[675:]]):
        s=io.StringIO();w=csv.writer(s,lineterminator='\n');w.writerow(['id']+[f'x{i}' for i in range(6)]+['y','p_true'])
        for i in ids:w.writerow([f'case_{i:04d}']+[f'{a:.12f}' for a in X[i]]+[int(y[i]),f'{p[i]:.12f}'])
        result[split+'.csv']=s.getvalue().encode()
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',default='outputs/generated-data');a=p.parse_args();out=Path(a.directory);out.mkdir(parents=True,exist_ok=True);raw=generated()
    for name,data in raw.items():(out/name).write_bytes(data)
    (out/'generation.json').write_text(json.dumps({'seed':SEED,'kind':'original synthetic independent binary quality-inspection examples','sha256':{k:hashlib.sha256(v).hexdigest() for k,v in raw.items()}},indent=2)+'\n')
if __name__=='__main__':main()
