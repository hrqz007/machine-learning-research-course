"""原创二维机制 + sklearn随包Digits真实数据；全程不下载。"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from sklearn.datasets import load_digits
ROOT=Path(__file__).resolve().parent
SEED=81021

def make_data():
    rng=np.random.default_rng(SEED)
    def normal(n):
        group=(rng.random(n)<.2).astype(int)
        means=np.array([[-2.,0.],[2.,0.]])
        scales=np.array([[.38,.45],[.9,.8]])
        return means[group]+rng.normal(size=(n,2))*scales[group],group
    data={}
    for split,n in [('train',600),('calibration',199),('normal_test',300)]:
        x,g=normal(n);data[split]={'x':x.tolist(),'group':g.tolist()}
    # 两种异常机制：桥区新密集模式，以及远离参考样本的上方模式。
    bridge=rng.normal([0.,0.],[.16,.2],size=(100,2))
    far=rng.normal([0.,3.5],[.55,.35],size=(100,2))
    data['anomaly_test']={'x':np.vstack([bridge,far]).tolist(),'kind':['bridge']*100+['far']*100}
    # 污染研究使用额外生成的点；不能把test异常反向塞进训练。
    data['contaminants']={'x':rng.normal([0.,0.],[.16,.2],size=(67,2)).tolist()}
    digits=load_digits();normal_indices=rng.permutation(np.flatnonzero(digits.target==0))
    selected={'train':normal_indices[:90],'calibration':normal_indices[90:135],
              'normal_test':normal_indices[135:],'anomaly_test':np.flatnonzero(digits.target==6)}
    real={key:{'x':(digits.data[idx]/16.).tolist(),'row_id':idx.tolist(),'digit':digits.target[idx].tolist()} for key,idx in selected.items()}
    return {'seed':SEED,'synthetic':data,'digits':real}

def write_data(directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    payload=(json.dumps(make_data(),sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    (directory/'dataset.json').write_bytes(payload)
    manifest={'seed':SEED,'dataset.json':{'sha256':hashlib.sha256(payload).hexdigest(),'bytes':len(payload)},'source':'sklearn.datasets.load_digits; fixed original row indices; values divided by 16'}
    (directory/'generation.json').write_text(json.dumps(manifest,indent=2)+'\n');return manifest
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',default=str(ROOT/'outputs/generated-data'));a=p.parse_args();print(json.dumps(write_data(a.directory),indent=2))
