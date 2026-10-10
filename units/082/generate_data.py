"""可重复合成真值、无混合负对照，以及随包Wine真实观测。"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from sklearn.datasets import load_wine
ROOT=Path(__file__).resolve().parent
SEED=82021

def make_data():
    rng=np.random.default_rng(SEED)
    weights=np.array([.45,.35,.2]);means=np.array([[-2.,-.7],[.9,1.6],[2.,-1.2]])
    cov=np.array([[[.5,.22],[.22,.45]],[[.5,-.18],[-.18,.35]],[[.25,.05],[.05,.55]]])
    def sample(n):
        z=rng.choice(3,n,p=weights);x=np.empty((n,2))
        for k in range(3):x[z==k]=rng.multivariate_normal(means[k],cov[k],size=int((z==k).sum()))
        return {'x':x.tolist(),'z':z.tolist()}
    synthetic={name:sample(n) for name,n in [('train',600),('validation',200),('test',400)]}
    synthetic['truth']={'weights':weights.tolist(),'means':means.tolist(),'covariances':cov.tolist()}
    negative={name:{'x':rng.normal(size=(n,2)).tolist()} for name,n in [('train',600),('validation',200),('test',400)]}
    wine=load_wine();indices=rng.permutation(len(wine.data));chunks={'train':indices[:106],'validation':indices[106:142],'test':indices[142:]}
    # 预先固定列0酒精与列6黄酮；标签完全不参与划分、拟合和K选择。
    real={name:{'x':wine.data[ix][:,[0,6]].tolist(),'row_id':ix.tolist(),'cultivar':wine.target[ix].tolist()} for name,ix in chunks.items()}
    return {'seed':SEED,'synthetic':synthetic,'negative':negative,'wine':real,'wine_features':[wine.feature_names[i] for i in [0,6]]}

def write_data(directory):
    out=Path(directory);out.mkdir(parents=True,exist_ok=True);payload=(json.dumps(make_data(),sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    (out/'dataset.json').write_bytes(payload);manifest={'seed':SEED,'dataset.json':{'sha256':hashlib.sha256(payload).hexdigest(),'bytes':len(payload)},'real_source':'sklearn.datasets.load_wine; columns [0,6]; original row indices retained'}
    (out/'generation.json').write_text(json.dumps(manifest,indent=2)+'\n');return manifest
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',default=str(ROOT/'outputs/generated-data'));a=p.parse_args();print(json.dumps(write_data(a.directory),indent=2))
