"""固定种子生成可审计的离线数据；训练真值与审计真值明确分文件。"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parent
SEEDS=[7,19,31,43,59]

def make_data(seed):
    rng=np.random.default_rng(seed)
    def split(n):
        y=np.r_[np.zeros(n//2,dtype=int),np.ones(n-n//2,dtype=int)];rng.shuffle(y)
        X=rng.normal(size=(n,2));X[:,0]+=1.25*(2*y-1);X[:,1]*=1.6
        return X.tolist(),y.tolist()
    pool,y_pool=split(240);validation,y_validation=split(100);test,y_test=split(600)
    return {'pool_X':pool,'validation_X':validation,'validation_y':y_validation,'test_X':test}, {'pool_y':y_pool,'test_y':y_test}

def generate(directory):
    d=Path(directory);d.mkdir(parents=True,exist_ok=True);records=[]
    for seed in SEEDS:
        visible,truth=make_data(seed)
        for name,data in [(f'seed-{seed}-features.json',visible),(f'seed-{seed}-audit.json',truth)]:
            raw=(json.dumps(data,indent=2)+'\n').encode();(d/name).write_bytes(raw);records.append({'file':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
    (d/'generation.json').write_text(json.dumps({'generator':'generate_data.py','numpy_rng':'default_rng PCG64','seeds':SEEDS,'files':records},indent=2)+'\n');return records
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',default=str(ROOT/'outputs/generated-data'));a=p.parse_args();print(json.dumps(generate(a.directory),indent=2))
