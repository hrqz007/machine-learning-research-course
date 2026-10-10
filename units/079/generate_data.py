"""重建固定教学模型；没有下载、隐藏随机状态或现实个人数据。"""
from pathlib import Path
import argparse, hashlib, json
ROOT=Path(__file__).resolve().parent

def models():
    return {'correlated':{'x':[1.0],'A':[[1.,1.]],'prior_mean':[0.,0.],'prior_cov':[[1.,0.],[0.,1.]],'noise_cov':[[.25]]},'independent':{'x':[1.,-1.],'A':[[1.,0.],[0.,1.]],'prior_mean':[0.,0.],'prior_cov':[[1.,0.],[0.,1.]],'noise_cov':[[.25,0.],[0.,.25]]}}

def generate(directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);records=[]
    for name,value in models().items():
        raw=(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode();path=directory/(name+'.json');path.write_bytes(raw)
        records.append({'file':path.name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
    (directory/'generation.json').write_text(json.dumps({'generator':'generate_data.py','randomness':'none','files':records},indent=2)+'\n')
    return records
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',default=str(ROOT/'outputs/generated-data'));a=p.parse_args();print(json.dumps(generate(a.directory),indent=2))
