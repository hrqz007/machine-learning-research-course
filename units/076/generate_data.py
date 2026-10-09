"""原创人工参数表和全部联合状态；无外部数据，无随机抽样。"""
from pathlib import Path
import argparse, hashlib, json
from graph_models import joint_from_bn
ROOT = Path(__file__).resolve().parent

def model():
    return {'chain': {'nodes':['A','B','C'], 'edges':[['A','B'],['B','C']], 'cpds':{'A':[0.4],'B':[0.1,0.8],'C':[0.2,0.9]}},
            'collider': {'nodes':['R','S','W'], 'edges':[['R','W'],['S','W']], 'cpds':{'R':[0.2],'S':[0.3],'W':[0.01,0.8,0.9,0.99]}}}

def payloads():
    models = model()  # 这些数是教学设计，不是现实天气估计。
    joints = {name:joint_from_bn(**spec) for name,spec in models.items()}
    return {'models.json': json.dumps(models,ensure_ascii=False,indent=2)+'\n',
            'joint-tables.json':json.dumps(joints,ensure_ascii=False,indent=2)+'\n'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',default=str(ROOT/'outputs/generated-data'));a=p.parse_args()
    out=Path(a.directory);out.mkdir(parents=True,exist_ok=True)
    meta={'kind':'original deterministic teaching tables','seed':None,'files':{}}
    for name,content in payloads().items():
        raw=content.encode();(out/name).write_bytes(raw)
        meta['files'][name]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    (out/'generation.json').write_text(json.dumps(meta,indent=2)+'\n')
if __name__=='__main__':main()
