"""无随机性、无下载的原创模型参数。"""
from pathlib import Path
import argparse,json,hashlib
ROOT=Path(__file__).resolve().parent

def payloads():
    p={'prior':[.6,.4],'transition':[[.85,.15],[.25,.75]],'length':4,'emission':[.1,.9]}
    return {'chain.json':json.dumps(p,indent=2)+'\n','star.json':json.dumps({'leaves':6,'center_prior':[.6,.4],'same_weight':3.,'different_weight':1.},indent=2)+'\n'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',default=str(ROOT/'outputs/generated-data'));a=p.parse_args();out=Path(a.directory);out.mkdir(parents=True,exist_ok=True)
    meta={'kind':'original deterministic teaching model','seed':None,'files':{}}
    for name,content in payloads().items():
        raw=content.encode();(out/name).write_bytes(raw);meta['files'][name]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    (out/'generation.json').write_text(json.dumps(meta,indent=2)+'\n')
if __name__=='__main__':main()
