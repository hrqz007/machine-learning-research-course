"""重建原创建模数据；记录哈希，禁止依赖网络和外部账户。"""
from pathlib import Path
import argparse,json,hashlib
DATA={'observations':[0.8,1.2,0.9,1.4,1.1,0.7,1.3,1.0], 'noise_sd':1.0,'prior_sd':2.0,'description':'Eight synthetic readings; exact fixed design, not field data.'}
def generate(directory):
    d=Path(directory);d.mkdir(parents=True,exist_ok=True)
    p=d/'model.json';p.write_text(json.dumps(DATA,ensure_ascii=False,indent=2)+'\n')
    (d/'generation.json').write_text(json.dumps({'file':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'randomness':'none; fixed original teaching readings'},indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',default=str(Path(__file__).resolve().parent/'data'));generate(p.parse_args().directory)
