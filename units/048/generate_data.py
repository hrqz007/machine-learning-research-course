"""Create original synthetic teaching data under a predeclared protocol."""
from pathlib import Path
import json, hashlib, itertools
import numpy as np
ROOT=Path(__file__).resolve().parent

def main():
    dest=ROOT/'data'; dest.mkdir(exist_ok=True)
    config={'unit':'048','seed':48019,'repetitions':400,'sigma':0.35,'complexity_n':40,'parameters':list(range(1,10)), 'learning_n':[16,32,64,128,256],'learning_p':[2,4,8],'test_n':256,'beta':[1.0,0.6,0.4,-0.25,0,0,0,0,0],'protocol':'random design, independent repetitions; within each repetition shared nested training prefixes and independent test set; no selection or tuning'}
    rng=np.random.default_rng(config['seed']); shape=(400,256)
    x=rng.uniform(-1,1,shape); eps=rng.normal(0,config['sigma'],shape)
    xt=rng.uniform(-1,1,shape); et=rng.normal(0,config['sigma'],shape)
    ef=rng.normal(0,config['sigma'],(400,40))
    np.savez_compressed(dest/'draws.npz',x=x,eps=eps,xt=xt,et=et,eps_fixed=ef)
    (dest/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
    (dest/'hand.csv').write_text('row,x,y,noise\nH1,-1,1,1\nH2,0,-1,-1\nH3,1,1,1\n')
    (dest/'enumeration.csv').write_text('rep,e1,e2,e3\n'+''.join(f'{r},{a},{b},{c}\n' for r,(a,b,c) in enumerate(itertools.product([-1,1],repeat=3))))
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(dest.iterdir()) if p.is_file()}
    (ROOT/'data_integrity.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(json.dumps({'seed':config['seed'],'data_hashes':hashes}))
if __name__=='__main__':main()
