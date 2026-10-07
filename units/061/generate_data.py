"""Original, non-personal synthetic observations; each split has its own RNG."""
from pathlib import Path
import argparse,hashlib,io,json
import numpy as np
ROOT=Path(__file__).resolve().parent

def generated():
    files={}
    for split,n,seed in [('train',180,61001),('validation',120,61002),('test',240,61003)]:
        rng=np.random.default_rng(seed);X=rng.uniform(-2.5,2.5,(n,2))
        score=X[:,0]+.8*np.sin(2*X[:,1]);clean=np.where(score>=0,1,-1)
        flipped=np.zeros(n,dtype=int)
        if split=='train':flipped[rng.choice(n,18,replace=False)]=1
        observed=clean*(1-2*flipped)
        out=io.StringIO();out.write('id,x0,x1,y_clean,y_noisy,flipped\n')
        for i,row in enumerate(X):out.write(f'{split}_{i:04d},{row[0]:.17g},{row[1]:.17g},{clean[i]},{observed[i]},{flipped[i]}\n')
        files[split+'.csv']=out.getvalue().encode()
    return files

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',default=str(ROOT/'outputs/generated-data'));a=p.parse_args()
    root=Path(a.directory);root.mkdir(parents=True,exist_ok=True)
    for name,content in generated().items():(root/name).write_bytes(content)
    (root/'generation.json').write_text(json.dumps({'schema':1,'lesson':'061','seeds':{'train':61001,'validation':61002,'test':61003},'generator':'NumPy default_rng PCG64','sha256':{k:hashlib.sha256(v).hexdigest() for k,v in generated().items()}},indent=2)+'\n')
if __name__=='__main__':main()
