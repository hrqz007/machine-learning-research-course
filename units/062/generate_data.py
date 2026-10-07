"""Original regression and binary-classification data with fixed separate RNGs."""
from pathlib import Path
import argparse,hashlib,io,json
import numpy as np
ROOT=Path(__file__).resolve().parent

def generated():
    files={}
    for split,n,seed in [('train',160,62001),('validation',100,62002),('test',220,62003)]:
        rng=np.random.default_rng(seed);x=rng.uniform(-3,3,n)
        truth=np.sin(1.6*x)+.25*x
        yr=truth+rng.normal(0,.16,n)
        logit=1.6*np.sin(x)+.35*x;p=1/(1+np.exp(-logit));yc=rng.binomial(1,p)
        out=io.StringIO();out.write('id,x,y_reg,y_class,true_mean,true_probability\n')
        for i in range(n):out.write(f'{split}_{i:04d},{x[i]:.17g},{yr[i]:.17g},{yc[i]},{truth[i]:.17g},{p[i]:.17g}\n')
        files[split+'.csv']=out.getvalue().encode()
    return files

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',default=str(ROOT/'outputs/generated-data'));a=p.parse_args();root=Path(a.directory);root.mkdir(parents=True,exist_ok=True)
    for name,content in generated().items():(root/name).write_bytes(content)
    (root/'generation.json').write_text(json.dumps({'schema':1,'lesson':'062','seeds':{'train':62001,'validation':62002,'test':62003},'generator':'NumPy default_rng PCG64','sha256':{k:hashlib.sha256(v).hexdigest() for k,v in generated().items()}},indent=2)+'\n')
if __name__=='__main__':main()
