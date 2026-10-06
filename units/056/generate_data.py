"""Deterministic synthetic rows; no external datasets, subjects, or downloads."""
from pathlib import Path
import argparse
import numpy as np
from common import ROOT, write_json

def generate(directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(5600);n=1800
    group=rng.binomial(1,.4,n)
    X=rng.normal(size=(n,3));X[:,0]+=.7*group;X[:,2]*=40
    logit=-1+1.2*X[:,0]-.8*X[:,1]+1.2*X[:,0]*X[:,1]+.3*group
    p=1/(1+np.exp(-logit));y=rng.binomial(1,p)
    X[rng.random(n)<.08,2]=np.nan
    for name,sl in [('development',slice(0,1200)),('test',slice(1200,None))]:
        a=np.column_stack([np.arange(n)[sl],X[sl],group[sl],y[sl]])
        np.savetxt(directory/(name+'.csv'),a,delimiter=',',header='row_id,x0,x1,x2,group,y',comments='',fmt=['%d','%.17g','%.17g','%.17g','%d','%d'])
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--directory',default=str(ROOT/'outputs/generated-data'));a=ap.parse_args();generate(a.directory)
