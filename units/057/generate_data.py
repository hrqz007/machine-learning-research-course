"""Generate the complete deterministic, independent-row teaching experiment."""
import argparse
from pathlib import Path
import numpy as np
from common import ROOT

def generate(directory):
    d=Path(directory);d.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(5700)
    X=rng.uniform(-2,2,(900,2));y=(X[:,1]>.55*np.sin(2*X[:,0])).astype(int);flip=rng.random(900)<.10;y=np.where(flip,1-y,y)
    noise=rng.normal(size=(900,20));X[:,1]*=40
    xr=rng.uniform(-3,3,380);yr=np.sin(2*xr)+rng.normal(0,.3,380)
    np.savez(d/'draws.npz',X=X,y=y,noise=noise,xr=xr,yr=yr)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',default=str(ROOT/'outputs/generated-data'));a=p.parse_args();generate(a.directory)
