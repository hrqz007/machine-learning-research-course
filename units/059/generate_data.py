"""Original synthetic classification data; fixed independent row splits."""
from pathlib import Path
import argparse
import numpy as np
from common import ROOT,write_json,require

def generate(directory):
    directory=Path(directory).absolute();require(directory.resolve()!=ROOT/'data','refuse overwrite published data')
    directory.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(59059);n=900
    X=rng.normal(size=(n,4));X[:,2]=X[:,0]+.22*rng.normal(size=n)
    p=np.where(X[:,0]<=-.3,.10,np.where(X[:,1]>.25,.82,.36));y=rng.binomial(1,p)
    for name,ids in [('train',slice(0,450)),('validation',slice(450,675)),('test',slice(675,900))]:
        a=np.column_stack([np.arange(n)[ids],X[ids],y[ids],p[ids]])
        np.savetxt(directory/(name+'.csv'),a,delimiter=',',header='row_id,x0,x1,x2,x3,y,true_probability',comments='',fmt=['%d']+['%.17g']*4+['%d','%.17g'])
    write_json(directory/'generation.json',{'seed':59059,'rows':900,'split':[450,225,225],'features':4,'original_synthetic':True,'noise_proxy':'x2=x0+0.22*normal','label_probability':'x0<=-0.3:0.10; otherwise x1>0.25:0.82; otherwise0.36'})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',default='outputs/new-data');a=p.parse_args();generate(a.directory)
