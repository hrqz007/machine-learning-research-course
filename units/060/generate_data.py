"""Original iid example and an intentionally group-correlated counterexample."""
from pathlib import Path
import argparse,numpy as np
from common import ROOT,write_json,require

def generate(directory):
    directory=Path(directory).absolute();require(directory.resolve()!=ROOT/'data','refuse overwrite released data');directory.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(60060);n=720;X=rng.normal(size=(n,8));X[:,2]=X[:,0]+.3*rng.normal(size=n)
    p=1/(1+np.exp(-(1.4*X[:,0]+1.8*X[:,1]-.8*X[:,0]*X[:,1])));y=rng.binomial(1,p)
    for name,ids in [('train',slice(0,360)),('test',slice(360,720))]:
        np.savetxt(directory/(name+'.csv'),np.column_stack([np.arange(n)[ids],X[ids],y[ids],p[ids]]),delimiter=',',header='row_id,'+','.join(f'x{k}' for k in range(8))+',y,true_probability',comments='',fmt=['%d']+['%.17g']*8+['%d','%.17g'])
    # Every object has six duplicate measurements. New groups have fresh,
    # independent labels, so identity memorization does not generalize.
    for name,offset in [('group_train',0),('group_test',60)]:
        G=rng.normal(size=(60,3));labels=rng.binomial(1,.5,size=60);a=np.column_stack([np.repeat(np.arange(offset,offset+60),6),np.repeat(G,6,axis=0),np.repeat(labels,6)])
        np.savetxt(directory/(name+'.csv'),a,delimiter=',',header='group_id,x0,x1,x2,y',comments='',fmt=['%d']+['%.17g']*3+['%d'])
    write_json(directory/'generation.json',{'seed':60060,'iid_rows':720,'iid_split':[360,360],'iid_features':8,'group_train_groups':60,'group_test_groups':60,'rows_per_group':6,'group_label_independent_of_features':True,'original_synthetic':True})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',default='outputs/new-data');a=p.parse_args();generate(a.directory)
