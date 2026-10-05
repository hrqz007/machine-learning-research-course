"""Three independent predeclared teaching mechanisms, deterministic data archive."""
from pathlib import Path
import argparse,csv,hashlib,io,json,zipfile
import numpy as np
from sklearn.model_selection import KFold,GroupKFold
ROOT=Path(__file__).resolve().parent

def savez(path,**arrays):
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,a in arrays.items():
            b=io.BytesIO();np.lib.format.write_array(b,np.asarray(a),allow_pickle=False)
            i=zipfile.ZipInfo(name+'.npy',(2020,1,1,0,0,0));i.compress_type=zipfile.ZIP_DEFLATED;z.writestr(i,b.getvalue())

def make(c,directory):
    directory=Path(directory).absolute()
    for p in (directory,*directory.parents):
        if p.is_symlink():raise ValueError('symlink generation directory')
    if directory.resolve()==ROOT/'data':raise ValueError('preserve shipped data; use a new directory')
    directory.mkdir(parents=True,exist_ok=True)
    if any(directory.iterdir()):raise ValueError('generation directory must be empty')
    R=c['independent_repetitions'];gc=c['group_experiment'];rng=np.random.default_rng(c['data_seed']);G,m=gc['devices'],gc['rows_per_device'];n=G*m;group=np.repeat(np.arange(G),m)
    latent=rng.normal(size=(R,n,2));effects=rng.normal(0,gc['device_sd'],size=(R,G));site=rng.integers(0,4,size=(R,G));site_rows=site[:,group];eps=rng.normal(0,gc['noise_sd'],size=(R,n));y=1+1.5*latent[:,:,0]-.7*latent[:,:,1]+.3*site_rows+effects[:,group]+eps
    missing=rng.random((R,n,2))<np.array(gc['missing_rate']);contam=rng.random((R,n,2))<gc['contamination_rate'];observed=latent+contam*gc['contamination_shift'];observed[missing]=np.nan
    b=c['selection_experiment'];sx=rng.normal(size=(R,b['rows'],b['noise_features']));sy=rng.normal(size=(R,b['rows']))
    q=c['encoding_experiment'];cat=rng.integers(0,q['categories'],size=(R,q['rows']));cy=rng.normal(0,q['target_sd'],size=(R,q['rows']))
    savez(directory/'draws.npz',group=group,latent=latent,effects=effects,site=site_rows,eps=eps,missing=missing,contamination=contam,numeric=observed,group_y=y,selection_X=sx,selection_y=sy,category=cat,category_y=cy)
    splits={}
    for k,(tr,va) in enumerate(GroupKFold(c['outer_folds']).split(observed[0],groups=group)):
        splits[f'group_{k}_train']=tr;splits[f'group_{k}_valid']=va
    for name,N in [('selection',b['rows']),('encoding',q['rows'])]:
        for k,(tr,va) in enumerate(KFold(c['outer_folds'],shuffle=True,random_state=c['outer_seed']).split(np.arange(N))):splits[f'{name}_{k}_train']=tr;splits[f'{name}_{k}_valid']=va
    savez(directory/'splits.npz',**splits)
    with (directory/'group_example.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['row_id','device','sensor_1','sensor_2','site','y'])
        for i in range(n):w.writerow([i,int(group[i]),*[('' if np.isnan(v) else format(v,'.17g')) for v in observed[0,i]],'S'+str(site_rows[0,i]),format(y[0,i],'.17g')])
    with (directory/'encoding_example.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['row_id','category','y'])
        for i in range(q['rows']):w.writerow([i,'C'+str(cat[0,i]),format(cy[0,i],'.17g')])
    with (directory/'split_membership.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['mechanism','fold','role','row_id'])
        for name in ['group','selection','encoding']:
            for k in range(c['outer_folds']):
                for role in ['train','valid']:
                    for i in splits[f'{name}_{k}_{role}']:w.writerow([name,k,role,int(i)])
    (directory/'hand.csv').write_text('row_id,x,type,y,role\nH1,1,A,1,train\nH2,,A,2,train\nH3,3,B,2,train\nH4,5,B,5,train\nH5,7,C,6,valid\nH6,,B,3,valid\n')
    (directory/'protocol.json').write_text(json.dumps(c,ensure_ascii=False,indent=2)+'\n')
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.iterdir()) if p.is_file()}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();print(json.dumps(make(json.loads((ROOT/'data/protocol.json').read_text()),a.directory),indent=2))
