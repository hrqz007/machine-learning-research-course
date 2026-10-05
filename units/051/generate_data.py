"""Original IID development/repeated samples and separately seeded final test."""
from pathlib import Path
import argparse,csv,io
import numpy as np
from numeric import ROOT,read_json
from protocol import validate

def draw(rng,n,c):
    x=rng.uniform(-1,1,n);mean=np.polynomial.legendre.legval(x,c['true_coefficients']);y=mean+c['noise_sd']*rng.normal(size=n);return x,y,mean

def csv_bytes(header,rows):
    f=io.StringIO(newline='');w=csv.writer(f,lineterminator='\n');w.writerow(header)
    for row in rows:w.writerow([format(float(v),'.17g') if isinstance(v,(float,np.floating)) else v for v in row])
    return f.getvalue().encode()

def generate(c):
    c=validate(c);x,y,mu=draw(np.random.default_rng(c['main_seed']),c['sample_count'],c);main=[[f'M{i+1:04}',float(x[i]),float(y[i]),float(mu[i])] for i in range(len(x))];rng=np.random.default_rng(c['repeated_seed']);rows=[]
    for rep in range(c['repetitions']):
        x,y,mu=draw(rng,c['sample_count'],c)
        rows.extend([[f'R{rep:03}_{i:04}',rep,float(x[i]),float(y[i]),float(mu[i])] for i in range(len(x))])
    return {'development.csv':csv_bytes(['id','x','y','true_mean'],main),'repeated_development.csv':csv_bytes(['id','repetition','x','y','true_mean'],rows)}

def final_test(c):
    c=validate(c);x,y,mu=draw(np.random.default_rng(c['test_seed']),c['test_count'],c)
    return {'id':[f'T{i+1:05}' for i in range(len(x))],'x':x,'y':y,'true_mean':mu}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default=str(ROOT/'data/protocol.json'));p.add_argument('--directory',default=str(ROOT/'outputs/regenerated-data'));a=p.parse_args();out=Path(a.directory).absolute()
    for part in (out,*out.parents):
        if part.is_symlink():raise ValueError('symlink data output')
    if out.resolve()==ROOT/'data':raise ValueError('rebuild into a new directory')
    generated=generate(read_json(a.config));out.mkdir(parents=True,exist_ok=True)
    for name,raw in generated.items():
        path=out/name
        if path.is_symlink() or (path.exists() and (not path.is_file() or path.stat().st_nlink>1)):raise ValueError('unsafe data output')
        path.write_bytes(raw)
    print('Generated two development CSVs; final test is drawn after selection in experiment.py')
