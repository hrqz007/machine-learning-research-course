"""Generate original CSV fixtures from a validated protocol; no external data."""
from pathlib import Path
import argparse,csv,io,json,math,os
import numpy as np
from numeric import ROOT,read_json
from protocol import validate
from calibration import sigmoid

def regression_draw(rng,n,c):
    x=rng.uniform(c['regression_feature_low'],c['regression_feature_high'],n)
    mu=c['regression_true_intercept']+c['regression_true_slope']*x
    sigma=c['regression_noise_intercept']+c['regression_noise_abs_slope']*abs(x)
    epsilon=rng.normal(size=n);y=mu+sigma*epsilon;shifted=mu+c['regression_noise_shift_factor']*sigma*epsilon
    return x,y,shifted,mu,sigma

def csv_bytes(header,rows):
    stream=io.StringIO(newline='');writer=csv.writer(stream,lineterminator='\n');writer.writerow(header)
    for row in rows:writer.writerow([format(float(v),'.17g') if isinstance(v,(float,np.floating)) else v for v in row])
    return stream.getvalue().encode()

def generate(c):
    c=validate(c);rng=np.random.default_rng(c['classification_seed']);classification=[]
    for role,key,prefix in [('calibration','classification_calibration_count','C'),('evaluation','classification_evaluation_count','E')]:
        n=c[key];x=rng.normal(size=n);base=c['classification_true_intercept']+c['classification_true_slope']*x;eta=sigmoid(base);z=c['classification_raw_logit_factor']*base;y=(rng.random(n)<eta).astype(int)
        classification.extend([[prefix+str(i+1).zfill(5),role,float(x[i]),float(z[i]),int(y[i]),float(eta[i])] for i in range(n)])
    out={'classification.csv':csv_bytes(['id','split','x','raw_logit','y','true_probability'],classification)}
    rng=np.random.default_rng(c['regression_seed'])
    for role,key,prefix in [('train','regression_train_count','T'),('calibration','regression_calibration_count','C'),('evaluation','regression_evaluation_count','E')]:
        x,y,shifted,mu,sigma=regression_draw(rng,c[key],c);rows=[]
        for i in range(len(x)):
            row=[prefix+str(i+1).zfill(5),float(x[i]),float(y[i]),float(mu[i]),float(sigma[i])]
            if role=='evaluation':row.append(float(shifted[i]))
            rows.append(row)
        out['regression_'+role+'.csv']=csv_bytes(['id','x','y','true_mean','noise_sd']+(['shifted_y'] if role=='evaluation' else []),rows)
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default=str(ROOT/'data/protocol.json'));p.add_argument('--directory',default=str(ROOT/'outputs/regenerated-data'));a=p.parse_args();out=Path(a.directory).absolute()
    for part in (out,*out.parents):
        if part.is_symlink():raise ValueError('symlink output directory')
    if out.resolve()==ROOT/'data':raise ValueError('rebuild into outputs/ or a new directory, not original teaching data')
    generated=generate(read_json(a.config));out.mkdir(parents=True,exist_ok=True)
    for name,raw in generated.items():
        pth=out/name
        if pth.is_symlink() or (pth.exists() and (not pth.is_file() or pth.stat().st_nlink>1)):raise ValueError('unsafe output data file')
        pth.write_bytes(raw)
    print('Generated',len(generated),'CSV files')
