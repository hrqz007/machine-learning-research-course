"""Generate original synthetic lessons without downloads. Never overwrite shipped data."""
from pathlib import Path
import argparse
import numpy as np
from common import ROOT, write_json, require

def make_data():
    rng = np.random.default_rng(58058)
    X = rng.uniform(-2, 2, size=(480, 2))
    p = np.where(X[:, 0] <= 0, .12, np.where(X[:, 1] <= .35, .30, .88))
    y = rng.binomial(1, p)
    rx = rng.uniform(-3, 3, size=(360, 1))
    mu = np.where(rx[:, 0] <= -1, -1., np.where(rx[:, 0] <= 1, 2., .5))
    ry = mu + rng.normal(0, .35, size=360)
    return {'classification_train':np.column_stack([X[:320],y[:320]]),
            'classification_test':np.column_stack([X[320:],y[320:]]),
            'regression_train':np.column_stack([rx[:240],ry[:240]]),
            'regression_test':np.column_stack([rx[240:],ry[240:]])}

def write_data(directory):
    directory=Path(directory).absolute()
    require(directory.resolve() != ROOT/'data', 'do not overwrite shipped data; choose outputs/data')
    directory.mkdir(parents=True, exist_ok=True)
    for name, arr in make_data().items():
        target=directory/(name+'.csv'); require(not target.exists(), 'data output already exists')
        header='x0,x1,y' if name.startswith('classification') else 'x0,y'
        np.savetxt(target, arr, delimiter=',', header=header, comments='', fmt='%.17g')
    write_json(directory/'generation.json', {'seed':58058,'kind':'original synthetic data','classification_rows':[320,160],'regression_rows':[240,120]})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',default=str(ROOT/'outputs/data'));args=p.parse_args();write_data(args.directory)
