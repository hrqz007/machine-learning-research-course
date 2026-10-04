"""Measured diagnostic-run timings, separate from deterministic scientific JSON."""
from pathlib import Path
import argparse
import hashlib
import os
import platform
import time
import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits
import experiment as e


def benchmark(hand,bench,selection,spec,repeats=3):
    h,b,d,s=e.validate(hand,bench,selection,spec)
    e.integer(repeats,'repeats',2,10)
    tasks=[]
    for case in e.CASES:
        X,y=e._arrays(b,case); X=X-X.mean(0);y=y-y.mean()
        for lam in (0.,.1):
            for method in e.METHODS: tasks.append((case,X,y,lam,method))
    X,y=e._arrays(b,'scaled');X=X-X.mean(0);y=y-y.mean();tasks.append(('scaled',X,y,.1,'gd_unsafe'))
    records=[]
    with threadpool_limits(limits=1):
        pools=[{k:p.get(k) for k in ('user_api','internal_api','num_threads','version','architecture')} for p in threadpool_info()]
        for case,X,y,lam,method in tasks:
            start=time.perf_counter_ns();warm=e._solve(X,y,lam,method,s,s['seed']); warm_ns=time.perf_counter_ns()-start
            samples=[]
            for j in range(repeats):
                start=time.perf_counter_ns();r=e._solve(X,y,lam,method,s,s['seed']);elapsed=time.perf_counter_ns()-start
                samples.append({'repeat':j+1,'nanoseconds':elapsed,'status':r['status'],'cost':r['cost'],'final_relative_gradient':r['final']['relative_gradient']})
            vals=[v['nanoseconds'] for v in samples]
            records.append({'case':case,'lambda':lam,'method':method,'warmup_ns':warm_ns,'warmup_status':warm['status'],'samples':samples,
                            'median_ns':float(np.median(vals)),'min_ns':min(vals),'max_ns':max(vals)})
    return {'unit':'041','measurement':'perf_counter_ns; diagnostic solver call includes Gram, eigenspectra, reference solve, all monitoring, trace construction, and unsuccessful attempts; excludes input loading/centering, selection, coverage, plotting, JSON I/O and interpreter startup',
            'protocol':'fixed task order; one recorded warmup per task; repeats execute sequentially; BLAS threads limited to one; no universal speed claim',
            'runtime':{'python':platform.python_version(),'numpy':np.__version__,'machine':platform.machine(),'system':platform.system(),'threadpools':pools},
            'input_hashes':{name:hashlib.sha256((e.ROOT/name).read_bytes()).hexdigest() for name in ['experiment.py','data/config.json','data/benchmark.csv']},
            'repeats':repeats,'records':records}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--repeats',type=int,default=3);a=p.parse_args()
    h,b,d,s=e.load_inputs(); e.integer(a.repeats,'repeats',2,10);e.check_destination(a.output)
    r=benchmark(h,b,d,s,a.repeats);e.atomic_json(a.output,r);print('Measured',len(r['records']),'configurations; all failed statuses retained.')
if __name__=='__main__':main()
