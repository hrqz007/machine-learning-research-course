"""Actual time and separately traced Python memory; no hardware performance claim."""
import argparse, json, platform, sys, time, tracemalloc
from pathlib import Path
import numpy as np
import scipy
from threadpoolctl import threadpool_limits, threadpool_info
from experiment import load_inputs, validate_config, optimize, METHODS, finite, checked_square, serialized, safe_write

class DenseRows:
    def __init__(self,A,y):
        self.A=A; self.y=y; self.n,self.d=A.shape; self.counts={'f':0,'g':0,'H':0}
    def evaluate(self,theta,gradient=True,hessian=False,rows=False):
        if rows:raise ValueError('benchmark uses trace=False')
        r=finite(self.A@theta-self.y);f=float(checked_square(r).mean()/2)
        self.counts['f']+=1;out={'f':f}
        if gradient:self.counts['g']+=1;out['g']=finite(self.A.T@r/self.n)
        if hessian:self.counts['H']+=1;out['H']=finite(self.A.T@self.A/self.n)
        return out

def run_benchmark(cfg):
    c=validate_config(cfg);rng=np.random.default_rng(c['benchmark_seed'])
    metadata={'python':sys.version.split()[0],'numpy':np.__version__,'scipy':scipy.__version__,
        'system':platform.system(),'architecture':platform.machine(),'timer':'perf_counter_ns',
        'thread_limit':1,'blas_before_limit':[{k:v for k,v in x.items() if k in ('user_api','internal_api','num_threads','version','threading_layer','architecture')} for x in threadpool_info()]}
    records=[]
    with threadpool_limits(limits=1):
        metadata['blas_during_measurement']=[{k:v for k,v in z.items() if k in ('user_api','internal_api','num_threads','version')} for z in threadpool_info()]
        for d in c['benchmark_dimensions']:
            t0=time.perf_counter_ns();n=2*d
            Q=np.linalg.qr(rng.normal(size=(n,d)),mode='reduced')[0]
            V=np.linalg.qr(rng.normal(size=(d,d)))[0]
            # eigenvalues(A.T A/n)=geomspace(1,20,d), so strictly quadratic SPD.
            A=np.sqrt(n)*Q@np.diag(np.sqrt(np.geomspace(1,20,d)))@V.T
            truth=rng.uniform(-.5,.5,size=d);y=A@truth;initial=np.zeros(d)
            generation_ns=time.perf_counter_ns()-t0
            g0=np.linalg.norm(A.T@(-y)/n);target=c['benchmark_gradient_relative_tolerance']*g0
            def solve(method):
                return optimize(DenseRows(A,y),initial,c,method,trace=False,gtol=float(target),max_updates=c['benchmark_max_updates'])
            # warm-up each algorithm, then rotate position in each repetition.
            for method in METHODS:solve(method)
            runs={m:[] for m in METHODS}
            orders=[]
            for rep in range(c['benchmark_repetitions']):
                order=METHODS[rep%4:]+METHODS[:rep%4];orders.append(list(order))
                for method in order:
                    begin=time.perf_counter_ns();result=solve(method);elapsed=time.perf_counter_ns()-begin
                    success=result['gradient_norm']<=target
                    runs[method].append({'elapsed_ns':elapsed,'target_met':success,'status':result['status'],
                        'gradient_norm':result['gradient_norm'],'updates':result['accepted_updates'],
                        'counts':result['counts'],'algorithm_state_nbytes':result['algorithm_state_nbytes']})
            summaries={}
            for method in METHODS:
                tracemalloc.start();before=tracemalloc.get_traced_memory()[0];tracemalloc.reset_peak()
                measured=solve(method);current,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
                times=[v['elapsed_ns'] for v in runs[method]]
                summaries[method]={'raw_runs':runs[method],'median_ns':float(np.median(times)),
                    'min_ns':min(times),'max_ns':max(times),'all_targets_met':all(z['target_met'] for z in runs[method]),
                    'python_traced_peak_increment_bytes':peak-before,'python_traced_retained_increment_bytes':current-before,
                    'algorithm_state_nbytes':measured['algorithm_state_nbytes'],
                    'temporary_dense_array_scale_bytes':8*d*d if method in ('newton','bfgs') else 8*d,
                    'memory_run_status':measured['status']}
            records.append({'d':d,'n':n,'conditioning_target':20.,'data_generation_ns_excluded':generation_ns,
                'gradient_initial_norm':float(g0),'common_gradient_target':float(target),'max_updates':c['benchmark_max_updates'],
                'orders':orders,'methods':summaries})
    return {'unit':'038','metadata':metadata,'designs':records,'measurement_contract':[
        'input generation and one warmup per method excluded from timed intervals',
        'entire optimize call included: validation, method setup, every requested row f/g/H computation, factorization, line search and final curvature diagnostic',
        'no precomputed Hessian or factorization cache offered to any method',
        'pedagogical BFGS literally forms V C V.T and checks Cholesky every accepted update; these O(d^3) operations are timed, unlike an optimized O(d^2) rank-two implementation',
        'all timed runs trace=False; scalar stopping state and operation counters remain enabled',
        'five default raw repetitions with rotating orders; median/range describe only this run',
        'tracemalloc run is separate from timing; Python-traced allocations are not complete native RSS or BLAS workspace',
        'resident array bytes count parameter, gradient and H/C/history state; exclude data, model, trace, temporaries and native workspace',
        'temporary scale is one representative array size, not a measured peak or full allocation bound',
        'target failures are retained and must not be ranked as successful speedups',
        'strict dense quadratic control permits a specialized direct least-squares alternative, not benchmarked here']}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,default=Path('outputs'));args=ap.parse_args()
    _,_,c=load_inputs();report=run_benchmark(c);safe_write(args.output_dir,'benchmark-result.json',serialized(report))
    print(json.dumps({'status':'measured','dimensions':c['benchmark_dimensions'],'repetitions':c['benchmark_repetitions']}))
if __name__=='__main__':main()
