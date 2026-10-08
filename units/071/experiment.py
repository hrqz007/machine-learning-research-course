"""执行手算、初始化、尺度、非球状结构及空簇反例。"""
import argparse,json
import numpy as np
from common import ROOT,load_data,safe_output,write_json
from kmeans import fit,squared_distances

def pack(m):
    return {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in m.items()}
def run():
    from sklearn.cluster import KMeans
    data=load_data();X=lambda key:np.c_[data[key]['x1'],data[key]['x2']]
    hand=fit([[0],[2],[3],[10],[11]],2,init=[[0],[3]],n_init=1)
    r=X('restarts');random=fit(r,2,init='random',n_init=40);plus=fit(r,2,n_init=40)
    same=KMeans(2,init=np.asarray(random['history'][0]['centers_before']),n_init=1,algorithm='lloyd',tol=1e-12,max_iter=300,random_state=71).fit(r)
    scale=X('scale');raw=scale*np.array([1,30]);standard=(raw-raw.mean(axis=0))/raw.std(axis=0,ddof=0)
    models={}
    for name,v in [('original',scale),('x2_times_30',raw),('standardized',standard)]:models[name]=pack(fit(v,2,n_init=30))
    ring=fit(X('rings'),2,n_init=20)
    empty=fit([[0],[0],[1],[10]],3,init=[[0],[0],[10]],n_init=1)
    duplicate=fit(np.ones((4,2)),3,n_init=2)
    return {'unit':'071','seed':71,'hand':pack(hand),'random':pack(random),'kmeans_plus_plus':pack(plus),'library_inertia_difference':float(abs(random['inertia']-same.inertia_)),'scale':models,'rings':pack(ring),'empty':pack(empty),'duplicate':pack(duplicate),'k_curve':[{'k':k,'inertia':fit(r,k,n_init=40)['inertia']} for k in range(1,7)],'storage_examples':[{'n':n,'k':k,'d':d,'distance_bytes':8*n*k,'broadcast_bytes':8*n*k*d} for n,k,d in [(200,2,2),(10000,20,50),(1000000,100,100)]]}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(ROOT/'outputs/result.json'));a=p.parse_args();report=run();write_json(safe_output(a.out),report);print(json.dumps({'hand_inertia':report['hand']['inertia'],'random_min':min(report['random']['restart_inertias']),'random_max':max(report['random']['restart_inertias']),'empty_steps':sum(bool(h['empty_clusters']) for h in report['empty']['history'])}))
