"""只把生成标签用于事后说明，不把它作为聚类或参数选择的输入。"""
import argparse,json
import numpy as np
from common import ROOT,load_data,safe_output,write_json
from density_hierarchy import dbscan,agglomerative,cut_k

def pack(m):return {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in m.items()}
def summarize(labels,truth):
    groups=sorted(int(v) for v in set(labels)-{-1});return {'clusters':len(groups),'sizes':[int(np.sum(labels==g)) for g in groups],'noise_fraction':float(np.mean(labels<0)),'generated_contingency':[[int(np.sum((truth==g)&(labels==c))) for c in [-1]+groups] for g in [-1,0,1]],'contingency_rows':[-1,0,1],'contingency_columns':[-1]+groups}
def run():
    from scipy.cluster.hierarchy import linkage
    from sklearn.cluster import KMeans,DBSCAN
    d=load_data();scenarios=[]
    for name,eps in [('rings',.13),('unequal',.13),('bridge',.24)]:
        X=np.c_[d[name]['x1'],d[name]['x2']];truth=d[name]['generated_group'].astype(int);methods={}
        km=KMeans(2,n_init=20,random_state=72).fit_predict(X);methods['kmeans']={'labels':km.tolist(),**summarize(km,truth)}
        for method in ['single','complete','average','ward']:
            Z=linkage(X,method=method);labels=cut_k(Z,2);methods[method]={'labels':labels.tolist(),**summarize(labels,truth)}
        for radius in ([.10,.13,.24] if name=='rings' else [.13,.24,.40]):
            m=dbscan(X,radius,6);lib=DBSCAN(eps=radius,min_samples=6,algorithm='brute').fit(X)
            methods['dbscan_'+str(radius)]={**pack(m),**summarize(m['labels'],truth),'eps':radius,'library_core_equal':bool(np.array_equal(np.flatnonzero(m['core']),lib.core_sample_indices_))}
        scenarios.append({'name':name,'n':len(X),'methods':methods})
    hand_X=np.array([[0],[.1],[.2],[.3],[.55],[1.5]])
    hierarchy_X=np.array([[0],[1],[4],[7]])
    return {'unit':'072','seed':72,'scenarios':scenarios,'hand_density':{'X':hand_X.tolist(),**pack(dbscan(hand_X,.3,4))},'hand_hierarchy':{'X':hierarchy_X.tolist(),'merges':{method:agglomerative(hierarchy_X,method).tolist() for method in ['single','complete','average']}},'complexity':[{'n':n,'dense_float64_bytes':8*n*n,'condensed_float64_bytes':8*n*(n-1)//2} for n in [1000,10000,100000]]}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(ROOT/'outputs/result.json'));a=p.parse_args();r=run();write_json(safe_output(a.out),r);print(json.dumps({'hand':r['hand_density'],'scenario_count':len(r['scenarios'])}))
