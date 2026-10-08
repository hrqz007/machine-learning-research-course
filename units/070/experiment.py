"""固定数据的探索性二维映射，非预测泛化评估；完整保存所有配置。"""
import os
os.environ.setdefault('NUMBA_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,importlib.metadata,json,time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE,trustworthiness
from umap import UMAP
from common import ROOT,load_data,write_json,safe_output
from neighborhoods import trust,recall,distance_rank_correlation

def matrix(d):return np.column_stack([d[k] for k in d if k.startswith('x')])
def summarize(X,Z,intrinsic=None):
    result={'trustworthiness_10':trust(X,Z,10),'neighbor_recall_10':recall(X,Z,10),'global_distance_spearman':distance_rank_correlation(X,Z),'library_trust_error':abs(trust(X,Z,10)-trustworthiness(X,Z,n_neighbors=10))}
    if intrinsic is not None:result.update({'intrinsic_neighbor_recall_10':recall(intrinsic,Z,10),'intrinsic_global_distance_spearman':distance_rank_correlation(intrinsic,Z)})
    return result

def run():
    d=load_data();X=matrix(d['roll']);G=matrix(d['gaussian']);intrinsic=np.column_stack([d['roll']['arc_length'],d['roll']['height']]);records=[]
    configs=[{'method':'PCA','seed':None,'parameter':2}]
    configs += [{'method':'t-SNE','seed':seed,'parameter':p} for seed in [0,1] for p in [5,30,80]]
    configs += [{'method':'UMAP','seed':seed,'parameter':n,'min_dist':.1} for seed in [0,1] for n in [5,30,80]]
    configs += [{'method':'UMAP','seed':0,'parameter':30,'min_dist':.8}]
    for c in configs:
        start=time.perf_counter()
        if c['method']=='PCA':Z=PCA(2,svd_solver='full').fit_transform(X)
        elif c['method']=='t-SNE':Z=TSNE(n_components=2,perplexity=c['parameter'],random_state=c['seed'],init='random',learning_rate='auto',max_iter=750,method='exact').fit_transform(X)
        else:Z=UMAP(n_components=2,n_neighbors=c['parameter'],min_dist=c['min_dist'],random_state=c['seed'],n_jobs=1,n_epochs=350).fit_transform(X)
        elapsed=time.perf_counter()-start
        records.append({**c,'fit_seconds':elapsed,'embedding':Z.tolist(),'metrics':summarize(X,Z,intrinsic)})
    null=[]
    for seed in [0,1]:
        Z=TSNE(2,perplexity=30,random_state=seed,init='random',learning_rate='auto',max_iter=750,method='exact').fit_transform(G);null.append({'seed':seed,'embedding':Z.tolist(),'metrics':summarize(G,Z)})
    # 明确构造一个容易手算的邻域破坏样例，n=4,k=1。
    H=np.array([[0.],[1.],[3.],[7.]]);Z=np.array([[0.],[3.],[1.],[7.]])
    return {'unit':'070','seed':70021,'versions':{p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn','umap-learn','numba']},'n_roll':len(X),'n_null':len(G),'neighborhood_k':10,'records':records,'null_control':null,'hand':{'X':H.tolist(),'Z':Z.tolist(),'k':1,'trustworthiness':trust(H,Z,1),'recall':recall(H,Z,1)},'protocol':'all fixed points embedded for exploratory visualization; no predictive train/test claim; color t and intrinsic arc length excluded from fit','scope':'ambient Euclidean neighborhoods and intrinsic known-sheet geometry are separate diagnostics; two seeds do not establish universal stability'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(ROOT/'outputs/result.json'));a=p.parse_args();r=run();write_json(safe_output(a.out),r);print(json.dumps({'records':len(r['records']),'hand':r['hand'],'trust_range':[min(z['metrics']['trustworthiness_10'] for z in r['records']),max(z['metrics']['trustworthiness_10'] for z in r['records'])]},ensure_ascii=False))
if __name__=='__main__':main()
