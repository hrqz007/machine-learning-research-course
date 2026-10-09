"""比较内部、外部、扰动稳定性与冻结下游任务；全部离线。"""
import argparse
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_samples
from common import ROOT, load_data, write_json, safe_output
from validation import silhouette, adjusted_rand

def matrix(data):return np.column_stack([data['x0'],data['x1']])
def fit(X,k,seed=0):return KMeans(n_clusters=k,n_init=10,random_state=seed).fit(X)
def run():
    data=load_data();X=matrix(data['development']);A=matrix(data['audit']);N=matrix(data['null']);rows=[]
    for k in [2,3,4,5,6]:
        model=fit(X,k);labels=model.labels_;sil=silhouette(X,labels)
        rows.append({'k':k,'silhouette':float(sil.mean()),'ari':adjusted_rand(data['development']['source'],labels)})
    chosen=max(rows,key=lambda r:r['silhouette'])['k']  # 只用开发集内部指标选择K。
    model=fit(X,chosen);base=model.predict(A);stabilities=[]
    for seed in range(20):
        rng=np.random.default_rng(seed);idx=rng.choice(len(X),size=int(.8*len(X)),replace=False)
        stabilities.append(adjusted_rand(base,fit(X[idx],chosen,seed).predict(A)))  # 在同一审计点上比较。
    null_rows=[]
    for k in [2,3,4,5,6]:
        m=fit(N,k);scores=[]
        for seed in range(20):
            idx=np.random.default_rng(seed).choice(len(N),size=192,replace=False)
            scores.append(adjusted_rand(m.labels_,fit(N[idx],k,seed).predict(N)))
        null_rows.append({'k':k,'silhouette':float(silhouette(N,m.labels_).mean()),'stability_mean':float(np.mean(scores))})
    y=data['development']['outcome'];ya=data['audit']['outcome'];train_labels=model.labels_
    group_means=np.array([y[train_labels==j].mean() for j in range(chosen)])
    coef=np.linalg.lstsq(np.column_stack([np.ones(len(X)),X]),y,rcond=None)[0]  # 冻结线性基线。
    pred_linear=np.column_stack([np.ones(len(A)),A])@coef
    errors={'global_mean':float(np.mean((ya-y.mean())**2)),'cluster_mean':float(np.mean((ya-group_means[base])**2)),'linear_features':float(np.mean((ya-pred_linear)**2))}
    return {'selection':rows,'chosen_k':chosen,'audit_ari':adjusted_rand(data['audit']['source'],base),'stability':stabilities,'null':null_rows,'downstream_mse':errors,'library_silhouette_max_error':float(np.max(np.abs(silhouette(X,train_labels)-silhouette_samples(X,train_labels)))),'library_ari_error':abs(adjusted_rand(data['development']['source'],train_labels)-adjusted_rand_score(data['development']['source'],train_labels))}
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',default=str(ROOT/'outputs/result.json'));a=parser.parse_args();report=run();write_json(safe_output(a.out),report);print(report)
if __name__=='__main__':main()
