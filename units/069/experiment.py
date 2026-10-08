"""PCA方差最大化/重构等价性、SVD库对照、尺度和预测信号反例。"""
import argparse,importlib.metadata,json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from common import ROOT,load_data,write_json,safe_output
from pca import fit,transform,reconstruct

def matrix(d):return np.column_stack([d[k] for k in d if k.startswith('x')])
def encode(m):return {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in m.items()}
def run():
    d=load_data();X=matrix(d['train']);T=matrix(d['test']);hand=np.array([[2.,1],[1,2],[-1,-2],[-2,-1]])
    h=fit(hand,1);hc=hand-hand.mean(0);cov=hc.T@hc/3;recon=reconstruct(h,hand)
    model=fit(X,2);eig=fit(X,2,'eigh');lib=PCA(n_components=2,svd_solver='full').fit(X);proj=model['components'].T@model['components'];libproj=lib.components_.T@lib.components_
    errs=[]
    for k in range(1,7):
        m=fit(X,k);tr=float(np.sum((X-reconstruct(m,X))**2));te=float(np.sum((T-reconstruct(m,T))**2));errs.append({'k':k,'train_sse':tr,'train_mean_squared_distance':tr/len(X),'test_mean_squared_distance':te/len(T),'discarded_eigenvalue_sum':float(m['eigenvalues'][k:].sum()),'cumulative_variance_ratio':float(m['explained_variance_ratio'][:k].sum())})
    # 改单位与标准化是不同建模假设。原始第三列乘100，拟合前仅训练集学习缩放。
    unit_scale=np.array([1,1,100,1,1,1]);changed=X*unit_scale;raw=fit(changed,2);scaler=StandardScaler().fit(changed);standardized=fit(scaler.transform(changed),2)
    # 未中心化SVD寻找穿过原点的空间，非平移后的PCA空间。
    _,_,vt=np.linalg.svd(X,full_matrices=False);uncentered=vt[:2];unc_sse=float(np.sum((X-X@uncentered.T@uncentered)**2))
    # 特意构造均值漂移以展示泄漏：正确转换仍用训练均值，不能用测试均值重置坐标。
    shift=np.array([3,0,0,0,0,0]);shifted=T+shift;correct=transform(model,shifted);wrong=(shifted-shifted.mean(0))@model['components'].T
    S=matrix(d['signal_train']);U=matrix(d['signal_test']);y=d['signal_train']['y'].astype(int);u=d['signal_test']['y'].astype(int)
    sig=fit(S,1);all_model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000)).fit(S,y);pc_model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000)).fit(transform(sig,S),y)
    full=all_model.predict(U);one=pc_model.predict(transform(sig,U))
    return {'unit':'069','seed':69021,'versions':{p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn']},'hand':{'X':hand.tolist(),'covariance':cov.tolist(),'model':encode(h),'scores':transform(h,hand).tolist(),'reconstruction':recon.tolist(),'sse':float(np.sum((hand-recon)**2))},'model':encode(model),'library_errors':{'projector':float(np.max(abs(proj-libproj))),'eigenvalues':float(np.max(abs(model['eigenvalues'][:2]-lib.explained_variance_))),'reconstruction':float(np.max(abs(reconstruct(model,T)-lib.inverse_transform(lib.transform(T))))),'eigh_projector':float(np.max(abs(proj-eig['components'].T@eig['components'])))},'reconstruction_curve':errs,'scaling':{'unit_multiplier':unit_scale.tolist(),'raw_components':raw['components'].tolist(),'standardized_components':standardized['components'].tolist(),'raw_ratio':raw['explained_variance_ratio'].tolist(),'standardized_ratio':standardized['explained_variance_ratio'].tolist(),'scaler_mean':scaler.mean_.tolist(),'scaler_scale':scaler.scale_.tolist()},'centering':{'centered_rank2_sse':errs[1]['train_sse'],'uncentered_rank2_sse':unc_sse,'train_mean':model['mean'].tolist(),'shifted_test_score_mean':correct.mean(0).tolist(),'incorrect_self_centered_mean':wrong.mean(0).tolist()},'prediction_counterexample':{'variance_ratio_pc1':float(sig['explained_variance_ratio'][0]),'axis':sig['components'][0].tolist(),'full_accuracy':float(np.mean(full==u)),'pc1_accuracy':float(np.mean(one==u)),'test_n':len(u),'full_predictions':full.tolist(),'pc1_predictions':one.tolist()},'limitations':'PCA optimizes Euclidean reconstruction, not predictive accuracy; component signs and repeated-eigenvalue bases are nonunique'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(ROOT/'outputs/result.json'));a=p.parse_args();r=run();write_json(safe_output(a.out),r);print(json.dumps({'hand_sse':r['hand']['sse'],'library_errors':r['library_errors'],'counterexample':{k:r['prediction_counterexample'][k] for k in ['variance_ratio_pc1','full_accuracy','pc1_accuracy']}},ensure_ascii=False))
if __name__=='__main__':main()
