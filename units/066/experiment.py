"""核选择与资源代价的冻结实验。所有模型选择仅看 validation，计时不参与选模。"""
from pathlib import Path
import argparse,json,pickle,platform,time,warnings
import numpy as np
import sklearn,scipy
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC,LinearSVC
from sklearn.kernel_ridge import KernelRidge
from sklearn.kernel_approximation import Nystroem,RBFSampler
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from common import ROOT,load_data,safe_output,write_json
from kernels import rbf,krr_fit,krr_predict

def candidates(family):
    if family=='linear':return [{'C':float(c)} for c in np.logspace(-4,4,9)]
    if family=='rbf':return [{'C':c,'gamma':g} for c in [.1,1.,10.] for g in [.1,1.,10.]]
    if family=='poly':return [{'C':c,'degree':d,'gamma':.5,'coef0':1.} for c in [.1,1.,10.] for d in [2,3,4]]
    raise ValueError('unknown family')

def make_model(family,params):
    # LinearSVC 与 SVC 的损失实现不同；这是实用模型族比较，不宣称只改变核。
    model=LinearSVC(dual=False,max_iter=20000,tol=1e-8,random_state=66,**params) if family=='linear' else SVC(kernel=family,cache_size=64,tol=1e-8,**params)
    return make_pipeline(StandardScaler(),model)

def xy(part):return np.column_stack([part['x1'],part['x2']]),part['y'].astype(int)

def timing(model,X,repeats=15):
    for _ in range(3):model.predict(X)
    values=[]
    for _ in range(repeats):
        start=time.perf_counter_ns();model.predict(X);values.append((time.perf_counter_ns()-start)/1e6)
    return {'repeats':repeats,'batch_size':len(X),'milliseconds':values,'median_ms':float(np.median(values)),'q25_ms':float(np.quantile(values,.25)),'q75_ms':float(np.quantile(values,.75))}

def run():
    with threadpool_limits(limits=1):return _run()

def _run():
    data=load_data();Xt,yt=xy(data['train']);Xv,yv=xy(data['validation']);Xs,ys=xy(data['test'])
    Xall=np.vstack([Xt,Xv]);yall=np.r_[yt,yv];models={};results={}
    xx,yy=np.meshgrid(np.linspace(-3.8,3.8,125),np.linspace(-.35,.35,105));grid=np.c_[xx.ravel(),yy.ravel()]
    warnings.simplefilter('error',ConvergenceWarning)
    for family in ['linear','rbf','poly']:
        trials=[]
        for params in candidates(family):
            model=make_model(family,params);start=time.perf_counter();model.fit(Xt,yt);elapsed=time.perf_counter()-start
            pred=model.predict(Xv);trials.append({'params':params,'validation_accuracy':float(np.mean(pred==yv)),'validation_prediction':pred.tolist(),'fit_seconds':elapsed,'scaler_mean':model[0].mean_.tolist()})
        # 同分取预先枚举中第一个，不用测试集或计时打破同分。
        best=max(range(len(trials)),key=lambda i:trials[i]['validation_accuracy'])
        model=make_model(family,trials[best]['params']);start=time.perf_counter();model.fit(Xall,yall);elapsed=time.perf_counter()-start;models[family]=model
        pred=model.predict(Xs);fitted=model[-1]
        arrays=[model[0].mean_,model[0].scale_]
        if family=='linear':arrays += [fitted.coef_,fitted.intercept_];sv=None
        else:arrays += [fitted.support_vectors_,fitted.dual_coef_,fitted.intercept_,fitted.support_];sv=int(len(fitted.support_))
        results[family]={'trials':trials,'best_trial':best,'test_accuracy':float(np.mean(pred==ys)),'test_prediction':pred.tolist(),'support_vectors':sv,'selected_prediction_arrays_bytes':sum(a.nbytes for a in arrays),'serialized_pipeline_bytes':len(pickle.dumps(model,protocol=5)),'final_fit_seconds':elapsed,'single_latency':timing(model,Xs[:1]),'batch_latency':timing(model,Xs),'grid_score':model.decision_function(grid).reshape(xx.shape).tolist(),'refit_scaler_mean':model[0].mean_.tolist()}
    # 非调参对照：相同 C、gamma 在不标准化时的变化，预先规定而非新增搜索。
    params=results['rbf']['trials'][results['rbf']['best_trial']]['params'];raw=SVC(kernel='rbf',tol=1e-8,**params).fit(Xall,yall)
    raw_result={'params':params,'test_accuracy':float(raw.score(Xs,ys)),'test_prediction':raw.predict(Xs).tolist(),'grid_score':raw.decision_function(grid).reshape(xx.shape).tolist()}
    # 二点手算：K 的非对角正好为 1/2。
    hX=np.array([[-.5],[.5]]);hy=np.array([1.,-1.]);hz=np.array([[-.5],[0.],[.5]]);gamma=float(np.log(2));hand=krr_fit(hX,hy,.5,gamma)
    lib=KernelRidge(alpha=.5,kernel='rbf',gamma=gamma).fit(hX,hy)
    hand_result={'K':rbf(hX,hX,gamma).tolist(),'coefficient':hand['coefficient'].tolist(),'query':hz.ravel().tolist(),'prediction':krr_predict(hand,hz).tolist(),'sklearn_prediction':lib.predict(hz).tolist(),'max_error':float(np.max(np.abs(krr_predict(hand,hz)-lib.predict(hz))))}
    # 固定 gamma=1 的训练输入核近似；绝不使用测试标签。
    Z=StandardScaler().fit_transform(Xt[:160]);K=rbf(Z,Z,1.);approx=[]
    for m in [8,16,32,64]:
        for method,cls in [('Nystroem',Nystroem),('RFF',RBFSampler)]:
            errs=[]
            for seed in range(5):
                feature=cls(gamma=1.,n_components=m,random_state=seed);F=feature.fit_transform(Z)
                errs.append(float(np.linalg.norm(K-F@F.T,'fro')/np.linalg.norm(K,'fro')))
            approx.append({'method':method,'components':m,'relative_frobenius_errors':errs,'mean_error':float(np.mean(errs)),'std_error':float(np.std(errs,ddof=1)),'feature_matrix_bytes':int(F.nbytes)})
    return {'unit':'066','seed':66021,'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__},'protocol':{'train':240,'validation':120,'test':120,'candidates_per_family':9,'validation_fits_per_family':9,'refits_per_family':1,'tie_break':'first enumerated candidate','blas_threads':1,'timing_note':'wall clock; shared host; not deterministic; no cross-machine speed guarantee','memory_note':'selected ndarray storage and serialized size, NOT peak RSS; SVC uses a bounded kernel cache'},'models':results,'unscaled_rbf':raw_result,'grid':{'x1':xx[0].tolist(),'x2':yy[:,0].tolist()},'hand_krr':hand_result,'approximation':approx,'storage':[{'n':n,'dense_float64_gram_bytes':8*n*n,'feature64_float64_bytes':8*n*64} for n in [100,1000,10000,100000]]}

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(ROOT/'outputs/result.json'));a=p.parse_args();out=safe_output(a.out);report=run();write_json(out,report)
    print(json.dumps({'status':'complete','out':str(out.relative_to(ROOT)) if out.is_relative_to(ROOT) else out.name,'models':{k:{'test_accuracy':v['test_accuracy'],'support_vectors':v['support_vectors']} for k,v in report['models'].items()}},ensure_ascii=False))
if __name__=='__main__':main()
