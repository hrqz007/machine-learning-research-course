"""ML057: neighbor implementation, held-out selection, dimensions and local smoothing."""
import argparse,json,time
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from common import ROOT,write_json
from knn import KNN
KS=[1,5,21,61]

def run(data_path=ROOT/'data/draws.npz'):
    a=np.load(data_path);X,y,noise=a['X'],a['y'],a['noise'];tr=slice(0,360);va=slice(360,540);te=slice(540,900)
    classification=[];scalers={}
    for route in ['raw','scaled','scaled_noise20']:
        Z=np.column_stack([X,noise]) if route=='scaled_noise20' else X.copy()
        if route!='raw':
            scaler=StandardScaler().fit(Z[tr]);Z=scaler.transform(Z);scalers[route]=dict(mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist())
        for weights in ['uniform','distance']:
            candidates=[]
            for k in KS:
                m=KNN(k,weights).fit(Z[tr],y[tr]);pv=m.predict_proba(Z[va])[:,1]
                candidates.append(dict(k=k,validation_error=float(np.mean((pv>.5)!=y[va])),validation_probability=pv.tolist()))
            best=min(candidates,key=lambda r:(r['validation_error'],r['k']));m=KNN(best['k'],weights).fit(Z[tr],y[tr]);p=m.predict_proba(Z[te])[:,1]
            reference=KNeighborsClassifier(n_neighbors=best['k'],weights=weights,algorithm='brute').fit(Z[tr],y[tr]);ref=reference.predict_proba(Z[te])[:,1]
            classification.append(dict(route=route,weights=weights,candidates=candidates,selected_k=best['k'],test_error=float(np.mean((p>.5)!=y[te])),test_probability=p.tolist(),library_max_error=float(np.max(np.abs(p-ref)))))
    # Regression has independent draws and an independent selection set.
    xr,yr=a['xr'],a['yr'];regression=[];grid=np.linspace(-3,3,161)[:,None]
    for weights in ['uniform','distance']:
        candidates=[]
        for k in KS:
            m=KNN(k,weights,'regression').fit(xr[:100,None],yr[:100]);pred=m.predict(xr[100:180,None])
            candidates.append(dict(k=k,validation_mse=float(np.mean((pred-yr[100:180])**2)),grid_prediction=m.predict(grid).tolist()))
        best=min(candidates,key=lambda r:(r['validation_mse'],r['k']));m=KNN(best['k'],weights,'regression').fit(xr[:100,None],yr[:100]);pred=m.predict(xr[180:,None])
        ref=KNeighborsRegressor(n_neighbors=best['k'],weights=weights,algorithm='brute').fit(xr[:100,None],yr[:100]).predict(xr[180:,None])
        regression.append(dict(weights=weights,candidates=candidates,selected_k=best['k'],test_prediction=pred.tolist(),test_mse=float(np.mean((pred-yr[180:])**2)),library_max_error=float(np.max(np.abs(pred-ref)))))
    # Finite-repetition decomposition. The identity uses population variance ddof=0.
    rng=np.random.default_rng(5790);predictions={k:[] for k in KS};g=np.linspace(-2.5,2.5,101)[:,None]
    for repetition in range(40):
        xx=rng.uniform(-3,3,(80,1));yy=np.sin(2*xx[:,0])+rng.normal(0,.3,80)
        for k in KS:predictions[k].append(KNN(k,task='regression').fit(xx,yy).predict(g))
    bv=[];truth=np.sin(2*g[:,0])
    for k in KS:
        p=np.array(predictions[k]);bias=float(np.mean((p.mean(0)-truth)**2));variance=float(np.mean(p.var(0,ddof=0)));error=float(np.mean((p-truth)**2))
        bv.append(dict(k=k,squared_bias=bias,variance=variance,mean_squared_error_to_truth=error,predictions=p.tolist()))
    # Controlled noise dimensions use the SAME signal/noise prefix and fixed k=21.
    dimensions=[]
    for q in [0,2,5,10,20]:
        z=np.column_stack([X,noise[:,:q]]);z=StandardScaler().fit(z[tr]).transform(z)
        m=KNN(21).fit(z[tr],y[tr]);p=m.predict(z[te]);dist=np.sqrt(((z[te][0]-z[tr])**2).sum(1))
        dimensions.append(dict(noise_dimensions=q,test_error=float(np.mean(p!=y[te])),nearest=float(dist.min()),median=float(np.median(dist)),nearest_over_median=float(dist.min()/np.median(dist))))
    return dict(unit='057',seed=5700,classification=classification,regression=regression,scalers=scalers,classification_test_y=y[te].tolist(),regression_test_y=yr[180:].tolist(),regression_grid=grid[:,0].tolist(),bias_variance_grid=g[:,0].tolist(),bias_variance=bv,dimensions=dimensions,protocol=dict(classification_split=[360,180,360],regression_split=[100,80,200],ks=KS,tie_break='smaller k',reference='sklearn brute force; random draws have no distance ties',all_routes_fixed_before_test=True))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',default=str(ROOT/'data/draws.npz'));ap.add_argument('--out',default=str(ROOT/'outputs/result.json'));a=ap.parse_args();r=run(a.data);write_json(a.out,r)
    print(json.dumps({'classification':[{k:v for k,v in x.items() if k in ['route','weights','selected_k','test_error','library_max_error']} for x in r['classification']],'regression':[{k:v for k,v in x.items() if k in ['weights','selected_k','test_mse','library_max_error']} for x in r['regression']],'bias_variance':[{k:v for k,v in x.items() if k!='predictions'} for x in r['bias_variance']]},indent=2))
