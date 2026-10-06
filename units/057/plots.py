"""Rebuild interpretable views from the fixed ML057 protocol."""
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from common import ROOT,figure_setup
from knn import KNN

def make_figures(r,directory):
    plt=figure_setup();d=Path(directory);d.mkdir(parents=True,exist_ok=True);paths=[];a=np.load(ROOT/'data/draws.npz')
    def save(fig,name):
        fig.tight_layout();p=d/name;fig.savefig(p);plt.close(fig);paths.append(p)
    fig,ax=plt.subplots(figsize=(7,3));x=np.array([0,2,5]);ax.scatter(x,[0,0,0],s=90,color='#276882');ax.scatter([1],[0],marker='*',s=220,color='#ba8752');ax.set(xlim=(-.7,5.7),ylim=(-.7,.7),yticks=[],xlabel='One-dimensional feature x',title='Query x=1; training x=(0,2,5), regression y=(0,4,10)')
    for xx,yy in zip(x,[0,4,10]):ax.text(xx,.16,f'y={yy}',ha='center')
    for xx in x:ax.plot([1,xx],[-.15,-.15],lw=1,color='#7c8a92')
    ax.text(1,-.45,'Distances: 1, 1, 4.  Uniform k=2 predicts 2.',ha='center');save(fig,'01_neighbors.png')
    fig,axes=plt.subplots(1,3,figsize=(10,3.3),sharey=True)
    for ax,route in zip(axes,['raw','scaled','scaled_noise20']):
        for row in r['classification']:
            if row['route']==route:ax.plot([v['k'] for v in row['candidates']],[v['validation_error'] for v in row['candidates']],'o-',label=row['weights'])
        ax.set(title=route,xlabel='k',xscale='log');ax.legend(fontsize=8)
    axes[0].set_ylabel('Validation error rate');save(fig,'02_validation.png')
    fig,axes=plt.subplots(1,2,figsize=(9,4));xx,yy=np.meshgrid(np.linspace(-2,2,100),np.linspace(-2,2,100));Q=np.column_stack([xx.ravel(),40*yy.ravel()]);X=a['X'][:360];y=a['y'][:360]
    for ax,route in zip(axes,['raw','scaled']):
        row=next(v for v in r['classification'] if v['route']==route and v['weights']=='uniform');Z=X;query=Q
        if route=='scaled':sc=StandardScaler().fit(X);Z=sc.transform(X);query=sc.transform(Q)
        p=KNN(row['selected_k']).fit(Z,y).predict(query).reshape(xx.shape)
        ax.contourf(xx,yy,p,levels=[-.5,.5,1.5],colors=['#d7e8f3','#f7e1c7'],alpha=.8)
        ax.scatter(X[:,0],X[:,1]/40,c=y,cmap='coolwarm',s=9,edgecolor='none');ax.set(title=f"{route}, validation-selected k={row['selected_k']}",xlabel='x0',ylabel='x1 / 40 (display units)')
    save(fig,'03_regions.png')
    fig,ax=plt.subplots(figsize=(7,3.5));g=np.array(r['regression_grid']);ax.plot(g,np.sin(2*g),'k--',label='True mean sin(2x)');ax.scatter(a['xr'][:100],a['yr'][:100],s=9,color='#b7bec6',label='Training labels')
    for row in r['regression'][0]['candidates']:ax.plot(g,row['grid_prediction'],label='uniform k='+str(row['k']),alpha=.8,lw=1)
    ax.set(xlabel='x',ylabel='Regression prediction',title='Increasing the neighborhood smooths and can erase structure');ax.legend(ncol=3,fontsize=8);save(fig,'04_regression.png')
    fig,ax=plt.subplots(figsize=(7,3.5));rows=r['bias_variance'];ks=[v['k'] for v in rows]
    for key,label in [('squared_bias','Squared bias'),('variance','Variance'),('mean_squared_error_to_truth','Error to true mean')]:ax.plot(ks,[v[key] for v in rows],'o-',label=label)
    ax.set(xscale='log',xlabel='k (40 independently redrawn training sets)',ylabel='Average squared quantity',title='Finite-repetition decomposition; observation noise excluded');ax.legend();save(fig,'05_bias_variance.png')
    fig,axes=plt.subplots(1,2,figsize=(8.8,3.3));rows=r['dimensions'];q=[v['noise_dimensions'] for v in rows]
    axes[0].plot(q,[v['test_error'] for v in rows],'o-',color='#276882');axes[0].set(xlabel='Added noise dimensions',ylabel='Test error',title='Fixed k=21, standardized features')
    axes[1].plot(q,[v['nearest_over_median'] for v in rows],'o-',color='#ba8752');axes[1].set(xlabel='Added noise dimensions',ylabel='Nearest / median distance',title='One fixed test query, 360 references',ylim=(0,1));save(fig,'06_dimensions.png')
    return paths
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',default=str(ROOT/'experiment-result.json'));p.add_argument('--directory',default=str(ROOT/'outputs/figures'));a=p.parse_args();print('\n'.join(map(str,make_figures(json.loads(Path(a.report).read_text()),a.directory))))
