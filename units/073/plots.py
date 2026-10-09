"""从冻结报告与真实数据重建图；颜色辅助区分，坐标注明含义。"""
import argparse,json
import numpy as np
from common import ROOT,load_data,figure_setup

def main():
    p=argparse.ArgumentParser();p.add_argument('--report',default=str(ROOT/'experiment-result.json'));p.add_argument('--directory',default=str(ROOT/'outputs/figures'));a=p.parse_args()
    from pathlib import Path
    d=Path(a.directory);d.mkdir(parents=True,exist_ok=True);r=json.loads(Path(a.report).read_text());data=load_data();plt=figure_setup()
    def save(fig,name):fig.tight_layout();fig.savefig(d/name);plt.close(fig)
    from experiment import matrix,fit
    fig,ax=plt.subplots(1,2,figsize=(9,3.5))
    for axis,name in zip(ax,['development','null']):
        X=matrix(data[name]);labels=fit(X,3).labels_;axis.scatter(*X.T,c=labels,cmap='viridis',s=13);axis.set(xlabel='x0 (synthetic unit)',ylabel='x1 (synthetic unit)',title=name+' / K=3');axis.set_aspect('equal')
    save(fig,'01_structures.png')
    fig,ax=plt.subplots(figsize=(8,3.2));x=[v['k'] for v in r['selection']]
    ax.plot(x,[v['silhouette'] for v in r['selection']],'o-',label='development silhouette',color='#0891b2');ax.plot(x,[v['ari'] for v in r['selection']],'s-',label='source ARI (audit only)',color='#ea580c');ax.plot(x,[v['silhouette'] for v in r['null']],'o--',label='single-Gaussian control silhouette',color='#7c3aed');ax.set(xlabel='K',ylabel='score',xticks=x);ax.legend(fontsize=8);save(fig,'02_metrics.png')
    fig,ax=plt.subplots(figsize=(8,3.2));ax.plot(range(1,21),r['stability'],'o-',color='#0891b2',label='structured K=3; common audit objects');ax.axhline(r['null'][0]['stability_mean'],color='#ea580c',linestyle='--',label='null K=2 mean (same-data anchors)');ax.set(xlabel='80% subsample replicate',ylabel='ARI to frozen partition',ylim=(0,1.04));ax.legend(fontsize=8);save(fig,'03_stability.png')
    fig,ax=plt.subplots(figsize=(8,3.2));names=list(r['downstream_mse']);ax.bar(names,list(r['downstream_mse'].values()),color=['#64748b','#0891b2','#ea580c']);ax.set(ylabel='audit mean squared error',title='Same outcome and untouched audit objects');save(fig,'04_downstream.png')
if __name__=='__main__':main()
