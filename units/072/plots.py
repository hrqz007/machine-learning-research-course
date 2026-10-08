"""相同坐标范围展示算法差异；噪声固定为灰色叉号。"""
from pathlib import Path
import argparse,json
import numpy as np
from common import ROOT,load_data,figure_setup,safe_output
from density_hierarchy import distances

def make(report,directory):
    from scipy.cluster.hierarchy import dendrogram,linkage
    plt=figure_setup();directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);data=load_data();get=lambda key:np.c_[data[key]['x1'],data[key]['x2']]
    def save(fig,name):fig.tight_layout();fig.savefig(safe_output(directory/name),bbox_inches='tight');plt.close(fig)
    def scatter(a,X,labels,title):
        labels=np.asarray(labels);keep=labels>=0;a.scatter(X[keep,0],X[keep,1],c=labels[keep],cmap='tab10',s=11,vmin=0,vmax=9);a.scatter(X[~keep,0],X[~keep,1],color='#8b8b8b',marker='x',s=17);a.set(title=title,xlabel='x1',ylabel='x2');a.set_aspect('equal')
    for i,s in enumerate(report['scenarios']):
        X=get(s['name']);fig,ax=plt.subplots(2,3,figsize=(10.4,6.2));names=['kmeans','single','complete','ward','dbscan_0.13','dbscan_0.24']
        for a,name in zip(ax.ravel(),names):
            m=s['methods'][name];scatter(a,X,m['labels'],name+' | noise '+str(round(m['noise_fraction']*100))+'%')
        save(fig,f'0{i+1}_{s["name"]}_comparison.png')
    fig,ax=plt.subplots(1,3,figsize=(10.5,3.3));X=np.array(report['hand_hierarchy']['X'])
    for a,method in zip(ax,['single','complete','average']):
        dendrogram(np.array(report['hand_hierarchy']['merges'][method]),labels=['0','1','4','7'],ax=a,color_threshold=0,above_threshold_color='#2563eb');a.axhline(3.2,color='#dc2626',ls='--');a.set_ylim(0,max(a.get_ylim()[1],3.6));a.set(title=method,xlabel='Observation value',ylabel='Linkage height')
    save(fig,'04_dendrograms.png')
    fig,ax=plt.subplots(1,2,figsize=(10,3.1));h=report['hand_density'];x=np.array(h['X']).ravel();core=np.array(h['core']);labels=np.array(h['labels']);a=ax[0]
    a.scatter(x[core],np.zeros(core.sum()),s=100,c='#2563eb',label='Core');a.scatter(x[(~core)&(labels>=0)],np.zeros(sum((~core)&(labels>=0))),s=50,c='#ef8a27',label='Border');a.scatter(x[labels<0],np.zeros(sum(labels<0)),s=60,c='gray',marker='x',label='Noise')
    a.plot([0,.6],[.3,.3],color='#2563eb',lw=2);a.text(.3,.36,'eps neighborhood of 0.3',ha='center',fontsize=9)
    for v,c in zip(x,h['neighbor_counts']):a.text(v,-.16,str(c),ha='center',fontsize=9)
    a.set(title='Counts include the query point',xlabel='x (count below point)',yticks=[],ylim=(-.4,.6));a.legend(loc='upper right',fontsize=8)
    for name in ['rings','unequal','bridge']:
        D=distances(get(name));r=np.sort(D,axis=1)[:,5];ax[1].plot(np.linspace(0,1,len(r)),np.sort(r),label=name)
    ax[1].set(xlabel='Sorted fraction of observations',ylabel='Distance to 6th point (self included)',title='A global threshold spans different densities');ax[1].legend();save(fig,'05_core_and_neighbor_distance.png')
    fig,ax=plt.subplots(1,3,figsize=(10.5,3.3));s=report['scenarios'][1]
    for a,eps in zip(ax,[.13,.24,.4]):
        m=s['methods']['dbscan_'+str(eps)];scatter(a,get('unequal'),m['labels'],f'eps={eps}: {m["clusters"]} clusters')
    save(fig,'06_unequal_density_sweep.png')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',default=str(ROOT/'experiment-result.json'));p.add_argument('--directory',default=str(ROOT/'outputs/figures'));a=p.parse_args();make(json.loads(Path(a.report).read_text()),a.directory)
