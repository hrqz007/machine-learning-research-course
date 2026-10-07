"""Geometric teaching figures and frozen numerical experiment curves."""
import argparse,json
from pathlib import Path
import numpy as np
from common import ROOT,figure_setup,load_data
from experiment import arrays

def build(r,directory):
    plt=figure_setup();d=Path(directory);d.mkdir(parents=True,exist_ok=True);names=[]
    def save(fig,name):fig.tight_layout();fig.savefig(d/name,bbox_inches='tight');plt.close(fig);names.append(name)
    def points(ax,X,y):
        for label,color,marker in [(-1,'#2563eb','o'),(1,'#dc2626','^')]:ax.scatter(*X[y==label].T,c=color,marker=marker,s=28,label=f'y={label}',alpha=.8)
    def boundary(ax,w,b,color='#334155',label=None,margins=True):
        xx,yy=np.meshgrid(np.linspace(-3.3,3.3,130),np.linspace(-3.1,3.1,130));f=w[0]*xx+w[1]*yy+b
        ax.contour(xx,yy,f,levels=[0],colors=[color],linewidths=1.8)
        if margins:ax.contour(xx,yy,f,levels=[-1,1],colors=[color],linestyles='--',linewidths=.9)
        if label:ax.plot([],[],color=color,label=label)
        ax.set(xlim=(-3.3,3.3),ylim=(-3.1,3.1),xlabel='x1',ylabel='x2');ax.set_aspect('equal',adjustable='box')
    fig,ax=plt.subplots(1,2,figsize=(8.5,3.5));X=np.array(r['hard_example']['X']);y=np.array(r['hard_example']['y'])
    for a,w,title in [(ax[0],[1,0],'Canonical w=(1,0), b=0'),(ax[1],[3,0],'Same boundary; w multiplied by 3')]:
        points(a,X,y);boundary(a,w,0);a.set(title=title);a.legend(fontsize=8)
    save(fig,'01_functional_geometric.png')
    fig,ax=plt.subplots(figsize=(8.2,3.2));m=np.linspace(-2,3,300);ax.plot(m,np.maximum(0,1-m),lw=2.5,color='#2563eb',label='hinge: max(0,1-margin)');ax.plot(m,(m<=0).astype(float),ls='--',color='#d97706',label='0-1 error (tie treated as error)');ax.axvline(0,color='#64748b',ls=':');ax.axvline(1,color='#64748b',ls=':');ax.set(xlabel='Functional margin y f(x)',ylabel='Loss',title='Correct prediction can still incur hinge loss');ax.legend();save(fig,'02_hinge_regions.png')
    X,y=arrays(load_data()['train']);fig,axs=plt.subplots(2,2,figsize=(8.4,6))
    for row,ax in zip(r['candidates'],axs.flat):
        points(ax,X,y);boundary(ax,row['w'],row['b']);sv=np.array(row['support_indices']);ax.scatter(*X[sv].T,s=85,facecolors='none',edgecolors='#475569',linewidths=.8);ax.scatter(*X[0],s=130,marker='*',color='#f59e0b',edgecolor='black');ax.set(title=f"C={row['C']}, SV={row['n_support']}")
    save(fig,'03_c_boundaries.png')
    fig,ax=plt.subplots(1,2,figsize=(8.5,3.4));cs=[v['C'] for v in r['candidates']];ax[0].semilogx(cs,[v['canonical_full_width'] for v in r['candidates']],'-o',color='#2563eb');ax[0].set(xlabel='C',ylabel='2 / ||w||',title='Canonical margin band width');ax[1].semilogx(cs,[sum(v['slack']) for v in r['candidates']],'-o',color='#d97706');ax[1].set(xlabel='C',ylabel='Sum hinge losses',title='Penalizing violations more strongly');save(fig,'04_c_tradeoff.png')
    fig,ax=plt.subplots(figsize=(8.2,3.5));points(ax,X,y)
    for name,col in [('with','#7c3aed'),('without','#0d9488')]:
        v=r['outlier_probe'][name];boundary(ax,v['w'],v['b'],col,name+' designated outlier',margins=False)
    ax.scatter(*X[0],s=150,marker='*',color='#f59e0b',edgecolor='black',label='designated outlier');ax.legend(fontsize=8,loc='upper left',bbox_to_anchor=(1.02,1));ax.set(title='Same selected C; remove only training row 0');save(fig,'05_outlier_probe.png')
    fig,ax=plt.subplots(figsize=(8.2,3.5));points(ax,X,y);s=r['scaling_probe'];boundary(ax,s['raw_units_w'],s['raw_units_b'],'#d97706','x2 multiplied by 100, unscaled',False);boundary(ax,s['standardized_original_w'],s['standardized_original_b'],'#0d9488','training-only standardization',False);ax.legend(fontsize=8,loc='upper left',bbox_to_anchor=(1.02,1));ax.set(title='Both boundaries transformed back to original coordinates');save(fig,'06_scaling_probe.png');return names
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',default=str(ROOT/'experiment-result.json'));p.add_argument('--directory',default=str(ROOT/'outputs/figures'));a=p.parse_args();print(build(json.loads(Path(a.report).read_text()),a.directory))
