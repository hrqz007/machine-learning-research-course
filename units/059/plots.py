"""Six figures regenerated entirely from released data and result records."""
from pathlib import Path
import argparse,json
import numpy as np
from sklearn.tree import plot_tree
from common import ROOT,figure_setup
from experiment import load,fit
from pruning import export_tree,predict_probability

def build(report,directory):
    plt=figure_setup();directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    def save(fig,name):fig.tight_layout();fig.savefig(directory/name,bbox_inches='tight');plt.close(fig)
    colors=['#2563eb','#d97706','#16836b'];alpha=np.linspace(0,.28,200)
    f,a=plt.subplots(figsize=(9,4))
    for r,l,c in zip([.14,.32,.5],[4,2,1],colors):a.plot(alpha,r+alpha*l,color=c,label=f'{l} leaves: risk={r:.2f}')
    a.axvline(.09,color='#777',ls=':');a.axvline(.18,color='#777',ls=':');a.set(xlabel='Complexity penalty alpha',ylabel='Risk + alpha x leaves',title='A hand-computable pruning path');a.legend();save(f,'01_cost_lines.png')
    path=report['path'];x=[r['alpha'] for r in path];f,ax=plt.subplots(1,2,figsize=(10,4));ax[0].step(x,[r['n_leaves'] for r in path],where='post',color=colors[0]);ax[0].set(xlabel='alpha',ylabel='Number of leaves');ax[1].plot(x,[r['risk'] for r in path],color=colors[1],label='Training Gini risk');ax[1].plot(x,[r['validation_brier'] for r in path],color=colors[2],label='Validation Brier');ax[1].axvline(report['choices']['post_alpha'],ls=':',color='black');ax[1].set(xlabel='alpha',ylabel='Criterion value');ax[1].legend();save(f,'02_pruning_path.png')
    f,a=plt.subplots(figsize=(8,4));matrix=np.array([r['validation_brier'] for r in report['pre_grid']]).reshape(6,3);im=a.imshow(matrix,cmap='YlGnBu',aspect='auto');a.set_xticks(range(3),[1,5,15]);a.set_yticks(range(6),[1,2,3,4,6,'unlimited']);a.set(xlabel='Minimum leaf samples',ylabel='Maximum depth',title='Pre-pruning validation Brier (lower is better)')
    for i in range(6):
        for j in range(3):a.text(j,i,f'{matrix[i,j]:.3f}',ha='center',va='center',color='white' if matrix[i,j]>.25 else 'black')
    f.colorbar(im,ax=a);save(f,'03_pre_grid.png')
    X,y=load('train');m=fit(X,y);tree=export_tree(m);pre=report['choices']['pre'];pm=fit(X,y,max_depth=pre['max_depth'],min_samples_leaf=pre['min_samples_leaf']);axis=np.linspace(-2,2,101);xx,yy=np.meshgrid(axis,axis);probe=np.column_stack([xx.ravel(),yy.ravel(),xx.ravel(),np.zeros(xx.size)])
    ps=[m.predict_proba(probe)[:,1],pm.predict_proba(probe)[:,1],predict_probability(tree,probe,path[report['choices']['post_index']]['leaves'])];f,ax=plt.subplots(1,3,figsize=(11,3.7))
    for a,p,title in zip(ax,ps,['Unrestricted: 95 leaves','Pre-pruned: 4 leaves','Post-pruned: 3 leaves']):im=a.imshow(p.reshape(xx.shape),origin='lower',extent=[-2,2,-2,2],vmin=0,vmax=1,cmap='viridis');a.set(xlabel='x0 (x2=x0)',ylabel='x1',title=title)
    f.subplots_adjust(right=.91,wspace=.34);cax=f.add_axes([.93,.2,.015,.62]);f.colorbar(im,cax=cax,label='P(class 1)');f.savefig(directory/'04_regions.png',bbox_inches='tight');plt.close(f)
    f,ax=plt.subplots(1,3,figsize=(11,3.7));axis=report['perturbation']['probe_axis'];mx=max(max(r['sd_probability']) for r in report['stability'].values())
    for a,(name,row) in zip(ax,report['stability'].items()):im=a.imshow(np.array(row['sd_probability']).reshape(len(axis),-1),origin='lower',extent=[-2,2,-2,2],vmin=0,vmax=mx,cmap='magma');a.set(xlabel='x0 (x2=x0)',ylabel='x1',title=f'{name}: prediction SD')
    f.subplots_adjust(right=.91,wspace=.34);cax=f.add_axes([.93,.2,.015,.62]);f.colorbar(im,cax=cax);f.savefig(directory/'05_stability.png',bbox_inches='tight');plt.close(f)
    f,a=plt.subplots(figsize=(10,4.5));model=fit(X,y,ccp_alpha=report['choices']['post_alpha']+1e-13);plot_tree(model,ax=a,feature_names=['x0','x1','x2','x3'],class_names=['0','1'],filled=True,rounded=True,precision=3,fontsize=10);save(f,'06_selected_tree.png')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',default='experiment-result.json');p.add_argument('--directory',default='outputs/figures');a=p.parse_args();build(json.loads(Path(a.report).read_text()),a.directory)
