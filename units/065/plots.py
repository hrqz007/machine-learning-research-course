"""Dual geometry, KKT roles and a verified exact polynomial feature map."""
import argparse,json
from pathlib import Path
import numpy as np
from common import ROOT,figure_setup,load_data
from experiment import arrays
from dual import gram,polynomial_features

def build(r,directory):
    plt=figure_setup();d=Path(directory);d.mkdir(parents=True,exist_ok=True);names=[]
    def save(fig,name):fig.tight_layout();fig.savefig(d/name,bbox_inches='tight');plt.close(fig);names.append(name)
    fig,ax=plt.subplots(figsize=(8,3.3));a=np.linspace(0,1,300);D=2*a-2*a*a;ax.plot(a,D,color='#2563eb',lw=2,label='D(a)=2a-2a²');ax.axvspan(0,.1,color='#f59e0b',alpha=.25,label='feasible when C=.1');ax.scatter([.5,.1],[.5,.18],color=['#0d9488','#d97706'],s=60);ax.set(xlabel='a = alpha1 = alpha2',ylabel='Dual lower bound',title='Two points: maximize a concave parabola');ax.legend(loc='lower center');save(fig,'01_two_point_dual.png')
    X,y=arrays(load_data()['train']);a=np.array(r['dual']['alpha']);m=np.array(r['dual']['functional_margin']);fig,ax=plt.subplots(1,2,figsize=(8.5,3.5));colors=np.where(a<1e-6,'#94a3b8',np.where(a>r['C']-1e-6,'#d97706','#2563eb'));ax[0].bar(range(len(a)),a,color=colors);ax[0].axhline(r['C'],ls='--',color='#64748b');ax[0].set(xlabel='Training index',ylabel='alpha',title='Coefficient bounds 0 <= alpha <= C');ax[1].scatter(a,m,c=colors,s=45);ax[1].axhline(1,ls='--',color='#64748b');ax[1].set(xlabel='alpha',ylabel='Functional margin',title='KKT links coefficient and margin');save(fig,'02_kkt_roles.png')
    K=gram(X);fig,ax=plt.subplots(1,2,figsize=(8.5,3.5));im=ax[0].imshow(K,cmap='viridis');fig.colorbar(im,ax=ax[0],shrink=.8);ax[0].set(title='Degree-2 Gram matrix',xlabel='Training index',ylabel='Training index');ax[1].bar(range(len(X)),r['gram_eigenvalues'],color='#2563eb');ax[1].axhline(0,color='#64748b');ax[1].set(title='Only 3 explicit feature dimensions',xlabel='Sorted eigenvalue index',ylabel='Eigenvalue');save(fig,'03_gram_psd.png')
    Phi=polynomial_features(X);fig,ax=plt.subplots(1,2,figsize=(8.5,3.7))
    for label,col,mark in [(-1,'#2563eb','o'),(1,'#dc2626','^')]:
        ax[0].scatter(*X[y==label].T,c=col,marker=mark,label=f'y={label}');ax[1].scatter(Phi[y==label,1],Phi[y==label,0]+Phi[y==label,2],c=col,marker=mark,label=f'y={label}')
    ax[0].set(xlabel='x1',ylabel='x2',title='XOR-like classes in input space');ax[0].set_aspect('equal');ax[1].axvline(0,ls='--',color='#64748b');ax[1].set(xlabel='sqrt(2) x1 x2',ylabel='x1² + x2²',title='A projection of the explicit feature map');ax[1].legend(loc='best');save(fig,'04_explicit_features.png')
    fig,ax=plt.subplots(figsize=(7.6,4));axis=np.array(r['grid']['axis']);xx,yy=np.meshgrid(axis,axis);z=np.array(r['grid']['scores']);ax.contourf(xx,yy,z,levels=[-100,0,100],colors=['#dbeafe','#fee2e2']);ax.contour(xx,yy,z,levels=[-1,0,1],colors=['#64748b','#111827','#64748b'],linestyles=['--','-','--'])
    for label,col,mark in [(-1,'#2563eb','o'),(1,'#dc2626','^')]:ax.scatter(*X[y==label].T,c=col,marker=mark,label=f'y={label}')
    sv=r['support_indices'];ax.scatter(*X[sv].T,s=120,edgecolors='#0f172a',facecolors='none',label='support vectors');ax.set(xlabel='x1',ylabel='x2',title='Same scores from explicit features and kernel');ax.set_aspect('equal');ax.legend(loc='upper left',bbox_to_anchor=(1.02,1));save(fig,'05_kernel_boundary.png')
    fig,ax=plt.subplots(1,2,figsize=(8.5,3.4));ep=np.array(r['test']['explicit']);kp=np.array(r['test']['precomputed']);ax[0].scatter(ep,kp,s=15,color='#0d9488');limits=[min(ep.min(),kp.min()),max(ep.max(),kp.max())];ax[0].plot(limits,limits,'--',color='#64748b');ax[0].set(xlabel='Explicit-map SVC score',ylabel='Precomputed-kernel SVC score',title='Held-out score equivalence');ax[1].bar(['valid toy K','invalid similarity'],[min(r['gram_eigenvalues']),min(r['invalid_similarity_eigenvalues'])],color=['#2563eb','#dc2626']);ax[1].axhline(0,color='#64748b');ax[1].set(ylabel='Smallest eigenvalue',title='Symmetric is not enough');save(fig,'06_equivalence_counterexample.png');return names
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',default=str(ROOT/'experiment-result.json'));p.add_argument('--directory',default=str(ROOT/'outputs/figures'));a=p.parse_args();print(build(json.loads(Path(a.report).read_text()),a.directory))
