"""固定报告的所有二维图；每面板独立坐标范围，不把跨面板距离当度量。"""
from pathlib import Path
import argparse,json
import numpy as np
from common import ROOT,figure_setup,load_data
from experiment import matrix

def draw(r,directory):
    plt=figure_setup();out=Path(directory);out.mkdir(parents=True,exist_ok=True);d=load_data();X=matrix(d['roll']);color=d['roll']['t'];vmin=color.min();vmax=color.max()
    def save(fig,name):fig.savefig(out/name,bbox_inches='tight');plt.close(fig)
    def scatter(ax,z,title):
        z=np.array(z);im=ax.scatter(z[:,0],z[:,1],c=color,cmap='viridis',vmin=vmin,vmax=vmax,s=9);ax.set_title(title,fontsize=10);ax.set_xticks([]);ax.set_yticks([]);ax.set_aspect('equal',adjustable='datalim');return im
    fig=plt.figure(figsize=(10,3.7));fig.subplots_adjust(wspace=.65);a=fig.add_subplot(121,projection='3d');a.scatter(X[:,0],X[:,2],X[:,1],c=color,cmap='viridis',s=10);a.set(xlabel='x0',ylabel='x2',zlabel='height',title='Observed3D continuous sheet');b=fig.add_subplot(122);im=b.scatter(d['roll']['arc_length'],d['roll']['height'],c=color,cmap='viridis',s=10);b.set(xlabel='True arc-length coordinate s(t)',ylabel='Height',title='Known intrinsic coordinates');fig.colorbar(im,ax=b,label='Continuous t',shrink=.7);save(fig,'01_known_geometry.png')
    fig,ax=plt.subplots(figsize=(6,3.8));im=scatter(ax,r['records'][0]['embedding'],'PCA2: overlap can hide the sheet');fig.colorbar(im,ax=ax,label='Continuous t');save(fig,'02_pca_projection.png')
    fig,axes=plt.subplots(2,3,figsize=(9,5.4))
    for ax,row in zip(axes.flat,[z for z in r['records'] if z['method']=='t-SNE']):scatter(ax,row['embedding'],f"seed{row['seed']} | perplexity{row['parameter']}\nT10={row['metrics']['trustworthiness_10']:.3f}")
    fig.subplots_adjust(wspace=.2,hspace=.38);save(fig,'03_tsne_sweep.png')
    fig,axes=plt.subplots(2,3,figsize=(9,5.4))
    for ax,row in zip(axes.flat,[z for z in r['records'] if z['method']=='UMAP' and z['min_dist']==.1]):scatter(ax,row['embedding'],f"seed{row['seed']} | neighbors{row['parameter']}\nT10={row['metrics']['trustworthiness_10']:.3f}")
    fig.subplots_adjust(wspace=.2,hspace=.38);save(fig,'04_umap_sweep.png')
    fig,axes=plt.subplots(1,2,figsize=(9,3.7))
    for ax,row in zip(axes,[z for z in r['records'] if z['method']=='UMAP' and z['seed']==0 and z['parameter']==30]):scatter(ax,row['embedding'],f"neighbors30 | min_dist{row['min_dist']}\nrecall10={row['metrics']['neighbor_recall_10']:.3f}")
    save(fig,'05_min_dist.png')
    fig,axes=plt.subplots(1,2,figsize=(9,3.5));palette={'PCA':'#64748b','t-SNE':'#2563eb','UMAP':'#e16b35'}
    for method,col in palette.items():
        rows=[z['metrics'] for z in r['records'] if z['method']==method];x=[z['trustworthiness_10'] for z in rows];axes[0].scatter(x,[z['neighbor_recall_10'] for z in rows],c=col,label=method);axes[1].scatter(x,[z['intrinsic_global_distance_spearman'] for z in rows],c=col,label=method)
    axes[0].set(xlabel='Input-space trustworthiness10',ylabel='Input-space neighbor recall10',xlim=(.88,1.005),ylim=(0,1));axes[1].set(xlabel='Input-space trustworthiness10',ylabel='Intrinsic global distance rank correlation',xlim=(.88,1.005),ylim=(0,1));axes[0].legend();save(fig,'06_local_global_metrics.png')
    fig,axes=plt.subplots(1,2,figsize=(9,3.7))
    for ax,z in zip(axes,r['null_control']):
        emb=np.array(z['embedding']);ax.scatter(emb[:,0],emb[:,1],s=10,c='#66748b');ax.set_title(f"Single6D Gaussian | seed{z['seed']}\nT10={z['metrics']['trustworthiness_10']:.3f}");ax.set_xticks([]);ax.set_yticks([]);ax.set_aspect('equal',adjustable='datalim')
    save(fig,'07_gaussian_null.png');return sorted(p.name for p in out.glob('*.png'))
def main():
    p=argparse.ArgumentParser();p.add_argument('--report',default=str(ROOT/'experiment-result.json'));p.add_argument('--directory',default=str(ROOT/'outputs/figures'));a=p.parse_args();print(json.dumps(draw(json.loads(Path(a.report).read_text()),a.directory)))
if __name__=='__main__':main()
