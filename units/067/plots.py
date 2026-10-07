"""七张 GP 解释图。每张图来自实际冻结数值或明示的成本公式。"""
from pathlib import Path
import argparse,json
import numpy as np
from common import ROOT,figure_setup,load_data,safe_output
from generate_data import truth

def make(report,directory):
    plt=figure_setup();directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);paths=[];colors=['#146c94','#eb8b23','#7a4d9b']
    def save(fig,name):
        p=safe_output(directory/name);fig.savefig(p,bbox_inches='tight',metadata={'Software':'ML course 067'});plt.close(fig);paths.append(p)
    d=load_data();tx=d['train']['x'];ty=d['train']['y'];x=np.array(report['grid_x'])
    fig,axs=plt.subplots(1,3,figsize=(11,3.3),layout='constrained')
    for ax,row in zip(axs,report['prior']['variants']):
        for line,c in zip(row['draws'],colors):ax.plot(report['prior']['x'],line,color=c,lw=1.5)
        ax.set(title=f'Prior length scale = {row["length_scale"]}',xlabel='x',ylabel='Function value',ylim=(-3,3));ax.axhline(0,color='#888',lw=.5)
    save(fig,'01_prior_samples.png')
    ref=report['variants'][1];mu=np.array(ref['grid_mean']);v=np.array(ref['grid_variance'])
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained');ax.fill_between(x,mu-1.96*np.sqrt(v+.0324),mu+1.96*np.sqrt(v+.0324),color=colors[1],alpha=.22,label='95% pointwise observation band');ax.fill_between(x,mu-1.96*np.sqrt(v),mu+1.96*np.sqrt(v),color=colors[0],alpha=.25,label='95% pointwise latent band');ax.plot(x,mu,color=colors[0],lw=2,label='Posterior mean');ax.plot(x,truth(x),'--',color='#333',label='Truth (evaluation only)');ax.scatter(tx,ty,s=25,c='#222',zorder=4,label='Noisy training observations');ax.axvspan(-.8,.7,color='#888',alpha=.07);ax.set(xlabel='x',ylabel='f(x) or y',title='Reference GP: length=1, noise variance=0.0324');ax.legend(fontsize=8,ncol=2,loc='lower left')
    save(fig,'02_posterior_bands.png')
    def panels(rows,name,title):
        fig,axs=plt.subplots(1,3,figsize=(11,3.4),layout='constrained')
        for ax,row,c in zip(axs,rows,colors):
            m=np.array(row['grid_mean']);s=np.sqrt(row['grid_variance']);ax.fill_between(x,m-1.96*s,m+1.96*s,color=c,alpha=.2);ax.plot(x,m,color=c,lw=1.8);ax.plot(x,truth(x),'--',c='#444',lw=1);ax.scatter(tx,ty,c='black',s=15);ax.axvline(3,color='#999',ls=':');ax.axvline(-3,color='#999',ls=':');ax.set(title=title(row),xlabel='x',ylabel='Latent f',ylim=(-3,3))
        save(fig,name)
    panels(report['variants'][:3],'03_length_scale.png',lambda r:f'length = {r["length_scale"]}')
    panels([report['variants'][3],report['variants'][1],report['variants'][4]],'04_noise_assumptions.png',lambda r:f'noise variance = {r["noise_variance"]}')
    fig,axs=plt.subplots(1,2,figsize=(10.5,3.8),layout='constrained');g=report['lml_grid'];ls=np.array(g['length_scales']);nv=np.array(g['noise_variances']);v=np.array(g['values']);im=axs[0].pcolormesh(ls,nv,np.maximum(v,-100),shading='auto',cmap='viridis');axs[0].set(xscale='log',yscale='log',xlabel='Length scale',ylabel='Noise variance',title='Training log marginal likelihood');fig.colorbar(im,ax=axs[0],label='LML (clipped below -100 for display)');axs[0].axhline(.0324,color='white',ls='--',lw=1)
    parts=np.array(g['fixed_noise_parts']);axs[1].semilogx(ls,parts[:,0],label='Data fit term',color=colors[0]);axs[1].semilogx(ls,parts[:,1],label='Log determinant term',color=colors[1]);axs[1].semilogx(ls,parts.sum(axis=1),label='Total LML',color=colors[2],lw=2);axs[1].axvline(report['optimized']['length_scale'],color='#777',ls=':');axs[1].set(xlabel='Length scale, noise variance fixed at 0.0324',ylabel='Log-density contribution',title='Balance terms, do not minimize training error');axs[1].legend(fontsize=8)
    save(fig,'05_marginal_likelihood.png')
    fig,axs=plt.subplots(1,2,figsize=(10.5,3.5),layout='constrained');rows=report['variants'][:3];labels=['short','reference','long'];loc=np.arange(3)
    axs[0].bar(loc-.17,[r['interpolation']['latent_rmse'] for r in rows],width=.34,label='Interpolation',color=colors[0]);axs[0].bar(loc+.17,[r['extrapolation']['latent_rmse'] for r in rows],width=.34,label='Extrapolation',color=colors[1]);axs[0].set(xticks=loc,xticklabels=labels,ylabel='RMSE versus latent truth',title='Accurate nearby need not mean accurate far away');axs[0].legend()
    axs[1].bar(loc-.17,[r['interpolation']['latent_pointwise_95_fraction'] for r in rows],width=.34,color=colors[0]);axs[1].bar(loc+.17,[r['extrapolation']['latent_pointwise_95_fraction'] for r in rows],width=.34,color=colors[1]);axs[1].axhline(.95,ls='--',color='#444');axs[1].set(xticks=loc,xticklabels=labels,ylabel='Fraction of grid truth inside pointwise band',ylim=(0,1.05),title='One deterministic grid is not a coverage theorem')
    save(fig,'06_extrapolation_audit.png')
    fig,axs=plt.subplots(1,2,figsize=(10,3.4),layout='constrained');rows=report['cost'];n=[r['n'] for r in rows]
    axs[0].loglog(n,[r['one_dense_matrix_bytes']/2**30 for r in rows],'o-',color=colors[0]);axs[0].set(xlabel='Training samples n',ylabel='GiB for ONE float64 n x n matrix',title='Quadratic storage (more arrays are needed)');axs[0].grid(alpha=.2)
    axs[1].loglog(n,[r['cholesky_leading_flops'] for r in rows],'o-',color=colors[1]);axs[1].set(xlabel='Training samples n',ylabel='Leading arithmetic count n^3 / 3',title='Dense factorization cost, not measured runtime');axs[1].grid(alpha=.2)
    save(fig,'07_dense_cost.png');return paths

def main():
    p=argparse.ArgumentParser();p.add_argument('--report',default=str(ROOT/'experiment-result.json'));p.add_argument('--directory',default=str(ROOT/'outputs/figures'));a=p.parse_args();paths=make(json.loads(Path(a.report).read_text()),a.directory);print(json.dumps([str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else p.name for p in paths]))
if __name__=='__main__':main()
