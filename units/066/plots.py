"""六张解释图。英文坐标避免中文字体依赖，中文逐图读法见讲义。"""
from pathlib import Path
import argparse,json
import numpy as np
from common import ROOT,figure_setup,load_data,safe_output
from experiment import make_model,xy
from kernels import rbf,polynomial

def make(report,directory):
    plt=figure_setup();directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);paths=[]
    def save(fig,name):
        p=safe_output(directory/name);fig.savefig(p,bbox_inches='tight',metadata={'Software':'ML course 066'});plt.close(fig);paths.append(p)
    colors=['#146c94','#eb8b23','#7a4d9b'];r=np.linspace(0,3,220)
    fig,ax=plt.subplots(1,2,figsize=(10,3.4),layout='constrained')
    for g,c in zip([.1,1,10],colors):ax[0].plot(r,np.exp(-g*r*r),label=f'gamma={g}',color=c,lw=2)
    ax[0].set(xlabel='Euclidean distance r',ylabel='RBF similarity',title='Same distance, different bandwidth');ax[0].legend()
    for d,c in zip([2,3,4],colors):ax[1].plot(r,(.5*r+1)**d,label=f'degree={d}',color=c,lw=2)
    ax[1].set(xlabel='Dot product',ylabel='Polynomial kernel value',title='Polynomial kernels are not bounded by 1');ax[1].legend()
    save(fig,'01_kernel_shapes.png')
    data=load_data();X,y=xy(data['train']);g=report['grid'];xx,yy=np.meshgrid(g['x1'],g['x2']);Z=np.c_[xx.ravel(),yy.ravel()]
    fig,axs=plt.subplots(1,3,figsize=(11,3.5),layout='constrained')
    for ax,gamma in zip(axs,[.03,1,40]):
        m=make_model('rbf',{'C':1.,'gamma':gamma}).fit(X,y);score=m.decision_function(Z).reshape(xx.shape)
        ax.contourf(xx,yy,score,levels=np.linspace(-3,3,19),cmap='RdBu',extend='both',alpha=.55);ax.contour(xx,yy,score,levels=[0],colors='black',linewidths=1)
        ax.scatter(X[:,0],X[:,1],c=y,cmap='RdBu',s=8,edgecolors='none');ax.set(title=f'C=1, gamma={gamma}',xlabel='x1 (raw units)',ylabel='x2 (raw units)')
    save(fig,'02_bandwidth_boundaries.png')
    fig,axs=plt.subplots(1,3,figsize=(11,3.5),layout='constrained')
    for ax,family in zip(axs,['linear','rbf','poly']):
        row=report['models'][family];s=np.array(row['grid_score']);ax.contourf(xx,yy,s,levels=np.linspace(-3,3,19),cmap='RdBu',extend='both',alpha=.55);ax.contour(xx,yy,s,levels=[0],colors='black',linewidths=1)
        ax.scatter(X[:,0],X[:,1],c=y,cmap='RdBu',s=8,edgecolors='none');ax.set(title=f'{family}: test={row["test_accuracy"]:.3f}',xlabel='x1',ylabel='x2')
    save(fig,'03_equal_budget.png')
    names=['linear','rbf','poly'];rows=[report['models'][k] for k in names]
    fig,axs=plt.subplots(1,3,figsize=(11,3.7),layout='constrained')
    med=[r['batch_latency']['median_ms'] for r in rows];lo=[med[i]-r['batch_latency']['q25_ms'] for i,r in enumerate(rows)];hi=[r['batch_latency']['q75_ms']-med[i] for i,r in enumerate(rows)]
    axs[0].bar(names,med,color=colors,yerr=[lo,hi],capsize=4);axs[0].set(title='Warm prediction batch of 120',ylabel='Median latency (ms), bars = IQR')
    axs[1].bar(names,[r['selected_prediction_arrays_bytes']/1024 for r in rows],color=colors);axs[1].set(title='Selected prediction arrays',ylabel='KiB (not peak RAM)')
    axs[2].bar(['rbf','poly'],[rows[1]['support_vectors'],rows[2]['support_vectors']],color=colors[1:]);axs[2].set(title='Kernel models retain support vectors',ylabel='Number of support vectors',ylim=(0,360));axs[2].text(.5,320,'LinearSVC: not applicable',ha='center',fontsize=9)
    save(fig,'04_resources.png')
    fig,ax=plt.subplots(figsize=(8.8,3.7),layout='constrained');rows=report['storage'];ns=[r['n'] for r in rows]
    ax.loglog(ns,[r['dense_float64_gram_bytes']/2**30 for r in rows],'o-',lw=2,label='Full float64 Gram matrix',color=colors[0]);ax.loglog(ns,[r['feature64_float64_bytes']/2**30 for r in rows],'s-',lw=2,label='n x 64 feature matrix',color=colors[1]);ax.axhline(8,color='#888',ls='--',label='8 GiB reference');ax.set(xlabel='Number of training samples n',ylabel='GiB for one array',title='Storage grows quadratically or linearly in n');ax.legend();ax.grid(alpha=.2)
    save(fig,'05_storage.png')
    fig,ax=plt.subplots(figsize=(8.8,3.7),layout='constrained')
    for method,c in [('Nystroem',colors[0]),('RFF',colors[1])]:
        rows=[v for v in report['approximation'] if v['method']==method];ax.errorbar([v['components'] for v in rows],[v['mean_error'] for v in rows],yerr=[v['std_error'] for v in rows],fmt='o-',color=c,capsize=4,label=method)
    ax.set(xlabel='Number of features m',ylabel='Relative Frobenius error',title='Five seeds: mean and sample standard deviation');ax.legend();ax.grid(alpha=.2)
    save(fig,'06_approximation.png');return paths

def main():
    p=argparse.ArgumentParser();p.add_argument('--report',default=str(ROOT/'experiment-result.json'));p.add_argument('--directory',default=str(ROOT/'outputs/figures'));a=p.parse_args();paths=make(json.loads(Path(a.report).read_text()),a.directory);print(json.dumps([str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else p.name for p in paths]))
if __name__=='__main__':main()
