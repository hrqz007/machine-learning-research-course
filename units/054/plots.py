from pathlib import Path
import argparse,json,os
os.environ.setdefault('MPLCONFIGDIR','/tmp/ml054-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from experiment import ROOT,METHODS
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
COLORS=['#2563a6','#c65b2c','#218573']

def draw(r,directory):
    d=Path(directory);d.mkdir(parents=True,exist_ok=True);paths=[]
    def save(fig,name):
        fig.tight_layout();p=d/name;fig.savefig(p,dpi=170,bbox_inches='tight');plt.close(fig);paths.append(p)
    h=r['controls']['interaction'];fig,axes=plt.subplots(1,2,figsize=(9,3.3))
    X=np.array(h['X']);axes[0].scatter(X[:,0],X[:,1],c=h['y'],cmap='coolwarm',vmin=-1,vmax=1,s=280)
    for x,y,z in zip(X[:,0],X[:,1],h['y']):axes[0].text(x,y,str(int(z)),ha='center',va='center',color='white')
    axes[0].set(xlim=(-1.5,1.5),ylim=(-1.5,1.5),xlabel='x1',ylabel='x2',title='y = x1 * x2; marginal correlation = 0')
    axes[1].plot(range(4),h['y'],'o-',label='truth');axes[1].plot(range(4),h['additive_prediction'],'s--',label='additive');axes[1].plot(range(4),h['interaction_prediction'],'x',ms=12,label='with interaction');axes[1].set(xlabel='Row',ylabel='Prediction',xticks=range(4));axes[1].legend();save(fig,'01_interaction.png')
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    for z in r['controls']['distance']:axes[0].hist(z['normalized_squared_distance'],bins=np.linspace(0,4,75),density=True,histtype='step',label=f"d={z['d']}")
    ds=[z['d'] for z in r['controls']['distance']];axes[1].plot(ds,[z['cv_squared_distance'] for z in r['controls']['distance']],'o-',label='4000 independent pairs');axes[1].plot(ds,[z['theory_cv'] for z in r['controls']['distance']],'--',label='sqrt(2/d)')
    axes[0].set(xlabel='Squared distance / (2d)',ylabel='Density',title='Independent standard-normal pair distances');axes[0].legend();axes[1].set(xscale='log',xlabel='Dimensions',ylabel='CV of squared distance');axes[1].legend();save(fig,'02_distance.png')
    fig,ax=plt.subplots(figsize=(9,3.8));ax.axis('off')
    ax.text(.02,.8,'Outer train 120',bbox=dict(fc='#e9f1fa',ec='#789'));ax.text(.72,.8,'Outer validation 40',bbox=dict(fc='#faede8',ec='#a87'))
    ax.text(.1,.46,'Inner train 80:\nrank + scale + ridge',bbox=dict(fc='#e9f1fa',ec='#789'));ax.text(.50,.46,'Inner validation 40:\nchoose k from losses',bbox=dict(fc='#e9f1fa',ec='#789'))
    ax.annotate('',xy=(.49,.5),xytext=(.35,.5),arrowprops=dict(arrowstyle='->'))
    ax.text(.08,.12,'Refit ranking, scaler, ridge on outer train; evaluate outer validation once.')
    ax.annotate('Forbidden label path',xy=(.29,.68),xytext=(.69,.68),color='#b64c2b',arrowprops=dict(arrowstyle='->',color='#b64c2b',ls='--'))
    save(fig,'03_boundaries.png')
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),sharey=True)
    for ax,mode in zip(axes,['signal','null']):
        x=np.arange(3);a=[r['summary'][mode][m]['outer_mse'] for m in METHODS];b=[r['summary'][mode][m]['confirmation_mse'] for m in METHODS];ax.bar(x-.18,a,.36,label='Outer CV');ax.bar(x+.18,b,.36,label='Fresh confirmation');ax.set(xticks=x,xticklabels=['nested','global leaky','all 80'],title=mode,ylabel='MSE; mean of 6 independent draws');ax.legend()
    save(fig,'04_scores.png')
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),sharey=True)
    for ax,mode in zip(axes,['signal','null']):
        for j,(m,c) in enumerate(zip(METHODS,COLORS)):
            a=[s['stability_mean_jaccard'] for s in r['runs'] if s['mode']==mode and s['method']==m];ax.scatter([j]*6,a,color=c);ax.plot([j-.18,j+.18],[np.mean(a)]*2,c=c,lw=3)
        ax.set(xticks=range(3),xticklabels=['nested','global leaky','all 80'],ylim=(-.03,1.08),ylabel='Mean pairwise fold Jaccard',title=mode)
    save(fig,'05_stability.png')
    fig,axes=plt.subplots(1,2,figsize=(10,3.3),sharey=True)
    for ax,mode in zip(axes,['signal','null']):
        s=next(s for s in r['runs'] if s['mode']==mode and s['rep']==0 and s['method']=='nested');ax.bar(range(15),s['selection_frequency'][:15],color=COLORS[0]);ax.set(xlabel='Original feature index (first 15 of 80)',ylabel='Fraction of 4 outer folds selected',title=f'{mode}, replicate 0',xticks=range(0,15,2),ylim=(0,1.08))
    save(fig,'06_frequency.png');return paths
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',default=str(ROOT/'experiment-result.json'));p.add_argument('--directory',default=str(ROOT/'outputs/figures'));a=p.parse_args();print('\n'.join(map(str,draw(json.loads(Path(a.report).read_text()),a.directory))))
