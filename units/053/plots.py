"""All figures are regenerated from actual JSON, except explicitly labeled toy curves."""
from pathlib import Path
import argparse,json,os
os.environ.setdefault('MPLCONFIGDIR','/tmp/ml053-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from experiment import ROOT,METHODS
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
COLORS=['#2563a6','#c65b2c','#218573']

def draw(r,directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);paths=[]
    def save(fig,name):
        fig.tight_layout();p=directory/name;fig.savefig(p,dpi=170,bbox_inches='tight');plt.close(fig);paths.append(p);return p
    fig,ax=plt.subplots(figsize=(9,3.6));ax.axis('off')
    boxes=[(.03,.5,'600 independent rows\nper replicate'),(.29,.7,'Development 240\n3-fold search'),(.57,.7,'Freeze winner\nRefit 60 trees'),(.78,.3,'Test 360\nEvaluate once')]
    for x,y,t in boxes:ax.text(x,y,t,ha='left',va='center',bbox=dict(boxstyle='round,pad=.5',fc='#eaf0f7',ec='#8babc8'))
    for a,b in [((.24,.56),(.29,.69)),((.48,.7),(.56,.7)),((.7,.6),(.8,.4)),((.19,.4),(.79,.3))]:ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#567'))
    ax.text(.03,.07,'Same data and folds for all methods. No test-to-search arrow.',color='#a34a27')
    save(fig,'01_protocol.png')
    fig,axes=plt.subplots(1,3,figsize=(10.2,3.2),sharex=True,sharey=True)
    for ax,m,c in zip(axes,METHODS,COLORS):
        s=next(z for z in r['runs'] if z['rep']==0 and z['method']==m)
        for depth,feat in s['configurations']:ax.scatter(10 if depth is None else depth,feat,color=c,s=55)
        ax.set(title=m,xlabel='max_depth (U = unlimited)',xticks=[2,4,6,8,10],xticklabels=['2','4','6','8','U'],yticks=[.2,.4,.6,.8,1]);ax.grid(alpha=.2)
    axes[0].set_ylabel('max_features fraction');save(fig,'02_search_space.png')
    fig,ax=plt.subplots(figsize=(8,3.7));s=next(z for z in r['runs'] if z['rep']==0 and z['method']=='halving')
    for cid in range(12):
        pts=[(st['resource'],v['mse']) for st in s['stages'] for v in st['ranking'] if v['candidate']==cid]
        ax.plot(*zip(*pts),marker='o',alpha=.8,label=f'C{cid}')
    ax.set(xlabel='Trees per fresh fit',ylabel='Mean 3-fold MSE',title='Replicate 0: discarded candidates have no later score',xticks=[15,30,60]);ax.legend(ncol=4,fontsize=8);save(fig,'03_halving.png')
    fig,axes=plt.subplots(1,2,figsize=(10,3.7))
    for i,(m,c) in enumerate(zip(METHODS,COLORS)):
        ys=[z['test_mse'] for z in r['runs'] if z['method']==m];axes[0].scatter([i]*6,ys,c=c);axes[0].plot([i-.18,i+.18],[np.mean(ys)]*2,color=c,lw=3)
    for m,c in zip(['random','halving'],COLORS[1:]):axes[1].plot(range(6),r['summary'][m]['paired_difference_vs_grid'],'o-',label=m,color=c)
    axes[0].set(xticks=range(3),xticklabels=METHODS,ylabel='Test MSE',title='Each dot = an independent data draw');axes[1].axhline(0,c='grey',ls='--');axes[1].set(xlabel='Replicate',ylabel='Test MSE minus grid',title='Positive = worse than grid');axes[1].legend();save(fig,'04_results.png')
    fig,axes=plt.subplots(1,2,figsize=(9,3.4));axes[0].bar(METHODS,[1080]*3,color=COLORS);axes[0].set(ylabel='Requested search trees / replicate',title='Equal proxy budget')
    axes[1].bar(METHODS,[r['summary'][m]['total_search_seconds'] for m in METHODS],color=COLORS);axes[1].set(ylabel='Measured seconds across 6 replicates',title='Timing is machine/load dependent');save(fig,'05_costs.png')
    fig,ax=plt.subplots(figsize=(7.8,3.5));h=r['hand'];ax.plot(h['noise_candidates'],h['validation_min'],'o-',label='Mean selected validation estimate');ax.axhline(1,ls='--',c='#999',label='True risk of every candidate');ax.set(xscale='log',xlabel='Number of candidates',ylabel='Estimated risk',title='Illustrative independent Gaussian selection noise');ax.legend();save(fig,'06_selection_noise.png')
    fig,ax=plt.subplots(figsize=(7.8,3.5));
    for k,v in h['curves'].items():ax.plot([1,2,3],v,'o-',label=k)
    ax.axvline(1,c='#999',ls='--');ax.set(xlabel='Resource rung (toy)',ylabel='Validation loss',title='Constructed counterexample: early ranking can reverse',xticks=[1,2,3]);ax.legend();save(fig,'07_early_stop.png')
    return paths
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',default=str(ROOT/'experiment-result.json'));p.add_argument('--directory',default=str(ROOT/'outputs/figures'));a=p.parse_args();print('\n'.join(map(str,draw(json.loads(Path(a.report).read_text()),a.directory))))
