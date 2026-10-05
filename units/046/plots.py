"""All chart pixels derive from the frozen CSV or explicit analytic formulas."""
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from experiment import ROOT, run, load_case, library_metrics, prior_weights, output_dir

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                     'axes.spines.right':False,'figure.dpi':150,'savefig.dpi':180})
COLORS=['#176b99','#ca563a','#729f38','#7655a4']

def create_figures(dest):
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    result=run();ids,y,s=load_case();files=[]
    def save(fig,name):
        path=dest/name
        if path.is_symlink() or (path.exists() and path.stat().st_nlink>1):raise ValueError('unsafe figure output')
        fig.savefig(path,bbox_inches='tight',facecolor='white');plt.close(fig);files.append(str(path))
    fig,ax=plt.subplots(figsize=(9,3.8))
    x=np.arange(len(y));ax.bar(x,s,color=[COLORS[1] if yi else COLORS[0] for yi in y],width=.72)
    ax.axhline(.8,color='black',linestyle='--',label='threshold 0.80 (include equality)')
    for i,(yi,si) in enumerate(zip(y,s)):ax.text(i,si+.018,f'{si:.2f}',ha='center',fontsize=9)
    ax.set(xticks=x,xticklabels=[f'{i}\ny={v}' for i,v in zip(ids,y)],ylim=(0,1.08),ylabel='Fixed score',title='Same 12 records throughout: 3 positive, 9 negative')
    ax.text(8,.94,'Orange: positive\nBlue: negative',fontsize=10);ax.text(6.8,.825,'threshold 0.80 (include equality)',fontsize=9)
    save(fig,'01_scores.png')
    fig,axs=plt.subplots(1,2,figsize=(9,3.8))
    for ax,key,title in zip(axs,['at_080','at_055'],['threshold 0.80','threshold 0.55']):
        c=result[key];mat=np.array([[c['TN'],c['FP']],[c['FN'],c['TP']]])
        ax.imshow(mat,cmap='Blues',vmin=0,vmax=9)
        for i in range(2):
            for j in range(2):ax.text(j,i,f'{[["TN","FP"],["FN","TP"]][i][j]} = {int(mat[i,j])}',ha='center',va='center',fontsize=13,color='white' if mat[i,j]>=6 else 'black')
        ax.set(xticks=[0,1],xticklabels=['predict 0','predict 1'],yticks=[0,1],yticklabels=['true 0','true 1'],title=title)
    fig.tight_layout();save(fig,'02_confusion.png')
    fig,ax=plt.subplots(figsize=(9,4.1))
    rows=result['sweep'];x=np.arange(len(rows))
    for k,c in zip(['precision','recall','f1','accuracy'],COLORS):
        values=[r[k] for r in rows];ax.plot(x,values,'o-',label=k,color=c,markersize=4)
    ax.scatter(0,0,facecolors='white',edgecolors=COLORS[0],s=95,zorder=6)
    ax.annotate('No positive predictions:\nprecision set to 0, undefined ratio',xy=(0,0),xytext=(1.3,.19),arrowprops={'arrowstyle':'->'},fontsize=9)
    ax.set(xticks=x,xticklabels=[str(r['threshold']) for r in rows],ylim=(-.04,1.05),xlabel='Threshold decreases left to right',ylabel='Metric',title='Thresholds change decisions; tied scores move together')
    ax.legend(ncol=4,loc='upper center',bbox_to_anchor=(.5,-.23),fontsize=9);save(fig,'03_thresholds.png')
    fig,ax=plt.subplots(figsize=(7.8,4.8))
    roc=result['curves']['roc'];ax.plot(roc['fpr'],roc['tpr'],'o-',color=COLORS[0],lw=2,label='Grouped empirical ROC')
    ax.plot([0,1],[0,1],'--',color='gray',label='Chance reference')
    for t in [.95,.9,.8,.55,.1]:
        i=roc['thresholds'].index(t);offset={.95:(8,2),.9:(-42,-20),.8:(9,-13),.55:(10,-20),.1:(-77,-25)}[t]
        ax.annotate(f't={t:.2f}',(roc['fpr'][i],roc['tpr'][i]),xytext=offset,textcoords='offset points',fontsize=9)
    ax.set(xlim=(-.02,1.03),ylim=(-.02,1.06),xlabel='FPR = FP / 9',ylabel='TPR = TP / 3',title='ROC AUC = 43/54 = 0.796296')
    ax.legend(loc='lower right');ax.grid(alpha=.2);save(fig,'04_roc.png')
    fig,ax=plt.subplots(figsize=(8.8,4.6))
    pr=result['curves']['pr'];rr=np.array(pr['recall'][::-1]);pp=np.array(pr['precision'][::-1])
    ax.step(rr,pp,where='pre',color=COLORS[0],lw=2,label='AP rectangles (grouped scores)')
    ax.fill_between(rr,pp,step='pre',alpha=.12,color=COLORS[0])
    ax.plot(rr,pp,'o--',color=COLORS[1],ms=4,label='Straight-line trapezoid')
    ax.axhline(.25,color='gray',ls=':',label='Prevalence 0.25')
    ax.set(xlim=(-.02,1.03),ylim=(0,1.07),xlabel='Recall',ylabel='Precision',title='AP = 9/14 = 0.642857; trapezoid = 79/126 = 0.626984')
    ax.legend(loc='lower left',fontsize=9);ax.grid(alpha=.2);save(fig,'05_pr_ap.png')
    # Every positive-negative pair: 1 win, 1/2 tie, 0 loss.
    pos=s[y==1];neg=s[y==0];pair=(pos[:,None]>neg[None,:]).astype(float)+.5*(pos[:,None]==neg[None,:])
    fig,ax=plt.subplots(figsize=(9,3.2));ax.imshow(pair,cmap=ListedColormap(['#f6d5cd','#e6dfad','#a9d6e5']),vmin=0,vmax=1,aspect='auto')
    for i in range(3):
        for j in range(9):ax.text(j,i,str(pair[i,j]).replace('.0',''),ha='center',va='center')
    ax.set(xticks=range(9),xticklabels=[f'{i}\n{s[j]:.2f}' for j,i in enumerate(ids) if y[j]==0],yticks=range(3),yticklabels=[f'{i} ({s[j]:.2f})' for j,i in enumerate(ids) if y[j]==1],xlabel='Negative record',ylabel='Positive record',title='27 pairs: 21 wins + 1 tie + 5 losses; AUC = (21 + 0.5)/27')
    save(fig,'06_pairs.png')
    fig,axs=plt.subplots(1,2,figsize=(10,4.2))
    for pi,c in zip([.1,.25,.5],COLORS):
        curves=result['prior_shift'][str(pi)]['curves'];r=curves['roc'];p=curves['pr']
        axs[0].plot(r['fpr'],r['tpr'],marker='o',ms=5 if pi==.1 else 3,alpha=.8,color=c,label=f'pi={pi:g}')
        axs[1].step(p['recall'][::-1],p['precision'][::-1],where='pre',color=c,label=f'pi={pi:g}, AP={curves["ap"]:.3f}')
        axs[1].axhline(pi,ls=':',color=c,alpha=.55)
    axs[0].set(xlabel='FPR',ylabel='TPR',title='ROC curves coincide',xlim=(-.03,1.03),ylim=(-.03,1.05))
    axs[1].set(xlabel='Recall',ylabel='Precision',title='PR depends on prevalence',xlim=(-.03,1.03),ylim=(-.03,1.05))
    for ax in axs:ax.legend(fontsize=9,loc='upper center',bbox_to_anchor=(.5,-.20),ncol=1);ax.grid(alpha=.2)
    fig.tight_layout();save(fig,'07_prevalence.png')
    fig,ax=plt.subplots(figsize=(8,3.8));names=['positive only','macro','micro','support weighted'];values=[result['averages_080'][a]['f1'] for a in ['binary','macro','micro','weighted']]
    ax.bar(names,values,color=COLORS)
    for i,v in enumerate(values):ax.text(i,v+.025,f'{v:.6f}',ha='center')
    ax.set(ylim=(0,1),ylabel='F1',title='Same predictions at t=0.80, four averaging questions')
    save(fig,'08_averages.png')
    fig,axs=plt.subplots(1,2,figsize=(9.5,3.9));axs[0].plot(s,s**3,'o',color=COLORS[0]);xx=np.linspace(0,1,101);axs[0].plot(xx,xx**3,color=COLORS[0]);axs[0].plot(xx,xx,':',color='gray')
    axs[0].set(xlabel='Original score',ylabel='Cubed score',title='Strictly increasing transform')
    ms=result['monotone'];names=['ROC AUC','AP','log loss','Brier'];x=np.arange(4)
    for j,(key,c) in enumerate(zip(['original','cubed'],COLORS)):
        vals=[ms[key][a] for a in ['auc','ap','log_loss_as_probability','brier_as_probability']];axs[1].bar(x+(j-.5)*.34,vals,width=.34,label=key,color=c)
    axs[1].set(xticks=x,xticklabels=names,title='Ranking unchanged\nProbability losses change',ylim=(0,1.2));axs[1].tick_params(axis='x',labelsize=9);axs[1].legend(fontsize=9)
    fig.tight_layout();save(fig,'09_rank_probability.png')
    return files

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(ROOT/'outputs/figures'));a=p.parse_args()
    for f in create_figures(output_dir(a.out)):print(f)
if __name__=='__main__':main()
