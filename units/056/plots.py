"""Figures reconstruct only saved run results; no new tuning or test searches."""
import argparse,json
from pathlib import Path
import numpy as np
from common import ROOT,figure_setup

def make_figures(r,directory):
    plt=figure_setup();d=Path(directory);d.mkdir(parents=True,exist_ok=True);paths=[]
    def save(fig,name):
        p=d/name;fig.tight_layout();fig.savefig(p);plt.close(fig);paths.append(p)
    fig,ax=plt.subplots(figsize=(9,3.8));ax.axis('off')
    labels=['Development 1200','Outer training 800\nInner 3-fold: choose degree, C, t','Outer holdout 400\nEvaluate procedure','Full development\nInner 3-fold + final fit','Freeze model + threshold','Test 600\nEvaluate once']
    loc=[(.16,.85),(.16,.45),(.51,.45),(.49,.85),(.80,.85),(.80,.25)]
    boxes=[]
    for (x,y),txt in zip(loc,labels):boxes.append(ax.text(x,y,txt,ha='center',va='center',fontsize=10,bbox=dict(boxstyle='round,pad=.6',facecolor='#e7eff7',edgecolor='#45698a')))
    for i,j in [(0,1),(1,2),(0,3),(3,4),(4,5)]:ax.annotate('',xy=loc[j],xytext=loc[i],arrowprops=dict(arrowstyle='->',color='#45698a',patchA=boxes[i].get_bbox_patch(),patchB=boxes[j].get_bbox_patch(),shrinkA=3,shrinkB=3))
    ax.text(.15,.10,'Repeat outer loop 3 times.\nNo test-driven decisions.',ha='center',fontsize=10);save(fig,'01_protocol.png')
    fig,ax=plt.subplots(figsize=(7,3.8))
    for deg in [1,2]:
        rows=[v for v in r['development']['final_selection']['candidates'] if v['degree']==deg]
        ax.plot([v['C'] for v in rows],[v['cost'] for v in rows],'o-',label=f'degree {deg}')
        for v in rows:ax.annotate('t='+str(v['threshold']),(v['C'],v['cost']),xytext=(3,6),textcoords='offset points',fontsize=8)
    ax.set(xscale='log',xlabel='C (weaker regularization to the right)',ylabel='Inner OOF cost per row',title='Development-only selection; each point selects its threshold');ax.legend();save(fig,'02_selection.png')
    fig,ax=plt.subplots(figsize=(7,3.6));val=[a['metrics']['cost'] for a in r['development']['outer']]+[r['test']['selected']['cost'],r['test']['baseline']['cost']]
    ax.bar(['Outer 1','Outer 2','Outer 3','Final test','Prior baseline'],val,color=['#739dbd']*3+['#276882','#ba8752'])
    ax.set(ylabel='Cost per row',title='Outer folds assess the procedure; final test assesses final fit')
    for i,v in enumerate(val):ax.text(i,v+.01,f'{v:.3f}',ha='center');ax.set_ylim(0,max(val)*1.2)
    save(fig,'03_costs.png')
    fig,ax=plt.subplots(figsize=(7,3.3))
    for i,g in enumerate(['0','1']):
        s=r['test']['groups'][g];lo,hi=s['recall_wilson'];v=s['recall'];ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',capsize=5,color='#276882');ax.text(.66,i+.17,f"n={s['n']}, positive={s['tp']+s['fn']}, FN={s['fn']}",fontsize=9)
    ax.set(xlim=(.65,1.01),ylim=(-.4,1.5),yticks=[0,1],yticklabels=['group 0','group 1'],xlabel='Recall with pointwise 95% Wilson interval',title='Descriptive group check, not a fairness certificate');save(fig,'04_groups.png')
    fig,ax=plt.subplots(figsize=(5,4));s=r['test']['selected'];m=np.array([[s['tn'],s['fp']],[s['fn'],s['tp']]])
    im=ax.imshow(m,cmap='Blues');ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Predict 0','Predict 1'],yticklabels=['True 0','True 1'],title='Final test confusion counts')
    for i in range(2):
        for j in range(2):ax.text(j,i,str(m[i,j]),ha='center',va='center',color='white' if m[i,j]>150 else 'black',fontsize=18)
    save(fig,'05_confusion.png')
    fig,ax=plt.subplots(figsize=(7,3.5));t=r['test'];ax.hist(t['bootstrap_distribution'],bins=35,color='#739dbd');ax.axvline(0,color='#9b3b35',ls='--',label='No cost difference')
    for v in t['paired_bootstrap_95']:ax.axvline(v,color='#276882',ls=':')
    ax.set(xlabel='Selected cost minus prior baseline cost',ylabel='Bootstrap replicates',title='Paired row bootstrap; fitted models held fixed');ax.legend();save(fig,'06_uncertainty.png')
    return paths
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',default=str(ROOT/'experiment-result.json'));p.add_argument('--directory',default=str(ROOT/'outputs/figures'));a=p.parse_args();print('\n'.join(map(str,make_figures(json.loads(Path(a.report).read_text()),a.directory))))
