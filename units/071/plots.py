"""由可复算结果绘图；颜色表示算法分组，不是已知类别。"""
from pathlib import Path
import argparse,json
import numpy as np
from common import ROOT,load_data,figure_setup,safe_output
from kmeans import fit

def make(report,directory):
    plt=figure_setup();directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);data=load_data();X=lambda key:np.c_[data[key]['x1'],data[key]['x2']]
    def save(fig,name):fig.tight_layout();fig.savefig(safe_output(directory/name),bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(1,3,figsize=(10.5,3));xx=np.array([0,2,3,10,11])
    for a,h in zip(ax,[report['hand']['history'][0],report['hand']['history'][1],report['hand']['history'][2]]):
        a.scatter(xx,np.zeros(5),c=h['labels_before'],cmap='coolwarm',s=75,vmin=0,vmax=1);a.scatter(np.array(h['centers_before']).ravel(),np.zeros(2)+.25,marker='X',s=110,c=['#2563eb','#dc2626']);a.set(ylim=(-.4,.7),yticks=[],xlabel='x',title='Before round '+str(h['iteration']))
        for x in xx:a.text(x,-.22,str(x),ha='center')
    save(fig,'01_manual_rounds.png')
    fig,ax=plt.subplots(1,2,figsize=(10,3.4));r=report['hand']['history'];vals=[r[0]['before']]
    for h in r[:2]:vals.extend([h['after_update'],h['after_assign']])
    ax[0].plot(range(5),vals,'o-',color='#2563eb');ax[0].set(xticks=range(5),xticklabels=['Initial','Update 1','Assign 2','Update 2','Assign 3'],ylabel='Squared-error objective',title='Every substep is non-increasing');ax[0].tick_params(axis='x',rotation=20)
    for k,label,marker in [('random','Random','o'),('kmeans_plus_plus','D-squared','x')]:
        values=report[k]['restart_inertias'];ax[1].scatter(range(len(values)),values,s=16,marker=marker,label=label+' runs',alpha=.7);ax[1].plot(np.minimum.accumulate(values),ls='--',label=label+' best')
    ax[1].set(xlabel='Restart index',ylabel='Objective (log scale)',title='Individual runs and retained best',yscale='log');ax[1].legend(fontsize=8);save(fig,'02_objective_restarts.png')
    fig,ax=plt.subplots(1,3,figsize=(10.5,3.3));r=X('restarts');v=report['random']['restart_inertias'];bad=fit(r,2,init='random',n_init=1,seed=71+int(np.argmax(v)))
    for a,m,t in zip(ax[:2],[report['random'],bad],['Best local solution','Worse stationary solution']):
        a.scatter(r[:,0],r[:,1],c=m['labels'],cmap='coolwarm',s=10);c=np.array(m['centers']);a.scatter(c[:,0],c[:,1],s=100,marker='X',c='black');a.set(title=t,xlabel='x1',ylabel='x2');a.set_aspect('equal')
    ring=X('rings');ax[2].scatter(ring[:,0],ring[:,1],c=report['rings']['labels'],cmap='coolwarm',s=10);ax[2].set(title='Two prototypes split rings',xlabel='x1',ylabel='x2');ax[2].set_aspect('equal');save(fig,'03_local_and_shape.png')
    fig,ax=plt.subplots(1,3,figsize=(10.5,3.3));s=X('scale')
    for a,(key,m) in zip(ax,report['scale'].items()):
        # 都画在原坐标，防止单位变化掩盖分组改变。
        a.scatter(s[:,0],s[:,1],c=m['labels'],cmap='coolwarm',s=12);a.set(title=key.replace('_',' '),xlabel='original x1',ylabel='original x2')
    save(fig,'04_scale_changes.png')
    fig,ax=plt.subplots(1,2,figsize=(9,3.2));ax[0].plot([v['k'] for v in report['k_curve']],[v['inertia'] for v in report['k_curve']],'o-');ax[0].set(xlabel='K',ylabel='Objective',title='More centers can reduce distortion')
    e=report['empty']['history'][0];x=np.array([0,0,1,10]);ax[1].scatter(x,[0,.1,0,0],c=e['labels_before'],cmap='tab10',s=65);ax[1].scatter(np.array(e['centers_after']).ravel(),[.35]*3,marker='X',s=95,c='black');ax[1].set(yticks=[0,.35],yticklabels=['Samples','New centers'],xlabel='x',ylim=(-.2,.6),title='Empty center moves to residual point')
    save(fig,'05_k_and_empty.png')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',default=str(ROOT/'experiment-result.json'));p.add_argument('--directory',default=str(ROOT/'outputs/figures'));a=p.parse_args();make(json.loads(Path(a.report).read_text()),a.directory)
