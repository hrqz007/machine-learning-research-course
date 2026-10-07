"""Teaching diagrams and empirical plots; every curve labels its source."""
import argparse,json
from pathlib import Path
import numpy as np
from matplotlib.patches import FancyBboxPatch
from common import ROOT,figure_setup,load_data
from boosting import leaf_weight,split_gain

def build(report,directory):
    plt=figure_setup();d=Path(directory);d.mkdir(parents=True,exist_ok=True);names=[]
    def save(fig,name):fig.tight_layout();fig.savefig(d/name,bbox_inches='tight');plt.close(fig);names.append(name)
    fig,ax=plt.subplots(figsize=(9,3.2));ax.axis('off')
    labels=['TRAIN 480\nfit trees & preprocessing','STOP 120\nchoose training length','TUNE 160\nchoose 1 of 4 settings','CALIBRATION 160\nfit fixed sigmoid','TEST 240\nfinal estimate only']
    for i,(label,color) in enumerate(zip(labels,['#dbeafe','#e0e7ff','#ccfbf1','#fef3c7','#fee2e2'])):
        x=i*1.9;ax.add_patch(FancyBboxPatch((x,0),1.7,1.3,boxstyle='round,pad=.04',facecolor=color,edgecolor='#64748b'));ax.text(x+.85,.65,label,ha='center',va='center',fontsize=8)
        if i<4:ax.annotate('',xy=(x+1.89,.65),xytext=(x+1.73,.65),arrowprops={'arrowstyle':'->'})
    ax.set(xlim=(-.2,9.5),ylim=(-.4,1.7),title='Different data answer different questions');save(fig,'01_data_roles.png')
    lam=np.linspace(0,8,150);fig,ax=plt.subplots(1,2,figsize=(8.5,3.3));ax[0].plot(lam,[-leaf_weight(1,.5,v) for v in lam],color='#2563eb');ax[0].set(xlabel='L2 penalty lambda',ylabel='Absolute leaf weight',title='G=1, H=0.5');ax[1].plot(lam,[split_gain(1,.5,-1,.5,v) for v in lam],color='#0d9488');ax[1].set(xlabel='L2 penalty lambda',ylabel='Toy split gain',title='Shrink a weakly supported update');save(fig,'02_newton_regularization.png')
    fig,axs=plt.subplots(2,2,figsize=(8.5,5.3),sharey=True)
    for row,ax in zip(report['hist_candidates'],axs.flat):
        ax.plot(row['train_loss'],label='training',color='#2563eb');ax.plot(row['stop_loss'],label='stopping set',color='#d97706');ax.axvline(row['iterations'],color='#64748b',ls=':');ax.set(title=f"Candidate {row['index']}, stopped at {row['iterations']}",xlabel='Boosting iteration',ylabel='Log loss');ax.legend(fontsize=8)
    save(fig,'03_stopping_curves.png')
    fig,ax=plt.subplots(figsize=(8.5,3.5));x=np.arange(4);ax.bar(x-.18,[r['tune_log_loss'] for r in report['hist_candidates']],.35,label='Histogram tree',color='#2563eb');ax.bar(x+.18,[r['tune_log_loss'] for r in report['linear_candidates']],.35,label='Linear logistic',color='#0d9488');ax.set(xticks=x,xlabel='Candidate index within each family',ylabel='Tuning log loss',title='Four candidates per family; computational budgets differ');ax.set_ylim(0, .85);ax.legend();save(fig,'04_budget_selection.png')
    fig,ax=plt.subplots(1,2,figsize=(8.5,3.6));ax[0].plot([0,1],[0,1],ls='--',color='#64748b',label='ideal')
    for name,color in [('raw','#2563eb'),('calibrated','#d97706')]:
        rows=[r for r in report['reliability'][name] if r['count']];ax[0].plot([r['mean_probability'] for r in rows],[r['fraction_positive'] for r in rows],'-o',label=name,color=color)
        ax[1].bar(np.arange(5)+(-.18 if name=='raw' else .18),[r['count'] for r in report['reliability'][name]],.35,label=name,color=color)
    ax[0].set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted probability',ylabel='Observed positive fraction',title='Test reliability, five fixed bins');ax[0].legend();ax[1].set(xticks=range(5),xticklabels=['0-.2','.2-.4','.4-.6','.6-.8','.8-1'],xlabel='Probability bin',ylabel='Number of test rows',title='Small bins are noisy');ax[1].set_ylim(0, 1.4*max(v['count'] for rows in report['reliability'].values() for v in rows));ax[1].legend();save(fig,'05_calibration.png')
    fig,ax=plt.subplots(1,2,figsize=(8.5,3.4));keys=['hist_raw','hist_sigmoid','linear'];colors=['#2563eb','#d97706','#0d9488']
    for a,metric in zip(ax,['log_loss','brier']):a.bar(keys,[report['test'][k][metric] for k in keys],color=colors);a.set(ylabel=metric,title='Final sealed test');a.tick_params(axis='x',labelsize=9)
    save(fig,'06_final_metrics.png');return names
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',default=str(ROOT/'experiment-result.json'));p.add_argument('--directory',default=str(ROOT/'outputs/figures'));a=p.parse_args();print(build(json.loads(Path(a.report).read_text()),a.directory))
