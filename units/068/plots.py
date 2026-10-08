"""所有图直接由已执行报告绘制；英文图标签在中文图注逐项说明。"""
import argparse,json
from pathlib import Path
import numpy as np
from common import ROOT,figure_setup,load_data
C=['#64748b','#2563eb','#e16b35','#159a84']
def draw(r,directory):
    plt=figure_setup();out=Path(directory);out.mkdir(parents=True,exist_ok=True)
    def save(fig,name):fig.savefig(out/name,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,2.7));ax.set_xlim(0,10);ax.set_ylim(0,3);ax.axis('off')
    for x,title,body,c in [(0,'TRAIN 450','fit scaling + models',C[1]),(3.5,'VALIDATION 225','9 configs -> lock winner',C[2]),(7,'TEST 225','one final audit',C[3])]:
        ax.text(x+.1,2,title,fontweight='bold',color=c,fontsize=13);ax.text(x+.1,1.15,body,fontsize=10);ax.add_patch(plt.Rectangle((x,.75),2.8,1.75,fill=False,edgecolor=c,lw=2))
    ax.annotate('',xy=(3.4,1.6),xytext=(2.85,1.6),arrowprops={'arrowstyle':'->'});ax.annotate('',xy=(6.9,1.6),xytext=(6.35,1.6),arrowprops={'arrowstyle':'->'});save(fig,'01_protocol.png')
    fig,axes=plt.subplots(1,3,figsize=(9,3),sharey=True)
    for ax,(family,col) in zip(axes,zip(r['finalists'],C[1:])):
        z=[x for x in r['candidate_rows'] if x['family']==family];ax.plot(range(3),[x['validation_log_loss'] for x in z],'o-',color=col);ax.set_xticks(range(3),[str(x['parameter']) for x in z]);ax.set_title(family);ax.set_xlabel('C / max_depth / k')
    axes[0].set_ylabel('Validation log loss (lower better)');save(fig,'02_fair_search.png')
    fam=list(r['test']);fig,axes=plt.subplots(1,2,figsize=(9,3.3));axes[0].bar(fam,[r['test'][f]['metrics']['log_loss'] for f in fam],color=C);axes[0].set_ylabel('Test log loss');acc=np.array([r['test'][f]['metrics']['accuracy'] for f in fam]);ci=np.array([r['test'][f]['metrics']['accuracy_wilson_95'] for f in fam]);axes[1].errorbar(range(4),acc,yerr=[acc-ci[:,0],ci[:,1]-acc],fmt='o',color=C[1],capsize=4);axes[1].set_xticks(range(4),fam);axes[1].set_ylim(.35,1);axes[1].set_ylabel('Accuracy with Wilson 95% interval');save(fig,'03_heldout_metrics.png')
    fig,ax=plt.subplots(figsize=(6,4));ax.plot([0,1],[0,1],'--',c='gray',label='ideal')
    for f,col in zip(fam,C):
        z=[a for a in r['test'][f]['metrics']['calibration'] if a['n']];ax.plot([a['mean_probability'] for a in z],[a['event_rate'] for a in z],'o-',c=col,label=f)
    ax.set(xlabel='Mean predicted probability in bin',ylabel='Observed event fraction',xlim=(0,1),ylim=(0,1));ax.legend();save(fig,'04_calibration.png')
    fig,ax=plt.subplots(figsize=(7,3));families=list(r['paired_vs_logistic']);z=[r['paired_vs_logistic'][f] for f in families];means=np.array([a['mean_delta'] for a in z]);interval=np.array([a['percentile_95'] for a in z]);ax.errorbar(means,range(3),xerr=[means-interval[:,0],interval[:,1]-means],fmt='o',capsize=4,color=C[2]);ax.axvline(0,c='gray',ls='--');ax.set_yticks(range(3),families);ax.set_xlabel('Paired log-loss difference versus logistic; negative is lower');save(fig,'05_paired_intervals.png')
    fig,axes=plt.subplots(1,2,figsize=(9,3));axes[0].bar(fam,[r['test'][f]['median_batch_seconds']*1e3 for f in fam],color=C);axes[0].set_ylabel('Median warm225-row batch (ms)');axes[1].bar(fam,[r['test'][f]['serialized_model_bytes']/1024 for f in fam],color=C);axes[1].set_ylabel('Serialized model size (KiB, not RAM)');save(fig,'06_costs.png')
    fig,ax=plt.subplots(figsize=(7,3.6));x=np.arange(4)
    for off,sl,col in [(-.18,'abs_x0_below_0.5',C[1]),(.18,'abs_x0_at_least_0.5',C[2])]:ax.bar(x+off,[r['test'][f]['slices'][sl]['log_loss'] for f in fam],width=.35,label=sl,color=col)
    ax.set_xticks(x,fam);ax.set_ylabel('Test slice log loss');ax.legend(fontsize=8);save(fig,'07_failure_slices.png')
    return sorted(p.name for p in out.glob('*.png'))
def main():
    p=argparse.ArgumentParser();p.add_argument('--report',default=str(ROOT/'experiment-result.json'));p.add_argument('--directory',default=str(ROOT/'outputs/figures'));a=p.parse_args();print(json.dumps(draw(json.loads(Path(a.report).read_text()),a.directory)))
if __name__=='__main__':main()
