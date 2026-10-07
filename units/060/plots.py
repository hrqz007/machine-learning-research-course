"""Original figures showing bootstrap, variance, OOB and group dependence."""
from pathlib import Path
import argparse,json,numpy as np
from common import ROOT,figure_setup

def build(r,directory):
    plt=figure_setup();directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);colors=['#2563eb','#d97706','#16836b']
    def save(f,name):f.tight_layout();f.savefig(directory/name,bbox_inches='tight');plt.close(f)
    samples=np.array(r['ensembles']['3']['bootstrap_samples']);counts=np.array([np.bincount(q,minlength=360) for q in samples])
    f,ax=plt.subplots(1,2,figsize=(10,4));im=ax[0].imshow(counts[:12,:24],aspect='auto',cmap='Blues',vmin=0,vmax=5);ax[0].set(xlabel='Original row (first 24)',ylabel='Bootstrap tree (first 12)',title='0 means out of bag');f.colorbar(im,ax=ax[0],label='Times selected');ax[1].hist((counts>0).sum(axis=1),bins=12,color=colors[0],alpha=.8);ax[1].axvline(r['bootstrap_theory']['expected_unique'],color=colors[1],ls='--',label='Finite-n expectation');ax[1].set(xlabel='Unique rows in a 360-draw bootstrap',ylabel='Number of trees');ax[1].legend(fontsize=8);save(f,'01_bootstrap.png')
    f,a=plt.subplots(figsize=(9,4));B=np.arange(1,101)
    for rho,c in zip([0,.2,.8],colors):
        a.plot(B,rho+(1-rho)/B,color=c,label=f'rho={rho}');rows=[q for q in r['variance_simulation'] if q['rho']==rho];a.scatter([q['B'] for q in rows],[q['empirical'] for q in rows],color=c,s=25)
    a.set(xlabel='Number of averaged predictors B',ylabel='Variance / individual variance',title='Lines: theory; dots: 12,000 simulated repetitions');a.legend();save(f,'02_variance.png')
    f,ax=plt.subplots(1,2,figsize=(10,4))
    for m,c in zip(['1','3','8'],colors):
        q=r['ensembles'][m];ax[0].plot([z['B'] for z in q['curves']],[z['brier'] for z in q['curves']],marker='o',color=c,label=f'm={m} test');ax[1].plot([z['B'] for z in q['oob']],[z['brier'] for z in q['oob']],marker='o',color=c,label=f'm={m} OOB')
    for a in ax:a.set(xlabel='B',ylabel='Brier (lower is better)',xscale='log');a.legend()
    ax[0].set_title('Frozen, predeclared test comparisons');ax[1].set_title('OOB: coverage changes for small B');save(f,'03_learning_curves.png')
    f,ax=plt.subplots(1,2,figsize=(10,4));q=r['ensembles']['3']['oob'];ax[0].plot([z['B'] for z in q],[z['coverage'] for z in q],marker='o',color=colors[2]);x=np.arange(1,81);p=r['bootstrap_theory']['row_absent_probability'];ax[0].plot(x,1-(1-p)**x,ls='--',color=colors[1],label='Expected coverage');ax[0].set(xlabel='B',ylabel='Fraction with OOB voters',ylim=(0,1.05));ax[0].legend();ax[1].hist(q[-1]['counts'],bins=14,color=colors[0]);ax[1].set(xlabel='OOB voters per row at B=80',ylabel='Training rows');save(f,'04_oob_coverage.png')
    f,ax=plt.subplots(1,2,figsize=(10,4));ms=[1,3,8];q=[r['ensembles'][str(m)] for m in ms];ax[0].plot(ms,[z['mean_individual_brier'] for z in q],'-o',color=colors[1],label='Mean single-tree Brier');ax[0].plot(ms,[z['curves'][-1]['brier'] for z in q],'-o',color=colors[2],label='Ensemble test Brier');ax[0].set(xlabel='Candidate features per split m',ylabel='Test Brier');ax[0].legend();ax[1].plot(ms,[z['empirical_error_correlation'] for z in q],'-o',color=colors[0]);ax[1].set(xlabel='Candidate features per split m',ylabel='Mean error correlation across test rows',title='Descriptive diagnostic, not fixed-x rho');save(f,'05_feature_tradeoff.png')
    f,ax=plt.subplots(1,2,figsize=(10,4));g=r['group_counterexample'];names=['Row OOB','Whole-group OOB','New-group test'];rows=[g['row_oob'],g['group_oob'],g['new_group_test']]
    for a,key in zip(ax,['accuracy','brier']):
        vals=[z[key] for z in rows];a.bar(names,vals,color=colors);a.set(ylabel=key,ylim=(0,max(vals)*1.22));a.tick_params(axis='x',labelsize=9)
        for i,val in enumerate(vals):a.text(i,val+max(vals)*.025,f'{val:.3f}',ha='center')
    save(f,'06_group_counterexample.png')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',default='experiment-result.json');p.add_argument('--directory',default='outputs/figures');a=p.parse_args();build(json.loads(Path(a.report).read_text()),a.directory)
