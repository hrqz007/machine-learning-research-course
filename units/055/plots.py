from pathlib import Path
import argparse,json,os
os.environ.setdefault('MPLCONFIGDIR','/tmp/ml055-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from experiment import ROOT,MAIN,prior_correct
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
LABELS=['natural .5','natural 1/9','natural tuned','weighted .5','weight corr .5','resample .5','resamp corr .5','oracle 1/9']

def draw(r,directory):
 d=Path(directory);d.mkdir(parents=True,exist_ok=True);paths=[]
 def save(fig,name):
  fig.tight_layout();p=d/name;fig.savefig(p,dpi=170,bbox_inches='tight');plt.close(fig);paths.append(p)
 fig,ax=plt.subplots(figsize=(9,3.6));ax.axis('off')
 for x,y,t in [(.02,.7,'Train 2000\nfit / weight / resample'),(.40,.7,'Validation 1200\nnatural prevalence\nchoose threshold'),(.77,.7,'Test 4000\nnatural prevalence\nreport once')]:ax.text(x,y,t,va='center',bbox=dict(fc='#e9f1fa',ec='#789',boxstyle='round,pad=.5'))
 ax.annotate('',xy=(.39,.7),xytext=(.31,.7),arrowprops=dict(arrowstyle='->'));ax.annotate('',xy=(.76,.7),xytext=(.66,.7),arrowprops=dict(arrowstyle='->'))
 ax.text(.03,.2,'Duplicated training source IDs must stay in [0, 1999].\nNo balancing or threshold tuning on test labels.',color='#a24a26');save(fig,'01_protocol.png')
 fig,axes=plt.subplots(1,2,figsize=(10,3.8));x=np.arange(8)
 for ax,key,title in [(axes[0],'recall','Minority recall'),(axes[1],'cost_per_case','Cost = (8 FN + FP) / N')]:
  ax.bar(x,[r['summary'][m][key] for m in MAIN],color=['#2563a6']*3+['#c65b2c']*2+['#218573']*2+['#777']);ax.set(xticks=x,xticklabels=LABELS,title=title);ax.tick_params(axis='x',rotation=45)
 save(fig,'02_decisions.png')
 fig,ax=plt.subplots(figsize=(8.6,3.7));s=r['runs'][0];xs=[z['threshold'] for z in s['threshold_trials']];ys=[z['cost'] for z in s['threshold_trials']];ax.plot(xs,ys);ax.axvline(s['selected_threshold'],ls='--',c='#c65b2c',label=f"Validation choice {s['selected_threshold']:.2f}");ax.axvline(1/9,ls=':',c='#218573',label='Analytic cost threshold 1/9');ax.set(xlabel='Threshold',ylabel='Validation cost per case',title='Replicate 0: threshold chosen without test labels');ax.legend();save(fig,'03_threshold.png')
 fig,axes=plt.subplots(1,2,figsize=(10,3.6));q=np.linspace(0,1,501);axes[0].plot(q,prior_correct(q,.5,.08),label='source prior .50 -> target .08');axes[0].plot(q,q,'--',c='#888',label='identity');axes[0].set(xlabel='Resampled-distribution probability q',ylabel='Target probability p',title='Exact prior-shift mapping under assumptions');axes[0].legend()
 methods=['natural_05','weighted_05','weighted_corrected_05','oversampled_05','oversampled_corrected_05'];labs=['natural','weighted','w corrected','resampled','r corrected'];axes[1].bar(labs,[r['summary'][m]['brier'] for m in methods],color=['#2563a6','#c65b2c','#d99b7a','#218573','#81b8a9']);axes[1].set(ylabel='Natural-test Brier score',title='Lower is better; correction is approximate in finite samples');axes[1].tick_params(axis='x',rotation=25);save(fig,'04_correction.png')
 fig,ax=plt.subplots(figsize=(8,4));s=r['runs'][0];methods=['natural_05','natural_cost','natural_tuned','weighted_05','oversampled_05'];labs=['natural .5','natural 1/9','natural tuned','weighted .5','resample .5']
 for i,m in enumerate(methods):
  a=s['methods'][m]['metrics'];lo,hi=a['recall_wilson95'];ax.errorbar(a['recall'],i,xerr=[[a['recall']-lo],[hi-a['recall']]],fmt='o',capsize=4,color='#2563a6');ax.text(1.02,i,f"{a['tp']}/{a['positives']}",va='center')
 ax.set(yticks=range(5),yticklabels=labs,xlabel='Recall and pointwise Wilson 95% interval',xlim=(0,1),title='Replicate 0; fixed classifier, natural test positives');ax.text(1.02,4.5,'TP / P',fontsize=9);save(fig,'05_minority_intervals.png')
 fig,axes=plt.subplots(1,2,figsize=(10,3.6));labs=['natural 1/9','selected only','oracle IPW','noisy labels','missing=0'];methods=['selected_only','oracle_ipw','noisy_labels','missing_as_negative']
 for ax,key in zip(axes,['cost_per_case','brier']):
  vals=[r['summary']['natural_cost'][key]]+[r['label_summary'][m][key] for m in methods];ax.bar(labs,vals,color=['#2563a6','#c65b2c','#218573','#9776ae','#777']);ax.set(ylabel=key,title='Fixed threshold 1/9; natural clean test labels');ax.tick_params(axis='x',rotation=25)
 save(fig,'06_biased_labels.png')
 fig,ax=plt.subplots(figsize=(8,3.8));s=r['runs'][0];y=np.array(s['test_labels']);bins=np.linspace(0,1,11)
 for m,label in [('natural_05','natural'),('weighted_05','weighted raw'),('weighted_corrected_05','weighted corrected')]:
  p=np.array(s['methods'][m]['probability']);xx=[];yy=[]
  for i in range(10):
   sel=(p>=bins[i])&(p<bins[i+1] if i<9 else p<=1)
   if sel.sum()>=10:xx.append(p[sel].mean());yy.append(y[sel].mean())
  ax.plot(xx,yy,'o-',label=label)
 ax.plot([0,1],[0,1],'--',c='#aaa');ax.set(xlabel='Mean predicted probability in fixed-width bin',ylabel='Observed positive fraction',xlim=(0,1),ylim=(0,1),title='Replicate 0; bins with fewer than 10 points omitted');ax.legend();save(fig,'07_reliability.png')
 return paths
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--report',default=str(ROOT/'experiment-result.json'));p.add_argument('--directory',default=str(ROOT/'outputs/figures'));a=p.parse_args();print('\n'.join(map(str,draw(json.loads(Path(a.report).read_text()),a.directory))))
