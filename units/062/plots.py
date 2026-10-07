"""Six reproducible figures linking derivatives to actual stage predictions."""
import argparse,json
from pathlib import Path
import numpy as np
from common import ROOT,load_data,figure_setup
from gbdt import GBDT,sigmoid

def main():
 p=argparse.ArgumentParser();p.add_argument('--report',default=str(ROOT/'experiment-result.json'));p.add_argument('--directory',default=str(ROOT/'outputs/figures'));a=p.parse_args();out=Path(a.directory);out.mkdir(parents=True,exist_ok=True);r=json.loads(Path(a.report).read_text());plt=figure_setup()
 def save(fig,name):fig.tight_layout();fig.savefig(out/name,bbox_inches='tight');plt.close(fig)
 F=np.linspace(-4,4,300);fig,axes=plt.subplots(1,2,figsize=(8,3.5));axes[0].plot(F,.5*(1-F)**2,label='half squared loss y=1');axes[0].plot(F,np.logaddexp(0,F)-F,label='log loss y=1');axes[1].plot(F,1-F,label='squared negative gradient');axes[1].plot(F,1-sigmoid(F),label='logistic negative gradient');[ax.set(xlabel='raw score F') for ax in axes];[ax.legend() for ax in axes];save(fig,'01_loss_derivatives.png')
 fig,axes=plt.subplots(1,2,figsize=(8,3.5));x=np.arange(1,5)
 for kind,ax in zip(['squared','logistic'],axes):
  for h in r['toy'][kind]:ax.plot(x,h['pseudo_residual'],'o-',label=f"residual round {h['round']}");ax.plot(x,h['direction'],'x--',label=f"tree round {h['round']}")
  ax.set(title=kind,xlabel='sample index',ylabel='update component');ax.legend(fontsize=8)
 save(fig,'02_hand_rounds.png')
 d=load_data();X=d['train']['x'][:,None];grid=np.linspace(-3,3,400)[:,None];m=GBDT('squared',40,.1,1,3).fit(X,d['train']['y_reg']);stages=list(m.staged_decision_function(grid));fig,ax=plt.subplots(figsize=(7,3.9));ax.scatter(X[:,0],d['train']['y_reg'],s=10,alpha=.3,label='train');ax.plot(grid[:,0],np.sin(1.6*grid[:,0])+.25*grid[:,0],color='black',label='true conditional mean')
 for i in [0,9,39]:ax.plot(grid[:,0],stages[i],label=f'round {i+1}')
 ax.set(xlabel='x',ylabel='regression prediction');ax.legend(ncol=2);save(fig,'03_additive_regression.png')
 fig,axes=plt.subplots(1,2,figsize=(8,3.5));
 for kind,ax in zip(['squared','logistic'],axes):
  for split,curve in r['results'][kind]['curves'].items():ax.plot(curve,label=split)
  ax.set(title=kind,xlabel='round (0 = initial constant)',ylabel='mean loss');ax.legend()
 save(fig,'04_loss_curves.png')
 fig,axes=plt.subplots(1,2,figsize=(8,3.5));m=GBDT('logistic',40,.1,1,3).fit(X,d['train']['y_class']);F=m.decision_function(grid);axes[0].plot(grid[:,0],F,label='learned full logit');axes[0].axhline(0,color='gray',linestyle='--');axes[1].plot(grid[:,0],sigmoid(F),label='sigmoid(learned logit)');axes[1].plot(grid[:,0],sigmoid(1.6*np.sin(grid[:,0])+.35*grid[:,0]),label='true probability');axes[1].axhline(.5,color='gray',linestyle='--');[ax.set(xlabel='x') for ax in axes];axes[0].set_ylabel('raw score');axes[1].set_ylabel('P(y = 1)');[ax.legend(fontsize=8) for ax in axes];save(fig,'05_logit_probability.png')
 fig,axes=plt.subplots(1,2,figsize=(8,3.5));
 for eta in [.05,.1,.5]:
  m=GBDT('logistic',40,eta,1,3).fit(X,d['train']['y_class']);axes[0].plot(m.losses_,label=f'eta = {eta}');
  from gbdt import loss
  axes[1].plot([loss(d['validation']['y_class'],F,'logistic') for F in m.staged_decision_function(d['validation']['x'][:,None])],label=f'eta = {eta}')
 axes[0].set(xlabel='round (including initial)',ylabel='training log loss');axes[1].set(xlabel='round index (0 means round 1)',ylabel='validation log loss');[ax.legend() for ax in axes];save(fig,'06_learning_rate.png')
if __name__=='__main__':main()
