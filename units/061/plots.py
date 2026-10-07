"""Six explanatory figures; all curves recomputed from the frozen experiment."""
import argparse,json
import numpy as np
from common import ROOT,figure_setup,load_data
from adaboost import AdaBoost

def main():
 p=argparse.ArgumentParser();p.add_argument('--report',default=str(ROOT/'experiment-result.json'));p.add_argument('--directory',default=str(ROOT/'outputs/figures'));a=p.parse_args();out=__import__('pathlib').Path(a.directory);out.mkdir(parents=True,exist_ok=True);r=json.loads(open(a.report).read());plt=figure_setup()
 def save(fig,name):fig.tight_layout();fig.savefig(out/name,bbox_inches='tight');plt.close(fig)
 z=np.linspace(-2,3,300);fig,ax=plt.subplots(figsize=(7,3.8));ax.plot(z,np.exp(-z),label='exp(-margin)');ax.plot(z,(z<=0).astype(float),label='zero-one upper envelope convention');ax.set(xlabel='signed margin y F(x)',ylabel='loss',title='One wrong confident vote can dominate exponential loss');ax.legend();save(fig,'01_margin_loss.png')
 fig,axes=plt.subplots(1,3,figsize=(9,3.5));weights=[np.full(6,1/6)]+[np.array(h['weights_after']) for h in r['toy']]
 for i,(ax,w) in enumerate(zip(axes,weights)):ax.bar(np.arange(1,7),w,color=['#357ca5']*3+['#db7048']+['#357ca5']*2);ax.set(title=f'Before round {i+1}',xlabel='sample index',ylim=(0,.56),ylabel='normalized weight')
 save(fig,'02_two_round_weights.png')
 fig,axes=plt.subplots(1,2,figsize=(8,3.5));
 for mode in ['clean','noisy']:
  v=r['results'][mode];axes[0].plot([h['exp_loss'] for h in v['history']],label=mode);axes[1].plot(v['curves']['test'],label=mode)
 axes[0].set(xlabel='round index (0 means round 1)',ylabel='training exponential loss',yscale='log');axes[1].set(xlabel='round index (0 means round 1)',ylabel='clean test error');[ax.legend() for ax in axes];save(fig,'03_learning_curves.png')
 fig,axes=plt.subplots(1,2,figsize=(8,3.5));
 for mode in ['clean','noisy']:
  v=r['results'][mode];axes[0].plot(np.arange(1,41),v['flipped_weight_mass'],label=mode);axes[1].plot(np.arange(1,41),v['effective_sample_size'],label=mode)
 axes[0].axhline(.1,color='gray',linestyle='--',label='row fraction = 0.10');axes[0].set(xlabel='round',ylabel='weight on designated 18 rows');axes[1].set(xlabel='round',ylabel='effective sample size 1 / sum(w^2)');[ax.legend() for ax in axes];save(fig,'04_noise_concentration.png')
 d=load_data()['train'];X=np.column_stack([d['x0'],d['x1']]);u=np.linspace(-2.5,2.5,160);xx,yy=np.meshgrid(u,u);grid=np.column_stack([xx.ravel(),yy.ravel()]);fig,axes=plt.subplots(1,2,figsize=(8,3.8))
 for ax,mode in zip(axes,['clean','noisy']):
  m=AdaBoost(40).fit(X,d['y_'+mode]);ax.contourf(xx,yy,m.predict(grid).reshape(xx.shape),levels=[-2,0,2],alpha=.3,cmap='coolwarm');ax.scatter(X[:,0],X[:,1],c=d['y_'+mode],s=12,cmap='coolwarm');ax.scatter(X[d['flipped']==1,0],X[d['flipped']==1,1],facecolors='none',edgecolors='black',s=45);ax.set(title=mode+' training labels',xlabel='x0',ylabel='x1')
 save(fig,'05_boundaries.png')
 fig,ax=plt.subplots(figsize=(7,3.7));
 for mode in ['clean','noisy']:
  m=AdaBoost(40).fit(X,d['y_'+mode]);margin=d['y_'+mode]*m.decision_function(X)/sum(m.alphas_);ax.hist(margin,bins=np.linspace(-.5,1,26),alpha=.5,label=mode)
 ax.axvline(0,color='black',linestyle='--');ax.set(xlabel='normalized signed training margin',ylabel='number of rows');ax.legend();save(fig,'06_margin_distribution.png')
if __name__=='__main__':main()
