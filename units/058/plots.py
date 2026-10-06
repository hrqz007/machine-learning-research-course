"""All figures are original and reconstructed from the verified experiment."""
from pathlib import Path
import argparse,json
import numpy as np
from common import ROOT,figure_setup
from experiment import load
from tree import ShallowTree

def build(report,directory):
    r=json.loads(Path(report).read_text());directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    plt=figure_setup();files=[]
    def save(fig,name):
        fig.tight_layout();path=directory/name;fig.savefig(path,bbox_inches='tight');plt.close(fig);files.append(path)
    p=np.linspace(0,1,501);h=np.zeros_like(p);m=(p>0)&(p<1);h[m]=-p[m]*np.log2(p[m])-(1-p[m])*np.log2(1-p[m])
    fig,ax=plt.subplots(figsize=(8,3.4));ax.plot(p,2*p*(1-p),label='Gini: 2p(1-p)',color='#147d92',lw=2);ax.plot(p,h,label='Entropy (bits)',color='#c35f35',lw=2)
    ax.set(xlabel='Positive-class fraction p',ylabel='Impurity',title='Both criteria are zero at pure nodes');ax.legend();ax.grid(alpha=.15);save(fig,'01_impurity.png')
    fig,axes=plt.subplots(1,2,figsize=(9,3.6))
    for crit,color,label in [('gini','#147d92','Gini gain'),('entropy','#c35f35','Entropy gain (bits)')]:
        rows=r['hand'][crit];axes[0].plot([a['threshold'] for a in rows],[a['gain'] for a in rows],'o-',color=color,label=label)
    axes[0].set(xlabel='Candidate threshold',ylabel='Local gain',title='Classification: ties at 2.5 and 4.5');axes[0].legend(fontsize=8)
    rows=r['hand']['squared_error'];axes[1].bar([a['threshold'] for a in rows],[a['gain'] for a in rows],width=.65,color='#147d92');axes[1].set(xlabel='Candidate threshold',ylabel='Variance reduction',title='Regression: best threshold = 3.5');save(fig,'02_candidate_gains.png')
    fig,ax=plt.subplots(figsize=(10,4.4));nodes=r['models']['gini']['nodes'];pos={0:(.5,.87),1:(.25,.52),4:(.75,.52),2:(.125,.12),3:(.375,.12),5:(.625,.12),6:(.875,.12)}
    for n in nodes:
        x,y=pos[n['id']]
        if n['feature'] is not None:
            for branch,label in [('left','yes'),('right','no')]:
                xx,yy=pos[n[branch]];ax.annotate('',(xx,yy+.075),(x,y-.07),arrowprops=dict(arrowstyle='->',color='#64748b'));ax.text((x+xx)/2,(y+yy)/2+.02,label,ha='center',fontsize=9)
            text=f"#{n['id']}  x{n['feature']} <= {n['threshold']:.4f}\nn={n['n']}; p={n['value']:.3f}\nGini={n['impurity']:.3f}"
        else:text=f"Leaf #{n['id']}\nn={n['n']}; p={n['value']:.3f}\nclass={int(n['value']>.5)}"
        ax.text(x,y,text,ha='center',va='center',fontsize=9,bbox=dict(boxstyle='round,pad=.5',facecolor='#e3f2f4' if n['feature'] is not None else '#fff2dc',edgecolor='#9baeb6'))
    ax.set(xlim=(0,1),ylim=(-.02,1),title='Fitted Gini tree: depth 0 at the root');ax.axis('off');save(fig,'03_tree_structure.png')
    X,y=load('classification_train');model=ShallowTree('gini',2,12).fit(X,y)
    a=np.linspace(-2,2,220);xx,yy=np.meshgrid(a,a);grid=np.column_stack([xx.ravel(),yy.ravel()]);true=np.where(xx<=0,.12,np.where(yy<=.35,.30,.88));est=model.predict_proba(grid)[:,1].reshape(xx.shape)
    fig,axes=plt.subplots(1,2,figsize=(8.5,3.8),constrained_layout=True)
    for ax,z,title in zip(axes,[true,est],['Generating probability (known only here)','Fitted leaf probability + training labels']):
        im=ax.pcolormesh(xx,yy,z,vmin=0,vmax=1,cmap='Blues',shading='auto');ax.set(xlabel='x0',ylabel='x1',title=title);ax.set_aspect('equal')
    axes[1].scatter(X[:,0],X[:,1],c=y,cmap='coolwarm',s=10,alpha=.65,edgecolors='none');fig.colorbar(im,ax=axes,label='P(Y = 1)',shrink=.85)
    path=directory/'04_regions.png';fig.savefig(path,bbox_inches='tight');plt.close(fig);files.append(path)
    X,y=load('regression_train');tx,ty=load('regression_test');model=ShallowTree('squared_error',2,12).fit(X,y)
    xx=np.linspace(-4,4,700);mu=np.where(xx<=-1,-1,np.where(xx<=1,2,.5))
    fig,ax=plt.subplots(figsize=(8.5,3.6));ax.axvspan(-4,-3,color='#ddd',alpha=.3);ax.axvspan(3,4,color='#ddd',alpha=.3)
    ax.scatter(X[:,0],y,s=12,alpha=.4,label='Train');ax.scatter(tx[:,0],ty,s=12,alpha=.45,label='Test',color='#c35f35');ax.plot(xx,mu,'--',color='#222',lw=1.3,label='Generating mean');ax.step(xx,model.predict(xx[:,None]),where='mid',color='#147d92',lw=2,label='Shallow tree')
    ax.set(xlabel='x0 (training support: -3 to 3)',ylabel='Target / prediction',title='Piecewise constants do not learn an extrapolation law');ax.legend(ncol=4,fontsize=8,loc='upper center');save(fig,'05_regression.png')
    fig,axes=plt.subplots(1,2,figsize=(8,3.5));X=np.array(r['xor']['X']);y=np.array(r['xor']['y'])
    for ax,title in zip(axes,['One-step gain = 0: our rule stops','Two-level construction predicts all 4']):
        ax.scatter(X[:,0],X[:,1],c=y,cmap='coolwarm',s=250,edgecolors='#333',vmin=0,vmax=1)
        for (a,b),v in zip(X,y):ax.text(a,b,str(v),ha='center',va='center',color='white',weight='bold')
        ax.set(xlabel='x0',ylabel='x1',xlim=(-.3,1.3),ylim=(-.3,1.3),xticks=[0,1],yticks=[0,1],title=title);ax.set_aspect('equal')
    axes[1].axvline(.5,color='#555',ls='--');axes[1].axhline(.5,color='#555',ls='--');save(fig,'06_xor.png')
    return files

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',default=str(ROOT/'experiment-result.json'));p.add_argument('--directory',default=str(ROOT/'outputs/figures'));a=p.parse_args()
    for f in build(a.report,a.directory):print(f)
