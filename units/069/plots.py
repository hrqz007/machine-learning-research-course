"""PCA投影几何、谱、重构、单位效应和预测反例。"""
from pathlib import Path
import argparse,json
import numpy as np
from common import ROOT,figure_setup,load_data
from experiment import matrix
from pca import fit,transform,reconstruct

def draw(r,directory):
    plt=figure_setup();out=Path(directory);out.mkdir(parents=True,exist_ok=True)
    def save(fig,name):fig.savefig(out/name,bbox_inches='tight');plt.close(fig)
    H=np.array(r['hand']['X']);R=np.array(r['hand']['reconstruction']);fig,ax=plt.subplots(figsize=(6,4));ax.scatter(H[:,0],H[:,1],c='#2563eb',s=65,label='original');ax.scatter(R[:,0],R[:,1],c='#e16b35',marker='s',s=45,label='projection')
    for i,(x,z) in enumerate(zip(H,R)):
        ax.plot([x[0],z[0]],[x[1],z[1]],c='gray',ls='--');ax.text(x[0]+.1,x[1]+.1,str(i+1))
    ax.plot([-2.5,2.5],[-2.5,2.5],c='#159a84',label='PC1');ax.set(xlim=(-2.7,2.7),ylim=(-2.7,2.7),xlabel='x0',ylabel='x1',aspect='equal');ax.legend(fontsize=9);save(fig,'01_hand_projection.png')
    angles=np.linspace(0,np.pi,181);cov=np.array(r['hand']['covariance']);v=np.column_stack([np.cos(angles),np.sin(angles)]);var=np.einsum('ij,jk,ik->i',v,cov,v);fig,ax=plt.subplots(figsize=(7,3));ax.plot(angles*180/np.pi,var,c='#2563eb');ax.axvline(45,c='#e16b35',ls='--');ax.set(xlabel='Projection direction (degrees)',ylabel='Sample variance',xticks=[0,45,90,135,180]);save(fig,'02_variance_direction.png')
    d=load_data();X=matrix(d['train']);T=matrix(d['test']);m=fit(X,2);fig,axes=plt.subplots(1,2,figsize=(9,3.3));ev=np.array(r['model']['eigenvalues']);axes[0].bar(range(1,7),ev,color='#2563eb');axes[0].set(xlabel='Principal component',ylabel='Sample eigenvalue');cur=r['reconstruction_curve'];axes[1].plot(range(1,7),[z['cumulative_variance_ratio'] for z in cur],'o-',c='#159a84');axes[1].set(xlabel='Number retained',ylabel='Cumulative variance ratio',ylim=(0,1.05));save(fig,'03_spectrum.png')
    fig,axes=plt.subplots(1,2,figsize=(9,3.3));axes[0].plot(range(1,7),[z['train_mean_squared_distance'] for z in cur],'o-',label='train');axes[0].plot(range(1,7),[z['test_mean_squared_distance'] for z in cur],'s-',label='test');axes[0].set(xlabel='Number retained',ylabel='Mean squared Euclidean residual');axes[0].legend();z=transform(m,T);axes[1].scatter(z[:,0],z[:,1],s=13,c='#e16b35',alpha=.7);axes[1].set(xlabel='PC1 score using training mean',ylabel='PC2 score using training mean');save(fig,'04_reconstruction.png')
    fig,axes=plt.subplots(1,2,figsize=(9,3.3));a=r['scaling'];axes[0].bar(range(6),np.abs(a['raw_components'][0]),color='#e16b35');axes[0].set(title='x2 unit multiplied by100',xlabel='Original feature index',ylabel='Absolute PC1 loading');axes[1].bar(range(6),np.abs(a['standardized_components'][0]),color='#159a84');axes[1].set(title='After train-only standardization',xlabel='Standardized feature index',ylabel='Absolute PC1 loading');save(fig,'05_scaling.png')
    S=matrix(d['signal_test']);y=d['signal_test']['y'];fig,axes=plt.subplots(1,2,figsize=(9,3.3));axes[0].scatter(S[:,0],S[:,1],c=y,cmap='coolwarm',s=15);axes[0].axhline(0,c='gray',ls='--');axes[0].set(xlabel='High-variance nuisance',ylabel='Low-variance predictive signal');axes[1].bar(['Both inputs','PC1 only'],[r['prediction_counterexample']['full_accuracy'],r['prediction_counterexample']['pc1_accuracy']],color=['#2563eb','#e16b35']);axes[1].set(ylabel='Held-out classification accuracy',ylim=(0,1));save(fig,'06_signal_counterexample.png')
    fig,ax=plt.subplots(figsize=(6,3));x=np.arange(2);ax.bar(x-.18,r['centering']['shifted_test_score_mean'],width=.35,label='correct: train mean',color='#2563eb');ax.bar(x+.18,r['centering']['incorrect_self_centered_mean'],width=.35,label='wrong: test own mean',color='#e16b35');ax.set_xticks(x,['PC1','PC2']);ax.set_ylabel('Mean score on shifted test data');ax.legend();save(fig,'07_centering_leakage.png')
    return sorted(p.name for p in out.glob('*.png'))
def main():
    p=argparse.ArgumentParser();p.add_argument('--report',default=str(ROOT/'experiment-result.json'));p.add_argument('--directory',default=str(ROOT/'outputs/figures'));a=p.parse_args();print(json.dumps(draw(json.loads(Path(a.report).read_text()),a.directory)))
if __name__=='__main__':main()
