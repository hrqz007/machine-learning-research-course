"""从冻结报告与真实数据重建图；颜色辅助区分，坐标注明含义。"""
import argparse,json
import numpy as np
from common import ROOT,load_data,figure_setup

def main():
    p=argparse.ArgumentParser();p.add_argument('--report',default=str(ROOT/'experiment-result.json'));p.add_argument('--directory',default=str(ROOT/'outputs/figures'));a=p.parse_args()
    from pathlib import Path
    d=Path(a.directory);d.mkdir(parents=True,exist_ok=True);r=json.loads(Path(a.report).read_text());data=load_data();plt=figure_setup()
    def save(fig,name):fig.tight_layout();fig.savefig(d/name);plt.close(fig)
    from latent import ppca,posterior
    from experiment import matrix,truth
    from generate_data import TRUE_W
    X=matrix(data['train']);A=matrix(data['audit']);mu,W,s,ev,U=ppca(X,2);R=np.array(r['rotation']);scores,rec=posterior(A,mu,W,s)
    fig,ax=plt.subplots(1,3,figsize=(9,3.4));maps=[TRUE_W,W,W@R]
    for axis,M,title in zip(ax,maps,['generator W','fitted W','equivalent W R']):
        im=axis.imshow(M,cmap='coolwarm',vmin=-1.6,vmax=1.6);axis.set(title=title,xlabel='factor coordinate',ylabel='observed feature',xticks=[0,1],yticks=range(4))
        for i in range(4):
            for j in range(2):axis.text(j,i,f'{M[i,j]:.2f}',ha='center',va='center',fontsize=9)
    save(fig,'01_loadings.png')
    fig,ax=plt.subplots(1,2,figsize=(8,3.3));z=truth(data['audit'])
    for axis,S,title in zip(ax,[scores,scores@R],['posterior coordinates','rotated posterior coordinates']):axis.scatter(*S.T,c=z[:,0],cmap='viridis',s=12);axis.set(xlabel='coordinate 0',ylabel='coordinate 1',title=title);axis.set_aspect('equal')
    save(fig,'02_rotation.png')
    fig,ax=plt.subplots(figsize=(8,3.2));ax.bar(range(1,5),ev,color=['#0891b2','#0891b2','#94a3b8','#94a3b8']);ax.axhline(s,color='#ea580c',label='PPCA noise = mean discarded eigenvalues');ax.set(xlabel='principal direction',ylabel='sample variance (divide by n)',xticks=range(1,5));ax.legend(fontsize=8);save(fig,'03_spectrum.png')
    fig,ax=plt.subplots(1,2,figsize=(9,3.2));ax[0].bar(['raw factors','aligned factors'],[r['raw_factor_mse'],r['aligned_factor_mse']],color=['#64748b','#0891b2']);ax[0].set(ylabel='audit factor mean squared error',title='Alignment learned on training truth')
    h=r['heteroscedastic'];ax[1].bar(['PPCA','factor analysis'],[h['ppca_audit_mean_log_density'],h['fa_audit_mean_log_density']],color=['#0891b2','#ea580c']);ax[1].set(ylabel='audit mean log density',title='Unequal-noise data; higher is better');save(fig,'04_validation.png')
if __name__=='__main__':main()
