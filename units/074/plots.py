"""从冻结报告与真实数据重建图；颜色辅助区分，坐标注明含义。"""
import argparse,json
import numpy as np
from common import ROOT,load_data,figure_setup

def main():
    p=argparse.ArgumentParser();p.add_argument('--report',default=str(ROOT/'experiment-result.json'));p.add_argument('--directory',default=str(ROOT/'outputs/figures'));a=p.parse_args()
    from pathlib import Path
    d=Path(a.directory);d.mkdir(parents=True,exist_ok=True);r=json.loads(Path(a.report).read_text());data=load_data();plt=figure_setup()
    def save(fig,name):fig.tight_layout();fig.savefig(d/name);plt.close(fig)
    from mixture import expectation
    fig,ax=plt.subplots(figsize=(8,3.2));x=data['train']['x'];grid=np.linspace(-5,5,500);best=r['runs'][r['best_run']];ax.hist(x,bins=40,density=True,color='#cbd5e1',label='training observations')
    density=np.zeros(len(grid))
    for j,(w,m,v) in enumerate(zip(best['weights'],best['means'],best['variances'])):
        part=w*np.exp(-.5*(grid-m)**2/v)/np.sqrt(2*np.pi*v);density+=part;ax.plot(grid,part,label='weighted component '+str(j))
    ax.plot(grid,density,color='#0f172a',lw=2,label='sum');ax.set(xlabel='x (synthetic unit)',ylabel='density (inverse unit)');ax.legend(fontsize=7,ncol=2);save(fig,'01_density.png')
    fig,ax=plt.subplots(figsize=(8,3.2));resp,_=expectation(grid,best['weights'],best['means'],best['variances'])
    for j in range(3):ax.plot(grid,resp[:,j],label='source '+str(j))
    ax.set(xlabel='observed x',ylabel='posterior responsibility',ylim=(-.03,1.03));ax.legend();save(fig,'02_responsibility.png')
    fig,ax=plt.subplots(figsize=(8,3.2))
    for i,run in enumerate(r['runs']):ax.plot(run['history'],label='restart '+str(i),alpha=.8)
    ax.set(xlabel='EM update',ylabel='training log likelihood');ax.legend(fontsize=7,ncol=4);save(fig,'03_restarts.png')
    fig,ax=plt.subplots(figsize=(8,3.2));ax.semilogx([v['variance'] for v in r['collapse']],[v['log_likelihood'] for v in r['collapse']],'o-',color='#dc2626');ax.invert_xaxis();ax.set(xlabel='one component variance decreases ->',ylabel='log likelihood of [0, 2, 4]',title='One narrow component centered exactly on 0');save(fig,'04_collapse.png')
if __name__=='__main__':main()
