"""五幅原创图均直接来自解析式或冻结实验；PNG用于PDF，SVG便于放大。"""
import argparse,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'outputs/mpl'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import font_manager
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
from experiment import run
ROOT=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':11,'figure.dpi':150,'savefig.facecolor':'white','svg.fonttype':'path'})
BLUE='#17658c';ORANGE='#d97920';GREEN='#33765c'

def contour(ax,mean,cov,color,label,scale=2.):
    vals,vecs=np.linalg.eigh(cov);angle=np.degrees(np.arctan2(vecs[1,-1],vecs[0,-1]))
    # 每条曲线是固定Mahalanobis半径；不是分别独立的95%边缘区间。
    for radius in [1.,scale]:
        ax.add_patch(Ellipse(mean,2*radius*np.sqrt(vals[-1]),2*radius*np.sqrt(vals[0]),angle=angle,fill=False,lw=2,color=color,label=label if radius==scale else None))
    ax.plot(*mean,'o',color=color,ms=4)

def save(fig,name,directory):
    fig.tight_layout();fig.savefig(directory/(name+'.png'),bbox_inches='tight');fig.savefig(directory/(name+'.svg'),bbox_inches='tight');plt.close(fig)

def main(directory):
    d=Path(directory);d.mkdir(parents=True,exist_ok=True);r=run();mu=np.array(r['posterior_mean']);S=np.array(r['posterior_covariance']);Q=np.diag(r['q_variance'])
    fig,ax=plt.subplots(figsize=(8.1,4.8));contour(ax,mu,S,BLUE,'解析后验：相关 −0.8');contour(ax,mu,Q,ORANGE,'mean-field：相关 0');ax.set(xlim=(-1.5,2.4),ylim=(-1.5,2.4),xlabel='z1',ylabel='z2',title='总和读数允许两个未知贡献互相补偿');ax.set_aspect('equal');ax.legend(loc='upper right',fontsize=9);ax.grid(alpha=.15);save(fig,'01_geometry',d)
    fig,ax=plt.subplots(figsize=(8.1,4.5));trace=r['trace'][:31];ax.plot(range(len(trace)),[v['elbo'] for v in trace],color=BLUE,marker='.',label='ELBO（每次坐标更新）');ax.axhline(r['log_evidence'],color=GREEN,ls='--',label='log p(x) = −1.5466');ax.axhline(r['elbo'],color=ORANGE,ls=':',label='族内最优 ELBO = −2.0575');ax.set(xlabel='坐标更新次数（0为初始化）',ylabel='自然对数目标',title='提高ELBO仍不能关闭全部证据差距');ax.legend();ax.grid(alpha=.15);save(fig,'02_elbo',d)
    fig,ax=plt.subplots(figsize=(8.1,4.5));v=np.array([t['mean'] for t in r['trace'][:17]]);ax.plot(v[:,0],v[:,1],'-o',color=BLUE,ms=4);ax.plot(*mu,'*',color=ORANGE,ms=16,label='解析均值 (4/9, 4/9)');ax.annotate('初始 (0,0)',v[0],xytext=(.03,.03));ax.annotate('第一轮 (.8,.16)',v[2],xytext=(.45,.08));ax.set(xlabel='当前 m1',ylabel='当前 m2',title='顺序坐标上升：一横一竖，不是同时移动');ax.legend();ax.grid(alpha=.2);save(fig,'03_coordinates',d)
    fig,axes=plt.subplots(1,2,figsize=(9.2,4.5));
    for ax,cov,label in [(axes[0],Q,'最小化 KL(q || p)：方差 0.2'),(axes[1],np.diag(np.diag(S)),'最小化 KL(p || q)：方差 5/9')]:
        contour(ax,mu,S,BLUE,'真实 p');contour(ax,mu,cov,ORANGE,'近似 q');ax.set(xlim=(-1.3,2.2),ylim=(-1.3,2.2),xlabel='z1',ylabel='z2',title=label);ax.set_aspect('equal');ax.legend(fontsize=9)
    save(fig,'04_direction',d)
    fig,axes=plt.subplots(1,2,figsize=(9.2,4.2));s=r['noise_sweep'];x=[v['noise_variance'] for v in s];axes[0].semilogx(x,[abs(v['correlation']) for v in s],'-o',color=BLUE);axes[0].set(xlabel='观测噪声方差 τ²',ylabel='后验相关系数的绝对值',ylim=(0,1),title='噪声小，总和约束更强');axes[1].semilogx(x,[v['variance_ratio'] for v in s],'-o',color=ORANGE);axes[1].set(xlabel='观测噪声方差 τ²',ylabel='q边缘方差 / p边缘方差',ylim=(0,1.05),title='相关强，原坐标方差低估更重');
    for ax in axes:ax.grid(alpha=.2)
    save(fig,'05_noise',d)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',default=str(ROOT/'outputs/figures'));a=p.parse_args();main(a.directory)
