"""五幅原创图：数据、模型边界、秩校准、预算与真实正常误报。"""
import argparse,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).parent/'outputs/mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
if Path(FONT).exists():font_manager.fontManager.addfont(FONT)
from experiment import run
ROOT=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':10,'axes.unicode_minus':False,'figure.dpi':130,'svg.fonttype':'none'})
COLORS=['#24789b','#d58b36','#7a55a4','#36866b']

def save(fig,out,name):
    fig.savefig(out/(name+'.png'),dpi=180,bbox_inches='tight',facecolor='white');fig.savefig(out/(name+'.svg'),bbox_inches='tight',facecolor='white');plt.close(fig)

def build(directory):
    out=Path(directory);out.mkdir(parents=True,exist_ok=True);result,a=run(True);s=a['synthetic'];raw=s['raw']
    fig,ax=plt.subplots(figsize=(8.5,4.3));x=np.asarray(raw['train']['x']);g=np.asarray(raw['train']['group']);q=np.asarray(raw['anomaly_test']['x'])
    for k,c in [(0,COLORS[0]),(1,COLORS[1])]:ax.scatter(x[g==k,0],x[g==k,1],s=12,alpha=.45,c=c,label=f'正常模式{k}（训练）')
    ax.scatter(q[:100,0],q[:100,1],s=12,c='#b94156',marker='x',label='桥区异常（测试）');ax.scatter(q[100:,0],q[100:,1],s=12,c='#784f99',marker='x',label='远端异常（测试）')
    ax.set(xlabel='原始特征1',ylabel='原始特征2',title='异常由生成任务定义：稀少的正常模式仍应保留');ax.legend(ncol=2,fontsize=8);ax.grid(alpha=.15);save(fig,out,'01_mechanisms')
    fig,axes=plt.subplots(2,2,figsize=(8.5,6));xx,yy=np.meshgrid(np.linspace(-4,4.8,160),np.linspace(-2.5,4.5,140));grid=np.c_[xx.ravel(),yy.ravel()];z=s['scaler'].transform(grid)
    for ax,(name,v) in zip(axes.flat,s['methods'].items()):
        score=v['model'].score(z).reshape(xx.shape);alarm=score>v['calibration'].threshold
        ax.contourf(xx,yy,alarm,levels=[-.5,.5,1.5],colors=['#e8f3f7','#fde3df'],alpha=.8)
        ax.contour(xx,yy,score,levels=[v['calibration'].threshold],colors=['#963b45'],linewidths=1)
        ax.scatter(x[:,0],x[:,1],s=3,c='#1c6e8c',alpha=.2);ax.set_title(name+'：蓝=接纳，红=报警',fontsize=10)
        ax.set(xlabel='原始特征1',ylabel='原始特征2')
    fig.tight_layout();save(fig,out,'02_boundaries')
    fig,ax=plt.subplots(figsize=(8.5,3.7));v=s['methods']['lof'];scores=np.sort(v['scores']['calibration']);ranks=np.arange(1,len(scores)+1)
    ax.plot(ranks,scores,c=COLORS[0]);ax.axvline(190,color=COLORS[1],ls='--',label='k=190');ax.axhline(v['calibration'].threshold,color='#b94156',ls=':',label='阈值t = 第190小分数')
    ax.set(xlabel='干净校准分数的升序位置',ylabel='LOF异常分数',title='m=199，α=0.05；严格大于阈值才报警');ax.legend();ax.grid(alpha=.15);save(fig,out,'03_calibration')
    fig,axes=plt.subplots(1,2,figsize=(8.5,3.7));names=list(result['synthetic']['methods']);p=np.arange(4)
    for g,c in [('0',COLORS[0]),('1',COLORS[1])]:axes[0].bar(p+(-.18 if g=='0' else .18),[result['synthetic']['methods'][n]['normal_group_fpr'][g] for n in names],.36,label='正常模式'+g,color=c)
    axes[0].axhline(.05,c='#b94156',ls='--');axes[0].set(xticks=p,xticklabels=names,ylabel='实际测试误报比例',title='总体预算不能保证每个子群');axes[0].legend(fontsize=8)
    fprs=np.linspace(0,.08,120);recall=.9;prev=.001;precision=prev*recall/(prev*recall+(1-prev)*fprs)
    axes[1].plot(fprs,precision,c=COLORS[2]);axes[1].axvline(.05,c='#b94156',ls='--');axes[1].set(xlabel='正常误报率',ylabel='报警中真正异常的比例',title='假设异常率0.1%、召回90%');axes[1].grid(alpha=.2);fig.tight_layout();save(fig,out,'04_budget')
    fig,axes=plt.subplots(2,6,figsize=(8.5,3.8));d=a['digits'];order=np.argsort(d['methods']['gaussian']['scores']['normal_test'])[::-1][:6];cal=d['methods']['gaussian']['calibration'];scores=d['methods']['gaussian']['scores']['normal_test'];anomal=np.asarray(d['raw']['anomaly_test']['x'])
    for j,i in enumerate(order):
        axes[0,j].imshow(d['x']['normal_test'][i].reshape(8,8),cmap='gray_r',vmin=0,vmax=1);axes[0,j].set_title(f'真0：{"误报" if scores[i]>cal.threshold else "接纳"}\n分数{scores[i]:.1f}',fontsize=8)
        axes[1,j].imshow(anomal[j].reshape(8,8),cmap='gray_r',vmin=0,vmax=1);axes[1,j].set_title('真6：任务外类别',fontsize=8)
    for ax in axes.flat:ax.axis('off')
    fig.suptitle('真实保留样本：上排为高分正常负例，下排为真实新类别',fontsize=12);fig.tight_layout();save(fig,out,'05_real_negatives')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',default=str(ROOT/'outputs/figures'));a=p.parse_args();build(a.directory)
