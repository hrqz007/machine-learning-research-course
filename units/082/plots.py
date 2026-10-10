"""五幅原创项目图，均从实际执行的数据与估计值生成。"""
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
plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':10,'axes.unicode_minus':False,'svg.fonttype':'none'})
C=['#24789b','#d58b36','#7a55a4','#36866b']

def save(fig,out,name):
    fig.savefig(out/(name+'.png'),dpi=180,bbox_inches='tight',facecolor='white');fig.savefig(out/(name+'.svg'),bbox_inches='tight',facecolor='white');plt.close(fig)

def build(directory):
    out=Path(directory);out.mkdir(parents=True,exist_ok=True);result,a=run(True)
    fig,axes=plt.subplots(1,2,figsize=(8.5,3.7));s=a['synthetic'];x=s['x']['test'];z=np.asarray(s['raw']['test']['z']);pred=s['model'].predict(x)
    for ax,lab,title in [(axes[0],z,'已知合成生成分量'),(axes[1],pred,'估计分量（颜色编号可交换）')]:
        ax.scatter(x[:,0],x[:,1],c=np.array(C)[lab],s=11,alpha=.65);ax.set(title=title,xlabel='训练标准化特征1',ylabel='训练标准化特征2')
    axes[1].scatter(s['model'].means_[:,0],s['model'].means_[:,1],c='black',marker='x',s=90,label='估计均值');axes[1].legend(fontsize=8);fig.tight_layout();save(fig,out,'01_truth_recovery')
    fig,axes=plt.subplots(1,3,figsize=(9,3.6))
    for ax,name,title in zip(axes,['synthetic','wine','negative'],['合成三分量','真实Wine','单高斯负对照']):
        rows=result[name]['candidates'];ks=[r['k'] for r in rows]
        ax.plot(ks,[r['train_log_density'] for r in rows],'-o',label='训练',c=C[0]);ax.plot(ks,[r['validation_log_density'] for r in rows],'-s',label='验证',c=C[1]);ax.axvline(result[name]['selected_k'],c='#777',ls=':');ax.set(xticks=ks,xlabel='候选分量数K',title=title)
    axes[0].set_ylabel('平均对数密度（标准化坐标）');axes[0].legend(fontsize=8);fig.tight_layout();save(fig,out,'02_model_selection')
    fig,axes=plt.subplots(1,2,figsize=(8.5,3.8));s=a['wine'];r=s['responsibilities'];order=np.argsort(r.max(axis=1));im=axes[0].imshow(r[order],aspect='auto',cmap='Blues',vmin=0,vmax=1);axes[0].set(xlabel='拟合分量编号',ylabel='测试样本（按最大责任度排序）',title='Wine：软责任度保留不确定性');fig.colorbar(im,ax=axes[0],fraction=.045)
    hard=-np.log(r.max(axis=1));axes[1].scatter(r.max(axis=1),hard,c=C[2]);axes[1].set(xlabel='最大责任度',ylabel='硬q的ELBO缺口（nat）',title='缺口 = -log(最大责任度)');axes[1].grid(alpha=.2);fig.tight_layout();save(fig,out,'03_responsibility_gap')
    fig,axes=plt.subplots(1,2,figsize=(8.5,3.5))
    for ax,name,title in zip(axes,['synthetic','wine'],['合成数据','真实Wine']):
        for field,color,label in [('restart_stability',C[0],'同数据换初始化'),('bootstrap_stability',C[1],'重采样训练行')]:
            values=[v['ari_against_reference'] for v in result[name][field]];ax.plot(range(len(values)),values,'o-',c=color,label=label)
        ax.set(ylim=(-.05,1.06),xlabel='预声明重复编号',ylabel='固定测试点上的ARI',title=title);ax.grid(alpha=.15)
    axes[0].legend(fontsize=8);fig.tight_layout();save(fig,out,'04_stability')
    fig,axes=plt.subplots(2,2,figsize=(8.5,5.4))
    for row,name in enumerate(['synthetic','wine']):
        for col,stat in enumerate(['correlation','tail_fraction']):
            ax=axes[row,col];values=[v[stat] for v in a[name]['predictive_replicates']];ax.hist(values,bins=15,color=C[row],alpha=.65)
            observed=result[name]['fixed_parameter_predictive_check'][stat]['observed'];ax.axvline(observed,c='#b94156',lw=2,label='保留测试集观测')
            ax.set(title=f'{name} / '+('相关系数' if stat=='correlation' else '尾部比例'),xlabel='固定参数复制统计量',ylabel='复制数据集数');ax.legend(fontsize=8)
    fig.tight_layout();save(fig,out,'05_predictive_checks')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',default=str(ROOT/'outputs/figures'));a=p.parse_args();build(a.directory)
