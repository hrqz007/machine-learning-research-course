"""Four original figures: analytic curves and frozen experiment measurements."""
from pathlib import Path
import argparse
import json
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/course-087-mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import experiment as ex

FONT=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
if FONT.exists():
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams['font.family']=font_manager.FontProperties(fname=str(FONT)).get_name()
plt.rcParams.update({'axes.unicode_minus':False,'font.size':10,'axes.grid':True,'grid.alpha':.22,'figure.facecolor':'#fbfcfe','axes.facecolor':'white','svg.fonttype':'none'})
COLORS={'small':'#777777','xavier':'#1877a5','he':'#db5b29','large':'#9b3c8b'}

def save(fig,out,name):
    fig.savefig(out/(name+'.png'),dpi=170,bbox_inches='tight')
    fig.savefig(out/(name+'.svg'),bbox_inches='tight');plt.close(fig)

def draw(out,result):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    z=np.linspace(-5,5,801)
    for k,c in [('sigmoid','#1877a5'),('tanh','#8c46a1'),('relu','#db5b29')]:
        axes[0,0].plot(z,ex.activate(z,k),label=k,color=c)
        axes[0,1].plot(z,ex.derivative(z,k),label=k,color=c)
    axes[0,0].set(title='函数值：平台、中心与单侧门',xlabel='预激活 z',ylabel='激活 φ(z)',ylim=(-1.2,2.2));axes[0,0].legend()
    axes[0,1].set(title='导数：局部敏感度',xlabel='预激活 z',ylabel="φ′(z)");axes[0,1].legend()
    zz=np.linspace(-4,4,1000);density=np.exp(-zz**2/2)/np.sqrt(2*np.pi)
    axes[1,0].plot(zz,density,label='标准正态 Z 的密度');axes[1,0].fill_between(zz,density,where=zz>0,alpha=.25,color='#db5b29')
    axes[1,0].annotate('负半边被映到 0：点质量 1/2',xy=(0,.08),xytext=(-3.8,.22),arrowprops={'arrowstyle':'->'})
    axes[1,0].set(title='ReLU：正半边保留，负半边归零',xlabel='Z',ylabel='连续部分密度')
    vals=[1/np.sqrt(2*np.pi),.5,.5-1/(2*np.pi)]
    axes[1,1].bar(['均值','二阶矩','方差'],vals,color=['#b5c5cf','#db5b29','#1877a5'])
    for i,v in enumerate(vals):axes[1,1].text(i,v+.015,f'{v:.4f}',ha='center')
    axes[1,1].set(title='A = ReLU(Z), Z ~ N(0,1)',ylabel='理论值',ylim=(0,.65))
    fig.suptitle('087 · 原创图1  激活导数与二阶矩',fontsize=16);save(fig,out,'01_activation_moments')

    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for mode,c in COLORS.items():
        group=[p for p in result['propagation'] if p['activation']=='relu' and p['initialization']==mode]
        for ax,key,title in [(axes[0,0],'final_second_moment','末层激活二阶矩：深度效应'),(axes[0,1],'gradient_rms_ratio','输入/输出梯度RMS比：深度效应')]:
            med=[];low=[];hi=[]
            for d in (2,8,24):
                values=[p[key] for p in group if p['depth']==d];med.append(np.median(values));low.append(min(values));hi.append(max(values))
            ax.plot([2,8,24],med,'o-',color=c,label=mode);ax.fill_between([2,8,24],low,hi,color=c,alpha=.15)
            ax.set(title=title,xlabel='隐藏层数',yscale='log')
        chosen=[p for p in group if p['depth']==24]
        for ax,key,title in [(axes[1,0],'activation','24层：逐层激活RMS'),(axes[1,1],'grad_activation','24层：逐层激活梯度RMS')]:
            arr=np.array([[r[key]['rms'] for r in p['layers']] for p in chosen])
            ax.plot(np.arange(1,25),np.median(arr,axis=0),color=c,label=mode)
            ax.fill_between(np.arange(1,25),arr.min(0),arr.max(0),color=c,alpha=.15)
            ax.set(title=title,xlabel='层号（输入 → 输出）',yscale='log')
    for ax in axes.flat:ax.legend(fontsize=8)
    fig.suptitle('087 · 原创图2  配对种子，宽度64；线为三种子中位数，带为最小–最大',fontsize=13);save(fig,out,'02_depth_propagation')

    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for col,depth in enumerate((2,8,24)):
        zero_labels=[]
        for mode in ('small','xavier','he'):
            a,g=ex.probe_samples(depth=depth,mode=mode)
            for row,values in ((0,a[-1]),(1,g[0])):
                nonzero=np.abs(values[values!=0]);logs=np.log10(nonzero)
                axes[row,col].hist(logs,bins=45,density=True,histtype='step',linewidth=1.4,color=COLORS[mode],label=mode)
            zero_labels.append(f'{mode}: {100*np.mean(a[-1]==0):.1f}%')
        axes[0,col].set(title=f'{depth}层 · 末层激活非零部分',xlabel='log10(|激活|)',ylabel='条件密度')
        axes[0,col].text(.02,.97,'零点质量\n'+'\n'.join(zero_labels),transform=axes[0,col].transAxes,va='top',fontsize=8,bbox={'facecolor':'white','alpha':.8,'edgecolor':'none'})
        axes[1,col].set(title=f'{depth}层 · 输入梯度非零部分',xlabel='log10(|输入梯度|)',ylabel='条件密度')
    for ax in axes.flat:ax.legend(fontsize=8,loc='upper right')
    fig.suptitle('087 · 原创图3  分布而非单个均值；固定种子8701，直方图不含零点',fontsize=14);save(fig,out,'03_distributions')

    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for mode in ('small','xavier','he'):
        group=[t for t in result['training'] if t['initialization']==mode]
        steps=[h['step'] for h in group[0]['history']]
        for ax,key in ((axes[0,0],'loss'),(axes[0,1],'accuracy')):
            arr=np.array([[h[key] for h in t['history']] for t in group]);ax.plot(steps,np.median(arr,0),color=COLORS[mode],label=mode);ax.fill_between(steps,arr.min(0),arr.max(0),color=COLORS[mode],alpha=.15)
    axes[0,0].set(title='30条真实Wine：交叉熵',xlabel='SGD步数',ylabel='平均交叉熵',yscale='log');axes[0,0].axhline(np.log(3),ls='--',color='black',lw=.8)
    axes[0,1].set(title='硬准确率掩盖近均匀概率',xlabel='SGD步数',ylabel='训练准确率',ylim=(0,1.05))
    for scale_name,c in [('未缩放','#9b3c8b'),('1/√L','#1e8b71')]:
        ms=[];gs=[]
        for d in (2,8,24):
            r=[r for r in result['residual'] if r['depth']==d and abs(r['branch_scale']-(1 if scale_name=='未缩放' else 1/np.sqrt(d)))<1e-12]
            ms.append(np.median([v['final_second_moment'] for v in r]));gs.append(np.median([v['gradient_rms_ratio'] for v in r]))
        axes[1,0].plot([2,8,24],ms,'o-',label=scale_name,color=c)
        axes[1,1].plot([2,8,24],gs,'o-',label=scale_name,color=c)
    axes[1,0].set(title='残差 h + α tanh(hW)：激活仍可能增长',xlabel='残差块数',ylabel='末层二阶矩')
    axes[1,1].set(title='残差：有恒等路径 ≠ 总梯度恒等',xlabel='残差块数',ylabel='输入/输出梯度RMS比')
    for ax in axes.flat:ax.legend(fontsize=8)
    fig.suptitle('087 · 原创图4  训练验收与残差边界（上：中位数/范围；下：中位数）',fontsize=13);save(fig,out,'04_training_residual')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',default='outputs/figures');p.add_argument('--result',default=str(ex.ROOT/'experiment-result.json'));args=p.parse_args()
    draw(args.directory,json.loads(Path(args.result).read_text()));print('Created four PNG and four SVG figures in',args.directory)
