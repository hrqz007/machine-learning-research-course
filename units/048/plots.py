"""Actual computed plots; fixed display replicates 0..19, never cherry-picked."""
from pathlib import Path
import argparse,json,os
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'outputs/mpl'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import experiment as e
NAMES=['01_hand_fit','02_exact_enumeration','03_complexity','04_bias_variance','05_learning_curves','06_prediction_variability','07_risk_distribution','08_fixed_random','09_test_noise','10_ddof_and_bias']
COL=['#137d92','#d16335','#7655a7','#42566e']

def style():
    paths=sorted([p for p in font_manager.findSystemFonts() if 'NotoSansCJK' in p],key=lambda p:('Regular' not in p,p))
    if paths:
        font_manager.fontManager.addfont(paths[0]);plt.rcParams['font.family']=font_manager.FontProperties(fname=paths[0]).get_name()
    plt.rcParams.update({'font.size':13.0,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.16,'axes.unicode_minus':False,'figure.facecolor':'white','savefig.dpi':170})

def legend(a,**kw):a.legend(loc='upper center',bbox_to_anchor=(.5,-.23),frameon=False,fontsize=10.5,**kw)

def make_figure(i,r):
    style();xx=np.linspace(-1,1,301);summ=[b['summary'] for b in r['complexity']];pp=np.arange(1,10)
    if i==1:
        f,ax=plt.subplots(1,2,figsize=(10.8,4.7),layout='constrained')
        a=ax[0];a.axhline(0,color='#444',ls=':',label='真实条件均值 0');a.scatter([-1,0,1],[1,-1,1],s=50,color='black',label='同一训练数据',zorder=4)
        a.plot(xx,np.ones_like(xx)/3,color=COL[0],label='常数预测 1/3');a.plot(xx,2*xx**2-1,color=COL[1],label='二次预测 2x²-1');a.set(xlabel='输入 x',ylabel='预测 / 标签',title='同一数据：插值不等于总体准确',ylim=(-1.3,1.5));legend(a,ncol=2)
        vals=r['hand']['realized'];a=ax[1];idx=np.arange(3)
        a.bar(idx-.18,[v['train_mse'] for v in vals],.36,color=COL[0],label='训练 MSE');a.bar(idx+.18,[v['risk'] for v in vals],.36,color=COL[1],label='真实含噪总体风险');a.set(xticks=idx,xticklabels=['常数 p=1','线性 p=2','二次 p=3'],ylabel='MSE',title='这一次：训练归零，总体风险上升',ylim=(0,1.8));legend(a,ncol=2)
    elif i==2:
        f,ax=plt.subplots(1,2,figsize=(10.8,4.7),layout='constrained');vals=r['hand']['all_eight_equiprobable_designs'];p=[1,2,3]
        ax[0].plot(p,[v['mean_train'] for v in vals],'-o',color=COL[0],label='八种训练 MSE 的精确平均');ax[0].plot(p,[v['risk'] for v in vals],'-s',color=COL[1],label='八种总体风险的精确平均');ax[0].set(xticks=p,xlabel='参数数 p',ylabel='MSE',title='固定设计全枚举：无需抽样近似',ylim=(-.08,2.05));legend(ax[0])
        ax[1].bar(p,[1,1,1],color='#ccd4dc',label='新标签噪声 = 1');ax[1].bar(p,[v['variance'] for v in vals],bottom=1,color=COL[2],label='训练集引起的预测方差');ax[1].set(xticks=p,xlabel='参数数 p',ylabel='风险分量',title='三类预测均无偏，风险仍增加',ylim=(0,2.05));legend(ax[1])
    elif i==3:
        f,ax=plt.subplots(1,2,figsize=(10.8,4.7),layout='constrained')
        a=ax[0];a.plot(pp,[s['train_mean'] for s in summ],'-o',color=COL[0],label='平均训练 MSE');a.plot(pp,[s['risk_mean'] for s in summ],'-s',color=COL[1],label='平均真实含噪风险');a.plot(pp,[s['test_mean'] for s in summ],'--^',color=COL[2],label='独立含噪测试 MSE 平均');a.set(xlabel='参数数 p（最高次数 p-1）',ylabel='MSE',title='随机设计 n=40，400 次重复',xticks=pp);legend(a)
        a=ax[1];a.fill_between(pp,[s['risk_q10'] for s in summ],[s['risk_q90'] for s in summ],alpha=.22,color=COL[1],label='各训练集风险的 10% 至 90% 分位');a.plot(pp,[s['risk_mean'] for s in summ],'-s',color=COL[1],label='400 次风险平均');a.plot(pp,[b['runs'][0]['risk'] for b in r['complexity']],':o',color=COL[3],label='预先固定的第 0 次');a.set(xlabel='参数数 p',ylabel='含噪总体风险',title='单次曲线不等于重复抽样规律',xticks=pp);legend(a)
    elif i==4:
        f,ax=plt.subplots(1,2,figsize=(10.8,4.7),layout='constrained')
        a=ax[0];bias=np.array([s['bias2_mc'] for s in summ]);var=np.array([s['variance_ddof0'] for s in summ]);noise=np.array([s['noise'] for s in summ]);a.bar(pp,noise,color='#ccd4dc',label='新标签噪声');a.bar(pp,bias,bottom=noise,color=COL[0],label='重复样本平均预测的偏差平方');a.bar(pp,var,bottom=noise+bias,color=COL[2],label='预测方差 ddof=0');a.set(xlabel='参数数 p',ylabel='MSE 分量',title='有限400次的精确代数恒等式',xticks=pp);legend(a)
        a=ax[1];a.plot(pp,bias,'-o',color=COL[0],label='MC 偏差平方');a.plot(pp,[s['approximation'] for s in summ],'--s',color=COL[1],label='真实假设类近似误差');a.set(xlabel='参数数 p',ylabel='平方误差',title='偏差平方与近似误差不是同一定义',xticks=pp);legend(a)
    elif i==5:
        f,ax=plt.subplots(1,2,figsize=(10.8,4.7),layout='constrained')
        for p,col in zip((2,4,8),COL):
            ss=[b['summary'] for b in r['learning'] if b['summary']['p']==p];n=[s['n'] for s in ss]
            ax[0].plot(n,[s['risk_mean'] for s in ss],'-o',color=col,label=f'p={p} 平均总体风险');ax[1].plot(n,[s['train_mean'] for s in ss],'-o',color=col,label=f'p={p} 平均训练 MSE')
        ax[0].axhline(.1225,color='#555',ls=':',label='噪声下限 0.1225')
        for a in ax:a.set_xscale('log',base=2);a.set_xticks([16,32,64,128,256],labels=['16','32','64','128','256']);a.set_xlabel('训练样本量 n')
        ax[0].set_yscale('log');ax[0].set(ylabel='含噪总体风险 对数轴',title='真实长尾保留，纵轴使用对数');ax[1].set(ylabel='训练 MSE',title='更多训练样本不保证训练误差下降');legend(ax[0],ncol=2);legend(ax[1],ncol=2)
    elif i==6:
        f,ax=plt.subplots(1,2,figsize=(10.8,4.7),layout='constrained');beta=np.array(r['protocol']['beta'])
        for a,p in zip(ax,(4,9)):
            C=np.array(r['complexity'][p-1]['coefficients']);P=e.basis(xx,9)@C.T
            for j in range(20):a.plot(xx,P[:,j],color=COL[1],alpha=.2,lw=.7,label='前20次预测' if j==0 else None)
            a.plot(xx,e.basis(xx,9)@beta,'k--',label='真实均值 f*');a.plot(xx,P.mean(axis=1),color=COL[0],lw=2,label='400次平均预测');a.set(xlabel='新输入 x',ylabel='预测值',title=f'p={p}，显示预定前20次，无筛选');legend(a)
    elif i==7:
        f,ax=plt.subplots(1,2,figsize=(10.8,4.7),layout='constrained')
        for a,p in zip(ax,(4,9)):
            risk=np.array([v['risk'] for v in r['complexity'][p-1]['runs']]);a.hist(np.log10(risk),bins=25,color=COL[0],alpha=.8,label='全部 400 次');a.axvline(np.log10(risk.mean()),color=COL[1],label='算术平均的位置');a.set(xlabel='log10 总体风险',ylabel='训练数据集个数',title=f'p={p} 风险分布，最大值 {risk.max():.3g}');legend(a,ncol=2)
    elif i==8:
        f,ax=plt.subplots(1,2,figsize=(10.8,4.7),layout='constrained');p=[2,4,8];idx=np.arange(3)
        for j,(kind,col,label) in enumerate([('fixed_design',COL[0],'固定均匀网格，重抽标签'),('complexity',COL[1],'随机输入和标签')]):
            ss=[next(b['summary'] for b in r[kind] if b['summary']['p']==v) for v in p]
            ax[0].bar(idx+(j-.5)*.34,[s['risk_mean'] for s in ss],.34,color=col,label=label)
            ax[1].bar(idx+(j-.5)*.34,[s['variance_ddof0'] for s in ss],.34,color=col,label=label)
        for a in ax:a.set(xticks=idx,xticklabels=['p=2','p=4','p=8']);legend(a)
        ax[0].set(ylabel='平均含噪总体风险',title='评价分布相同，训练随机性不同');ax[1].set(ylabel='积分预测方差 ddof=0',title='固定设计结论不能直接移给随机设计')
    elif i==9:
        f,ax=plt.subplots(1,2,figsize=(10.8,4.7),layout='constrained')
        for a,p in zip(ax,(4,9)):
            vals=r['complexity'][p-1]['runs'];risk=np.array([v['risk'] for v in vals]);test=np.array([v['test'] for v in vals]);a.scatter(risk,test,s=10,alpha=.5,color=COL[0]);lo=min(risk.min(),test.min());hi=max(risk.max(),test.max());a.plot([lo,hi],[lo,hi],'--',color=COL[1],label='测试 MSE = 总体风险');a.set(xscale='log',yscale='log',xlabel='该训练集的真实含噪风险 对数轴',ylabel='独立256行测试 MSE 对数轴',title=f'p={p}：测试集也在随机波动');legend(a)
    elif i==10:
        f,ax=plt.subplots(1,2,figsize=(10.8,4.7),layout='constrained');sample=np.array([-1,0,1,2.]);mu=sample.mean();var0=sample.var(ddof=0);var1=sample.var(ddof=1)
        ax[0].bar(['ddof=0','ddof=1'],[var0,var1],color=COL[:2]);ax[0].set(ylabel='四个数的方差',title='同一组数，分母4或3',ylim=(0,2));
        for j,v in enumerate([var0,var1]):ax[0].text(j,v+.06,f'{v:.6f}',ha='center')
        ax[1].bar(pp,[s['bias2_unbiased_estimate'] for s in summ],color=COL[2]);ax[1].axhline(0,color='black',lw=.7);ax[1].set_yscale('symlog',linthresh=1e-4);ax[1].set(xticks=pp,xlabel='参数数 p',ylabel='修正后偏差平方估计 对称对数轴',title='无偏估计可为负：保留原值，不裁零')
    else:raise ValueError('figure number 1..10 required')
    return f

def render_all(report,directory):
    directory=Path(directory).absolute()
    for part in (directory,*directory.parents):
        if part.is_symlink():raise ValueError('symlink directory')
    if directory.resolve()==e.ROOT/'figures':raise ValueError('render to a fresh directory, not shipped figures')
    directory.mkdir(parents=True,exist_ok=True)
    for i,name in enumerate(NAMES,1):
        path=directory/(name+'.png')
        if path.is_symlink() or (path.exists() and (not path.is_file() or path.stat().st_nlink>1)):raise ValueError('unsafe output')
        f=make_figure(i,report)
        try:f.savefig(path,bbox_inches='tight')
        finally:plt.close(f)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',required=True);p.add_argument('--directory',required=True);a=p.parse_args();render_all(json.loads(Path(a.report).read_text()),a.directory)
