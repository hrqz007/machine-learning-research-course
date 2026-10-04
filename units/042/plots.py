"""Nine original, computed teaching figures. No downloaded source illustrations."""
from pathlib import Path
import io, os, tempfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
import experiment as e
NAMES=['01_sigmoid_logodds','02_stable_likelihood_chain','03_two_round_ledger','04_logistic_curvature','05_logistic_optimization','06_complete_separation','07_probability_and_decision','08_linear_probability_contrast','09_numerical_diagnostics']
C=['#167F94','#DD633F','#795AAF','#B59B23']

def style():
    fonts=sorted([p for p in font_manager.findSystemFonts() if 'NotoSansCJK' in p or 'SourceHanSans' in p],key=lambda p:('Regular' not in p,p))
    if fonts:font_manager.fontManager.addfont(fonts[0]);plt.rcParams['font.family']=font_manager.FontProperties(fname=fonts[0]).get_name()
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'axes.unicode_minus':False,'figure.facecolor':'#FAFCFE','axes.facecolor':'#FAFCFE','savefig.dpi':150})

def box(ax,x,y,w,h,s,color):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.01',facecolor=color,alpha=.13,edgecolor='none'));ax.text(x+w/2,y+h/2,s,ha='center',va='center',color='#203344',fontsize=10)

def make_figure(number,r):
    style();x=np.array(r['data']['x']);y=np.array(r['data']['y']);wstar=float(r['finite_mle_certificate']['w']) if r['finite_mle_certificate'] else r['newton']['final']['theta'][1]
    if number==1:
        f,ax=plt.subplots(1,3,figsize=(11.4,3.5),layout='constrained');z=np.linspace(-7,7,500);p=e.sigmoid(z)
        ax[0].plot(z,p,color=C[0],lw=2);ax[0].axhline(.5,ls=':',color=C[1]);ax[0].set(xlabel='线性得分 z',ylabel='条件概率 p',title='① sigmoid：范围被约束')
        ax[1].plot(p,z,color=C[1],lw=2);ax[1].set(xlabel='概率 p（不含0、1）',ylabel='log[p/(1−p)]',title='② logit：反变换到实数')
        ax[2].plot(z,p*e.sigmoid(-z),color=C[2],lw=2);ax[2].set(xlabel='z',ylabel='dp/dz',title='③ 概率变化不是常数');ax[2].annotate('最大斜率1/4',xy=(0,.25),xytext=(1.4,.2),arrowprops={'arrowstyle':'->','color':C[2]})
    elif number==2:
        f=plt.figure(figsize=(11.4,5.1),layout='constrained');gs=f.add_gridspec(2,2,height_ratios=[.8,1.3]);top=f.add_subplot(gs[0,:]);top.axis('off');top.set(xlim=(0,1),ylim=(0,1))
        names=['输入 (1,x)\n参数 (b,w)','得分 z=b+wx','signed softplus\nℓ=softplus(−tz)','稳定导数\ny=0:σ(z); y=1:−σ(−z)']
        for i,s in enumerate(names):
            xx=.015+.25*i;box(top,xx,.2,.215,.57,s,C[i])
            if i<3:top.annotate('',xy=(xx+.244,.48),xytext=(xx+.219,.48),arrowprops={'arrowstyle':'->','color':'#536571','lw':1.5})
        a=f.add_subplot(gs[1,0]);b=f.add_subplot(gs[1,1]);z=np.linspace(-6,6,300)
        for yy,color in [(0,C[0]),(1,C[1])]:a.plot(z,e.softplus((1-2*yy)*z),label=f'y={yy}',color=color);b.plot(z,e.sigmoid(z) if yy==0 else -e.sigmoid(-z),label=f'y={yy}',color=color)
        a.set(xlabel='z',ylabel='单行损失',title='错误且自信时损失增大');b.set(xlabel='z',ylabel='dℓ/dz',title='导数有符号，损失非负');a.legend();b.legend()
    elif number==3:
        f,ax=plt.subplots(1,2,figsize=(11.6,4.2),layout='constrained');states=r['hand_states'];k=range(len(states))
        for row,col in zip(range(4),C):ax[0].plot(k,[s['rows'][row]['loss'] for s in states],'-o',color=col,label=f'B{row+1}')
        ax[0].plot(k,[s['data_loss'] for s in states],'--',color='#243842',lw=2,label='四行平均F');ax[0].set(xlabel='完成同步更新数',ylabel='单行 / 平均损失',xticks=list(k),title='中间两行变差，总损失仍下降');ax[0].legend(ncol=2,fontsize=9)
        ax[1].axis('off');rows=[]
        for s in states:rows.append([str(len(rows)),f"{s['theta'][1]:.9f}",f"{s['data_loss']:.9f}",f"{s['gradient'][1]:.9f}"])
        t=ax[1].table(cellText=rows,colLabels=['k','w','平均F','总梯度gw'],loc='center',cellLoc='center');t.auto_set_font_size(False);t.set_fontsize(9.5);t.scale(1,2.2)
        ax[1].set_title('同一批四行，b始终为0');ax[1].text(.5,.09,'旧前向 → 每行链式梯度 → 求平均\n同一旧参数同步更新 → 四行新前向',ha='center',transform=ax[1].transAxes,color='#243842')
    elif number==4:
        f,ax=plt.subplots(1,2,figsize=(11.4,3.8),layout='constrained');idx=np.arange(4)
        for k,s in enumerate(r['hand_states']):ax[0].bar(idx+(k-1)*.23,[v['local_dp_dz'] for v in s['rows']],width=.23,color=C[k],label=f'k={k}')
        ax[0].set(xticks=idx,xticklabels=['B1','B2','B3','B4'],ylabel='σ(z)σ(−z)',title='行级曲率：越饱和越小');ax[0].legend()
        ws=np.linspace(0,8,160);Hs=np.array([e.row_ledger(x,y,[0,w])['hessian'] for w in ws]);ax[1].semilogy(ws,Hs[:,0,0],label='Hbb',color=C[0]);ax[1].semilogy(ws,Hs[:,1,1],label='Hww',color=C[1]);ax[1].axvline(wstar,ls=':',color=C[2],label='有限MLE');ax[1].set(xlabel='对称截面 b=0 上的 w',ylabel='Hessian 对角元',title='主例H正定，但不是常量');ax[1].legend()
    elif number==5:
        f,ax=plt.subplots(1,2,figsize=(11.4,4),layout='constrained');bs=np.linspace(-.5,.5,180);ws=np.linspace(-.15,.85,180);B,W=np.meshgrid(bs,ws);Z=B[...,None]+W[...,None]*x;J=e.softplus(-(2*y-1)*Z).mean(-1);levels=[.642,.65,.67,.7,.74,.8];ax[0].contour(W,B,J,levels=levels,colors='#BAC6CB');ax[0].plot(wstar,0,'*',color=C[2],ms=12,label='独立三次方程MLE')
        for key,color in [('gd',C[0]),('newton',C[1])]:
            states=r[key]['trace'];t=np.array([s['theta'] for s in states]);ax[0].plot(t[:,1],t[:,0],'-o',markersize=3,color=color,alpha=.8,label=key);ax[1].semilogy(range(len(states)),[s['gradient_norm'] for s in states],'-o',markersize=3,color=color,label=f"{key}: {r[key]['updates']}次更新")
        ax[0].set(xlabel='斜率 w',ylabel='截距 b',title='相同目标的不同更新路径');ax[0].legend(fontsize=8);ax[1].axhline(r['config']['gradient_tolerance'],ls=':',color='#677');ax[1].set(xlabel='完成更新数',ylabel='完整梯度L2范数',title='共同停止标准；不据此比较墙钟成本');ax[1].legend(fontsize=9)
    elif number==6:
        f,ax=plt.subplots(1,2,figsize=(11.4,4),layout='constrained');ray=r['separation_ray'];s=[v['scale'] for v in ray]
        for key,label,col in [('loss','平均负对数似然',C[0]),('gradient_norm','梯度L2范数',C[1])]:ax[0].semilogy(s,[v[key] for v in ray],'-o',label=label,color=col)
        ax[0].axhline(1e-10,ls=':',color='#677',label='数值容差示意');ax[0].set(xlabel='已知分离方向的尺度 s',ylabel='数值（对数轴）',title='有限点全为正，但可任意接近0');ax[0].legend(fontsize=9)
        ax[1].plot(s,[v['theta_norm'] for v in ray],'-o',label='参数范数',color=C[2]);ax[1].plot(s,[v['minimum_margin'] for v in ray],'--',label='最小正确margin（本例重合）',color=C[1]);ax[1].set(xlabel='s',ylabel='范数 / margin',title='无有限MLE：参数继续增大');ax[1].legend(fontsize=9)
    elif number==7:
        f,ax=plt.subplots(1,2,figsize=(11.4,4),layout='constrained');xx=np.linspace(-6,6,300);ax[0].plot(xx,e.sigmoid(wstar*xx),color='#263F4B',lw=2)
        for tau,col in zip(r['config']['thresholds'],C):ax[0].axhline(tau,ls='--',color=col,label=f'τ={tau:g}')
        ax[0].scatter(x,e.sigmoid(wstar*x),c=y,cmap='coolwarm',s=65,edgecolors='#333');ax[0].set(xlabel='x',ylabel='拟合概率',title='概率固定，行动阈值可改变');ax[0].legend()
        rows=[]
        for v in r['decisions']:rows.append([f"{v['threshold']:.1f}",' '.join(map(str,v['predictions'])),f"{v['accuracy']:.2f}"])
        ax[1].axis('off');t=ax[1].table(cellText=rows,colLabels=['τ','B1 B2 B3 B4预测','准确率'],cellLoc='center',loc='center');t.auto_set_font_size(False);t.set_fontsize(10);t.scale(1,2.2);ax[1].set_title('这四行：三种阈值都只对一半');ax[1].text(.5,.08,'C_FP=1, C_FN=4 ⇒ τ=0.2\n相等时本讲判正；库零得分probe判0',ha='center',transform=ax[1].transAxes,fontsize=10)
    elif number==8:
        f,ax=plt.subplots(1,2,figsize=(11.4,4),layout='constrained');xx=np.linspace(-12,12,350);ols=r['linear_probability']['theta']
        ax[0].plot(xx,ols[0]+ols[1]*xx,color=C[1],label='线性概率 OLS');ax[0].plot(xx,e.sigmoid(wstar*xx),color=C[0],label='Bernoulli logit');ax[0].axhspan(0,1,color=C[0],alpha=.05);ax[0].axhline(0,color='#999',ls=':');ax[0].axhline(1,color='#999',ls=':');ax[0].set(xlabel='含训练范围外的 x',ylabel='模型输出',title='外推合法范围不同');ax[0].legend()
        ax[1].scatter(x,y,color='#243E50',s=70,label='实际0/1标签');ax[1].plot(x,ols[0]+ols[1]*x,'s-',color=C[1],label='OLS');ax[1].plot(x,e.sigmoid(wstar*x),'o-',color=C[0],label='logit');ax[1].set(xlabel='四行训练 x',ylabel='概率/标签',title='两种目标给不同折中');ax[1].legend(fontsize=9)
    elif number==9:
        f,ax=plt.subplots(1,2,figsize=(11.4,4.1),layout='constrained');scan=r['diagnostics']['finite_difference']
        for i in range(3):
            a=scan[i*len(r['config']['fd_steps']):(i+1)*len(r['config']['fd_steps'])];ax[0].loglog([v['h'] for v in a],[v['gradient_error'] for v in a],'-o',color=C[i],label=f'非驻点{i+1}: 梯度');ax[0].loglog([v['h'] for v in a],[v['hvp_error'] for v in a],'--',color=C[i],alpha=.6)
        ax[0].set(xlabel='中心差分步长 h',ylabel='L2误差',title='实线梯度；虚线Hessian-vector');ax[0].legend(fontsize=8)
        z=np.linspace(20,44,240);stable=e.softplus(-z);naive=np.log1p(np.exp(z))-z;ax[1].semilogy(z,stable,color=C[0],lw=2,label='signed stable loss');valid=naive>0;ax[1].semilogy(z[valid],naive[valid],'.',color=C[1],label='相减式：仅画正值');ax[1].axvspan(z[np.flatnonzero(~valid)[0]],44,color=C[1],alpha=.08);ax[1].text(.44,.84,'浅色区出现舍入为0\n零值未伪装成小正数',transform=ax[1].transAxes,fontsize=9);ax[1].set(xlabel='正确正类的正得分 z',ylabel='损失（对数轴）',title='浮点相消会早于指数下溢');ax[1].legend(fontsize=8,loc='lower left')
    else:raise ValueError('figure number 1..9')
    return f

def render_all(report,directory):
    # Public rebuilding targets a new directory or outputs/, never original assets.
    directory=Path(directory).absolute()
    for part in (directory,*directory.parents):
        if part.is_symlink():raise ValueError('symlink figure directory')
    root=e.ROOT.resolve();resolved=directory.resolve()
    if resolved==root or (root in resolved.parents and root/'outputs' not in (resolved,*resolved.parents)):
        raise ValueError('rebuild figures under outputs/ or outside the unit')
    targets=[directory/(name+'.png') for name in NAMES]
    for target in targets:
        if target.is_symlink() or (target.exists() and (not target.is_file() or target.stat().st_nlink>1)):
            raise ValueError('unsafe figure target')
    # Compute and serialize each figure completely before changing its target.
    for i,target in enumerate(targets,1):
        fig=make_figure(i,report);buffer=io.BytesIO()
        try:fig.savefig(buffer,format='png',dpi=150,bbox_inches='tight')
        finally:plt.close(fig)
        data=buffer.getvalue();directory.mkdir(parents=True,exist_ok=True);temp=None
        try:
            fd,temp=tempfile.mkstemp(prefix='.ml042-figure-',dir=directory)
            with os.fdopen(fd,'wb') as file:file.write(data);file.flush();os.fsync(file.fileno())
            os.replace(temp,target);temp=None
        finally:
            if temp is not None and Path(temp).exists():Path(temp).unlink()
if __name__=='__main__':
    import argparse,json
    p=argparse.ArgumentParser();p.add_argument('--report',required=True);p.add_argument('--directory',required=True);args=p.parse_args();render_all(json.loads(Path(args.report).read_text()),args.directory)
