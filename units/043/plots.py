"""Nine original figures, computed from the experiment and explicit definitions."""
from pathlib import Path
import argparse,io,json,os,tempfile
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'outputs/mpl-cache'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import experiment as e
NAMES=['01_task_and_normalization','02_simplex_and_shift','03_forward_and_derivative',
       '04_two_round_rows','05_jacobian_geometry','06_fit_and_certificate',
       '07_information_decomposition','08_support_and_direction','09_numerical_diagnostics']
COLORS=['#147D92','#D5633D','#7962AB']


def style():
    fonts=sorted([p for p in font_manager.findSystemFonts() if 'NotoSansCJK' in p],key=lambda p:('Regular' not in p,p))
    if fonts:
        font_manager.fontManager.addfont(fonts[0]);plt.rcParams['font.family']=font_manager.FontProperties(fname=fonts[0]).get_name()
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
      'axes.grid':True,'grid.alpha':.16,'axes.unicode_minus':False,'figure.facecolor':'white',
      'axes.facecolor':'white','savefig.dpi':160})


def make_figure(number,r):
    style();hand=r['hand_states']
    if number==1:
        f,ax=plt.subplots(1,2,figsize=(10.5,3.8),layout='constrained')
        p,_=e.stable_softmax([[2,1,0]]);sig=1/(1+np.exp(-np.array([2,1,0])))
        for a,values,title in [(ax[0],p[0],'互斥三类：一次只取一个类别'),(ax[1],sig,'三个标签：可以同时为真')]:
            a.bar(['甲','乙','丙'],values,color=COLORS);a.set(ylim=(0,1),ylabel='概率',title=title)
            for i,v in enumerate(values):a.text(i,v+.04,f'{v:.3f}',ha='center')
            a.text(.5,.92,f'本例概率和 = {sum(values):.3f}',ha='center',transform=a.transAxes,fontsize=10)
    elif number==2:
        f,ax=plt.subplots(1,2,figsize=(10.5,3.5),layout='constrained')
        vertex=np.array([[0,0],[1,0],[.5,np.sqrt(3)/2]])
        ax[0].plot(*vertex[[0,1,2,0]].T,color='#667788');ax[0].set_aspect('equal');ax[0].set(xlim=(-.10,1.10),ylim=(-.15,1.06));ax[0].axis('off')
        for i,(x,y) in enumerate(vertex):ax[0].text(x,y+(.045 if i==2 else -.055),f'{["甲","乙","丙"][i]} = 1',ha='center')
        for pp,label,col in [([1/3]*3,'均匀',COLORS[2]),([.5,.25,.25],'x=-1 的最优',COLORS[0]),([.25,.25,.5],'x=+1 的最优',COLORS[1])]:
            pos=np.array(pp)@vertex;ax[0].scatter(*pos,color=col,s=65)
            offset=(0,-18) if label=='x=-1 的最优' else (12,6)
            ax[0].annotate(label,pos,xytext=offset,textcoords='offset points',fontsize=9,
                ha='center' if label=='x=-1 的最优' else 'left',
                va='top' if label=='x=-1 的最优' else 'bottom',
                arrowprops={'arrowstyle':'-','color':col,'lw':.8,'shrinkA':3,'shrinkB':4})
        ax[0].set_title('三类概率位于二维单纯形')
        z=np.array([[.2,-.4,.7]]);idx=np.arange(3)
        for j,s in enumerate([-7,0,11]):p,_=e.stable_softmax(z+s);ax[1].bar(idx+(j-1)*.24,p[0],.24,color=COLORS[j],label=f'所有得分 + {s}')
        ax[1].set(xticks=idx,xticklabels=['甲','乙','丙'],ylabel='概率',title='共同平移不改变分布');ax[1].legend(fontsize=9)
    elif number==3:
        f=plt.figure(figsize=(10.5,4.8),layout='constrained');gs=f.add_gridspec(2,2,height_ratios=[.7,1.3]);top=f.add_subplot(gs[0,:]);top.axis('off')
        text=['一行特征与参数\nz = a^TΘ','先减同一行最大值\nlog p = shifted − logden','取真实类的负log p\nℓ = −log p_y','局部反向与汇总\ndℓ/dz = p − onehot']
        for i,s in enumerate(text):
            top.text(.12+.25*i,.55,s,ha='center',va='center',fontsize=10)
            if i<3:top.annotate('',xy=(.255+.25*i,.55),xytext=(.23+.25*i,.55),arrowprops={'arrowstyle':'->'})
        a=f.add_subplot(gs[1,0]);b=f.add_subplot(gs[1,1]);t=np.linspace(-5,5,200);z=np.column_stack([t,np.zeros((len(t),2))]);p,lp=e.stable_softmax(z)
        for k in range(3):a.plot(t,p[:,k],color=COLORS[k],ls='--' if k==2 else '-',label=['甲','乙','丙'][k])
        a.set(xlabel='只改变甲类得分，另两类得分为0',ylabel='概率',title='一类增加，其他类共同让出质量');a.legend()
        b.plot(t,-lp[:,0],color=COLORS[0],label='真类=甲的损失');b.plot(t,p[:,0]-1,color=COLORS[1],label='对甲类得分的导数');b.axhline(0,color='#777',lw=.6);b.set(xlabel='甲类得分',ylabel='损失 / 导数',title='下降方向要看导数符号');b.legend(fontsize=9)
    elif number==4:
        f,ax=plt.subplots(1,2,figsize=(10.5,4.0),layout='constrained');ks=range(len(hand))
        for row in range(8):
            favored=row in (0,1,6,7)
            ax[0].plot(ks,[s['rows'][row]['loss'] for s in hand],'-o' if favored else '--s',alpha=.6,color=COLORS[0] if favored else COLORS[1],label=('M1 M2 M7 M8' if row==0 else 'M3 M4 M5 M6' if row==2 else None))
        ax[0].plot(ks,[s['objective'] for s in hand],'k.-',lw=2,label='八行平均');ax[0].set(xlabel='已完成的同步更新数',ylabel='单行损失 / 平均损失',xticks=list(ks),title='平均下降允许部分行变差');ax[0].legend(fontsize=9)
        idx=np.arange(3)
        for j,s in enumerate(hand):ax[1].bar(idx+(j-1)*.24,s['probabilities'][0],.24,color=COLORS[j],label=f'k={j}')
        ax[1].set(xticks=idx,xticklabels=['甲','乙','丙'],ylabel='x=-1 的预测概率',ylim=(0,.5),title='重算前向，不沿用旧概率');ax[1].legend(ncol=3,loc='upper center',fontsize=9,frameon=False)
    elif number==5:
        f,ax=plt.subplots(1,2,figsize=(10.5,3.8),layout='constrained');J=np.array(hand[0]['rows'][0]['local_softmax_jacobian']);im=ax[0].imshow(J,cmap='coolwarm',vmin=-2/9,vmax=2/9)
        for i,j in np.ndindex(J.shape):ax[0].text(j,i,'2/9' if i==j else '−1/9',ha='center',va='center')
        ax[0].set(xticks=range(3),yticks=range(3),xticklabels=['z甲','z乙','z丙'],yticklabels=['p甲','p乙','p丙'],title='均匀概率下的局部Jacobian');ax[0].grid(False)
        evals=np.linalg.eigvalsh(J);ax[1].bar(['共同平移','差异方向1','差异方向2'],evals,color=COLORS);ax[1].set(ylabel='特征值',title='一个零方向，两条可变方向');ax[1].text(.5,.7,'J·(1,1,1)^T = 0\n参数不可识别不是优化器故障',ha='center',transform=ax[1].transAxes,fontsize=10)
    elif number==6:
        f,ax=plt.subplots(1,2,figsize=(10.5,4.2),layout='constrained');trace=r['fit']['trace'];loss=[s['objective'] for s in trace];grad=[s['gradient_norm'] for s in trace]
        ax[0].plot(loss,color=COLORS[0],label='GD平均负对数似然');ax[0].axhline(r['certificate']['minimum_objective'],color=COLORS[1],ls='--',label='独立解析最小值');ax[0].set(xlabel='同步更新数',ylabel='F',title='85步到数值阈值；无泛化声明');ax[0].legend(fontsize=9)
        ax[1].semilogy(grad,color=COLORS[2],label='GD完整梯度范数');ax[1].axhline(1e-10,color='#777',ls=':',label='预声明停止阈值');ax[1].set(xlabel='同步更新数',ylabel='梯度L2范数',title='库状态与重算残差分开报告');ax[1].legend(fontsize=9);ax[1].text(.50,.78,'BFGS: precision loss\nsuccess=False\n重算梯度约7.5e-11',transform=ax[1].transAxes,fontsize=9.5,va='top')
    elif number==7:
        f,ax=plt.subplots(1,2,figsize=(10.5,3.9),layout='constrained');I=r['information'];R=r['reverse_information']
        ax[0].bar([0],I['entropy'],color=COLORS[0],label='H(q)');ax[0].bar([0],I['kl'],bottom=I['entropy'],color=COLORS[1],label='D(q||p)');ax[0].bar([1],I['cross_entropy'],color=COLORS[2],label='H(q,p)');ax[0].set(xticks=[0,1],xticklabels=['熵 + KL','交叉熵'],ylabel='nat',title='固定q时，优化交叉熵等价优化KL');ax[0].legend(fontsize=9,loc='lower right')
        idx=np.arange(3);ax[1].bar(idx-.16,I['q'],.32,color=COLORS[0],label='q=(.5,.3,.2)');ax[1].bar(idx+.16,I['p'],.32,color=COLORS[1],label='p=(.7,.2,.1)');ax[1].set(xticks=idx,xticklabels=['甲','乙','丙'],ylabel='概率',ylim=(0,.88),title='交换方向，权重分布也改变');ax[1].legend(fontsize=9);ax[1].text(.45,.60,f'D(q||p)={I["kl"]:.6f}\nD(p||q)={R["kl"]:.6f}',transform=ax[1].transAxes,fontsize=10)
    elif number==8:
        f,ax=plt.subplots(1,2,figsize=(10.5,4),layout='constrained');eps=np.logspace(-6,-.01,180);forward=[];reverse=[]
        q=[.5,.3,.2]
        for s in eps:
            p=[s,(1-s)*.6,(1-s)*.4];forward.append(e.information(q,p)['kl']);reverse.append(e.information(p,q)['kl'])
        ax[0].semilogx(eps,forward,color=COLORS[0],label='D(q||p)');ax[0].semilogx(eps,reverse,color=COLORS[1],label='D(p||q)');ax[0].set(xlabel='p甲 = ε；其他质量按3:2分配',ylabel='KL / nat',title='q甲>0而p甲趋零，正向KL发散');ax[0].legend(fontsize=9)
        t=np.linspace(0,1,301);H=[]
        for v in t:H.append(e.information([v,(1-v)/2,(1-v)/2],[1/3]*3)['entropy'])
        ax[1].plot(t,H,color=COLORS[2]);ax[1].axhline(np.log(3),color='#888',ls=':',label='log 3');ax[1].axvline(1/3,color=COLORS[0],ls=':');ax[1].set(xlabel='q甲 = t；q乙 = q丙 = (1-t)/2',ylabel='H(q) / nat',title='均匀分布的熵最大');ax[1].legend()
    elif number==9:
        f,ax=plt.subplots(1,2,figsize=(10.5,4.1),layout='constrained');scan=r['diagnostics']['finite_difference']
        for s,col in zip([1.,-.7],COLORS):
            v=[a for a in scan if a['scale']==s];ax[0].loglog([a['h'] for a in v],[a['maximum_gradient_error'] for a in v],'-o',color=col,label=f'非驻点scale={s}')
        ax[0].set(xlabel='中心差分步长 h',ylabel='最大梯度绝对误差',title='步长太小会出现消减');ax[0].legend(fontsize=9)
        t=np.linspace(20,42,180);z=np.column_stack([np.zeros(len(t)),-t,-t]);p,lp=e.stable_softmax(z);naive=np.log(np.exp(z).sum(axis=1))
        ax[1].semilogy(t,-lp[:,0],color=COLORS[0],lw=2,label='log1p保留赢者小损失');valid=naive>0;ax[1].semilogy(t[valid],naive[valid],'.',color=COLORS[1],label='log(sum(exp))的正值');ax[1].set(xlabel='另两类得分均为 −t，真类得分0',ylabel='正确类负对数损失',title='先减max仍可能丢掉微小尾项');ax[1].legend(fontsize=8,loc='lower left');ax[1].text(.39,.80,'naive的0点未画在对数轴',transform=ax[1].transAxes,fontsize=8)
    else:raise ValueError('figure index 1..9')
    return f


def render_all(r,directory):
    directory=Path(directory).absolute()
    for p in (directory,*directory.parents):
        if p.is_symlink():raise ValueError('symlink figure directory')
    if directory.resolve()==e.ROOT/'figures':raise ValueError('use outputs/ or a new directory, not original figures')
    directory.mkdir(parents=True,exist_ok=True)
    for i,name in enumerate(NAMES,1):
        target=directory/(name+'.png')
        if target.is_symlink() or (target.exists() and (not target.is_file() or target.stat().st_nlink>1)):raise ValueError('unsafe figure target')
        fig=make_figure(i,r);stream=io.BytesIO()
        try:fig.savefig(stream,format='png',bbox_inches='tight')
        finally:plt.close(fig)
        target.write_bytes(stream.getvalue())
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',required=True);p.add_argument('--directory',required=True);a=p.parse_args();render_all(json.loads(Path(a.report).read_text()),a.directory)
