"""Rebuild four original PNG/SVG teaching figures from frozen measurements."""
from pathlib import Path
import argparse,json,os
os.environ.setdefault('MPLCONFIGDIR','/tmp/course-085-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from numpy_core import load_data,loss_and_grad
FONT=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
if FONT.exists():font_manager.fontManager.addfont(str(FONT));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(FONT)).get_name()
plt.rcParams.update({'font.size':11,'axes.unicode_minus':False,'svg.fonttype':'path','figure.facecolor':'#f8fafc','axes.facecolor':'#ffffff'})
ROOT=Path(__file__).parent

def save(fig,name,directory):
    for ext in ['png','svg']:fig.savefig(directory/f'{name}.{ext}',dpi=160,bbox_inches='tight')
    plt.close(fig)

def main(directory='outputs/figures',result=None):
    d=Path(directory);d.mkdir(parents=True,exist_ok=True)
    r=json.loads(Path(result or ROOT/'experiment-result.json').read_text())
    fig,ax=plt.subplots(figsize=(11,4));ax.set(xlim=(0,10),ylim=(0,4));ax.axis('off')
    names=['输入 X\nB × D','预激活 Z\nB × H','激活 A\nB × H','分数 S\nB × C','平均损失 L\n标量']
    for i,name in enumerate(names):
        x=.9+2.05*i
        ax.text(x,2.7,name,ha='center',va='center',bbox=dict(boxstyle='round,pad=.5',facecolor='#dbeafe',edgecolor='#2563eb'))
        if i<4:ax.annotate('',xy=(x+1.45,2.7),xytext=(x+.6,2.7),arrowprops=dict(arrowstyle='->',lw=2,color='#2563eb'))
    labels=['G_X = G_Z W1.T','G_Z = G_A ⊙ φ′(Z)','G_A = G_S W2.T','G_S = (P − E) / B']
    for i,label in enumerate(labels):
        x=.9+2.05*i;ax.text(x,1.1,label,ha='center',fontsize=10,bbox=dict(boxstyle='round,pad=.35',facecolor='#ffedd5',edgecolor='#ea580c'))
        if i<3:ax.annotate('',xy=(x+.5,1.65),xytext=(x+1.6,1.65),arrowprops=dict(arrowstyle='->',color='#ea580c'))
    ax.text(5,.1,'参数梯度：W1 与 G_W1 同形；b1 与 G_b1 同形；第二层同理',ha='center',color='#334155')
    ax.set_title('前向保留中间值，反向按形状逐层传递',fontsize=15,pad=16);save(fig,'01_shape_chain',d)
    fig,axes=plt.subplots(1,3,figsize=(10,3.4),constrained_layout=True)
    for ax,key,title in zip(axes,['dS','dH','dZ'],['分数梯度 G_S','隐藏激活梯度 G_A','ReLU 后的 G_Z']):
        v=np.array(r['hand_example']['cache'][key]);ax.imshow(v,cmap='RdBu_r',vmin=-.3,vmax=.3)
        for i in range(2):
            for j in range(2):ax.text(j,i,f'{v[i,j]:.6f}',ha='center',va='center',fontsize=11)
        ax.set_xticks([0,1],['坐标 1','坐标 2']);ax.set_yticks([0,1],['样本 1','样本 2']);ax.set_title(title)
    fig.suptitle('共享参数累加贡献；关闭的 ReLU 通道只截断自己的路径',fontsize=13);save(fig,'02_hand_gradients',d)
    fig,ax=plt.subplots(figsize=(7.5,4.5));s=r['gradient_audit']['epsilon_sweep']
    ax.loglog([x['epsilon'] for x in s],[x['max_abs'] for x in s],'o-',color='#2563eb',lw=2)
    ax.set(xlabel='中心差分步长 ε',ylabel='17 个参数中的最大绝对误差',title='步长减小：先减少截断误差，后放大舍入误差')
    ax.grid(True,which='both',alpha=.25);save(fig,'03_difference_sweep',d)
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),constrained_layout=True);h=r['training']['history']
    for key,label,col in [('train','训练','#2563eb'),('validation','验证','#ea580c')]:axes[0].semilogy([v['step'] for v in h],[v[key]['loss'] for v in h],label=label,color=col)
    axes[0].set(xlabel='参数更新次数',ylabel='平均交叉熵',title='固定预算：1200 次全批量更新');axes[0].legend();axes[0].grid(alpha=.2)
    a=np.linspace(-2,2,180);xx,yy=np.meshgrid(a,a);grid=np.c_[xx.ravel(),yy.ravel()]
    p={k:np.array(v) for k,v in r['final_parameters'].items()};prob=loss_and_grad(grid,np.zeros(len(grid),dtype=int),p)[2]['P'][:,1].reshape(xx.shape)
    im=axes[1].contourf(xx,yy,prob,levels=np.linspace(0,1,21),cmap='RdBu_r',alpha=.65)
    X,y=load_data()['test'];axes[1].scatter(X[:,0],X[:,1],c=y,cmap='RdBu_r',edgecolors='#334155',s=17,linewidths=.35)
    axes[1].set(xlabel='输入特征 1',ylabel='输入特征 2',title='冻结后测试点与类别 1 概率');fig.colorbar(im,ax=axes[1],label='P(y = 1 | x)');save(fig,'04_training_boundary',d)
    return sorted(str(p) for p in d.glob('*.png'))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--directory',default='outputs/figures');ap.add_argument('--result');args=ap.parse_args();print('\n'.join(main(args.directory,args.result)))
