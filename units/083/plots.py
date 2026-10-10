"""Rebuild four original scientific figures from executable lesson functions."""
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'outputs/mpl-cache'))
import argparse, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
_font = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
if _font.exists(): font_manager.fontManager.addfont(str(_font))
from experiment import fixed_xor, alternative_xor, interpolate_square, run
plt.rcParams.update({'font.family':['Noto Sans CJK JP','DejaVu Sans'],'axes.unicode_minus':False,'font.size':11,'figure.facecolor':'#fbfaf7','axes.facecolor':'#ffffff','savefig.facecolor':'#fbfaf7','svg.fonttype':'path'})
BLUE='#245e8e'; RED='#c05245'; GREEN='#28836c'

def build(directory):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    def save(fig,name):
        fig.savefig(directory/(name+'.png'),dpi=160,bbox_inches='tight'); fig.savefig(directory/(name+'.svg'),bbox_inches='tight'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(11.5,4.7)); ax.axis('off')
    for x,text,c in [(.04,'输入 x\n2个数',BLUE),(.30,'仿射层 1\nW₁x+b₁',GREEN),(.59,'仿射层 2\nW₂h+b₂',GREEN),(.84,'输出 y\n1个数',BLUE)]:
        ax.text(x,.68,text,ha='center',va='center',transform=ax.transAxes,bbox=dict(boxstyle='round,pad=.6',fc='white',ec=c,lw=2))
    for a,b in [(.10,.23),(.39,.50),(.68,.79)]: ax.annotate('',xy=(b,.68),xytext=(a,.68),xycoords='axes fraction',arrowprops=dict(arrowstyle='->',lw=2,color=BLUE))
    ax.text(.47,.32,'y = (W₂W₁)x + (W₂b₁+b₂)\n仍然是一次仿射变换',ha='center',va='center',fontsize=19,color=BLUE,transform=ax.transAxes)
    ax.text(.47,.09,'增加中间非线性才可能打破此合并；更深并非自动表达更多函数',ha='center',transform=ax.transAxes,color='#555555')
    ax.set_title('图1 逐层计算可以更长，函数形式却未必更丰富',loc='left',pad=20,fontsize=16); save(fig,'01_affine_collapse')
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.8)); rows=run()['xor']
    for r in rows:
        color=RED if r['y'] else BLUE
        axes[0].scatter(*r['x'],s=140,color=color,zorder=3); axes[0].annotate(str(tuple(int(v) for v in r['x'])),r['x'],xytext=(7,7),textcoords='offset points')
    axes[0].plot([0,1],[0,1],':',color=BLUE); axes[0].plot([0,1],[1,0],':',color=RED)
    axes[0].set(xlim=(-.2,1.35),ylim=(-.2,1.3),xlabel='x₁',ylabel='x₂',title='输入空间：两类线段在中心相交')
    for h,label,color in [([0,0],'(0,0) → 类0',BLUE),([1,0],'(0,1)与(1,0)重合 → 类1',RED),([2,1],'(1,1) → 类0',BLUE)]:
        axes[1].scatter(*h,s=150,color=color,zorder=3); axes[1].annotate(label,h,xytext=(-20,16),textcoords='offset points',fontsize=9)
    h1=np.linspace(-.1,2.4,100); axes[1].plot(h1,(h1-.5)/2,'--',color=GREEN,label='h₁−2h₂ = 0.5')
    axes[1].set(xlim=(-.4,2.5),ylim=(-.45,1.4),xlabel='h₁=ReLU(x₁+x₂)',ylabel='h₂=ReLU(x₁+x₂−1)',title='隐藏空间：直线已经能分开三处位置'); axes[1].legend(loc='lower right')
    for ax in axes: ax.grid(alpha=.2)
    fig.tight_layout(); save(fig,'02_xor_representation')
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.8)); t=np.linspace(0,1,151); X,Y=np.meshgrid(t,t)
    for ax,Z,title in [(axes[0],np.maximum(0,X+Y)-2*np.maximum(0,X+Y-1),'和的三角帽：中心输出1'),(axes[1],np.abs(X-Y),'差的绝对值：中心输出0')]:
        im=ax.imshow(Z,origin='lower',extent=(0,1,0,1),cmap='viridis',vmin=0,vmax=1,aspect='equal'); ax.contour(X,Y,Z,levels=[.5],colors='white',linewidths=1.5); ax.scatter([0,0,1,1],[0,1,0,1],c=[0,1,1,0],cmap='coolwarm',edgecolors='black',s=75,clip_on=False); ax.set(xlabel='x₁',ylabel='x₂',title=title)
    fig.colorbar(im,ax=axes.ravel().tolist(),label='固定网络原始分数',shrink=.85); fig.suptitle('图3 同样拟合四个顶点，内部延拓可以不同',fontsize=16); save(fig,'03_extension_ambiguity')
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.8)); x=np.linspace(0,1,1025); axes[0].plot(x,x*x,color='black',lw=2,label='目标 x²')
    for n,c in [(2,RED),(4,BLUE),(8,GREEN)]: axes[0].plot(x,[interpolate_square(v,n) for v in x],color=c,label=f'{n}段线性插值')
    axes[0].set(xlabel='x',ylabel='函数值',title='分段线性可逐渐逼近曲线'); axes[0].legend()
    result=run()['square_interpolation']; n=[r['segments'] for r in result]; e=[r['max_grid_error'] for r in result]
    axes[1].loglog(n,e,'o-',color=BLUE,label='1025点网格实测'); axes[1].loglog(n,[1/(4*v*v) for v in n],'x--',color=RED,label='区间精确上界 1/(4n²)'); axes[1].set(xlabel='分段数 n',ylabel='最大绝对误差',title='这是一种构造，不是训练成功定理'); axes[1].legend()
    for ax in axes: ax.grid(alpha=.2)
    fig.tight_layout(); save(fig,'04_piecewise_approximation')
    return sorted(str(p) for p in directory.glob('*') if p.suffix in ['.png','.svg'])
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--directory',default='outputs/figures'); a=p.parse_args(); print('\n'.join(build(a.directory)))
