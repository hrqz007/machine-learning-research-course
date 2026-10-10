"""Four original figures; numerical panels come from experiment.run()."""
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'outputs/mpl-cache'))
import argparse,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
_font = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
if _font.exists(): font_manager.fontManager.addfont(str(_font))
from experiment import run
plt.rcParams.update({'font.family':['Noto Sans CJK JP','DejaVu Sans'],'axes.unicode_minus':False,'font.size':11,'figure.facecolor':'#fbfaf7','axes.facecolor':'white','svg.fonttype':'path'})
BLUE='#245e8e'; RED='#c05245'; GREEN='#28836c'

def build(directory):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True); r=run()
    def save(fig,name):
        fig.savefig(directory/(name+'.png'),dpi=160,bbox_inches='tight'); fig.savefig(directory/(name+'.svg'),bbox_inches='tight'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(11.5,5)); ax.axis('off'); positions={'x':(.06,.75),'y':(.06,.25),'t':(.35,.5),'q':(.65,.72),'z':(.92,.5)}
    labels={'x':'x=2\n∂z/∂x=39','y':'y=3\n∂z/∂y=26','t':'t=x·y=6\n∂z/∂t=13','q':'q=t·t=36\n∂z/∂q=1','z':'z=q+t=42\n种子=1'}
    for node,pos in positions.items(): ax.text(*pos,labels[node],ha='center',va='center',bbox=dict(boxstyle='round,pad=.6',fc='white',ec=BLUE,lw=2),transform=ax.transAxes,zorder=3)
    for a,b,rad in [('x','t',0),('y','t',0),('t','q',.25),('t','q',-.10),('q','z',0),('t','z',.38)]:
        ax.annotate('',xy=positions[b],xytext=positions[a],xycoords='axes fraction',arrowprops=dict(arrowstyle='->',color=GREEN,lw=2,shrinkA=45,shrinkB=47,connectionstyle=f'arc3,rad={rad}'))
    ax.text(.49,.80,'两个输入槽位',fontsize=10,color=GREEN,transform=ax.transAxes); ax.text(.46,.15,'同一个 t 节点，三条使用路径；不能覆盖贡献',ha='center',transform=ax.transAxes,color=RED)
    ax.set_title('图1 共享节点图：绿色为前向依赖，框中保存反向梯度',loc='left',pad=14,fontsize=15); save(fig,'01_shared_graph')
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.8)); axes[0].bar(['t·t 左槽位','t·t 右槽位','直接 +t'],[6,6,1],color=[BLUE,BLUE,RED]); axes[0].set(ylabel='对 ∂z/∂t 的贡献',ylim=(0,8),title='局部乘积规则产生两份6，加法再贡献1')
    for i,v in enumerate([6,6,1]): axes[0].text(i,v+.15,str(v),ha='center')
    axes[1].bar(['∂z/∂t','∂z/∂x','∂z/∂y'],[13,39,26],color=[GREEN,BLUE,RED]); axes[1].set(ylabel='累积后的梯度',ylim=(0,45),title='先把13攒齐，再经 t=x·y 向下传播')
    for i,v in enumerate([13,39,26]): axes[1].text(i,v+.8,str(v),ha='center')
    fig.tight_layout(); save(fig,'02_accumulated_paths')
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.8)); sweep=r['finite_difference_sweep']; axes[0].loglog([v['h'] for v in sweep],[max(v['max_abs_error'],1e-18) for v in sweep],'o-',color=BLUE); axes[0].set(xlabel='中心差分步长 h',ylabel='与反向梯度的最大绝对差',title='平滑点：截断误差与浮点消减竞争'); axes[0].grid(alpha=.25)
    t=np.linspace(-.3,.3,101); axes[1].plot(t,np.maximum(0,t),color=BLUE,lw=3,label='ReLU(x)'); axes[1].plot([-.2,.2],[0,.2],'--',color=RED,label='跨拐点中心割线斜率 0.5'); axes[1].scatter([0],[0],color=GREEN,s=60); axes[1].set(xlabel='x',ylabel='ReLU(x)',title='在0处不可微：实现选择0，不等于导数存在'); axes[1].legend(fontsize=9)
    fig.tight_layout(); save(fig,'03_gradient_check')
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.8));
    for ax,title,formula,vals,cats in [(axes[0],'前向：给输入方向，得到输出变化','J · (1,−1)ᵀ = (1,3)ᵀ',[1,3],['输出1切向量','输出2切向量']),(axes[1],'反向：给输出权重，得到输入敏感度','Jᵀ · (2,−1)ᵀ = (2,3)ᵀ',[2,3],['输入x伴随量','输入y伴随量'])]:
        ax.bar(cats,vals,color=[BLUE,GREEN],width=.5); ax.set(ylim=(0,4),title=title,ylabel='计算结果'); ax.text(.5,.9,formula,ha='center',transform=ax.transAxes,color=RED,fontsize=12)
    fig.suptitle('图4 f(x,y)=(xy,x²+y)，在(2,3)处 J=[[3,2],[4,1]]',fontsize=14); fig.tight_layout(rect=[0,0,1,.93]); save(fig,'04_jvp_vjp')
    return sorted(str(p) for p in directory.glob('*') if p.suffix in ['.png','.svg'])
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--directory',default='outputs/figures'); a=p.parse_args(); print('\n'.join(build(a.directory)))
