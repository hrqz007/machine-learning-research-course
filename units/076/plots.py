"""重建五幅原创机制图；中文字体来自系统Noto CJK。"""
from pathlib import Path
import argparse,os
os.environ.setdefault('MPLCONFIGDIR','/tmp/ml076-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from matplotlib import font_manager
from experiment import run
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
if Path(FONT).exists():font_manager.fontManager.addfont(FONT)
plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':12})
BLUE='#d9edf7';ORANGE='#ffd18c';INK='#15364d';GREEN='#d6eedf'
def canvas(title,w=10,h=4):
    fig,ax=plt.subplots(figsize=(w,h));ax.set(xlim=(0,10),ylim=(0,4));ax.axis('off');ax.set_title(title,loc='left',fontsize=16,color=INK,pad=16);return fig,ax
def node(ax,x,y,label,observed=False,square=False):
    patch=Rectangle((x-.28,y-.28),.56,.56,facecolor=GREEN,edgecolor=INK,lw=1.8) if square else Circle((x,y),.28,facecolor=ORANGE if observed else BLUE,edgecolor=INK,lw=1.8)
    ax.add_patch(patch);ax.text(x,y,label,ha='center',va='center',fontsize=12,color=INK)
def edge(ax,a,b,arrow=True):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>' if arrow else '-',mutation_scale=16,shrinkA=22,shrinkB=22,color=INK,lw=1.8))
def save(fig,out,name):
    fig.savefig(out/(name+'.png'),dpi=190,bbox_inches='tight',facecolor='white');fig.savefig(out/(name+'.svg'),bbox_inches='tight',facecolor='white');plt.close(fig)
def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',default='outputs/figures');a=p.parse_args();out=Path(a.directory);out.mkdir(parents=True,exist_ok=True);r=run()
    fig,ax=canvas('局部条件表 → 完整联合分布')
    for x,s in [(1,'A'),(4,'B'),(7,'C')]:node(ax,x,2.8,s)
    edge(ax,(1,2.8),(4,2.8));edge(ax,(4,2.8),(7,2.8))
    for x,s in [(1,'P(A=1)=0.4'),(4,'P(B=1|A)\n0 → 0.1；1 → 0.8'),(7,'P(C=1|B)\n0 → 0.2；1 → 0.9')]:ax.text(x,1.6,s,ha='center',va='center')
    ax.text(4.8,.45,'完整状态 101：0.4 × 0.2 × 0.2 = 0.016\n枚举 000 到 111，八行总和为 1',ha='center',bbox=dict(boxstyle='round',fc=GREEN,ec='none'))
    save(fig,out,'01_factorization')
    fig,ax=canvas('三种路径：中间点的身份决定观测的作用',h=5)
    rows=[('链',.7,'right','right'),('叉',2,'left','right'),('碰撞',3.3,'right','left')]
    for label,y,d1,d2 in rows:
        ax.text(.1,y,label,va='center');points=[(1.8,y),(3.5,y),(5.2,y)]
        for pt,n in zip(points,'ABC'):node(ax,*pt,n)
        edge(ax,points[0] if d1=='right' else points[1],points[1] if d1=='right' else points[0]);edge(ax,points[1] if d2=='right' else points[2],points[2] if d2=='right' else points[1])
        ax.text(6,y,'未观测B：可活跃\n观测B：阻断' if label!='碰撞' else 'B及后代未观测：阻断\nB或后代观测：可活跃',va='center',fontsize=12)
    save(fig,out,'02_paths')
    fig,ax=plt.subplots(figsize=(9,4));values=[.2,r['rain_given_wet'],r['rain_given_wet_sprinkler']];bars=ax.bar(['无证据','只知 W=1','知 W=1 且 S=1'],values,color=['#5ba3c5','#e5a34a','#64a67e']);ax.set_ylim(0,.6);ax.set_ylabel('P(R=1 | 指定证据)');ax.set_title('碰撞点条件化：湿地的两个解释',loc='left');ax.bar_label(bars,fmt='%.6f',padding=5);save(fig,out,'03_explaining_away')
    fig,ax=canvas('Markov等价：同骨架，同未屏蔽碰撞结构',h=5)
    for y,label,dirs in [(3.4,'等价1',('right','right')),(2.4,'等价2',('left','right')),(1.4,'等价3',('left','left')),(.4,'不等价',('right','left'))]:
        ax.text(.1,y,label,va='center');pts=[(2,y),(4,y),(6,y)]
        for pt,s in zip(pts,'ABC'):node(ax,*pt,s)
        for i,d in enumerate(dirs):edge(ax,pts[i] if d=='right' else pts[i+1],pts[i+1] if d=='right' else pts[i])
        ax.text(7,y,'A 与 C | B 独立' if label!='不等价' else 'A 与 C 边缘独立',va='center',fontsize=11)
    save(fig,out,'04_equivalence')
    fig,ax=canvas('变量图、因子图与归一化流程',h=4.4)
    for x,s in [(1,'A'),(3,'B'),(5,'C')]:node(ax,x,3,s)
    edge(ax,(1,3),(3,3),False);edge(ax,(3,3),(5,3),False);ax.text(6,3,'无向变量图',va='center')
    for x,s in [(1,'A'),(3,'B'),(5,'C')]:node(ax,x,1.65,s)
    for x,s in [(2,'φAB'),(4,'φBC')]:node(ax,x,1.65,s,square=True)
    for x in range(1,5):edge(ax,(x,1.65),(x+1,1.65),False)
    ax.text(6,1.65,'因子图：圆点变量 / 方块因子',va='center',fontsize=11)
    ax.text(4.8,.25,'局部权重相乘 → 汇总八个状态得 Z=24 → 每个权重除以24',ha='center',bbox=dict(boxstyle='round',fc=GREEN,ec='none'))
    save(fig,out,'05_factor_graph')
if __name__=='__main__':main()
