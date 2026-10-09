"""由冻结模型和实时计算重建推断机制图。"""
from pathlib import Path
import argparse,os,math
os.environ.setdefault('MPLCONFIGDIR','/tmp/ml077-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,FancyArrowPatch
from matplotlib import font_manager
from experiment import run
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
if Path(FONT).exists():font_manager.fontManager.addfont(FONT)
plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':12})
INK='#15364d'
def canvas(title,h=4):
    fig,ax=plt.subplots(figsize=(10,h));ax.set(xlim=(0,10),ylim=(0,4));ax.axis('off');ax.set_title(title,loc='left',fontsize=16,color=INK,pad=16);return fig,ax
def node(ax,x,y,s,color='#d9edf7'):
    ax.add_patch(Circle((x,y),.28,facecolor=color,edgecolor=INK,lw=1.8));ax.text(x,y,s,ha='center',va='center',color=INK)
def edge(ax,a,b,arrow=True,color=INK):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>' if arrow else '-',shrinkA=22,shrinkB=22,mutation_scale=16,lw=1.8,color=color))
def save(fig,out,name):
    fig.savefig(out/(name+'.png'),dpi=190,bbox_inches='tight',facecolor='white');fig.savefig(out/(name+'.svg'),bbox_inches='tight',facecolor='white');plt.close(fig)
def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',default='outputs/figures');a=p.parse_args();out=Path(a.directory);out.mkdir(parents=True,exist_ok=True);r=run()
    fig,ax=canvas('查询、证据与隐藏变量先分工')
    for i,label in enumerate(['X0','X1','X2','X3','E']):node(ax,1+2*i,2.8,label,'#ffd18c' if i==4 else '#d9edf7')
    for i in range(4):edge(ax,(1+2*i,2.8),(3+2*i,2.8))
    for x,s in [(1,'查询\n保留X0'),(4,'隐藏\n求和X1、X2、X3'),(9,'证据\n固定E=1')]:ax.text(x,1.5,s,ha='center')
    ax.text(5,.35,'相容世界的权重相加 → 得到两种X0的分数 → 除以证据概率',ha='center',bbox=dict(boxstyle='round',fc='#d6eedf',ec='none'));save(fig,out,'01_query')
    fig,ax=canvas('消元X3：先乘含X3的两张表，再把X3求和')
    boxes=[(1.5,'P(X3|X2)\n范围：X2,X3'),(5,'P(E=1|X3)\n范围：X3'),(8.5,'m(X2)\n范围：X2')]
    for x,s in boxes:ax.text(x,2.7,s,ha='center',va='center',bbox=dict(boxstyle='round,pad=.7',fc='#d9edf7',ec=INK))
    ax.text(3.25,2.7,'×',ha='center',fontsize=24);ax.text(6.75,2.7,'求和\nX3 →',ha='center',fontsize=13)
    ax.text(5,1.2,'m(0) = 0.85 × 0.1 + 0.15 × 0.9 = 0.22\nm(1) = 0.25 × 0.1 + 0.75 × 0.9 = 0.70',ha='center',fontsize=15)
    ax.text(5,.25,'消息不是P(X2)：0.22 + 0.70 ≠ 1；它描述末端证据对每个X2状态的支持。',ha='center',fontsize=11);save(fig,out,'02_eliminate')
    fig,ax=canvas('链的两遍sum-product：两端信息在查询点相乘')
    for i in range(4):node(ax,1.5+2.3*i,2,f'X{i}')
    for i in range(3):
        edge(ax,(1.5+2.3*i,2),(3.8+2.3*i,2),False)
        ax.annotate('',(3.8+2.3*i,2.9),(1.5+2.3*i,2.9),arrowprops=dict(arrowstyle='->',color='#238dba',lw=2))
        ax.annotate('',(1.5+2.3*i,1.1),(3.8+2.3*i,1.1),arrowprops=dict(arrowstyle='->',color='#d28b2d',lw=2))
    ax.text(5,3.5,'前向 α：累积左侧先验与转移',ha='center',color='#238dba');ax.text(5,.3,'后向 β：累积右侧末端证据；局部分布 ∝ α × β',ha='center',color='#b47728');save(fig,out,'03_messages')
    fig,ax=plt.subplots(figsize=(9,4));prior=[row[1] for row in r['forward']];posterior=r['chain_p1_given_e1'];ax.plot(range(4),prior,'o-',label='未观测末端E',color='#559dbe');ax.plot(range(4),posterior,'o-',label='已观测E=1',color='#d78d31');ax.set(xticks=range(4),xticklabels=['X0','X1','X2','X3'],ylim=(0,1),ylabel='状态1的概率',title='同一证据改变整条链上的后验');ax.legend();ax.grid(alpha=.2);save(fig,out,'04_posteriors')
    fig,ax=canvas('星图先消中心：六个邻居被连成团',h=4.7)
    for center,fill in [(2.5,False),(7.5,True)]:
        coords=[(center+1.45*math.cos(i*math.pi/3),2+1.45*math.sin(i*math.pi/3)) for i in range(6)]
        for i,pt in enumerate(coords):node(ax,*pt,f'L{i}','#ffd18c' if i==5 else '#d9edf7')
        if not fill:
            node(ax,center,2,'C')
            for pt in coords:edge(ax,(center,2),pt,False)
        else:
            for i in range(6):
                for j in range(i+1,6):edge(ax,coords[i],coords[j],False,color='#d38e51')
    ax.text(2.5,.05,'先消叶：临时表最多4项',ha='center');ax.text(7.5,.05,'先消C：临时表128项，结果因子64项',ha='center',fontsize=11);save(fig,out,'05_order')
if __name__=='__main__':main()
