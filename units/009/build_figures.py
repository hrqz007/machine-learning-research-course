"""第九讲原创数值几何图；Matplotlib 与 Noto CJK 字体。"""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).parent/'.mplconfig'))
os.environ.setdefault('XDG_CACHE_HOME',str(Path(__file__).parent/'.cache'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Arc, Circle, Polygon, FancyBboxPatch
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
font=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
if font.exists():
 font_manager.fontManager.addfont(str(font));family=font_manager.FontProperties(fname=str(font)).get_name()
else:family='sans-serif'
plt.rcParams.update({'font.family':family,'font.size':12,'axes.unicode_minus':False,'figure.facecolor':'white','savefig.facecolor':'white'})
C={'blue':'#2463A5','teal':'#137C83','orange':'#C46726','red':'#B83D49','gray':'#687789','ink':'#203044','pale':'#ECF3FA','green':'#388156'}
x=np.array([4.,1.]);y=np.array([3.,4.]);p=np.array([1.92,2.56]);r=x-p

def title(fig,s):fig.suptitle(s,fontsize=20,fontweight='bold',color=C['ink'],y=.98)
def coords(ax,xlim,ylim):
 ax.set_xlim(*xlim);ax.set_ylim(*ylim);ax.set_aspect('equal',adjustable='box')
 ax.axhline(0,color=C['gray'],lw=.9);ax.axvline(0,color=C['gray'],lw=.9)
 ax.grid(alpha=.17);ax.set_xlabel('第一分量');ax.set_ylabel('第二分量')
 for spine in ax.spines.values():spine.set_visible(False)
def arrow(ax,start,end,color='blue',lw=2.6,ls='-'):
 ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=16,lw=lw,color=C[color],linestyle=ls,zorder=3))
def label(ax,at,text,color='ink',ha='left'):ax.text(*at,text,ha=ha,va='center',color=C[color],fontsize=12)
def box(ax,xywh,text,color='pale',size=15):
 xx,yy,w,h=xywh;ax.add_patch(FancyBboxPatch((xx,yy),w,h,boxstyle='round,pad=.015',facecolor=C.get(color,color),edgecolor='white'))
 ax.text(xx+w/2,yy+h/2,text,ha='center',va='center',fontsize=size,color=C['ink'],linespacing=1.8)
def save(fig,name):fig.savefig(OUT/name,dpi=180,bbox_inches='tight',pad_inches=.18);plt.close(fig)

fig,(ax,bx)=plt.subplots(1,2,figsize=(11,5.5),gridspec_kw={'width_ratios':[1.1,1]});title(fig,'同一个二维向量 三种表示')
coords(ax,(-.5,5),(-.5,3));arrow(ax,(0,0),x);ax.plot([4,4],[0,1],'--',color=C['gray']);ax.plot([0,4],[1,1],'--',color=C['gray']);label(ax,(3.15,1.45),'x = (4, 1)','blue')
bx.axis('off');bx.set_xlim(0,1);bx.set_ylim(0,1)
box(bx,(.05,.61,.87,.22),'数学坐标  (4, 1)\n列写法仍是同两个分量')
box(bx,(.05,.33,.87,.21),'单个向量数组\nshape = (2,)')
box(bx,(.05,.04,.87,.21),'一批二维样本\nshape = (n, 2)')
fig.tight_layout(rect=[0,0,1,.93]);save(fig,'01_coordinates.png')

fig,(ax,bx)=plt.subplots(1,2,figsize=(12,5.9));title(fig,'加法连接位移 数乘改变长度与方向')
coords(ax,(-1,8),(-1,6));arrow(ax,(0,0),x);arrow(ax,x,x+y,'teal');arrow(ax,(0,0),x+y,'orange');arrow(ax,(0,0),y,'teal',1.3,'--');label(ax,(2.4,.3),'x','blue');label(ax,(5.8,2.5),'平移后的 y','teal');label(ax,(5.8,5.5),'x+y=(7,5)','orange');ax.set_title('首尾相接')
coords(bx,(-5,9),(-3,4));arrow(bx,(0,0),2*x,'teal');arrow(bx,(0,0),x);arrow(bx,(0,0),-x,'orange');label(bx,(7,2.8),'2x','teal');label(bx,(3.8,.3),'x','blue');label(bx,(-4.3,-1.8),'−x','orange');bx.scatter([0],[0],c=C['red'],s=35,zorder=5);label(bx,(.3,-.7),'0x 是零向量','red');bx.set_title('同一方向上的数乘')
fig.tight_layout(rect=[0,0,1,.93]);save(fig,'02_add_scale.png')

fig,ax=plt.subplots(figsize=(11,4.7));title(fig,'内积是对应相乘之后再求和');ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1)
box(ax,(.04,.39,.24,.36),'x = (4, 1)\ny = (3, 4)')
box(ax,(.37,.39,.25,.36),'逐元素相乘\n(12, 4)',size=16)
box(ax,(.72,.39,.23,.36),'求和\n12 + 4 = 16',color='#E6F3ED',size=16)
for aa,bb in [(.29,.36),(.63,.71)]:ax.annotate('',xy=(bb,.57),xytext=(aa,.57),arrowprops={'arrowstyle':'->','lw':2,'color':C['gray']})
ax.text(.16,.22,'两个输入都是 (2,)',ha='center');ax.text(.495,.22,'输出仍为 (2,)',ha='center');ax.text(.835,.22,'输出是标量',ha='center')
fig.tight_layout(rect=[0,0,1,.93]);save(fig,'03_dot.png')

fig,ax=plt.subplots(figsize=(9,5.8));title(fig,'同一个位移 L1 与 L2 衡量不同路径');coords(ax,(-.4,4.9),(-.4,2.5));arrow(ax,(0,0),x,'blue',3);ax.plot([0,4,4],[0,0,1],color=C['orange'],lw=3,marker='o');label(ax,(1.1,.65),'L2 = √17 ≈ 4.123','blue');label(ax,(2,-.23),'沿横轴 4','orange',ha='center');label(ax,(4.25,.55),'再走 1','orange');label(ax,(1.7,1.7),'L1 = 4 + 1 = 5','orange');label(ax,(4,1.33),'(4, 1)','blue');fig.tight_layout(rect=[0,0,1,.93]);save(fig,'04_norms.png')

fig,ax=plt.subplots(figsize=(8.8,6.3));title(fig,'距离来自点与点的差向量');coords(ax,(0,5.4),(0,6.1));z=np.array([1.,5.]);w=np.array([3.,4.]);arrow(ax,z,x,'blue',3);ax.plot([1,3,4],[5,4,1],'o--',color=C['orange'],lw=2);label(ax,(.6,5.35),'z=(1,5)','blue');label(ax,(3.45,.7),'x=(4,1)','blue');label(ax,(3.25,4.3),'途经 (3,4)','orange');label(ax,(1,2.3),'x−z=(3,−4)\nL2 距离 = 5','blue');label(ax,(2.6,5.65),'绕路：√5 + √10 > 5','orange',ha='center');fig.tight_layout(rect=[0,0,1,.93]);save(fig,'05_distance.png')

fig,ax=plt.subplots(figsize=(8.9,6.5));title(fig,'内积为零对应平方长度相加');coords(ax,(-.8,6.3),(-5,2.5));rr=np.array([1.,-4.]);arrow(ax,(0,0),x);arrow(ax,x,x+rr,'teal');arrow(ax,(0,0),rr,'teal',1.5,'--');arrow(ax,(0,0),x+rr,'orange');label(ax,(2.6,1.3),'x=(4,1)','blue');label(ax,(5.2,-1),'r=(1,−4)','teal');label(ax,(4.6,-3.55),'x+r=(5,−3)','orange');label(ax,(-.2,-4.5),'x·r = 4 − 4 = 0；17 + 17 = 34','ink');u=-x/np.linalg.norm(x)*.35;v=rr/np.linalg.norm(rr)*.35;points=np.array([x+u,x+u+v,x+v]);ax.plot(points[:,0],points[:,1],color=C['gray'],lw=1.5);fig.tight_layout(rect=[0,0,1,.93]);save(fig,'06_orthogonal.png')

fig,ax=plt.subplots(figsize=(9.8,6.5));title(fig,'正交投影将 x 拆成沿线部分与残差');coords(ax,(-.5,5.7),(-.5,5.2));line=np.array([-.1*y,1.15*y]);ax.plot(line[:,0],line[:,1],color=C['gray'],ls='--',lw=1.2);arrow(ax,(0,0),y,'teal');arrow(ax,(0,0),x,'blue');arrow(ax,(0,0),p,'orange',4);arrow(ax,p,x,'red');ax.scatter(*p,c=C['orange'],s=35,zorder=5);label(ax,(3.2,4.3),'y=(3,4)','teal');label(ax,(4.05,.58),'x=(4,1)','blue');label(ax,(.15,3.15),'p=(1.92,2.56)','orange');label(ax,(3.1,2.35),'r=(2.08,−1.56)','red');label(ax,(.4,-.29),'a=16/25；p=a y；x=p+r','ink');u=y/5*.25;v=r/np.linalg.norm(r)*.25;points=np.array([p+u,p+u+v,p+v]);ax.plot(points[:,0],points[:,1],color=C['gray'],lw=1.5);fig.tight_layout(rect=[0,0,1,.93]);save(fig,'07_projection.png')

fig,(ax,bx)=plt.subplots(1,2,figsize=(12,5.6),gridspec_kw={'width_ratios':[1.2,1]});title(fig,'内积界中的差额就是非负残差平方')
ax.barh([0],[10.24],height=.4,color=C['orange'],label='投影平方长度 256/25');ax.barh([0],[6.76],left=[10.24],height=.4,color=C['teal'],label='残差平方长度 169/25');ax.text(5.12,0,'10.24',ha='center',va='center',color='white',fontsize=16);ax.text(13.62,0,'6.76',ha='center',va='center',color='white',fontsize=16);ax.set_xlim(0,18);ax.set_ylim(-.7,.9);ax.set_yticks([]);ax.set_xlabel('平方长度');ax.set_title('17 = 10.24 + 6.76');ax.legend(loc='upper left',frameon=False);ax.spines[['top','right','left']].set_visible(False);ax.grid(axis='x',alpha=.12)
bx.axis('off');bx.set_xlim(0,1);bx.set_ylim(0,1);box(bx,(.04,.55,.9,.31),'本例不共线\n16 < 5√17',size=16);box(bx,(.04,.11,.9,.31),'同向或反向共线\n绝对内积 = 长度乘积',color='#E6F3ED',size=15);fig.tight_layout(rect=[0,0,1,.9]);save(fig,'08_cauchy.png')

fig,(ax,bx)=plt.subplots(1,2,figsize=(12,5.9),gridspec_kw={'width_ratios':[1.1,1]});title(fig,'单位圆中的横坐标是余弦');coords(ax,(-1.3,1.35),(-1.15,1.3));theta=np.linspace(0,2*np.pi,400);ax.plot(np.cos(theta),np.sin(theta),color=C['gray'],lw=1.5);u=np.array([.6,.8]);arrow(ax,(0,0),u,'teal');arrow(ax,(0,0),(1,0),'blue');ax.plot([.6,.6],[0,.8],'--',color=C['orange']);ax.plot([0,.6],[0,0],color=C['orange'],lw=4);ax.add_patch(Arc((0,0),.8,.8,theta1=0,theta2=np.degrees(np.arccos(.6)),color=C['teal'],lw=2));label(ax,(.55,.98),'单位方向 (0.6,0.8)','teal',ha='center');label(ax,(.32,-.21),'cos θ = 0.6','orange',ha='center');label(ax,(-.92,.7),'θ ≈ 53.13°','teal');ax.set_xlabel('横坐标');ax.set_ylabel('纵坐标')
bx.axis('off');bx.set_xlim(0,1);bx.set_ylim(0,1);box(bx,(.04,.52,.9,.37),'0° → 1\n90° → 0\n180° → −1',size=17);box(bx,(.04,.09,.9,.31),'本讲 x 与 y 的夹角\n约 39.0939°\n零向量角度未定义',color='#FFF1E4',size=14);fig.tight_layout(rect=[0,0,1,.91]);save(fig,'09_angle.png')

fig,ax=plt.subplots(figsize=(8,6.9));title(fig,'相同坐标尺度下的 L1 与 L2 单位球');coords(ax,(-1.4,1.4),(-1.4,1.4));ax.add_patch(Circle((0,0),1,facecolor=C['blue'],alpha=.12,edgecolor=C['blue'],lw=2));diamond=np.array([[1,0],[0,1],[-1,0],[0,-1]]);ax.add_patch(Polygon(diamond,facecolor=C['orange'],alpha=.2,edgecolor=C['orange'],lw=2));ax.plot(np.cos(theta),np.sin(theta),color=C['blue'],lw=2,label='L2 边界：a²+b²=1');closed=np.vstack([diamond,diamond[:1]]);ax.plot(closed[:,0],closed[:,1],color=C['orange'],lw=2,label='L1 边界：|a|+|b|=1');ax.set_xlabel('第一分量',loc='right',labelpad=3);ax.legend(loc='lower center',bbox_to_anchor=(.43,-.32),frameon=False);fig.tight_layout(rect=[0,.1,1,.92]);save(fig,'10_unit_balls.png')

fig,axes=plt.subplots(1,3,figsize=(13.2,5.7));title(fig,'面积换单位 原始距离排名会翻转')
raw=np.sqrt([101.,37.]);conv=np.sqrt([10100.,360001.]);scaled=np.sqrt([4.01,.4])
for ax,vals,sub,ylab in zip(axes,[raw,conv,scaled],['原始数字：平方米 + 年','原始数字：平方分米 + 年','同步换参考尺度后'],['数值距离（混合单位）','数值距离（混合单位）','无量纲距离']):
 bars=ax.bar(['A','B'],vals,color=[C['orange'],C['blue']],width=.6)
 ax.set_ylim(0,max(vals)*1.28);ax.set_title(sub,fontsize=12);ax.set_ylabel(ylab);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15)
 for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+max(vals)*.04,f'{v:.3f}',ha='center',fontsize=12)
 ax.text(.5,.94,'更近：'+('A' if np.argmin(vals)==0 else 'B'),ha='center',transform=ax.transAxes,color=C['teal'])
fig.text(.5,.025,'Q=(50,5)，A=(51,15)，B=(56,6)；参考尺度 (10 平方米, 5 年) 同步改为 (1000 平方分米, 5 年)',ha='center',fontsize=11,color=C['gray']);fig.tight_layout(rect=[0,.07,1,.9]);save(fig,'11_scale.png')
print('Generated 11 original geometry figures.')
