"""生成本讲的原创教学图；需要 Matplotlib 与 Noto Sans CJK 字体。"""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).parent / '.mplconfig'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)
font_file = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
if font_file.exists():
    font_manager.fontManager.addfont(str(font_file))
    font_name = font_manager.FontProperties(fname=str(font_file)).get_name()
else:
    font_name = 'sans-serif'
plt.rcParams.update({'font.family':font_name, 'font.size':12, 'axes.unicode_minus':False,
                     'figure.facecolor':'white', 'savefig.facecolor':'white'})
C = {'blue':'#2463A5','lightblue':'#E8F1FA','teal':'#167D8D','lightteal':'#E5F4F2',
     'orange':'#C96B23','lightorange':'#FFF0E1','red':'#B83C43','lightred':'#FCEBEC',
     'ink':'#233346','gray':'#607181','pale':'#F3F6F8','green':'#2F8157','lightgreen':'#E6F5EC'}

def canvas(title, sub='', size=(11,5.7)):
    fig, ax = plt.subplots(figsize=size)
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
    ax.text(.035,.95,title,size=21,weight='bold',color=C['ink'],va='top')
    if sub:ax.text(.035,.86,sub,size=12,color=C['gray'],va='top')
    return fig,ax

def box(ax,x,y,w,h,text,color='lightblue',fontsize=14, edge='white'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.009,rounding_size=0.015',
                              linewidth=1.2,edgecolor=C.get(edge,edge),facecolor=C.get(color,color)))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',size=fontsize,color=C['ink'],linespacing=1.7)

def arrow(ax,x1,y1,x2,y2,color='gray'):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops={'arrowstyle':'->','lw':2,'color':C[color]})

def grid(ax,values,x,y,w,h,colors=None,fs=15):
    r=len(values); c=len(values[0]);cw=w/c;rh=h/r
    for i,row in enumerate(values):
        for j,v in enumerate(row):
            fc=colors[i][j] if colors is not None else ('lightblue' if j%2==0 else 'lightteal')
            ax.add_patch(Rectangle((x+j*cw,y+(r-i-1)*rh),cw,rh,facecolor=C.get(fc,fc),edgecolor='white',lw=3))
            ax.text(x+(j+.5)*cw,y+(r-i-.5)*rh,str(v),ha='center',va='center',size=fs,color=C['ink'])

def save(fig,name):
    fig.savefig(OUT/name,dpi=180,bbox_inches='tight',pad_inches=.15);plt.close(fig)

fig,ax=canvas('样本轴与特征轴','X.shape = (3, 2)     X.ndim = 2     X.size = 6')
grid(ax,[[50,1],[60,2],[80,1]],.28,.23,.4,.47)
ax.text(.38,.74,'面积 / 平方米',ha='center',color=C['blue'])
ax.text(.58,.74,'房间数 / 间',ha='center',color=C['teal'])
for y,label in [(.62,'样本 0'),(.465,'样本 1'),(.31,'样本 2')]:ax.text(.23,y,label,ha='right',va='center')
arrow(ax,.75,.7,.75,.25,'orange');ax.text(.79,.48,'axis=0\n样本轴\n长度 3',va='center',color=C['orange'])
arrow(ax,.29,.15,.67,.15,'blue');ax.text(.48,.07,'axis=1  特征轴  长度 2',ha='center',color=C['blue'])
save(fig,'01_shape_map.png')

fig,ax=canvas('数字相同 形状不同','整数索引移除所选轴；范围切片保留所选轴')
box(ax,.04,.5,.42,.25,'X[0] → [50, 1]\nshape = (2,)','lightorange')
box(ax,.54,.5,.42,.25,'X[0:1] → [[50, 1]]\nshape = (1, 2)','lightblue')
box(ax,.04,.14,.42,.25,'X[:, 0] → [50, 60, 80]\nshape = (3,)','lightorange')
box(ax,.54,.14,.42,.25,'X[:, 0:1] → 三行一列\nshape = (3, 1)','lightblue')
ax.text(.25,.43,'只有特征轴',ha='center',color=C['orange'])
ax.text(.75,.43,'仍有样本轴与特征轴',ha='center',color=C['blue'])
save(fig,'02_slicing.png')

fig,ax=canvas('模型 C 的误差逐项可查','面积 [50, 60, 80, 90] 平方米；标签 [110, 130, 170, 190] 万元',size=(12,5.3))
items=[('预测','[115, 130, 160, 175]','lightblue'),('预测 − 标签','[5, 0, −10, −15]','lightorange'),('取绝对值','[5, 0, 10, 15]','lightteal')]
for i,(title,val,col) in enumerate(items):
    x=.03+i*.33;box(ax,x,.44,.28,.28,title+'\n'+val,col,13)
    if i<2:arrow(ax,x+.285,.58,x+.325,.58)
box(ax,.24,.12,.52,.17,'(5 + 0 + 10 + 15) / 4 = 7.5 万元','lightgreen')
arrow(ax,.82,.42,.75,.3)
ax.text(.04,.33,'每一步形状仍为 (4,)',color=C['gray'])
save(fig,'03_error_pipeline.png')

fig,ax=canvas('axis 选择被合并的轴','示例 Z 使用无单位数字，先单独理解归约方向',size=(11,6.4))
grid(ax,[[1,2],[3,4],[5,6]],.28,.33,.32,.4)
arrow(ax,.62,.66,.73,.66,'teal');arrow(ax,.62,.53,.73,.53,'teal');arrow(ax,.62,.4,.73,.4,'teal')
grid(ax,[[3],[7],[11]],.76,.33,.13,.4,colors=[['lightteal']]*3)
ax.text(.82,.78,'sum(axis=1)',ha='center',color=C['teal'])
ax.text(.82,.26,'shape (3,)',ha='center',color=C['teal'])
arrow(ax,.36,.3,.36,.23,'blue');arrow(ax,.52,.3,.52,.23,'blue')
grid(ax,[[9,12]],.28,.1,.32,.11)
ax.text(.23,.155,'sum(axis=0)',ha='right',va='center',color=C['blue'])
ax.text(.64,.155,'shape (2,)',va='center',color=C['blue'])
save(fig,'04_axis_reductions.png')

fig,ax=canvas('先按特征乘 再按样本求和','每项贡献变为万元，才可以在一行内相加',size=(12,6.3))
grid(ax,[[50,1],[60,2],[80,1]],.03,.29,.25,.32)
grid(ax,[[2,5]],.03,.67,.25,.08)
ax.text(.155,.21,'X: (3, 2)\n系数: (2,)',ha='center',va='top')
arrow(ax,.30,.45,.37,.45)
grid(ax,[[100,5],[120,10],[160,5]],.39,.29,.25,.32)
ax.text(.515,.70,'逐元素贡献 / 万元',ha='center',color=C['blue'])
ax.text(.515,.21,'每行求和\naxis=1',ha='center',va='top')
arrow(ax,.66,.45,.73,.45)
grid(ax,[[115],[140],[175]],.76,.29,.16,.32,colors=[['lightgreen']]*3)
ax.text(.84,.70,'再加 10 万元',ha='center',color=C['green'])
ax.text(.84,.21,'预测: (3,)',ha='center',va='top')
save(fig,'05_weighted_broadcast.png')

fig,ax=canvas('错误广播把配对变成交叉比较','(3, 1) − (3,) → (3, 3)，只有对角线比较同一样本',size=(11,6.5))
values=[[0,-25,-60],[25,0,-35],[60,35,0]]
colors=[['lightgreen' if i==j else 'lightred' for j in range(3)] for i in range(3)]
grid(ax,values,.28,.28,.49,.44,colors=colors,fs=18)
for j,v in enumerate([115,140,175]):ax.text(.28+(j+.5)*.49/3,.765,f'y={v}',ha='center',size=12)
for i,v in enumerate([115,140,175]):ax.text(.25,.28+(2-i+.5)*.44/3,f'p={v}',ha='right',va='center',size=12)
box(ax,.12,.08,.77,.12,'正确 MAE = 0；错误 MAE = 240 / 9 ≈ 26.67 万元','lightorange',14)
save(fig,'06_broadcast_trap.png')

fig,ax=canvas('形状契约与独立核验','先让位置含义明确，再让函数执行计算',size=(11,6.7))
box(ax,.06,.62,.39,.18,'入口检查\nX: (n,d)；weights: (d,)','lightblue',14)
box(ax,.56,.62,.37,.18,'有效数据\n非空、实数、有限、单位明确','lightteal',14)
arrow(ax,.47,.71,.54,.71)
box(ax,.28,.35,.44,.18,'数组计算\n逐元素乘 → axis=1 求和','lightorange',14)
arrow(ax,.74,.60,.61,.55);arrow(ax,.24,.60,.39,.55)
box(ax,.06,.07,.39,.18,'出口检查\n输出 (n,)；输入未修改','lightblue',14)
box(ax,.56,.07,.37,.18,'独立证据\n手算 + 循环 + 边界测试','lightgreen',14)
arrow(ax,.4,.33,.26,.27);arrow(ax,.6,.33,.74,.27)
save(fig,'07_contract_checks.png')

fig,ax=canvas('视图共享 副本独立','a = [50, 60, 80, 90]；在修改前创建视图与副本',size=(11,6.1))
box(ax,.035,.48,.29,.29,'原数组 a\n[50, 600, 80, 90]','lightblue',14)
box(ax,.385,.48,.27,.29,'view = a[1:3]\n[600, 80]','lightorange',14)
box(ax,.72,.48,.24,.29,'copy = a[1:3].copy()\n[60, 80]','lightgreen',12)
arrow(ax,.37,.61,.34,.61,'orange')
ax.text(.36,.41,'共享数据',ha='center',color=C['orange'])
box(ax,.11,.13,.52,.16,'执行 view[0] = 600\n视图与原数组同时看到改动','lightorange',13)
box(ax,.71,.13,.25,.16,'副本不变\n不共享数据','lightgreen',13)
save(fig,'08_views_copies.png')

fig,ax=canvas('格式会限制范围与精度','示例只展示具体风险，不意味着低精度计算一定错误',size=(12,6.0))
box(ax,.04,.41,.42,.36,'整数范围\nint8: −128 至 127\n120 + 10 → −126（溢出）\n先转 int64 → 130','lightorange',14)
box(ax,.54,.41,.42,.36,'浮点累加\n[100000000, 1, −100000000]\nfloat32 求和 → 0\nfloat64 累加 → 1','lightblue',14)
box(ax,.11,.12,.78,.16,'先以 float32 存入 100000001 → 100000000\n再转 float64，也不能恢复已丢失的 1','lightred',14)
save(fig,'09_dtype_precision.png')

fig,ax=plt.subplots(figsize=(10.5,6.1))
n=np.array([10,100,1000,10000,100000])
ax.loglog(n,8*n,'o-',lw=2.5,color=C['teal'],label='正确配对：8n 字节')
ax.loglog(n,8*n*n,'s-',lw=2.5,color=C['red'],label='错误交叉：8n² 字节')
ax.set_title('相同样本数 不同形状的内存代价',loc='left',fontsize=20,pad=22,color=C['ink'])
ax.set_xlabel('样本数 n（对数刻度）',labelpad=12);ax.set_ylabel('仅结果数字的存储 / 字节（对数刻度）',labelpad=12)
ax.grid(True,which='major',alpha=.23);ax.legend(loc='upper left',frameon=False)
ax.annotate('n=100000\n错误输出约 80 GB',xy=(100000,8e10),xytext=(1800,5e8),
            arrowprops={'arrowstyle':'->','color':C['red']},color=C['red'],fontsize=13)
fig.text(.12,-.005,'float64 每个元素 8 字节；不包含输入、临时数组和运行环境',fontsize=11,color=C['gray'])
fig.tight_layout();save(fig,'10_cost_growth.png')
print('Generated 10 teaching figures.')
