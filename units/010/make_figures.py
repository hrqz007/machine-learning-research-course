"""Original explanatory diagrams; no external images. Optional CJK font."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/ml010-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from experiment import HERE,load_data,rotation
FONT=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
if FONT.exists():
 font_manager.fontManager.addfont(str(FONT));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(FONT)).get_name()
plt.rcParams.update({'font.size':12,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'})
BLUE='#2667A8';ORANGE='#D77925';GREEN='#23846A';RED='#B64A43';GRAY='#AAB3BD';PURPLE='#7457A0'
OUT=HERE/'figures';OUT.mkdir(exist_ok=True)
ids,X,cfg=load_data();A=np.array(cfg['A_rect']);B=np.array(cfg['B_rect']);H=np.array(cfg['shear']);D=np.array(cfg['stretch'])
def save(fig,name):
 fig.savefig(OUT/name,dpi=170,bbox_inches='tight',pad_inches=.22);plt.close(fig)
def title(ax,s):ax.set_title(s,loc='left',fontsize=15,pad=15,weight='bold')
def plane(ax,lim=4):
 ax.set(xlim=(-lim,lim),ylim=(-lim,lim),xlabel='第1坐标 无量纲',ylabel='第2坐标 无量纲');ax.set_aspect('equal');ax.axhline(0,color=GRAY,lw=.8);ax.axvline(0,color=GRAY,lw=.8);ax.grid(alpha=.18)
def arrow(ax,v,color,label=None,start=(0,0),lw=2.5):
 start=np.array(start);v=np.array(v);ax.annotate('',xy=start+v,xytext=start,arrowprops=dict(arrowstyle='->',color=color,lw=lw))
 if label:ax.annotate(label,start+v,xytext=(6,6),textcoords='offset points',color=color)
def matrix(ax,M,name,highlight=None):
 M=np.asarray(M);ax.axis('off');t=ax.table(cellText=[[f'{x:g}' for x in row] for row in M],cellLoc='center',loc='center');t.scale(1,2.3);t.auto_set_font_size(False);t.set_fontsize(16)
 for (i,j),cell in t.get_celld().items():cell.set_facecolor('#E9EFF5');cell.set_edgecolor('white')
 if highlight:
  kind,num=highlight
  for (i,j),cell in t.get_celld().items():
   if (kind=='row' and i==num) or (kind=='col' and j==num):cell.set_facecolor('#F5D6A9')
 title(ax,name)
def grid(ax,M,caption,lim=4.5):
 plane(ax,lim);t=np.linspace(-2,2,61)
 for s in np.arange(-2,2.1,.5):
  for pts in [np.column_stack([t,np.full_like(t,s)]),np.column_stack([np.full_like(t,s),t])]:
   ax.plot(pts[:,0],pts[:,1],color=GRAY,lw=.6,alpha=.55)
   q=pts@M.T;ax.plot(q[:,0],q[:,1],color=BLUE,lw=1,alpha=.65)
 for v,col,name in [(np.array([1.,0.]),GREEN,'A e1'),(np.array([0.,1.]),ORANGE,'A e2')]:
  q=M@v
  if np.linalg.norm(q)>0:arrow(ax,q,col,name)
  else:ax.scatter([0],[0],color=col,s=60,zorder=4)
 title(ax,caption)

fig,axs=plt.subplots(1,2,figsize=(11,4.6));matrix(axs[0],X,'X：4行样本 × 3列特征');matrix(axs[1],A,'A：2行输出规则 × 3列输入坐标')
axs[0].text(.05,.03,'每一行回答：这是哪条样本？',transform=axs[0].transAxes);axs[1].text(.02,.03,'每一列回答：此输入坐标贡献到哪里？',transform=axs[1].transAxes)
save(fig,'01_shapes_meanings.png')
fig,axs=plt.subplots(1,3,figsize=(11,4.5));matrix(axs[0],A,'A 为 2×3',('row',0));matrix(axs[1],B,'B 为 3×2',('col',0));matrix(axs[2],A@B,'AB 为 2×2')
fig.text(.13,.04,'输出左上角：1×2 + 2×1 + (−1)×(−1) = 5',fontsize=14,color=BLUE);save(fig,'02_row_column_product.png')
fig,axs=plt.subplots(1,2,figsize=(11,5));M=np.array(cfg['column_demo']);x=np.array([1,2])
for ax in axs:plane(ax,5)
arrow(axs[0],[1,0],GREEN,'e1');arrow(axs[0],[0,1],ORANGE,'e2');arrow(axs[0],x,BLUE,'x = e1 + 2 e2');title(axs[0],'输入先拆成单位方向')
arrow(axs[1],M[:,0],GREEN,'M e1 = (2,1)');arrow(axs[1],2*M[:,1],ORANGE,start=M[:,0]);axs[1].text(2.6,.35,'2 M e2',color=ORANGE);arrow(axs[1],M@x,BLUE,'M x = (4,3)');title(axs[1],'输出是矩阵列的相同系数组合');save(fig,'03_columns_as_images.png')
fig,axs=plt.subplots(1,2,figsize=(11,5.2));grid(axs[0],np.array(cfg['scale']),'缩放：横向×2 纵向×0.5');grid(axs[1],H,'剪切：新横坐标 = x1 + x2')
fig.suptitle('灰网格为输入，蓝网格为输出；列向量作用 y = A x',y=1.02);save(fig,'04_scale_shear_grid.png')
fig,axs=plt.subplots(1,2,figsize=(11,5.2));grid(axs[0],rotation(np.pi/4),'逆时针旋转45°',3.5);grid(axs[1],np.array(cfg['collapse']),'压扁：保留x1 丢掉x2',3.5)
axs[1].text(.05,.92,'不同高度落到同一水平线',transform=axs[1].transAxes,color=RED);fig.suptitle('灰网格为输入，蓝网格为输出；原点仍在原点',y=1.02);save(fig,'05_rotate_collapse_grid.png')
fig,axs=plt.subplots(1,2,figsize=(11,5));x=np.array([1.,1.])
for ax,first,second,label in [(axs[0],D,H,'先D后H：H D x = (3,1)'),(axs[1],H,D,'先H后D：D H x = (4,1)')]:
 plane(ax,5);pts=[x,first@x,second@first@x];ax.scatter(*np.array(pts).T,c=[GRAY,ORANGE,BLUE],s=75,zorder=5)
 for i in range(2):arrow(ax,pts[i+1]-pts[i],[ORANGE,BLUE][i],start=pts[i],lw=2)
 for point,name in zip(pts,['x','中间','最后']):ax.annotate(name,point,xytext=(2,12),textcoords='offset points')
 title(ax,label)
save(fig,'06_order_matters.png')
fig,axs=plt.subplots(1,3,figsize=(11,4.7));matrix(axs[0],A@B,'AB：先得到2×2');matrix(axs[1],(A@B).T,'(AB)^T：行列交换');matrix(axs[2],B.T@A.T,'B^T A^T：同一结果')
fig.text(.07,.05,'顺序反转：B^T 为 2×3，A^T 为 3×2；先检查内侧维数3相等',fontsize=13);save(fig,'07_transpose_product.png')
fig,axs=plt.subplots(1,2,figsize=(11,5));v=np.array([3,1]);b1=np.array([1,1]);b2=np.array([1,-1])
for ax in axs:plane(ax,4)
arrow(axs[0],[3,0],GREEN);axs[0].text(1,-.45,'3 e1',color=GREEN);arrow(axs[0],[0,1],ORANGE,start=[3,0]);axs[0].text(3.15,.4,'1 e2',color=ORANGE);arrow(axs[0],v,BLUE,'x = (3,1)');title(axs[0],'标准坐标：3 与 1')
arrow(axs[1],2*b1,GREEN,'2 b1');arrow(axs[1],b2,ORANGE,start=2*b1);axs[1].text(2.8,1.85,'1 b2',color=ORANGE);arrow(axs[1],v,BLUE,'同一个 x');title(axs[1],'新基坐标：2 与 1');save(fig,'08_basis_coordinates.png')
fig,ax=plt.subplots(figsize=(7.8,6));plane(ax,1.4);t=np.linspace(0,2*np.pi,400);ax.plot(np.cos(t),np.sin(t),color=GRAY);q=np.sqrt(.5);arrow(ax,[q,q],BLUE,'(cos θ, sin θ)')
ax.plot([0,q],[0,0],color=GREEN,lw=4);ax.plot([q,q],[0,q],color=ORANGE,lw=4);ax.text(.2,-.14,'cos θ',color=GREEN);ax.text(q+.06,.3,'sin θ',color=ORANGE)
a=np.linspace(0,np.pi/4,40);ax.plot(.28*np.cos(a),.28*np.sin(a),color=RED,lw=2);ax.text(.38,.17,'θ');ax.text(-1.27,1.19,'半径1；45° = π/4弧度',fontsize=13);title(ax,'单位圆定义正弦与余弦');save(fig,'09_unit_circle.png')
fig,axs=plt.subplots(1,2,figsize=(11,5));x=np.array([2.,1.]);z=np.array([1.,3.]);R=rotation(np.pi/4)
for ax,M,label in [(axs[0],np.eye(2),'旋转之前'),(axs[1],R,'逆时针旋转45°之后')]:
 plane(ax,4);arrow(ax,M@x,BLUE,'x' if label=='旋转之前' else 'R x');arrow(ax,M@z,ORANGE,'z' if label=='旋转之前' else 'R z');pts=np.vstack([M@x,M@z]);ax.plot(pts[:,0],pts[:,1],'--',color=GREEN);title(ax,label)
 ax.text(.03,.04,'内积=5；两箭头长度与端点距离不变',transform=ax.transAxes,fontsize=11)
save(fig,'10_rotation_preserves_geometry.png')
fig,axs=plt.subplots(1,2,figsize=(11,5));b=np.array([1.,-1.]);pts=np.array([[0,0],[1,0],[0,1],[1,1]])
for ax,bias,label in [(axs[0],np.zeros(2),'线性：H 0 = 0'),(axs[1],b,'仿射：H 0 + b = b')]:
 plane(ax,3.5);q=pts@H.T+bias;ax.scatter(q[:,0],q[:,1],c=[RED,BLUE,BLUE,BLUE],s=75,zorder=3);order=[0,1,3,2,0];ax.plot(q[order,0],q[order,1],color=BLUE);ax.annotate('原点的去向',q[0],xytext=(8,-20),textcoords='offset points',color=RED);title(ax,label)
save(fig,'11_affine_bias.png')
fig,axs=plt.subplots(1,3,figsize=(12,5));matrix(axs[0],X,'X：4×3');matrix(axs[1],A.T,'A^T：3×2');matrix(axs[2],X@A.T,'Y：4×2')
fig.text(.12,.05,'样本轴4保留；输入特征轴3求和消去；输出坐标轴2保留',fontsize=14,color=BLUE);save(fig,'12_batch_axes.png')
print('Created 12 original teaching figures.')
