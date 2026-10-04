"""Optional figures: matplotlib and a Noto Sans CJK font; Linux font path below."""
from pathlib import Path
import os,json
os.environ.setdefault('MPLCONFIGDIR','/tmp/ml-course-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch,Rectangle
from matplotlib import font_manager
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':11,'axes.unicode_minus':False,'savefig.facecolor':'white'})
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
B='#2166ac';G='#16826b';O='#d16e22';R='#b13b45';GRAY='#64748b'
def canvas(h=3):
 f,a=plt.subplots(figsize=(9,h));a.set(xlim=(0,10),ylim=(0,4));a.axis('off');return f,a
def box(a,x,y,w,h,text,c=B):
 a.add_patch(Rectangle((x,y),w,h,facecolor=c,alpha=.1,edgecolor=c,lw=1.5));a.text(x+w/2,y+h/2,text,ha='center',va='center',color=c)
def arrow(a,x1,y1,x2,y2,c=GRAY):a.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=13,color=c,lw=1.5))
def save(f,name):f.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight',pad_inches=.18);plt.close(f)
f,a=canvas();labs=['任务与单位','固定数据','编号划分','候选与选择','指标与限制']
for i,label in enumerate(labs):
 x=.1+i*2;box(a,x,2,1.75,1.1,label,B if i<3 else G)
 if i<4:arrow(a,x+1.75,2.55,x+2,2.55)
a.text(5,.75,'代码版本、输入摘要与实际环境贯穿整条链',ha='center',color=O);save(f,'01_chain')
f,a=canvas();box(a,.3,2.3,2.4,1,'生成器与种子',B);box(a,3.7,2.3,2.6,1,'固定 houses.csv',G);box(a,7.4,2.3,2.3,1,'重复读取与训练',G);arrow(a,2.7,2.8,3.7,2.8);arrow(a,6.3,2.8,7.4,2.8);a.text(5,.85,'换一个新种子重新生成 → 新的数据版本',ha='center',color=O);save(f,'02_fixed_data')
f,a=canvas();
for x,n,label,c in [(.3,36,'训练：拟合参数',B),(3.6,12,'验证：选择方案',G),(6.9,12,'测试：评价冻结选择',O)]:box(a,x,1.8,2.8,1.4,f'{n} 条\n{label}',c)
arrow(a,3.1,2.5,3.6,2.5);arrow(a,6.4,2.5,6.9,2.5);a.text(5,.6,'同一条样本只承担一种角色',ha='center',color=GRAY);save(f,'03_roles')
f,a=canvas();box(a,.3,1.5,2.4,1.4,'稳定编号\nH001 … H060',B);box(a,3.7,1.5,2.6,1.4,'split.csv\n每个编号唯一映射',G);box(a,7.3,2.5,2.4,.7,'train 36',B);box(a,7.3,1.5,2.4,.7,'validation 12',G);box(a,7.3,.5,2.4,.7,'test 12',O);arrow(a,2.7,2.2,3.7,2.2)
for y in [2.85,1.85,.85]:arrow(a,6.3,2.2,7.3,y)
save(f,'04_split')
f,a=canvas();box(a,.2,2.3,2.3,1,'种子 20261004',B);box(a,3.2,2.3,2.7,1,'先抽60个面积\n再抽60个波动',G);box(a,6.8,2.3,2.9,1,'固定数据序列',G);arrow(a,2.5,2.8,3.2,2.8);arrow(a,5.9,2.8,6.8,2.8);a.text(5,1,'种子相同 + 调用顺序不同 → 序列可能不同',ha='center',color=R);a.text(5,.4,'另用 seed=7 产生划分，分开管理随机来源',ha='center',color=O);save(f,'05_seed')
import experiment as e
record=e.run();scores=[r['training_mae_wan'] for r in record['training']['grid_scores']]
f,a=plt.subplots(figsize=(7.5,3.8));import numpy as np
values=np.asarray(scores).reshape(3,3);im=a.imshow(values,cmap='Blues',vmin=0,vmax=50);a.set(xticks=[0,1,2],xticklabels=['b=0','b=10','b=20'],yticks=[0,1,2],yticklabels=['w=1.5','w=2','w=2.5'],xlabel='截距 万元',ylabel='斜率 万元每平方米')
for i in range(3):
 for j in range(3):a.text(j,i,f'{values[i,j]:.3f}',ha='center',va='center',color='white' if values[i,j]>30 else '#18202a')
a.add_patch(Rectangle((.5,.5),1,1,fill=False,edgecolor=O,lw=3));f.colorbar(im,ax=a,label='训练 MAE 万元');f.tight_layout();save(f,'06_grid')
f,axs=plt.subplots(1,2,figsize=(9,3));axs[0].bar(['常数','直线网格最优'],[record['training']['constant_training_mae_wan'],min(scores)],color=[B,G]);axs[0].set(title='训练角色',ylabel='MAE 万元',ylim=(0,48));axs[1].bar(['常数','直线网格最优'],[x['validation_mae_wan'] for x in record['validation']],color=[B,G]);axs[1].set(title='验证角色 决定最终方案',ylim=(0,48),ylabel='MAE 万元')
for ax in axs:
 for p in ax.patches:ax.text(p.get_x()+p.get_width()/2,p.get_height()+1,f'{p.get_height():.2f}',ha='center')
f.tight_layout();save(f,'07_selection')
f,a=canvas();box(a,.2,2.1,2.4,1.3,'冻结模型\n参数与方案已定',B);box(a,3.5,2.1,2.6,1.3,'打开测试\n记录误差与边界',O);box(a,7,2.1,2.7,1.3,'新研究问题\n另行设计评价',G);arrow(a,2.6,2.7,3.5,2.7);arrow(a,6.1,2.7,7,2.7);a.text(5,.8,'看过反馈后，不再称同一标签完全未参与开发',ha='center',color=R);save(f,'08_unseal')
f,a=canvas();box(a,.2,2.3,2.7,1,'同名文件\nhouses.csv',B);box(a,3.8,2.3,2.5,1,'读取全部字节',G);box(a,7.2,2.3,2.5,1,'SHA-256 摘要',O);arrow(a,2.9,2.8,3.8,2.8);arrow(a,6.3,2.8,7.2,2.8);a.text(5,1.05,'字节不同 → 摘要通常不同',ha='center',color=O);a.text(5,.4,'摘要不证明真实性、合法性或模型正确性',ha='center',color=GRAY);save(f,'09_hash')
f,a=canvas();box(a,.3,1.7,3.4,1.6,'声明的环境\nrequirements.txt\n需要哪些版本',B);box(a,6.3,1.7,3.4,1.6,'实际运行环境\nPython / NumPy 版本\n本次到底用了什么',G);arrow(a,3.7,2.5,6.3,2.5);a.text(5,.75,'安装说明存在，不等于所有平台都已实测',ha='center',color=O);save(f,'10_environment')
f,a=canvas();box(a,.2,2.2,2.5,1.1,'同一固定材料',B);box(a,3.7,2.6,2.4,.9,'新进程 1',G);box(a,3.7,.8,2.4,.9,'新进程 2',G);box(a,7.3,1.5,2.4,1.3,'比较记录\n输入、参数、预测',O);arrow(a,2.7,2.7,3.7,3.05);arrow(a,2.7,2.7,3.7,1.25);arrow(a,6.1,3.05,7.3,2.4);arrow(a,6.1,1.25,7.3,1.8);save(f,'11_rerun')
