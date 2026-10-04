from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/ml-course-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch,Rectangle
from matplotlib import font_manager
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':11,'axes.unicode_minus':False,'savefig.facecolor':'white'})
OUT=Path(__file__).resolve().parent/'figures';OUT.mkdir(exist_ok=True)
blue='#2166ac';green='#16826b';orange='#db7625';gray='#64748b'
def save(fig,name):
 fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight',pad_inches=.16);plt.close(fig)
def box(ax,x,y,w,h,label,color):
 ax.add_patch(Rectangle((x,y),w,h,facecolor=color,alpha=.10,edgecolor=color,lw=1.8));ax.text(x+w/2,y+h/2,label,ha='center',va='center',color=color)
def arrow(ax,a,b,c=gray):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,lw=1.5,color=c))
fig,ax=plt.subplots(figsize=(9,3.2));ax.set(xlim=(0,10),ylim=(0,4));ax.axis('off')
box(ax,.1,2.3,2.1,1,'train.csv\n4 条训练记录',blue);box(ax,3,2.3,2.6,1,'冻结参数\nB  w=2  b=10',blue);arrow(ax,(2.2,2.8),(3,2.8))
box(ax,.1,.45,2.1,1,'test_inputs.csv\n65 与 85',green);box(ax,3,.45,2.6,1,'predictions.json\n140 与 180',green);arrow(ax,(2.2,.95),(3,.95));arrow(ax,(4.3,2.3),(4.3,1.45))
box(ax,7,2.3,2.6,1,'test_labels.csv\n144 与 174',orange);box(ax,7,.45,2.6,1,'metrics.json\nMAE=5',orange);arrow(ax,(8.3,2.3),(8.3,1.45),orange);arrow(ax,(5.6,.95),(7,.95),orange)
save(fig,'01_files')
fig,ax=plt.subplots(figsize=(8.8,3));ax.axis('off');table=ax.table(cellText=[['50','115','110','+5','5'],['60','130','130','0','0'],['80','160','170','-10','10'],['90','175','190','-15','15']],colLabels=['面积','C 的预测','标签','预测减标签','绝对误差'],loc='center',cellLoc='center');table.scale(1,1.7)
for (r,c),cell in table.get_celld().items():cell.set_edgecolor('#d4dde8');cell.set_facecolor('#e8eff7' if r==0 else '#ffffff');cell.set_text_props(color=orange if c==4 and r else '#18202a')
ax.text(.5,.02,'最后一步：(5 + 0 + 10 + 15) ÷ 4 = 7.5 万元',ha='center',color=orange,transform=ax.transAxes);save(fig,'02_table')
fig,axs=plt.subplots(1,2,figsize=(8.8,2.8));axs[0].bar(['第一条','第二条'],[10,-10],color=[blue,orange]);axs[0].axhline(0,color=gray);axs[0].set(ylim=(-13,13),title='带方向误差  平均 0',ylabel='万元');axs[1].bar(['第一条','第二条'],[10,10],color=[blue,orange]);axs[1].set(ylim=(0,13),title='绝对误差  平均 10',ylabel='万元');fig.tight_layout();save(fig,'03_errors')
fig,ax=plt.subplots(figsize=(9,2.5));ax.set(xlim=(0,10),ylim=(0,3));ax.axis('off');arrow(ax,(.3,1.5),(9.6,1.5));
for x,label,c in [(1.2,'比较训练误差',blue),(3.5,'冻结模型与基线',blue),(6,'写下测试预测',green),(8.4,'打开标签并评价',orange)]:ax.plot(x,1.5,'o',color=c,ms=9);ax.text(x,2.05,label,ha='center',color=c,fontsize=11)
ax.axvline(4.4,ymin=.17,ymax=.85,color=gray,ls='--');ax.text(2.2,.55,'允许用训练信息选参数',ha='center',color=blue);ax.text(7,.55,'这次评价不再改选择',ha='center',color=orange);save(fig,'04_freeze')
fig,axs=plt.subplots(1,2,figsize=(8.8,3));axs[0].bar(['A','B','C'],[10,0,7.5],color=[gray,blue,gray]);axs[0].set(title='训练集上的候选比较',ylabel='MAE 万元',ylim=(0,17));axs[1].bar(['冻结模型 B','固定基线'],[5,15],color=[blue,orange]);axs[1].set(title='同两条测试记录上的比较',ylabel='MAE 万元',ylim=(0,17))
for ax in axs:
 for patch in ax.patches:ax.text(patch.get_x()+patch.get_width()/2,patch.get_height()+.5,f'{patch.get_height():g}',ha='center')
fig.tight_layout();save(fig,'05_results')
fig,axs=plt.subplots(1,2,figsize=(8.8,3));axs[0].scatter([1,2],[0,0],s=90,c=[blue,orange]);axs[0].set(xticks=[1,2],xticklabels=['直线 B','记忆反例 M'],xlim=(.4,2.6),ylim=(-1,10),ylabel='训练 MAE 万元',title='相同的训练表现');axs[1].bar(['直线 B','记忆反例 M'],[5,840],color=[blue,orange]);axs[1].set(ylabel='测试 MAE 万元',ylim=(0,950),title='不同的新输入行为');axs[1].text(0,25,'5',ha='center');axs[1].text(1,860,'840',ha='center');fig.tight_layout();save(fig,'06_counterexample')
