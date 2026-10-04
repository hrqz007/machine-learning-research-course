"""Optional figure rebuild: matplotlib + Noto Sans CJK font on Linux."""
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
B='#2166ac';G='#16826b';O='#d16e22';R='#b13b45';GRAY='#64748b'
def canvas(h=3):
 f,a=plt.subplots(figsize=(9,h));a.set(xlim=(0,10),ylim=(0,4));a.axis('off');return f,a
def box(a,x,y,w,h,text,c=B):
 a.add_patch(Rectangle((x,y),w,h,facecolor=c,alpha=.1,edgecolor=c,lw=1.5));a.text(x+w/2,y+h/2,text,ha='center',va='center',color=c)
def arrow(a,x1,y1,x2,y2,c=GRAY):a.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=13,color=c,lw=1.5))
def save(f,name):f.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight',pad_inches=.18);plt.close(f)
f,a=canvas();box(a,.2,1.5,2.5,1,'纸笔\n2 × 70 + 10');box(a,3.6,2.3,1.5,.8,'2 × 70');box(a,5.8,2.3,1.5,.8,'+ 10');box(a,4.5,.7,3,.8,'print(2 * 70 + 10)',G);arrow(a,2.7,2,3.6,2.7);arrow(a,5.1,2.7,5.8,2.7);arrow(a,6.5,2.3,6,.8);box(a,8.2,1.5,1.5,1,'150',O);arrow(a,7.5,1.1,8.2,1.8);save(f,'01_expression')
f,a=canvas();box(a,.2,2.1,2.7,1.2,'屏幕源码\narea = 80',B);box(a,3.7,2.1,2.5,1.2,'实际执行\nShift + Enter',G);box(a,7,2.1,2.7,1.2,'内核状态\narea 关联 80',O);arrow(a,2.9,2.7,3.7,2.7);arrow(a,6.2,2.7,7,2.7);a.text(5,.85,'只编辑、不执行：内核仍可能保留之前的 70',ha='center',color=R);save(f,'02_execution')
f,a=canvas();box(a,.3,2.2,2.3,1,'执行前\narea → 70');box(a,3.7,2.2,2.5,1,'右侧求值\n70 + 5 → 75',G);box(a,7.3,2.2,2.3,1,'重新关联\narea → 75',O);arrow(a,2.6,2.7,3.7,2.7);arrow(a,6.2,2.7,7.3,2.7);a.text(5,.9,'area = area + 5  表示一次状态更新',ha='center');save(f,'03_assignment')
f,a=canvas();
for x,t,expr,result,c in [(0.3,'int 整数','70 + 5','75',B),(3.6,'float 浮点','70.0 + 5','75.0',G),(6.9,'str 文本','"70" + "5"','"705"',O)]:box(a,x,2.4,2.8,.8,t,c);a.text(x+1.4,1.8,expr,ha='center');arrow(a,x+1.4,1.5,x+1.4,1);a.text(x+1.4,.65,result,ha='center',color=c);save(f,'04_types')
f,a=canvas();a.text(.5,3,'正确',color=G);a.text(5,3,'(5 + 0 + 10 + 15) / 4',ha='center',fontsize=15);a.text(9,3,'7.5',color=G,ha='center');a.plot([2.5,7.6],[2.6,2.6],color=G,lw=3);a.text(5,2.1,'全部相加，再除以 4',ha='center',color=G);a.text(.5,1.1,'错误',color=R);a.text(5,1.1,'5 + 0 + 10 + 15 / 4',ha='center',fontsize=15);a.text(9,1.1,'18.75',color=R,ha='center');a.plot([6.2,7.6],[.75,.75],color=R,lw=3);a.text(5,.15,'只有最后一项被除',ha='center',color=R);save(f,'05_precedence')
f,a=canvas();a.text(.3,3.2,'索引',ha='left');a.text(.3,2.2,'areas',ha='left');a.text(.3,1.1,'prices',ha='left');
for j,(x,y) in enumerate(zip([50,60,80,90],[110,130,170,190])):
 xx=2+1.8*j;a.text(xx+.6,3.2,str(j),ha='center',color=O);box(a,xx,1.8,1.2,.8,str(x));box(a,xx,.7,1.2,.8,str(y),G);arrow(a,xx+.6,1.8,xx+.6,1.5)
a.text(5,.05,'同一个位置 → 同一条样本',ha='center',color=GRAY);save(f,'06_index')
f,a=canvas();box(a,.3,2.5,2.1,.7,'original');box(a,.3,1.2,2.1,.7,'other_name');box(a,4,2,2.7,1,'[55, 60, 80, 90]',O);arrow(a,2.4,2.8,4,2.7);arrow(a,2.4,1.6,4,2.3);box(a,7.5,2.5,2.1,.7,'independent',G);box(a,7.2,.9,2.7,1,'[999, 60, 80, 90]',G);arrow(a,8.55,2.5,8.55,1.9);a.text(4.5,.45,'共享对象',ha='center',color=O);a.text(8.5,.45,'独立一层列表',ha='center',color=G);save(f,'07_alias')
f,a=canvas();box(a,3.5,2.8,3,.8,'score < best_score ?',B);box(a,.5,.7,3.4,1,'更新 best_score',G);box(a,6.1,.7,3.4,1,'保留原选择',O);arrow(a,4,2.8,2.2,1.7,G);arrow(a,6,2.8,7.8,1.7,O);a.text(2.5,2.4,'True',color=G);a.text(7.1,2.4,'False',color=O);save(f,'08_branch')
f,a=canvas();
for i,(x,y) in enumerate(zip([50,60,80,90],[110,130,170,190])):
 xx=.25+2.45*i;box(a,xx,2.3,2.1,1.1,f'index={i}\n面积 {x}  标签 {y}',B);box(a,xx,.5,2.1,1.1,f'预测 {2*x+10}\n绝对误差 0',G);arrow(a,xx+1.05,2.3,xx+1.05,1.6)
a.text(5,.05,'每一列是一次循环，预测规则没有改变',ha='center',color=GRAY);save(f,'09_loop')
f,a=plt.subplots(figsize=(9,3));a.step([0,1,2,3,4],[0,5,5,15,30],where='post',color=B,lw=2);a.plot([0,1,2,3,4],[0,5,5,15,30],'o',color=G);a.set(xlabel='已处理样本数',ylabel='累计绝对误差 万元',xticks=[0,1,2,3,4],ylim=(-2,36));
for x,y in zip([0,1,2,3,4],[0,5,5,15,30]):a.text(x,y+2,str(y),ha='center')
a.text(2.1,28,'循环后：30 ÷ 4 = 7.5',color=O);a.spines[['top','right']].set_visible(False);save(f,'10_accumulator')
f,a=plt.subplots(figsize=(9,3));a.axis('off');t=a.table(cellText=[['100 / 10','120 / 10','160 / 10','180 / 10','10'],['110 / 0','130 / 0','170 / 0','190 / 0','0'],['115 / 5','130 / 0','160 / 10','175 / 15','7.5']],rowLabels=['A','B','C'],colLabels=['H1','H2','H3','H4','该行 MAE'],cellLoc='center',loc='center');t.scale(1,1.8)
for (r,c),cell in t.get_celld().items():cell.set_edgecolor('#d4dde8');cell.set_facecolor('#e8f3ed' if r==2 else '#edf2f7' if r==0 else '#ffffff')
a.text(.5,.03,'每格为 预测 / 绝对误差；每行开始重新清零累计量',ha='center',transform=a.transAxes);save(f,'11_nested')
f,a=plt.subplots(figsize=(9,2.3));a.set(xlim=(0,10),ylim=(0,2));a.axis('off');a.plot([.5,9.5],[1,1],color=GRAY,lw=1.5)
for x in range(1,10):a.plot([x,x],[.88,1.12],color=B)
a.plot(5.4,1,'o',color=O,ms=9);arrow(a,5.4,1.2,5,1.7,O);a.text(5,1.85,'用邻近可表示值保存',ha='center',color=O);a.text(5.4,.4,'数学上的目标值',ha='center');a.text(5,.02,'示意刻度被夸张放大，并非真实浮点间距',ha='center',color=GRAY);save(f,'12_float')
