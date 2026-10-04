"""Original explanatory figures. Optional Chinese font improves regeneration."""
from pathlib import Path
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ml006-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from experiment import run, HERE

FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
if FONT.exists():
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(FONT)).get_name()
plt.rcParams.update({"axes.unicode_minus":False, "font.size":12, "axes.spines.top":False,
                     "axes.spines.right":False, "figure.facecolor":"white", "savefig.facecolor":"white"})
C = {"blue":"#2667A8", "orange":"#D97C26", "green":"#23846A", "red":"#BB4A45", "gray":"#74808B", "light":"#E9EFF5"}
OUT=HERE/"figures"; OUT.mkdir(exist_ok=True)
r=run(); d=r["clean"]; s=r["summary"]; g=r["groups"]

def save(fig,name):
    fig.savefig(OUT/name,dpi=170,bbox_inches="tight",pad_inches=.22)
    plt.close(fig)

def title(ax,text): ax.set_title(text,loc="left",fontsize=15,pad=18,fontweight="bold")

fig,ax=plt.subplots(figsize=(11,4.9));ax.axis("off")
rows=[["0","L001","H01","50","100"],["1","L002","H02","60","120"],["2","L003","H03","80","160"],["3","L004","H04","空白","180"]]
t=ax.table(cellText=rows,colLabels=["位置／默认索引","挂牌身份","房屋身份","面积 平方米","挂牌价 万元"],cellLoc="center",loc="center",colWidths=[.18,.19,.19,.22,.22]);t.scale(1,2.2);t.auto_set_font_size(False);t.set_fontsize(12)
for (i,j),cell in t.get_celld().items():
    cell.set_edgecolor("white")
    cell.set_facecolor(C["blue"] if i==0 else ("#FFF0DD" if i==4 and j==3 else C["light"]))
    if i==0: cell.set_text_props(color="white")
ax.text(.02,.92,"一行是一条记录；一列是一种字段",fontsize=17,weight="bold",transform=ax.transAxes)
ax.text(.02,.08,"第一个位置是 0。L001 是身份，不会因为排序就变成 L004。",fontsize=13,transform=ax.transAxes)
save(fig,"01_table_identity.png")

fig,axs=plt.subplots(1,2,figsize=(11,4.5))
for ax in axs: ax.set_xlim(0,10);ax.set_ylim(0,6);ax.axis("off")
title(axs[0],"同一房屋可以有两次挂牌")
for y,txt in [(4,"L001   1月2日   100万元"),(1.5,"L013   1月20日   110万元")]:
    axs[0].text(3.2,y,txt,bbox=dict(boxstyle="round,pad=.6",facecolor="#E0F0EA",edgecolor=C["green"]),va="center")
    axs[0].annotate("",xy=(3,y),xytext=(1.4,2.8),arrowprops=dict(arrowstyle="->",color=C["green"],lw=2))
axs[0].text(.3,2.8,"H01",weight="bold");axs[0].text(2,.1,"保留两次事件",color=C["green"],fontsize=15)
title(axs[1],"同一事件的完全复制")
axs[1].text(.6,4,"原始第4行：L003 / H03 / 80 / 160",fontsize=12)
axs[1].text(.6,2.5,"原始第16行：L003 / H03 / 80 / 160",fontsize=12,color=C["red"])
axs[1].annotate("六列都相同",xy=(5,2.8),xytext=(5,3.5),ha="center",color=C["red"])
axs[1].text(.6,.8,"衍生表留一份；审计表记录去掉哪一份",fontsize=12)
save(fig,"02_duplicate_vs_relisting.png")

fig,ax=plt.subplots(figsize=(11,4.5))
observed=np.vstack([np.ones(14),d.area_sqm.notna().to_numpy(dtype=int),np.ones(14)])
from matplotlib.colors import ListedColormap
ax.imshow(observed,aspect="auto",cmap=ListedColormap(["#E9A450",C["blue"]]),vmin=0,vmax=1)
ax.set_xticks(range(14),d.event_id,rotation=50,ha="right");ax.set_yticks(range(3),["事件身份 14/14","面积 12/14","挂牌价 14/14"])
for j in [3,9]: ax.text(j,1,"缺",ha="center",va="center",color="black",weight="bold")
ax.set_xticks(np.arange(-.5,14,1),minor=True);ax.set_yticks(np.arange(-.5,3,1),minor=True);ax.grid(which="minor",color="white",lw=3);ax.tick_params(which="minor",bottom=False,left=False)
title(ax,"同一张表，每一列的有效观测数可以不同")
save(fig,"03_missing_matrix.png")

fig,axs=plt.subplots(1,2,figsize=(11,5))
ids=["L001","L002","L009","L013","L014"]; vals=[100,120,240,110,118]
for ax in axs: ax.set_xlim(0,4);ax.set_ylim(-.7,5.2);ax.axis("off")
title(axs[0],"右表 N 只有一个条目")
for i,(eid,val) in enumerate(zip(ids,vals)):
    y=4-i;axs[0].text(.1,y,f"{eid}  {val}",va="center",color=C["blue"])
    axs[0].annotate("",xy=(2.8,2),xytext=(1.4,y),arrowprops=dict(arrowstyle="->",color=C["blue"],alpha=.65))
axs[0].text(2.9,2,"N : A",va="center");axs[0].text(.2,-.55,"5次挂牌 → 5个匹配组合",weight="bold")
title(axs[1],"右表 N 同时有 A 和 B")
for i,(eid,val) in enumerate(zip(ids,vals)):
    y=4-i;axs[1].text(.1,y,f"{eid}  {val}",va="center",color=C["blue"])
    for yy in [1,3]: axs[1].annotate("",xy=(2.8,yy),xytext=(1.4,y),arrowprops=dict(arrowstyle="->",color=C["orange"],alpha=.55))
axs[1].text(2.9,3,"N : A",va="center");axs[1].text(2.9,1,"N : B",va="center");axs[1].text(.2,-.55,"5次挂牌 → 10个匹配组合",weight="bold",color=C["red"])
save(fig,"04_join_cardinality.png")

fig,axs=plt.subplots(1,2,figsize=(11,4.8))
labels=["正确左合并","冲突键合并","只留匹配行"]
for ax,values,ylabel in [(axs[0],[14,19,13],"衍生表行数"),(axs[1],[2888/14,3576/19,176],"平均挂牌价 万元")]:
    bars=ax.bar(labels,values,color=[C["blue"],C["orange"],C["red"]],width=.6)
    ax.bar_label(bars,labels=[str(v) if isinstance(v,int) else f"{v:.2f}" for v in values],padding=6)
    ax.set_ylim(0,max(values)*1.25);ax.set_ylabel(ylabel);ax.grid(axis="y",alpha=.15)
title(axs[0],"多行和少行都可能改变问题");title(axs[1],"数值改变不代表发现了新规律")
save(fig,"05_join_changes_mean.png")

fig,axs=plt.subplots(1,2,figsize=(11,4.8)); order=["N","S","E","W","X"]
z=g.loc[order]
axs[0].bar(order,z.observed_area_count,color=C["blue"],label="面积已观测")
axs[0].bar(order,z.missing_area_count,bottom=z.observed_area_count,color=C["orange"],label="面积缺失")
axs[0].set_ylim(0,6);axs[0].set_ylabel("挂牌事件数");axs[0].legend();title(axs[0],"组内样本数与缺失一起报告")
axs[1].bar(order,z.mean_area_sqm.astype(float),color=[C["blue"],C["orange"],C["blue"],C["blue"],C["gray"]]);axs[1].set_ylim(0,370);axs[1].set_ylabel("已观测面积的均值 平方米")
for i,key in enumerate(order): axs[1].text(i,float(z.loc[key,"mean_area_sqm"])+12,f"n={z.loc[key,'observed_area_count']}",ha="center")
title(axs[1],"南片区的均值只来自 1 个观测")
save(fig,"06_group_denominators.png")

fig,ax=plt.subplots(figsize=(10.8,5))
x=np.array([0,1,2,3]);y=np.array([50,60,100,130]);ax.plot(x,y,"o-",lw=2,color=C["blue"],ms=9)
for xi,yi in zip(x,y): ax.annotate(str(yi),(xi,yi),xytext=(0,12),textcoords="offset points",ha="center")
for xx,yy,txt,col in [(.75,57.5,"q=0.25 → 57.5",C["green"]),(1.5,80,"q=0.50 → 80",C["orange"]),(2.25,107.5,"q=0.75 → 107.5",C["red"])]:
    ax.plot([xx,xx],[40,yy],"--",color=col);ax.scatter(xx,yy,s=85,color=col,zorder=3)
    ax.annotate(txt,(xx,yy),xytext=(4,-28),textcoords="offset points",fontsize=11,color=col)
ax.set_xticks(x);ax.set_xlim(-.25,3.25);ax.set_ylim(40,150);ax.set_xlabel("排序后的位置 从 0 开始");ax.set_ylabel("面积 平方米")
title(ax,"线性插值先找位置 h = (n − 1)q，再在相邻值间移动")
save(fig,"07_quantile_interpolation.png")

fig,axs=plt.subplots(1,2,figsize=(11,4.7));areas=d.area_sqm.dropna().to_numpy(dtype=float)
for ax,bins,name in [(axs[0],[40,180,320],"2 个宽箱 每箱宽 140"),(axs[1],list(range(40,321,40)),"7 个细箱 每箱宽 40")]:
    n,b,p=ax.hist(areas,bins=bins,color=C["blue"],edgecolor="white",linewidth=1.5)
    for a,bh in zip(p,n):
        if bh: ax.text(a.get_x()+a.get_width()/2,bh+.18,str(int(bh)),ha="center")
    ax.set_xlim(40,320);ax.set_ylim(0,13);ax.set_xlabel("面积 平方米");ax.set_ylabel("观测个数");title(ax,name)
fig.suptitle("相同 12 个面积观测；另外 2 次挂牌的面积缺失",y=1.01,fontsize=13)
save(fig,"08_histogram_bins.png")

fig,axs=plt.subplots(1,2,figsize=(11,4.9));complete=d.loc[d.area_sqm.notna()].copy()
colors={"N":C["blue"],"S":C["green"],"E":C["orange"],"W":"#7855A5","X":C["red"]}
for ax in axs:
    for key,part in complete.groupby("district_code"):
        ax.scatter(part.area_sqm.astype(float),part.list_price_wan.astype(float),label=key,color=colors[key],s=70,alpha=.85,edgecolor="white")
    ax.set_xlabel("建筑面积 平方米");ax.set_ylabel("总挂牌价 万元");ax.grid(alpha=.15)
axs[0].set_xlim(30,320);axs[0].set_ylim(70,650);axs[0].legend(title="片区",ncol=3,loc="upper left");axs[0].annotate("L012 保留",(300,600),xytext=(220,500),arrowprops=dict(arrowstyle="->"))
axs[1].set_xlim(40,140);axs[1].set_ylim(80,290);axs[1].text(.03,.96,"局部放大；L012 位于视窗外",transform=axs[1].transAxes,va="top",fontsize=11,color=C["red"])
title(axs[0],"全体可画点记录 n=12");title(axs[1],"放大不能悄悄冒充全体")
save(fig,"09_scatter_scope.png")

fig,ax=plt.subplots(figsize=(11,4.8))
train=[50,60,80,100,70,90,110,50,60];test=[120,130,300]
ax.scatter(train,np.ones(9),s=90,alpha=.65,color=C["blue"],label="训练已观测 9 个")
ax.scatter(test,np.zeros(3),s=90,alpha=.8,color=C["orange"],label="测试已观测 3 个")
ax.axvline(670/9,color=C["green"],lw=2,label="只用训练学得 74.44")
ax.axvline(1220/12,color=C["red"],lw=2,ls="--",label="混入测试得到 101.67")
ax.set_yticks([0,1],["测试 另有1个缺失","训练 另有1个缺失"]);ax.set_ylim(-.5,1.8);ax.set_xlim(35,320);ax.set_xlabel("建筑面积 平方米")
ax.legend(loc="upper right",fontsize=11);title(ax,"训练参数由哪些数据决定，是可检查的边界")
save(fig,"10_train_only_imputation.png")
print("Created 10 original figures.")
