"""原创机制图、冻结协议结果及弱标签重复计数反例。"""
import argparse,os,json
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'outputs/mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
import numpy as np
from experiment import run,select_labels
ROOT=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':11,'figure.dpi':150,'svg.fonttype':'path'})
BLUE='#17658c';ORANGE='#d97920'
def save(fig,name,d):
    fig.tight_layout();fig.savefig(d/(name+'.png'),bbox_inches='tight',facecolor='white');fig.savefig(d/(name+'.svg'),bbox_inches='tight',facecolor='white');plt.close(fig)
def main(directory):
    d=Path(directory);d.mkdir(parents=True,exist_ok=True);r=run();vis=json.loads((ROOT/'data/seed-7-features.json').read_text());audit=json.loads((ROOT/'data/seed-7-audit.json').read_text());X=np.array(vis['pool_X']);y=np.array(audit['pool_y']);ix,lab,_=select_labels(y,.1,0,7)
    fig,axes=plt.subplots(1,2,figsize=(9.2,4.2));axes[0].scatter(X[:,0],X[:,1],c='#ccd3da',s=18,label='未标注训练池');axes[0].scatter(X[ix,0],X[ix,1],c=lab,cmap='coolwarm',s=42,edgecolor='black',label='24条可用标签');axes[1].scatter(X[:,0],X[:,1],c=y,cmap='coolwarm',s=20);axes[0].set_title('算法看到：少量标签和大量位置');axes[1].set_title('仅审计可见：完整隐藏真值');
    for ax in axes:ax.set(xlabel='x1：含类别信号',ylabel='x2：干扰特征')
    axes[0].legend(fontsize=8);save(fig,'01_information',d)
    fig,ax=plt.subplots(figsize=(9.2,3.8));ax.axis('off');positions=[(.12,.6,'人工标签\n监督拟合'),(.39,.6,'未标注池预测\n置信度 ≥ 0.8'),(.68,.6,'冻结新伪标签\n单样本权重 0.5'),(.89,.6,'重新拟合\n再筛选')]
    for x,yy,label in positions:ax.text(x,yy,label,ha='center',va='center',bbox=dict(boxstyle='round,pad=.6',fc='#e6f0f6',ec=BLUE))
    for a,b in zip(positions,positions[1:]):ax.annotate('',xy=(b[0]-.09,.6),xytext=(a[0]+.09,.6),arrowprops={'arrowstyle':'->','color':BLUE})
    ax.text(.5,.17,'隐藏真值只在训练完成后审计；测试集不参与筛选、阈值或停止规则',ha='center',color='#8b3b1c');ax.set_title('伪标签的闭环会同时传播正确和错误判断');save(fig,'02_cycle',d)
    fig,axes=plt.subplots(1,3,figsize=(10.2,4.2),sharey=True)
    for ax,noise in zip(axes,[0.,.25,.5]):
        s=[v for v in r['summaries'] if v['noise']==noise];xx=np.arange(3);ax.plot(xx,[v['supervised_mean_accuracy'] for v in s],'-o',color=BLUE,label='监督基线');ax.plot(xx,[v['pseudo_mean_accuracy'] for v in s],'-s',color=ORANGE,label='伪标签');ax.set(xticks=xx,xticklabels=['约2%','10%','30%'],ylim=(.25,1),xlabel='名义标注比例',title=f'标签翻转概率 {noise:.0%}');ax.grid(alpha=.2)
    axes[0].set_ylabel('封存测试准确率（种子均值）');axes[0].legend(fontsize=9);fig.suptitle('固定45个配置；2% / 25%噪声一例单类不可训练，均值用4例',fontsize=11);save(fig,'03_results',d)
    fig,axes=plt.subplots(1,2,figsize=(9.2,4.3));axes[0].bar(['错误人工种子','新增错误伪标签'],[2,120],color=[BLUE,ORANGE]);axes[0].set(ylabel='错误标签条数',title='压力反例：错误从2条扩展到122条');stage=r['stress']['stages'];axes[1].bar(['初始模型','自训练后'],[v['mean_wrong_confidence'] for v in stage],color=[BLUE,ORANGE]);axes[1].set(ylim=(0,1.05),ylabel='错误预测的平均置信度',title='准确率均为0；置信度仍高，不必递增');save(fig,'04_reinforcement',d)
    fig,axes=plt.subplots(1,2,figsize=(9.2,4.2));w=r['weak_labels'];axes[0].bar(['规则1','规则2','规则3','多数票'],w['rule_coverage']+[w['coverage']],color=[BLUE,BLUE,BLUE,ORANGE]);axes[0].set(ylim=(0,1),ylabel='覆盖率',title='弃权与冲突决定有多少可用弱标签');axes[1].bar(['一票正、一票负','复制正票后'],[.5,.8],color=[BLUE,ORANGE]);axes[1].set(ylim=(0,1),ylabel='错误独立假设给出的 P(y=1)',title='准确率都假定0.8：复制不是新证据');save(fig,'05_weak_labels',d)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',default=str(ROOT/'outputs/figures'));a=p.parse_args();main(a.directory)
