"""Original real-measurement figures; no framework execution is simulated."""
from pathlib import Path
import argparse,json,os
os.environ.setdefault('MPLCONFIGDIR','/tmp/course-086-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
FONT=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
if FONT.exists():font_manager.fontManager.addfont(str(FONT));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(FONT)).get_name()
plt.rcParams.update({'font.size':11,'axes.unicode_minus':False,'svg.fonttype':'path','figure.facecolor':'#f8fafc','axes.facecolor':'white'})
ROOT=Path(__file__).parent

def save(fig,name,directory):
    for ext in ['png','svg']:fig.savefig(directory/f'{name}.{ext}',dpi=160,bbox_inches='tight')
    plt.close(fig)

def main(directory='outputs/figures',result=None):
    d=Path(directory);d.mkdir(parents=True,exist_ok=True);r=json.loads(Path(result or ROOT/'experiment-result.json').read_text())
    fig,ax=plt.subplots(figsize=(10,4.5));ax.axis('off');ax.set(xlim=(0,10),ylim=(0,5))
    boxes=[(2.3,3.6,'NumPy\nW1: D × H\nZ = X @ W1 + b1','#dbeafe'),(7.6,3.6,'PyTorch nn.Linear\nweight: H × D\nZ = X @ weight.T + bias','#dcfce7'),(2.3,1.1,'NumPy 梯度\nG_W1: D × H','#dbeafe'),(7.6,1.1,'PyTorch weight.grad\nH × D','#dcfce7')]
    for x,y,t,c in boxes:ax.text(x,y,t,ha='center',va='center',bbox=dict(boxstyle='round,pad=.8',facecolor=c,edgecolor='#64748b'),fontsize=12)
    ax.annotate('',xy=(5.8,3.6),xytext=(4.2,3.6),arrowprops=dict(arrowstyle='->',lw=2));ax.text(5,4.15,'转置后复制',ha='center')
    ax.annotate('',xy=(4.2,1.1),xytext=(5.8,1.1),arrowprops=dict(arrowstyle='->',lw=2));ax.text(5,1.65,'转置后比较',ha='center')
    ax.set_title('相同函数，不同权重存储布局',fontsize=16);save(fig,'01_layout_bridge',d)
    a=r['accumulation'];fig,ax=plt.subplots(figsize=(8,4.5))
    values=[a['first'],a['second_without_zero'],a['after_reset']];bars=ax.bar(['第一次 backward','未清空，再 backward','清空后，再 backward'],values,color=['#2563eb','#ea580c','#16a34a'],width=.55)
    for b,v in zip(bars,values):ax.text(b.get_x()+b.get_width()/2,v+1,str(v),ha='center',fontsize=13)
    ax.set(ylim=(0,70),ylabel='w.grad',title='L = (3w − 1)²，w = 2 保持不变');ax.grid(axis='y',alpha=.2);save(fig,'02_gradient_accumulation',d)
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),constrained_layout=True);h=r['training']['history'];best=r['training']['best_epoch']
    for k,label,color in [('train_frozen','固定模型：训练','#2563eb'),('validation','验证','#ea580c')]:
        axes[0].semilogy([x['epoch'] for x in h],[x[k]['loss'] for x in h],color=color,label=label)
        axes[1].plot([x['epoch'] for x in h],[x[k]['accuracy'] for x in h],color=color,label=label)
    for ax in axes:ax.axvline(best,ls='--',color='#64748b',label=f'选中 epoch {best}');ax.set_xlabel('epoch');ax.grid(alpha=.2);ax.legend(fontsize=9)
    axes[0].set(ylabel='按样本平均交叉熵',title='训练/验证：尾批按实际样本数加权');axes[1].set(ylabel='准确率',ylim=(.8,1.015),title='验证选择检查点；测试仅最后评价');save(fig,'03_training_validation',d)
    fig,ax=plt.subplots(figsize=(8.5,4.7));p=r['parity'];keys=list(p['gradient_max_abs']);x=np.arange(len(keys))
    for j,(section,label,color) in enumerate([('gradient_max_abs','全部梯度','#2563eb'),('step_max_abs','一次 SGD 更新','#ea580c')]):
        vals=[max(1e-18,p[section][k]) for k in keys];bars=ax.bar(x+(j-.5)*.33,vals,width=.33,label=label,color=color)
        for b,k in zip(bars,keys):
            if p[section][k]==0:ax.text(b.get_x()+b.get_width()/2,1.25e-18,'0',ha='center',fontsize=10)
    ax.set_yscale('log');ax.set_xticks(x,keys);ax.set(ylabel='与 NumPy 的最大绝对差',title='CPU float64：四块参数逐项审计',ylim=(5e-19,1e-15));ax.legend();ax.grid(axis='y',alpha=.2)
    fig.text(.5,.015,'零误差仅为绘图显示在 1e-18；JSON 保留真实零值。',ha='center',fontsize=10);save(fig,'04_parity_errors',d)
    return sorted(str(p) for p in d.glob('*.png'))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--directory',default='outputs/figures');ap.add_argument('--result');a=ap.parse_args();print('\n'.join(main(a.directory,a.result)))
