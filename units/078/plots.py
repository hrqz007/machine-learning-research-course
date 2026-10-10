"""五幅原创机制与实测图；PNG和SVG同源生成。"""
from pathlib import Path
import argparse,os
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'outputs/mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from experiment import run
ROOT=Path(__file__).resolve().parent
font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font_manager.fontManager.addfont(font)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'axes.unicode_minus':False,'font.size':11,'svg.fonttype':'path','figure.dpi':120})
COLORS=['#156b99','#d37b17','#258572','#9a4f9a']
def save(fig,out,name):
    fig.savefig(out/(name+'.png'),dpi=160,bbox_inches='tight');fig.savefig(out/(name+'.svg'),bbox_inches='tight');plt.close(fig)
def main(directory):
    out=Path(directory);out.mkdir(parents=True,exist_ok=True);result,chains=run(True)
    fig,ax=plt.subplots(figsize=(10,3));ax.axis('off')
    for i,(title,body) in enumerate([('模型输入','先验 + 8个读数'),('目标后验','密度比例 / 解析对照'),('随机计算','独立样本 / Markov链'),('有限估计','均值 + MCSE + 诊断')]):
        x=.02+i*.25;ax.text(x,.55,title+'\n\n'+body,transform=ax.transAxes,ha='left',va='center',bbox=dict(boxstyle='round,pad=.7',fc='#e8f2f7',ec=COLORS[0]),fontsize=11)
        if i<3:ax.annotate('',xy=(x+.24,.55),xytext=(x+.205,.55),xycoords='axes fraction',arrowprops={'arrowstyle':'->','color':COLORS[1],'lw':2})
    ax.text(.03,.06,'参数后验标准差 ≈ 0.348：数据不确定性；MCSE：有限计算不确定性。',transform=ax.transAxes,color=COLORS[0]);save(fig,out,'01_pipeline')
    x=np.linspace(-2,3,800);m=result['posterior']['mean'];v=result['posterior']['variance']
    fig,ax=plt.subplots(figsize=(9,3.5));ax.plot(x,np.exp(-.5*(x-m)**2/v)/np.sqrt(2*np.pi*v),label='目标后验',color=COLORS[0],lw=2)
    for s,c in [(.2,COLORS[1]),(2,COLORS[2])]:ax.plot(x,np.exp(-.5*(x/s)**2)/(s*np.sqrt(2*np.pi)),label=f'提议 q: SD={s}',color=c)
    ax.set(xlabel='θ（无单位）',ylabel='概率密度',title='支持相同不代表有限样本覆盖充分');ax.legend();ax.grid(alpha=.2);save(fig,out,'02_importance')
    fig,ax=plt.subplots(figsize=(9,3.5));ax.axis('off')
    boxes=[(.05,.65,'当前 x\n保存目标 log 密度'),(.39,.65,'提议 y = x + 随机步长\n计算对数比 Δ'),(.76,.65,'抽 U，比较 log U\n与 min(0, Δ)'),(.41,.15,'接受：记录 y\n拒绝：仍然记录 x')]
    for x0,y0,t in boxes:ax.text(x0,y0,t,transform=ax.transAxes,va='center',bbox=dict(boxstyle='round,pad=.6',fc='#e9f5fb',ec=COLORS[0]))
    for a,b in [((.28,.65),(.37,.65)),((.67,.65),(.74,.65)),((.85,.43),(.65,.18)),((.38,.17),(.13,.42))]:ax.annotate('',xy=b,xytext=a,xycoords='axes fraction',arrowprops=dict(arrowstyle='->',lw=2,color=COLORS[1]))
    save(fig,out,'03_mh')
    fig,axs=plt.subplots(3,1,figsize=(9,7),sharex=True)
    for ax,(name,a) in zip(axs,chains.items()):
        for i,c in enumerate(a):ax.plot(np.arange(len(c))[::8],c[::8],lw=.55,alpha=.8,color=COLORS[i],label=f'链{i+1}')
        ax.axhline(m,color='black',ls='--',lw=1);ax.set(ylabel='θ',title=f'{name}: 步长 {result["mh"][name]["proposal_sd"]}, split-Rhat {result["mh"][name]["split_rhat"]:.3f}');ax.grid(alpha=.2)
    axs[0].legend(ncol=4,fontsize=8);axs[-1].set_xlabel('预热后保存步数（绘图每8步显示一次，计算未抽稀）');fig.tight_layout();save(fig,out,'04_trace')
    from samplers import autocorrelation
    fig,axs=plt.subplots(1,2,figsize=(10,3.7))
    for (name,a),c in zip(chains.items(),COLORS):
        acf=autocorrelation(a[0],100);axs[0].plot(acf,label=name,color=c)
    axs[0].set(xlabel='滞后 k',ylabel='链1自相关',title='仅首100个滞后');axs[0].legend();axs[0].grid(alpha=.2)
    names=list(chains);values=[np.mean(result['mh'][name]['ess_per_chain']) for name in names]
    axs[1].bar(names,values,color=COLORS[:3]);axs[1].set(ylabel='平均每链ESS',title='每链都保留12000步')
    for i,v0 in enumerate(values):axs[1].text(i,v0+40,f'{v0:.0f}',ha='center')
    fig.tight_layout();save(fig,out,'05_diagnostics')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',default=str(ROOT/'outputs/figures'));main(p.parse_args().directory)
