"""Original figures for supplied-posterior Bayes decisions, no hidden model training."""
from pathlib import Path
import argparse,io,json,os
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'outputs/mpl'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import experiment as e
NAMES=['01_probability_to_action','02_conditional_risk_lines','03_eight_row_ledger',
       '04_cost_and_accuracy','05_reject_decision','06_multiclass_costs',
       '07_prior_shift','08_exact_and_observed','09_threshold_and_protocol']
C=['#137C91','#D0603A','#7662AB','#5B8773']

def style():
    fonts=sorted([p for p in font_manager.findSystemFonts() if 'NotoSansCJK' in p],key=lambda p:('Regular' not in p,p))
    if fonts:font_manager.fontManager.addfont(fonts[0]);plt.rcParams['font.family']=font_manager.FontProperties(fname=fonts[0]).get_name()
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.15,'axes.unicode_minus':False,'savefig.dpi':160})

def figure(i,r):
    p0,w0,ids0,q00,q10,sids0,c0=e.load_inputs()
    expected={'ids':ids0,'positive_posterior':p0.tolist(),'weights':w0.tolist(),'shift_ids':sids0,'class_likelihood_negative':q00.tolist(),'class_likelihood_positive':q10.tolist()}
    if r.get('unit')!='045' or r.get('config')!=c0 or r.get('data')!=expected:
        raise ValueError('Teaching figure layouts require the unchanged default data/config report; use the numerical API for custom plots')
    style();p=np.array(r['data']['positive_posterior']);w=np.array(r['data']['weights']);cfg=r['config']
    if i==1:
        f,ax=plt.subplots(1,2,figsize=(10.5,3.8),layout='constrained');idx=np.arange(len(p))
        ax[0].bar(idx,p,color=C[0]);ax[0].axhline(.25,color=C[1],ls='--',label='代价阈值1/4');ax[0].axhline(.5,color=C[2],ls=':',label='等错误代价阈值1/2');ax[0].set(xticks=idx,xticklabels=r['data']['ids'],ylabel='已给定 P(Y=1|X)',title='先固定概率，不用行动反改概率');ax[0].legend(fontsize=9)
        data=np.array([r['accuracy_decision']['actions'],r['decision']['actions'],r['rejection']['actions']]);ax[1].imshow(data,vmin=0,vmax=2,cmap=matplotlib.colors.ListedColormap([C[0],C[1],C[2]]),aspect='auto')
        for a,b in np.ndindex(data.shape):ax[1].text(b,a,['判0','判1','拒绝'][data[a,b]],ha='center',va='center',color='white',fontsize=9)
        ax[1].set(xticks=idx,xticklabels=r['data']['ids'],yticks=range(3),yticklabels=['等代价','FP1 FN3','另加拒绝'],title='同一概率，行动随代价改变');ax[1].grid(False)
    elif i==2:
        f,ax=plt.subplots(1,2,figsize=(10.5,3.8),layout='constrained');t=np.linspace(0,1,401)
        for a,fp,fn in [(ax[0],1,1),(ax[1],1,3)]:
            a.plot(t,fn*t,color=C[0],label='R(判0)=FN×p');a.plot(t,fp*(1-t),color=C[1],label='R(判1)=FP×(1-p)');a.plot(t,np.minimum(fn*t,fp*(1-t)),color='#222',ls='--',lw=2,label='最小条件风险');tau=fp/(fp+fn);a.axvline(tau,color=C[2],ls=':');a.set(xlabel='正类概率 p',ylabel='条件风险',title=f'FP={fp}, FN={fn}；交点p={tau:g}');a.legend(fontsize=8)
    elif i==3:
        f,ax=plt.subplots(1,2,figsize=(10.5,4.0),layout='constrained');idx=np.arange(8);risk=np.array(r['decision']['risks']);minimum=np.min(risk,axis=1)
        ax[0].bar(idx-.17,risk[:,0],.34,color=C[0],label='判0风险');ax[0].bar(idx+.17,risk[:,1],.34,color=C[1],label='判1风险');ax[0].plot(idx,minimum,'ko',label='取较小值');ax[0].set(xticks=idx,xticklabels=r['data']['ids'],ylabel='逐组条件风险',title='D3并列：两行动风险都为3/4');ax[0].legend(fontsize=8)
        contributions=r['evaluation']['cost_optimal']['weighted_risk_contribution'];ax[1].bar(idx,contributions,color=C[3]);ax[1].set(xticks=idx,xticklabels=r['data']['ids'],ylabel='人口权重1/8 × 条件风险',title='逐组贡献相加为11/32');ax[1].text(.98,.90,'不是先对概率取平均再决策',transform=ax[1].transAxes,ha='right',fontsize=10)
    elif i==4:
        f,ax=plt.subplots(1,2,figsize=(10.5,3.8),layout='constrained');names=['accuracy_optimal','cost_optimal'];labels=['准确率最优','代价最优'];cost=[r['evaluation'][n]['expected_cost'] for n in names];acc=[1-r['evaluation'][n]['expected_error_rate'] for n in names]
        for a,values,title,yl in [(ax[0],acc,'准确率的排名','总体期望准确率'),(ax[1],cost,'指定代价的排名','FP1 FN3 下的总体期望代价')]:
            a.bar(labels,values,color=C[:2]);a.set(ylabel=yl,title=title,ylim=(0,max(values)*1.22))
            for j,v in enumerate(values):a.text(j,v+.02,f'{v:.5f}',ha='center')
    elif i==5:
        f,ax=plt.subplots(1,2,figsize=(10.5,4.0),layout='constrained');t=np.linspace(0,1,401);risk=np.column_stack([3*t,1-t,np.full_like(t,.375)])
        for k,label in enumerate(['判0','判1','拒绝']):ax[0].plot(t,risk[:,k],color=C[k],label=label)
        ax[0].plot(t,risk.min(axis=1),'k--',lw=2,label='最小风险');ax[0].axvspan(.125,.625,color=C[2],alpha=.1);ax[0].set(ylim=(0,1.1),xlabel='p',ylabel='条件风险（显示0至1.1）',title='拒绝最优的开区间：1/8 < p < 5/8');ax[0].legend(fontsize=9)
        idx=np.arange(8);actions=r['rejection']['actions'];ax[1].scatter(p,actions,c=[C[a] for a in actions],s=90,zorder=3);ax[1].set(yticks=[0,1,2],yticklabels=['判0','判1','拒绝'],xlabel='八组给定p',ylabel='按顺序择优的行动',title='左边界判0，右边界判1');ax[1].text(.04,.60,'拒绝质量=3/8\n包括拒绝费的总代价=1/4',transform=ax[1].transAxes,fontsize=10)
    elif i==6:
        f,ax=plt.subplots(1,2,figsize=(10.5,3.8),layout='constrained');cost=np.array(cfg['multiclass_costs']);ax[0].imshow(cost,cmap='Blues',vmin=0,vmax=4)
        for a,b in np.ndindex(cost.shape):ax[0].text(b,a,str(int(cost[a,b])),ha='center',va='center',color='white' if cost[a,b]>=3 else 'black')
        ax[0].set(xticks=range(3),xticklabels=['真甲','真乙','真丙'],yticks=range(3),yticklabels=['行动甲','行动乙','行动丙'],title='行是行动，列是真实状态');ax[0].grid(False)
        risks=np.array(r['multiclass']['risks']);idx=np.arange(3)
        for k in range(2):ax[1].bar(idx+(k-.5)*.32,risks[k],.32,color=C[k],label=['p=(1/2,1/4,1/4)','p=(1/4,1/4,1/2)'][k])
        ax[1].set(xticks=idx,xticklabels=['行动甲','行动乙','行动丙'],ylabel='条件风险',title='两种分布都选择概率不最大的乙');ax[1].legend(fontsize=9)
    elif i==7:
        f,ax=plt.subplots(1,2,figsize=(10.5,4.0),layout='constrained');shift=r['label_shift'];idx=np.arange(3)
        for j,(key,col) in enumerate([('source',C[0]),('target',C[1])]):ax[0].bar(idx+(j-.5)*.34,shift[key]['positive_posterior'],.34,color=col,label=f"正类先验={shift[key]['prior']}")
        ax[0].axhline(.25,ls=':',color=C[2],label='代价阈值仍为1/4');ax[0].set(xticks=idx,xticklabels=['low','middle','high'],ylabel='P(Y=1|X)',title='同一P(X|Y)，先验改变后验');ax[0].legend(fontsize=8)
        vals=[shift['stale_expected_target_cost']['expected_cost'],shift['corrected_expected_target_cost']['expected_cost']];ax[1].bar(['沿用旧后验决策','先修正后验再决策'],vals,color=C[:2]);ax[1].set(ylabel='目标总体期望代价',title='评价权重也来自目标总体')
        for j,v in enumerate(vals):ax[1].text(j,v+.025,f'{v:.5f}',ha='center')
        ax[1].set_ylim(0,max(vals)*1.2)
    elif i==8:
        f,ax=plt.subplots(1,2,figsize=(10.5,4.0),layout='constrained');names=['accuracy_optimal','cost_optimal','with_reject'];labels=['准确率规则','代价规则','含拒绝规则'];sim=r['simulation']['results'];idx=np.arange(3)
        expected=[sim[n]['expected_cost'] for n in names];observed=[sim[n]['observed_mean_cost'] for n in names];se=[sim[n]['stratified_mean_standard_error'] for n in names]
        ax[0].errorbar(idx,observed,yerr=np.array(se)*2,fmt='o',color=C[1],capsize=5,label='观察均值 ± 2×理论标准误');ax[0].scatter(idx,expected,marker='_',s=400,color=C[0],label='精确总体期望',zorder=4);ax[0].set(xticks=idx,xticklabels=labels,ylabel='代价',title='固定种子4505：2048个模拟标签');ax[0].legend(fontsize=8)
        counts=sim['cost_optimal']['observed_positive_counts'];ax[1].bar(np.arange(8),counts,color=C[0],label='真实模拟计数');ax[1].plot(np.arange(8),p*cfg['samples_per_group'],'o',color=C[1],label='每组期望计数');ax[1].set(xticks=range(8),xticklabels=r['data']['ids'],ylabel='每组256次中的正类数',title='分层固定样本数，与总体随机抽样不同');ax[1].legend(fontsize=8)
    elif i==9:
        f,ax=plt.subplots(1,2,figsize=(10.5,3.8),layout='constrained');s=r['threshold_sweep'];ax[0].step([v['threshold'] for v in s],[v['expected_cost'] for v in s],where='post',color=C[0]);ax[0].plot([v['threshold'] for v in s],[v['expected_cost'] for v in s],'o',color=C[0]);ax[0].axvline(.25,color=C[1],ls=':',label='代价推导τ=1/4');ax[0].set(xlabel='预声明阈值τ；相等判0',ylabel='已知总体的期望代价',title='有限概率格点造成同代价平台');ax[0].legend(fontsize=9)
        ax[1].axis('off');ax[1].set(xlim=(0,1),ylim=(0,1))
        steps=[('训练数据','估计概率模型'),('验证数据 / 预声明代价','选择阈值、拒绝成本与规则'),('锁定模型和规则','保存选定方案'),('独立测试','只报告性能，不反向调阈值')]
        for j,(head,body) in enumerate(steps):
            yy=.88-.24*j;ax[1].text(.5,yy,head,ha='center',fontsize=12,color=C[j%4]);ax[1].text(.5,yy-.075,body,ha='center',fontsize=10)
            if j<3:ax[1].annotate('',xy=(.5,yy-.195),xytext=(.5,yy-.115),arrowprops={'arrowstyle':'->','color':'#666'})
        ax[1].set_title('实际数据流程；本讲不伪装成盲测')
    else:raise ValueError('figure 1..9 required')
    return f


def render_all(report,directory):
    directory=Path(directory).absolute()
    for p in (directory,*directory.parents):
        if p.is_symlink():raise ValueError('symlink figure directory')
    if directory.resolve()==e.ROOT/'figures':raise ValueError('use outputs/figures or a new directory')
    directory.mkdir(parents=True,exist_ok=True)
    for i,name in enumerate(NAMES,1):
        out=directory/(name+'.png')
        if out.is_symlink() or (out.exists() and (not out.is_file() or out.stat().st_nlink>1)):raise ValueError('unsafe output figure')
        fig=figure(i,report);stream=io.BytesIO()
        try:fig.savefig(stream,format='png',dpi=160,bbox_inches='tight')
        finally:plt.close(fig)
        out.write_bytes(stream.getvalue())
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',required=True);p.add_argument('--directory',required=True);a=p.parse_args();render_all(json.loads(Path(a.report).read_text()),a.directory)
