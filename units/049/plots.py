"""Ten original figures for exact finite-class learning-theory examples."""
from pathlib import Path
import argparse,io,json,os
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'outputs/mpl'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.stats import binom
import experiment as e
NAMES=['01_population_and_rules','02_complete_erm_ledger','03_exact_tails_and_bounds','04_gap_and_excess_curves','05_selection_failure','06_minimum_error_distribution','07_sample_complexity','08_independence_counterexample','09_independent_holdout','10_vc_shattering']
C=['#137C91','#D0603A','#7662AB','#5B8773']
def style():
    fonts=sorted([p for p in font_manager.findSystemFonts() if 'NotoSansCJK' in p],key=lambda p:('Regular' not in p,p))
    if fonts:font_manager.fontManager.addfont(fonts[0]);plt.rcParams['font.family']=font_manager.FontProperties(fname=fonts[0]).get_name()
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.15,'axes.unicode_minus':False})

def figure(i,r):
    pop,hyp,sample,c=e.load_inputs()
    if r.get('unit')!='049' or r.get('protocol')!=c or r.get('data')!={'population':pop,'hypotheses':hyp,'hand_sample':sample}:raise ValueError('teaching figures require unchanged default data and protocol')
    style();f,ax=plt.subplots(1,2,figsize=(10.8,4.2),layout='constrained');ids=[v['id'] for v in hyp]
    if i==1:
        H=np.array([v['predictions'] for v in hyp]);ax[0].imshow(H,cmap=matplotlib.colors.ListedColormap(['#edf3f6',C[0]]),vmin=0,vmax=1,aspect='auto')
        for a,b in np.ndindex(H.shape):ax[0].text(b,a,str(H[a,b]),ha='center',va='center',color='white' if H[a,b] else '#222')
        ax[0].set(xticks=range(4),xticklabels=['−2','−1','1','2'],yticks=range(5),yticklabels=ids,xlabel='四个可能的输入x',title='数据出现前就固定的五个规则');ax[0].grid(False)
        risk=r['population']['risks'];ax[1].bar(ids,risk,color=[C[0] if j!=2 else C[1] for j in range(5)]);ax[1].set(ylim=(0,.65),ylabel='对八种(x,y)精确求和的风险',title='真实风险，不是大型测试集估计')
        for j,v in enumerate(risk):ax[1].text(j,v+.015,f'{v:.5f}',ha='center',fontsize=9)
    elif i==2:
        h=r['hand'];loss=np.array([v['losses'] for v in h['initial']['rows']]);ax[0].imshow(loss,cmap=matplotlib.colors.ListedColormap(['#edf3f6',C[1]]),vmin=0,vmax=1,aspect='auto')
        for a,b in np.ndindex(loss.shape):ax[0].text(b,a,str(loss[a,b]),ha='center',va='center',color='white' if loss[a,b] else '#222')
        ax[0].set(xticks=range(5),xticklabels=ids,yticks=range(6),yticklabels=[f"{v['id']}: x={v['x']}, y={v['y']}" for v in h['initial']['rows']],title='六行逐样本0-1损失');ax[0].grid(False)
        for j,(values,label) in enumerate([(h['initial']['empirical_risks'],'初始6行经验风险'),(h['after_batch']['empirical_risks'],'追加整批后8行风险'),(r['population']['risks'],'真实总体风险')]):ax[1].bar(np.arange(5)+(j-1)*.24,values,.24,color=C[j],label=label)
        ax[1].set(xticks=range(5),xticklabels=ids,ylim=(0,.78),ylabel='平均0-1损失',title='并列先选H1；加入整批后选H2');ax[1].legend(fontsize=8,loc='upper center')
    elif i==3:
        exact=next(v for v in r['exact'] if v['n']==8);xx=[float(__import__('fractions').Fraction(v['epsilon'])) for v in exact['tails']]
        for a,key,bkey,title in [(ax[0],'fixed_tail_probability','fixed_Hoeffding','事前固定最优规则H2'),(ax[1],'uniform_tail_probability','uniform_Hoeffding_union','五个规则同时控制')]:
            a.semilogy(xx,[v[key] for v in exact['tails']],'o-',c=C[0],label='完整6435状态精确概率');a.semilogy(xx,[v[bkey]['probability_bound'] for v in exact['tails']],'s--',c=C[1],label='理论上界截断到1');a.set(xticks=xx,xlabel='绝对偏差阈值 ε；事件为 > ε',ylabel='尾概率（对数轴）',title=title,ylim=(.001,1.3));a.legend(fontsize=8,loc='lower left')
    elif i==4:
        blocks=r['main_repetitions'];n=[v['n'] for v in blocks];q=[v['summary']['uniform_absolute_gap'] for v in blocks]
        ax[0].fill_between(n,[v['p05'] for v in q],[v['p95'] for v in q],alpha=.18,color=C[0],label='实测5%至95%分位带');ax[0].plot(n,[v['mean'] for v in q],'o-',c=C[0],label='实测最大绝对偏差均值');ax[0].plot(n,[v['uniform_radius'] for v in blocks],'s--',c=C[1],label='同时界 ε(n,M,δ)');ax[0].set(xscale='log',xlabel='IID样本数n（对数轴）',ylabel='max_h |经验风险−真实风险|',title='每个n重复2000次，δ=0.05');ax[0].legend(fontsize=8)
        ax[1].plot(n,[v['summary']['excess_risk']['mean'] for v in blocks],'o-',c=C[0],label='真实ERM超额风险均值');ax[1].plot(n,[v['ERM_excess_bound_raw'] for v in blocks],'s--',c=C[1],label='原始2ε上界（可能大于1）');ax[1].plot(n,[v['ERM_excess_bound_using_risk_range'] for v in blocks],':',c=C[2],label='仅用风险范围截断');ax[1].set(xscale='log',xlabel='n（对数轴）',ylabel='R(选中规则)−min R',title='上界不是预言实际误差');ax[1].legend(fontsize=8)
    elif i==5:
        rows=r['selection_noise'][0]['scenarios'];M=[v['M'] for v in rows]
        ax[0].plot(M,[v['analytic']['expected_selected_training_risk'] for v in rows],'-',c=C[0],label='解析期望训练风险');ax[0].scatter(M,[v['mean_training_risk'] for v in rows],c=C[0],marker='o',label='2000次实测均值');ax[0].axhline(.5,c=C[1],ls='--',label='所有规则真实风险都为1/2');ax[0].set(xscale='log',xlabel='事前固定候选数M（对数轴）',ylabel='风险',ylim=(.1,.64),title='n=20：筛选更多纯噪声规则');ax[0].legend(fontsize=8,loc='lower left')
        ax[1].plot(M,[v['analytic']['selected_fixed_radius_violation_probability'] for v in rows],'o-',c=C[1],label='误套单模型界：解析失败率');ax[1].scatter(M,[v['fixed_radius_violations']['frequency'] for v in rows],c='black',marker='x',s=45,label='误套界：实测失败率');ax[1].plot(M,[v['analytic']['selected_uniform_radius_violation_probability'] for v in rows],'s-',c=C[0],label='正确同时界：解析失败率');ax[1].axhline(.05,c=C[2],ls=':',label='声称的δ=0.05');ax[1].set(xscale='log',xlabel='候选数M（对数轴）',ylabel='|R−经验R| 超过相应界的概率',ylim=(-.01,.65),title='筛选后不能把M改写成1');ax[1].legend(fontsize=8,loc='upper left')
    elif i==6:
        rows=r['selection_noise'][0]['scenarios'];n=20;k=np.arange(21)
        for j,v in enumerate(rows):ax[0].plot(k/n,v['analytic']['minimum_pmf'],'o-',ms=3,c=C[j],label=f"M={v['M']}")
        ax[0].set(xlabel='选中规则的训练错误率 min K/n',ylabel='解析概率质量',title='每个K_j独立服从Binomial(20,1/2)');ax[0].legend(fontsize=9,loc='upper right')
        v=rows[-1];ax[1].bar(k/n,np.array(v['selected_error_count_histogram'])/2000,width=.03,color=C[0],label='实际2000次频率');ax[1].plot(k/n,v['analytic']['minimum_pmf'],'o',c=C[1],ms=4,label='解析概率质量');ax[1].set(xlim=(-.025,.45),xlabel='选中训练错误率（显示0至0.45）',ylabel='频率 / 概率质量',title='M=512：没有提高真实预测能力');ax[1].legend(fontsize=8)
    elif i==7:
        eps=np.array([.05,.1,.2]);table=r['sample_complexity_table']
        for j,M in enumerate([1,5,512]):
            rows=[v for v in table if v['M']==M];ax[0].plot(eps,[v['agnostic']['sufficient_sample_size'] for v in rows],'o-',c=C[j],label=f'M={M}');ax[1].plot(eps,[v['realizable']['sufficient_sample_size'] for v in rows],'o-',c=C[j],label=f'M={M}')
        ax[0].set(yscale='log',xlabel='目标超额风险 ε',ylabel='充分样本数（对数轴）',title='一般有噪声：2 log(2M/δ) / ε²');ax[1].set(yscale='log',xlabel='目标真实错误率 ε',ylabel='充分样本数（对数轴）',title='可实现且一致：log(M/δ) / ε');ax[0].legend(fontsize=9);ax[1].legend(fontsize=9);f.suptitle('δ=0.05；右图前提不适用于本讲有噪声主总体',fontsize=11)
    elif i==8:
        n=20;k=np.arange(n+1);ax[0].bar(k/n,binom.pmf(k,n,.5),width=.035,color=C[0]);ax[0].set(xlabel='经验错误率',ylabel='精确概率质量',title='20个独立损失：均值会集中',xlim=(-.05,1.05),ylim=(0,.55))
        ax[1].bar([0,1],[.5,.5],width=.08,color=C[1]);ax[1].set(xlabel='经验错误率',ylabel='精确概率质量',title='一个损失复制20次：不增加信息',xlim=(-.05,1.05),ylim=(0,.55))
        for a in ax:a.axvline(.5,c='black',ls=':',label='相同边缘真实风险1/2');a.legend(fontsize=8,loc='upper center')
    elif i==9:
        blocks=r['selection_noise'];labels=[str(v['n']) for v in blocks];xx=np.arange(3)
        vals=[[b['scenarios'][-1]['fixed_radius_violations']['frequency'] for b in blocks],[b['independent_holdout']['violations']['frequency'] for b in blocks]]
        for j,label in enumerate(['复用筛选数据并误套单模型界','筛选后独立100行留出评价']):ax[0].bar(xx+(j-.5)*.32,vals[j],.32,color=C[j],label=label)
        ax[0].axhline(.05,color=C[2],ls=':');ax[0].set(xticks=xx,xticklabels=labels,xlabel='选模型时的训练样本数n',ylabel='实际2000次违约频率',ylim=(0,1.05),title='各自使用对应样本数的单模型界');ax[0].legend(fontsize=8,loc='upper left')
        for j in range(2):
            for k,v in enumerate(vals[j]):ax[0].text(k+(j-.5)*.32,v+.015,f'{v:.4f}',ha='center',fontsize=9)
        ax[1].axis('off');steps=[('训练数据','在512个固定候选中选择索引'),('锁定选择','保存训练选择记录摘要'),('独立留出100行','条件于已选索引再评价'),('不得反馈调选择','否则留出集又成了选择集')]
        for j,(head,body) in enumerate(steps):
            yy=.9-.24*j;ax[1].text(.5,yy,head,ha='center',color=C[j],fontsize=12);ax[1].text(.5,yy-.07,body,ha='center',fontsize=10)
            if j<3:ax[1].annotate('',xy=(.5,yy-.20),xytext=(.5,yy-.11),arrowprops={'arrowstyle':'->','color':'#777'})
    elif i==10:
        import itertools
        for a,m,key,title in [(ax[0],2,'threshold_patterns','固定方向阈值：两点缺少10'),(ax[1],3,'interval_patterns','区间：三点缺少101')]:
            row=r['vc']['ordered_distinct_point_patterns'][m-1];possible={tuple(v) for v in row[key]};allp=list(itertools.product([0,1],repeat=m));mat=np.array(allp);a.imshow(mat,cmap=matplotlib.colors.ListedColormap(['#edf3f6',C[0]]),vmin=0,vmax=1,aspect='auto')
            for rr,cc in np.ndindex(mat.shape):a.text(cc,rr,str(mat[rr,cc]),ha='center',va='center',color='white' if mat[rr,cc] else '#222')
            labels=[''.join(map(str,v))+(' 可实现' if v in possible else ' 缺少') for v in allp];a.set(xticks=range(m),xticklabels=[f'x{j+1}' for j in range(m)],yticks=range(len(allp)),yticklabels=labels,xlabel='按从小到大排列的不同实数点',title=title);a.grid(False)
    else:raise ValueError('figure index1..10')
    return f

def render_all(report,directory):
    directory=Path(directory).absolute()
    for part in (directory,*directory.parents):
        if part.is_symlink():raise ValueError('symlink output')
    if directory.resolve()==e.ROOT/'figures':raise ValueError('use a new output directory')
    directory.mkdir(parents=True,exist_ok=True)
    for i,name in enumerate(NAMES,1):
        out=directory/(name+'.png')
        if out.is_symlink() or (out.exists() and (not out.is_file() or out.stat().st_nlink>1)):raise ValueError('unsafe figure output')
        fig=figure(i,report);stream=io.BytesIO()
        try:fig.savefig(stream,format='png',dpi=160,bbox_inches='tight')
        finally:plt.close(fig)
        out.write_bytes(stream.getvalue())
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',required=True);p.add_argument('--directory',required=True);a=p.parse_args();render_all(json.loads(Path(a.report).read_text()),a.directory)
