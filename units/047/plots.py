"""Ten original, data-bound calibration and conformal figures."""
from pathlib import Path
import argparse,io,json,os
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'outputs/mpl'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import experiment as e
from numeric import canonical_bytes
NAMES=['01_two_reliability_objects','02_hand_update_chain','03_proper_scores','04_temperature_selection','05_evaluation_reliability','06_uncertainty_components','07_calibration_rank','08_prediction_intervals','09_marginal_and_shift','10_quantile_and_reuse']
C=['#137C91','#D0603A','#7662AB','#5B8773']

def style():
    fonts=sorted([p for p in font_manager.findSystemFonts() if 'NotoSansCJK' in p],key=lambda p:('Regular' not in p,p))
    if fonts:font_manager.fontManager.addfont(fonts[0]);plt.rcParams['font.family']=font_manager.FontProperties(fname=fonts[0]).get_name()
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.16,'axes.unicode_minus':False})

def figure(i,r):
    classification,regression,c=e.load_inputs();expected={'classification':classification,'regression':regression}
    expected={kind:{role:{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in row.items()} for role,row in splits.items()} for kind,splits in expected.items()}
    if r.get('unit')!='047' or r.get('protocol')!=c or r.get('data')!=expected:raise ValueError('Original teaching figures require the unchanged default data and protocol')
    style();f,ax=plt.subplots(1,2,figsize=(10.6,4),layout='constrained');h=r['hand'];con=r['conformal'];main=con['main']
    if i==1:
        for side,key,title in [(0,'positive_class_reliability','正类可靠性：p 对 Y'),(1,'top_label_reliability','最高置信度：max(p,1-p) 对正确性')]:
            a=ax[side];a.plot([0,1],[0,1],'k:',lw=1,label='理想对角线')
            for j,(name,label) in enumerate([('initial_metrics','T=1'),('optimum_metrics','T=2')]):
                bins=h[name][key]['bins'];xx=[b['mean_score'] for b in bins if b['count']];yy=[b['empirical_outcome_rate'] for b in bins if b['count']]
                a.scatter(xx,yy,s=100,marker=['o','s'][j],color=C[j],label=f"{label}；ECE={h[name][key]['ece']:.2f}",zorder=4)
            a.set(xlim=(-.04,1.04),ylim=(-.05,1.08),xlabel='分箱平均预测值',ylabel='分箱实际比例',title=title);a.legend(loc='lower right',fontsize=8)
    elif i==2:
        states=h['states'];xx=np.arange(4)
        for j,s in enumerate(states):ax[0].bar(xx+(j-1)*.23,s['probabilities'],.23,color=C[j],label=f"第{j}次前向 a={s['inverse_temperature']:.3f}")
        ax[0].scatter(xx,h['labels'],marker='x',c='black',s=45,label='真实y',zorder=4);ax[0].set(xticks=xx,xticklabels=['1','2','3','4'],xlabel='四个样本',ylabel='正类概率',ylim=(-.02,1.38),title='两次同步更新后重新前向');ax[0].legend(fontsize=8,loc='upper center',ncol=2)
        for j,s in enumerate(states):ax[1].bar(xx+(j-1)*.23,[v['gradient_contribution'] for v in s['rows']],.23,color=C[j],label=f"阶段{j}：总梯度={s['gradient']:.3f}")
        ax[1].axhline(0,c='black',lw=.7);ax[1].set(xticks=xx,xticklabels=['1','2','3','4'],xlabel='样本（贡献已除以4）',ylabel='平均损失的梯度贡献',ylim=(-.14,.75),title='错误且自信的第2行主导梯度');ax[1].legend(fontsize=8,loc='upper right')
    elif i==3:
        p=np.linspace(.005,.995,500);eta=.75
        ax[0].plot(p,eta*(1-p)**2+(1-eta)*p**2,color=C[0],label='期望二元 Brier');ax[0].axvline(eta,color=C[1],ls=':',label='真实概率 η=0.75');ax[0].set(xlabel='报告的概率 p',ylabel='期望损失',title='(p−η)² + η(1−η)');ax[0].legend(fontsize=9)
        ax[1].plot(p,-eta*np.log(p)-(1-eta)*np.log1p(-p),color=C[2]);ax[1].axvline(eta,color=C[1],ls=':');ax[1].set(xlabel='报告的概率 p',ylabel='期望对数损失',title='对错误且极端自信的预测惩罚大')
    elif i==4:
        t=r['temperature'];profile=t['calibration_profile'];chosen=t['frozen_choice']['fit']['final'];ax[0].plot([v['a'] for v in profile],[v['calibration_log_loss'] for v in profile],color=C[0]);ax[0].axvline(chosen['inverse_temperature'],color=C[1],ls=':',label=f"a={chosen['inverse_temperature']:.3f}, T={chosen['temperature']:.3f}");ax[0].set(xlabel='逆温度 a=1/T',ylabel='400个校准样本的平均 NLL',title='只用校准集选择参数');ax[0].legend(fontsize=9,loc='upper left')
        ax[1].axis('off');steps=[('固定合成分数 z=2(1.2x−0.3)','不是训练神经网络'),('独立校准400行','在[0,4]上最小化 NLL'),('锁定a、分箱和选择记录','保存选择摘要 SHA-256'),('独立评价4000行','不给拟合器任何评价标签')]
        for j,(title,body) in enumerate(steps):
            yy=.92-.25*j;ax[1].text(.5,yy,title,ha='center',fontsize=11,color=C[j%4]);ax[1].text(.5,yy-.07,body,ha='center',fontsize=9)
            if j<3:ax[1].annotate('',xy=(.5,yy-.20),xytext=(.5,yy-.11),arrowprops={'arrowstyle':'->','color':'#777'})
        ax[1].set(xlim=(0,1),ylim=(0,1))
    elif i==5:
        m=r['temperature']['metrics']['evaluation'];xx=np.arange(5)
        for j,(key,label) in enumerate([('before','校准前'),('after','校准后')]):
            b=m[key]['positive_class_reliability']['bins'];ax[0].plot([v['mean_score'] for v in b],[v['empirical_outcome_rate'] for v in b],'o-',c=C[j],label=f"{label} ECE={m[key]['positive_class_reliability']['ece']:.4f}");ax[1].bar(xx+(j-.5)*.35,[v['count'] for v in b],.35,color=C[j],label=label)
        ax[0].plot([0,1],[0,1],'k:',lw=1);ax[0].set(xlim=(0,1),ylim=(0,1),xlabel='分箱平均正类概率',ylabel='正类比例',title='独立评价集上的正类可靠性');ax[0].legend(fontsize=8,loc='lower right');ax[1].set(xticks=xx,xticklabels=['[0,.2)','[.2,.4)','[.4,.6)','[.6,.8)','[.8,1]'],xlabel='固定等宽分箱',ylabel='样本数',title='可靠性图必须同时看样本量');ax[1].tick_params(axis='x',labelsize=9);ax[1].legend(fontsize=9)
    elif i==6:
        u=con['uncertainty_components'];x=np.array(u['x']);noise=np.array(u['known_conditional_noise_variance']);var=np.array(u['repeated_training_prediction_variance_ddof1']);ax[0].plot(x,noise,color=C[0],label='已知 Var(Y|X=x)');ax[0].plot(x,var,color=C[1],label='120次训练预测的样本方差');ax[0].set(xlabel='x',ylabel='方差（同一量纲）',title='噪声与估计变动是不同对象');ax[0].legend(fontsize=8,loc='upper center')
        ax[1].plot(x,var,color=C[1]);ax[1].set(xlabel='x',ylabel='训练预测样本方差（放大纵轴）',title='仅此线性估计器的抽样变动');ax[1].text(.5,.96,'不是所有模型不确定性的总和',ha='center',va='top',transform=ax[1].transAxes,fontsize=9,bbox={'facecolor':'white','alpha':1,'edgecolor':'none'})
    elif i==7:
        scores=np.sort(main['calibration_scores']);k=main['quantile']['rank'];q=main['quantile']['quantile'];ax[0].plot(np.arange(1,len(scores)+1),scores,'o',ms=3,c=C[0]);ax[0].axvline(k,c=C[1],ls=':');ax[0].axhline(q,c=C[1],ls=':',label=f'第{k}个；q={q:.4f}');ax[0].set(xlabel='从小到大的秩（从1开始）',ylabel='绝对残差',title='99个独立校准残差');ax[0].legend(fontsize=9,loc='upper left')
        ranks=np.arange(1,101);ax[1].bar(ranks,np.ones(100),color=[C[0] if j<=90 else C[1] for j in ranks],width=1);ax[1].set(xlim=(.5,100.5),ylim=(0,1.5),yticks=[],xlabel='新分数在100个可交换分数中的秩',title='无并列时，90/100个秩被覆盖');ax[1].text(45,1.18,'覆盖：秩1至90',ha='center',fontsize=10);ax[1].text(95,1.18,'漏掉',ha='center',fontsize=9)
    elif i==8:
        d=r['data']['regression']['evaluation'];count=c['interval_display_count'];order=np.argsort(d['x'][:count]);x=np.array(d['x'][:count])[order];pred=np.array(main['test_prediction'][:count])[order];q=main['quantile']['quantile']
        for a,key,title in [(ax[0],'y','原机制：预先取前24个测试样本'),(ax[1],'shifted_y','相同x和噪声抽样，噪声幅度×2.5')]:
            y=np.array(d[key][:count])[order];covered=abs(y-pred)<=q;a.vlines(x,pred-q,pred+q,color='#a1c7ce',lw=3,label='同一冻结预测区间');a.plot(x,pred,'_',c=C[0],ms=9);a.scatter(x,y,c=[C[0] if v else C[1] for v in covered],s=25,zorder=3,label='实点：红色表示漏覆盖');a.set(xlabel='x（先取前24行，再按x排序）',ylabel='新的响应Y',title=title);a.legend(fontsize=8,loc='upper left')
    elif i==9:
        s=con['summary'];keys=['coverage','low_abs_x','high_abs_x','shifted_coverage'];labels=['原机制总体','|x|≤1','|x|>1','噪声改变总体'];means=[s[k]['mean'] for k in keys];errs=[2*s[k]['mc_standard_error'] for k in keys]
        ax[0].errorbar(np.arange(4),means,yerr=errs,fmt='o',color=C[0],capsize=5);ax[0].axhline(.9,c=C[1],ls=':',label='目标边际覆盖0.9');ax[0].set(xticks=range(4),xticklabels=labels,ylim=(.5,1.01),ylabel='120次独立重复的覆盖率均值',title='误差棒：±2个重复层面的 MCSE');ax[0].tick_params(axis='x',labelsize=9);ax[0].legend(fontsize=8,loc='lower left')
        for key,label,col in [('coverage','原机制',C[0]),('shifted_coverage','噪声改变',C[1])]:ax[1].hist([v[key] for v in con['repetitions']],bins=np.linspace(.35,1,27),alpha=.65,color=col,label=label)
        ax[1].axvline(.9,c='black',ls=':',lw=1);ax[1].set(xlabel='每次400测试点的经验覆盖率',ylabel='独立重复次数',title='不要求每次实验恰好覆盖90%');ax[1].legend(fontsize=9,loc='upper left')
    elif i==10:
        from conformal import finite_sample_quantile
        from sklearn.neighbors import KNeighborsRegressor
        correct=[];naive=[]
        for held in range(10):
            cal=np.delete(np.arange(10.),held);correct.append(held<=finite_sample_quantile(cal,.1)['quantile']);naive.append(held<=np.quantile(cal,.9,method='linear'))
        ax[0].imshow(np.array([correct,naive]),cmap=matplotlib.colors.ListedColormap([C[1],C[0]]),vmin=0,vmax=1,aspect='auto')
        for j in range(10):
            for row,values in enumerate([correct,naive]):ax[0].text(j,row,'✓' if values[j] else '×',ha='center',va='center',color='white')
        ax[0].set(xticks=range(10),xlabel='轮流作为新分数的值',yticks=[0,1],yticklabels=['修正秩：9/10','普通插值：8/10'],title='固定10分数，均匀抽一个留出');ax[0].grid(False)
        cal=r['data']['regression']['calibration'];test=r['data']['regression']['evaluation'];nn=KNeighborsRegressor(1).fit(np.array(cal['x'])[:,None],cal['y']);indices=np.argsort(test['x'][:24]);xx=np.array(test['x'][:24])[indices];yp=nn.predict(xx[:,None]);yy=np.array(test['y'][:24])[indices];ax[1].plot(xx,yp,'s',c=C[2],label='错误复用的零宽度区间');ax[1].plot(xx,yy,'x',c=C[1],label='新的真实Y');ax[1].set(xlabel='同样预先选取的24个测试x',ylabel='响应Y',title='在校准标签上训练1-NN：q=0');ax[1].legend(fontsize=8,loc='upper left')
    else:raise ValueError('figure number must be 1..10')
    return f

def render_all(report,directory):
    directory=Path(directory).absolute()
    for part in (directory,*directory.parents):
        if part.is_symlink():raise ValueError('symlink figure directory')
    if directory.resolve()==e.ROOT/'figures':raise ValueError('use outputs/figures or a new directory')
    directory.mkdir(parents=True,exist_ok=True)
    for i,name in enumerate(NAMES,1):
        p=directory/(name+'.png')
        if p.is_symlink() or (p.exists() and (not p.is_file() or p.stat().st_nlink>1)):raise ValueError('unsafe figure output')
        fig=figure(i,report);stream=io.BytesIO()
        try:fig.savefig(stream,format='png',dpi=160,bbox_inches='tight')
        finally:plt.close(fig)
        p.write_bytes(stream.getvalue())
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',required=True);p.add_argument('--directory',required=True);a=p.parse_args();render_all(json.loads(Path(a.report).read_text()),a.directory)
