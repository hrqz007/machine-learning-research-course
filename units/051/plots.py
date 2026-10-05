"""Ten original data-bound figures for nested model selection."""
from pathlib import Path
import argparse,io,json,os,math
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'outputs/mpl'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import experiment as e
NAMES=['01_nested_data_roles','02_hand_inner_selection','03_hand_gradient_chain','04_inner_candidate_scores','05_outer_scores_and_oracles','06_repeated_optimism','07_reversals_and_choices','08_splitter_indices','09_overlapping_folds','10_final_sealed_test']
C=['#137C91','#D0603A','#7662AB','#5B8773']
def style():
    fonts=sorted([p for p in font_manager.findSystemFonts() if 'NotoSansCJK' in p],key=lambda p:('Regular' not in p,p))
    if fonts:font_manager.fontManager.addfont(fonts[0]);plt.rcParams['font.family']=font_manager.FontProperties(fname=fonts[0]).get_name()
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.15,'axes.unicode_minus':False})
def figure(i,r):
    main,rep,c=e.load_inputs();expected={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in main.items()}
    expected_repeated=[{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in row.items()} for row in rep]
    if r.get('unit')!='051' or r.get('protocol')!=c or r['data']['development']!=expected or r['data']['repeated_development']!=expected_repeated:raise ValueError('original figures require unchanged default development data and protocol')
    style();p=r['primary'];h=r['hand'];f,ax=plt.subplots(1,2,figsize=(10.8,4.2),layout='constrained')
    if i==1:
        first=p['nested']['folds'][0];mat=np.zeros((5,80));mat[0,first['training_indices']]=1;mat[0,first['validation_indices']]=3
        for j,row in enumerate(first['choice']['selection']['folds'],1):mat[j,row['training_indices_global']]=1;mat[j,row['validation_indices_global']]=2;mat[j,first['validation_indices']]=3
        ax[0].imshow(mat,aspect='auto',vmin=0,vmax=3,cmap=matplotlib.colors.ListedColormap(['white',C[0],C[1],C[2]]));ax[0].set(yticks=range(5),yticklabels=['外层第1折','内层1','内层2','内层3','内层4'],xlabel='原始80行的全局索引',title='蓝：拟合　橙：内验　紫：外测');ax[0].grid(False)
        ax[1].axis('off');steps=[('留出外层16行','该轮选择不读取其标签'),('剩余64行内部4折','每次48拟合、16选参'),('选中候选后重拟合64行','保存候选与模型摘要'),('只在外层16行评价','每行外测一次，再合并误差')]
        for j,(title,body) in enumerate(steps):
            yy=.92-.25*j;ax[1].text(.5,yy,title,ha='center',fontsize=12,color=C[j%4]);ax[1].text(.5,yy-.075,body,ha='center',fontsize=10)
            if j<3:ax[1].annotate('',xy=(.5,yy-.20),xytext=(.5,yy-.11),arrowprops={'arrowstyle':'->','color':'#777'})
    elif i==2:
        for j,row in enumerate(h['candidates']):
            ax[0].bar(np.arange(2)+(j-.5)*.32,[v['validation_MSE'] for v in row['folds']],.32,color=C[j],label=f"λ={row['penalty']:g}")
        ax[0].set(xticks=[0,1],xticklabels=['验证端点行0,3','验证中间行1,2'],ylabel='该内层验证折MSE',ylim=(0,2.6),title='同一候选要接受两折评价');ax[0].legend(fontsize=9,loc='upper right')
        vals=[v['pooled_CV_MSE'] for v in h['candidates']];ax[1].bar(['λ=0','λ=1'],vals,color=C[:2]);ax[1].set(ylim=(0,1.35),ylabel='四个验证预测的合并MSE',title='选择λ=0，然后用全部四行重拟合')
        for j,v in enumerate(vals):ax[1].text(j,v+.025,f'{v:.2f}',ha='center')
    elif i==3:
        x=np.arange(4)
        for j,s in enumerate(h['gradient_trace']):
            ax[0].bar(x+(j-1)*.23,[v['mean_gradient_contribution'] for v in s['rows']],.23,color=C[j],label=f"w={s['weight']:.4f}")
        ax[0].set(xticks=x,xticklabels=['x=-2','x=-1','x=1','x=2'],ylabel='逐行平均数据梯度贡献',title='固定λ=1的另一个优化演示',ylim=(-1.65,.3));ax[0].axhline(0,c='black',lw=.8);ax[0].legend(fontsize=8,loc='lower left')
        states=h['gradient_trace'];ax[1].plot(range(3),[s['total_objective'] for s in states],'o-',c=C[0],label='数据半MSE + λw²/2');ax[1].plot(range(3),[s['penalty_value'] for s in states],'s--',c=C[1],label='正则项');ax[1].set(xticks=range(3),xlabel='第几次前向（两次更新）',ylabel='目标值',title='0 → 0.25 → 0.4125');ax[1].legend(fontsize=9,loc='upper right')
    elif i==4:
        mat=np.array([row['choice']['selection']['candidate_scores'] for row in p['nested']['folds']]);im=ax[0].imshow(mat,aspect='auto',cmap='viridis',norm=matplotlib.colors.LogNorm(vmin=float(mat.min()),vmax=float(mat.max())));f.colorbar(im,ax=ax[0],label='内层MSE（对数色标）',shrink=.8)
        for j,row in enumerate(p['nested']['folds']):ax[0].plot(row['choice']['selection']['selected_index'],j,'r*',ms=11)
        labels=[f"{v['degree']}/{v['penalty']:g}" for v in c['candidates']];ax[0].set(xticks=range(len(labels)),xticklabels=labels,yticks=range(5),yticklabels=[f'外折{k+1}' for k in range(5)],xlabel='候选 degree / λ',title='每个外层训练集重新选参；星号为选择');ax[0].tick_params(axis='x',labelrotation=90,labelsize=8);ax[0].grid(False)
        scores=p['flat']['selection']['candidate_scores'];ax[1].plot(range(len(scores)),scores,'o-',c=C[0]);j=p['flat']['selection']['selected_index'];ax[1].plot(j,scores[j],'r*',ms=12);ax[1].set(xticks=range(len(labels)),xticklabels=labels,xlabel='相同17候选',ylabel='复用同一5折的CV分数',title='非嵌套选择：取这条曲线最低点');ax[1].tick_params(axis='x',labelrotation=90,labelsize=8)
    elif i==5:
        xx=np.arange(5);rows=p['nested']['folds'];actual=[v['validation_MSE'] for v in rows];truth=[v['oracle_risk_same_fitted_model']['total_MSE'] for v in rows]
        ax[0].bar(xx-.17,actual,.34,color=C[0],label='真实外层测试MSE');ax[0].bar(xx+.17,truth,.34,color=C[1],label='同一已拟合模型的总体MSE');ax[0].set(xticks=xx,xticklabels=range(1,6),xlabel='外层折',ylabel='MSE',ylim=(0,max(actual+truth)*1.32),title='不拿最终80行模型冒充外层模型');ax[0].legend(fontsize=8)
        names=['非嵌套复用分数','其同折模型总体','嵌套外层分数','其同折模型总体'];vals=[p['flat']['minimum_reused_CV_MSE'],p['flat']['mean_oracle_risk_same_selected_fold_models'],p['nested']['pooled_outer_MSE'],p['nested']['mean_oracle_risk_same_outer_models']];ax[1].bar(range(4),vals,color=[C[0],C[1],C[0],C[1]]);ax[1].set(xticks=range(4),xticklabels=names,ylim=(0,.9),ylabel='主数据集的MSE',title='各自分数与各自评价对象配对');ax[1].tick_params(axis='x',labelrotation=25,labelsize=9)
        for j,v in enumerate(vals):ax[1].text(j,v+.018,f'{v:.4f}',ha='center',fontsize=9)
    elif i==6:
        s=r['repeated_summary'];keys=['flat_optimism','nested_optimism'];vals=[s[k]['mean'] for k in keys];se=[2*s[k]['MCSE_mean'] for k in keys];ax[0].errorbar([0,1],vals,yerr=se,fmt='o',capsize=5,c=C[0]);ax[0].axhline(0,c='black',lw=.8);ax[0].set(xticks=[0,1],xticklabels=['非嵌套','嵌套'],ylabel='同模型总体MSE − 报告CV分数',title='80个独立数据集；误差棒±2 MCSE')
        for j,(key,col) in enumerate([('flat',C[0]),('nested',C[1])]):ax[1].plot(range(80),[v['result'][key]['oracle_minus_CV'] for v in r['repeated']],'.',c=col,label=['非嵌套','嵌套'][j])
        ax[1].axhline(0,c='black',lw=.8);ax[1].set(xlabel='独立数据集编号（未排序筛选）',ylabel='带符号乐观差',ylim=(-.35,.44),title='单次差可以为负，全部保留');ax[1].legend(fontsize=9)
    elif i==7:
        diff=np.array([v['result']['flat']['minimum_reused_CV_MSE']-v['result']['nested']['pooled_outer_MSE'] for v in r['repeated']]);ax[0].hist(diff,bins=18,color=C[0],edgecolor='white');ax[0].axvline(0,c=C[1],ls='--');ax[0].set(xlabel='非嵌套CV − 嵌套CV',ylabel='独立数据集数',title=f'80次中{int((diff>0).sum())}次出现相反排序')
        counts=np.bincount([v['result']['final_choice']['selected_index'] for v in r['repeated']],minlength=17);labels=[f"{v['degree']}/{v['penalty']:g}" for v in c['candidates']];ax[1].bar(range(17),counts,color=C[2]);ax[1].set(xticks=range(17),xticklabels=labels,xlabel='全数据CV选择的 degree / λ',ylabel='被选中的数据集数',title='调参结果本身也随数据改变');ax[1].tick_params(axis='x',labelrotation=90,labelsize=8)
    elif i==8:
        plt.close(f);f,axs=plt.subplots(2,2,figsize=(10.8,6.2),layout='constrained');demo=r['splitter_demo']
        for a,name in zip(axs.flat,['KFold','StratifiedKFold','GroupKFold','TimeSeriesSplit']):
            rows=demo['splitters'][name]
            mat=np.zeros((len(rows),12))
            for k,row in enumerate(rows):mat[k,row['training_indices']]=1;mat[k,row['validation_indices']]=2
            a.imshow(mat,aspect='auto',vmin=0,vmax=2,cmap=matplotlib.colors.ListedColormap(['#eee',C[0],C[1]]));a.set(xticks=range(12),yticks=range(len(rows)),yticklabels=[str(k+1) for k in range(len(rows))],xlabel='行/时间索引0至11',ylabel='折号',title=name);a.grid(False)
        f.suptitle('蓝=训练，橙=验证，灰=未用；组ID每三行相同；时间gap=1',fontsize=11)
    elif i==9:
        sets=[set(v['training_indices']) for v in p['outer_splits']];overlap=np.array([[len(a&b)/len(a) for b in sets] for a in sets]);ax[0].imshow(overlap,vmin=0,vmax=1,cmap='Blues')
        for j,k in np.ndindex(overlap.shape):ax[0].text(k,j,f'{overlap[j,k]:.0%}',ha='center',va='center',color='white')
        ax[0].set(xticks=range(5),xticklabels=range(1,6),yticks=range(5),yticklabels=range(1,6),xlabel='另一外层训练折',ylabel='当前外层训练折',title='训练行共享比例；不是相关系数');ax[0].grid(False)
        ax[1].axis('off');ax[1].text(.04,.92,'同一数据集的五折',fontsize=15,color=C[1],va='top');ax[1].text(.04,.80,'训练集合重叠，误差会有关联。\n不能把五个折分数当五次独立实验\n做简单t检验或直接套标准误。',fontsize=11,linespacing=1.8,va='top');ax[1].text(.04,.43,'本讲80个独立数据集',fontsize=15,color=C[0],va='top');ax[1].text(.04,.31,'每次重做全部分折、选参、重拟合。\n标准差和MCSE在这个重复层面计算。\n现实只有一个数据集时，不能凭空造80份。',fontsize=11,linespacing=1.8,va='top')
    elif i==10:
        d=r['data']['development'];grid=np.linspace(-1,1,201);import models
        model=p['final_choice']['model'];pred=models.predict(model,grid);true=np.polynomial.legendre.legval(grid,c['true_coefficients']);ax[0].scatter(d['x'],d['y'],s=12,alpha=.5,c='#777',label='80行开发数据');ax[0].plot(grid,true,c=C[1],label='合成真均值，仅诊断');ax[0].plot(grid,pred,c=C[0],label='锁定后的最终80行模型');ax[0].set(xlabel='x',ylabel='y',title='最终候选：degree3，λ=0.01');ax[0].legend(fontsize=8,loc='upper left')
        test=r['final_test'];vals=[p['nested']['pooled_outer_MSE'],test['MSE'],p['final_model_oracle_risk']['total_MSE']];ax[1].bar([0,1,2],vals,color=[C[2],C[0],C[1]]);ax[1].errorbar(1,vals[1],yerr=2*test['conditional_test_mean_loss_SE'],fmt='none',c='black',capsize=5);ax[1].set(xticks=[0,1,2],xticklabels=['嵌套外层\n64行训练目标','独立4000行测试\n固定80行模型','精确总体MSE\n固定80行模型'],ylabel='MSE',ylim=(0,.85),title='误差棒只描述独立测试抽样')
        for j,v in enumerate(vals):ax[1].text(j,v+.055,f'{v:.4f}',ha='center',fontsize=9)
    else:raise ValueError('figure1..10')
    return f

def render_all(r,directory):
    directory=Path(directory).absolute()
    for part in (directory,*directory.parents):
        if part.is_symlink():raise ValueError('symlink figure output')
    if directory.resolve()==e.ROOT/'figures':raise ValueError('use a new output directory')
    directory.mkdir(parents=True,exist_ok=True)
    for i,name in enumerate(NAMES,1):
        out=directory/(name+'.png')
        if out.is_symlink() or (out.exists() and (not out.is_file() or out.stat().st_nlink>1)):raise ValueError('unsafe figure output')
        fig=figure(i,r);stream=io.BytesIO()
        try:fig.savefig(stream,format='png',dpi=160,bbox_inches='tight')
        finally:plt.close(fig)
        out.write_bytes(stream.getvalue())
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',required=True);p.add_argument('--directory',required=True);a=p.parse_args();render_all(json.loads(Path(a.report).read_text()),a.directory)
