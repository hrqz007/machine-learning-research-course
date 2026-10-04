"""Original, computed ML040 teaching figures; reusable from the notebook."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import experiment as e

COLORS=['#147D92','#D65F38','#7457A4','#B59722']
NAMES=['01_regularization_workflow','02_collinearity_geometry','03_ridge_spectral_shrinkage','04_regularized_forward_chain','05_bias_variance','06_lasso_geometry','07_elastic_grouping','08_scaling_and_leakage','09_paths_and_selection']

def style():
    candidates=sorted([p for p in font_manager.findSystemFonts() if 'NotoSansCJK' in p or 'SourceHanSans' in p], key=lambda p: ('Regular' not in p, p))
    if candidates:
        font_manager.fontManager.addfont(candidates[0]);plt.rcParams['font.family']=font_manager.FontProperties(fname=candidates[0]).get_name()
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.2,'figure.facecolor':'#FAFCFE','axes.facecolor':'#FAFCFE','axes.unicode_minus':False,'savefig.dpi':150})

def box(ax,x,y,w,h,text,color):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.018',facecolor=color,edgecolor='none',alpha=.14));ax.text(x+w/2,y+h/2,text,ha='center',va='center',color='#172D3D',fontsize=11)

def contours(ax,X,y):
    u=np.linspace(-.7,2.7,220);v=np.linspace(-.8,3,220);a,b=np.meshgrid(u,v);p=np.stack([a,b],axis=-1);H=X.T@X/len(y);c=X.T@y/len(y);F=.5*np.einsum('...i,ij,...j->...',p,H,p)-p@c+(y@y)/(2*len(y))
    ax.contour(a,b,F,levels=[.13,.16,.23,.4,.7,1.2,2.],colors='#A8BDC5',linewidths=1);ax.set(xlabel=r'$\beta_1$',ylabel=r'$\beta_2$');return a,b,F

def make_figure(number,report,X,y,splits,c):
    style()
    if number==1:
        fig,ax=plt.subplots(figsize=(11,3));ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1))
        labels=['训练数据\n只拟合训练均值/尺度','预声明候选\n拟合系数；检查KKT','验证MSE\n锁定λ、ρ与模型','最终测试\n报告锁定模型+OLS']
        for i,label in enumerate(labels):
            x=.02+i*.25;box(ax,x,.32,.205,.42,label,COLORS[i])
            if i<3:ax.annotate('',xy=(x+.239,.53),xytext=(x+.214,.53),arrowprops={'arrowstyle':'->','color':'#536571','lw':2})
        ax.text(.5,.1,'训练 F、惩罚 P、总目标 J、验证/测试 MSE 分别记录；测试无返回箭头',ha='center',fontsize=12)
    elif number==2:
        fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained');contours(axes[0],X,y)
        ols=np.linalg.lstsq(X,y,rcond=None)[0];ridge=np.array(report['main_fits']['ridge']['beta']);axes[0].plot(*ols,'o',color=COLORS[0],label='OLS (0.25, 2)');axes[0].plot(*ridge,'s',color=COLORS[1],label='Ridge λ=0.25');axes[0].legend();axes[0].set_title('同一未惩罚损失的窄谷')
        axes[1].scatter(X[:,0],X[:,1],s=180,c=COLORS[0]);
        for i,r in enumerate(X):axes[1].annotate(f'S{i+1}',r,xytext=(6,9 if i%2 else -17),textcoords='offset points')
        axes[1].set(xlabel='$x_1$',ylabel='$x_2$',title='四行共线设计：det(H)=1/64');axes[1].set_xlim(-1.4,1.4);axes[1].set_ylim(-1.4,1.4)
    elif number==3:
        fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained');lam=np.r_[0,np.logspace(-3,1,150)];s=np.linalg.svd(X,compute_uv=False)
        for i,v in enumerate(s):axes[0].plot(lam,v*v/(v*v+len(y)*lam),label=f'σ={v:.3f}',color=COLORS[i])
        axes[0].set_xscale('symlog',linthresh=.001);axes[0].set(xlabel='λ',ylabel='相对OLS的谱收缩因子',title='弱奇异方向收缩更快');axes[0].legend()
        path=report['paths']['ridge'];b=np.array([r['beta'] for r in path]);l=[r['lambda'] for r in path]
        for j in range(2):axes[1].plot(l,b[:,j],'-o',label=f'原坐标 β{j+1}',color=COLORS[j])
        axes[1].set_xscale('symlog',linthresh=.001);axes[1].set(xlabel='λ',ylabel='原单位系数',title='原第一系数可比OLS更大');axes[1].legend()
    elif number==4:
        fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained');trace=report['hand_trace'];k=np.arange(3)
        for key,label,color in [('data_loss','数据F',COLORS[0]),('l2_penalty','惩罚P',COLORS[1]),('objective','目标J',COLORS[2])]:axes[0].plot(k,[v[key] for v in trace],'-o',label=label,color=color)
        axes[0].set(xlabel='完成同步更新的轮数',ylabel='目标项',xticks=k,title='第二步：F略升，J继续降');axes[0].legend();axes[0].text(.04,.53,'k=1→2: ΔF=+0.004718\nΔJ=−0.000272',transform=axes[0].transAxes,fontsize=10,bbox={'facecolor':'white','edgecolor':'none','alpha':.96})
        axes[1].axis('off');rows=[]
        for row in trace[1]['rows']:rows.append([f"S{row['id']}",f"{row['prediction']:.5f}",f"{row['residual']:.5f}",f"{row['gradient_contribution'][0]:.5f}",f"{row['gradient_contribution'][1]:.5f}"])
        table=axes[1].table(cellText=rows,colLabels=['旧行','预测','残差','贡献g1','贡献g2'],cellLoc='center',loc='center');table.auto_set_font_size(False);table.set_fontsize(10);table.scale(1,2)
        axes[1].set_title('第二轮先读同一个旧参数，再加 λβ');axes[1].text(.5,.12,'数据梯度 + 惩罚梯度 → 同步更新 → 新前向',ha='center',transform=axes[1].transAxes,fontsize=10)
    elif number==5:
        fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained');records=report['monte_carlo']['records'];l=np.array([r['lambda'] for r in records]);bias=np.array([np.dot(r['analytic_bias'],r['analytic_bias']) for r in records]);var=np.array([np.trace(r['analytic_covariance']) for r in records])
        for values,label,color in [(bias,'理论偏差平方',COLORS[1]),(var,'理论方差迹',COLORS[0]),(bias+var,'理论系数MSE',COLORS[2])]:axes[0].plot(l,values,'-o',label=label,color=color)
        axes[0].scatter(l,[r['empirical_coefficient_mse'] for r in records],marker='x',s=65,color='#222',label='1000次实测MSE');axes[0].set_xscale('symlog',linthresh=.01);axes[0].set(yscale='log',xlabel='λ',ylabel='系数误差',title='理论与有限样本模拟分别报告');axes[0].legend(fontsize=9)
        for key,label,color in [('analytic_noiseless_prediction_mse_on_fixed_X','固定X上无噪声预测MSE',COLORS[0]),('analytic_new_label_mse_on_fixed_X','新独立标签MSE（+0.25）',COLORS[1])]:axes[1].plot(l,[r[key] for r in records],'-o',label=label,color=color)
        axes[1].set_xscale('symlog',linthresh=.01);axes[1].set(xlabel='λ',ylabel='预测误差',title='与系数MSE不是同一个量');axes[1].legend(fontsize=9)
    elif number==6:
        fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
        grids=[]
        for ax in axes:
            grids.append(contours(ax,X,y));ax.set_aspect('equal');ax.set_xlim(-2,2.7);ax.set_ylim(-2,3)
        axes[0].plot([0,1.5,0,-1.5,0],[1.5,0,-1.5,0,1.5],color=COLORS[1],lw=2);axes[0].plot(1.5,0,'o',color=COLORS[1]);axes[0].set_title('L1半径1.5：本例角点相切')
        b=np.array(report['main_fits']['ridge']['beta']);t=np.linalg.norm(b);u=np.linspace(0,2*np.pi,200);axes[1].plot(t*np.cos(u),t*np.sin(u),color=COLORS[0],lw=2);axes[1].plot(*b,'o',color=COLORS[0]);axes[1].set_title(f'L2半径{t:.3f}：约束半径不是λ')
        for ax,grid,point,color in zip(axes,grids,([1.5,0],b),(COLORS[1],COLORS[0])):
            ax.contour(*grid,levels=[e.row_ledger(X,y,point)['data_loss']],colors=[color],linewidths=1.5)
    elif number==7:
        fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained');D=np.column_stack(([-1.,-1.,1.,1.],[-1.,-1.,1.,1.]));fits=[e.fit_library(D,y,.5,f,c,.5) for f in ('lasso','elastic')];z=np.arange(2)
        for j in range(2):axes[0].bar(z+(j-.5)*.3,[r['beta'][j] for r in fits],width=.3,label=f'β{j+1}',color=COLORS[j])
        axes[0].set(xticks=z,xticklabels=['Lasso','Elastic Net'],ylabel='重复列的系数',title='相同列：正L2带来唯一且对称的分配');axes[0].legend()
        for f,color in zip(fits,COLORS):axes[1].plot(D@np.array(f['beta']),'-o',label=f['family'],color=color)
        axes[1].plot(D@np.array(fits[0]['beta'])[::-1],'--',label='Lasso交换分配（同预测）',color=COLORS[2]);axes[1].plot(y,'x',label='观测y',color='#333');axes[1].set(xlabel='行索引（0至3）',ylabel='预测/标签',title='不同惩罚仍可改变预测值');axes[1].legend()
    elif number==8:
        fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained');select=report['selection'];model=select['chosen'];V=splits['validation']['X'];mu=np.array(select['preprocessing']['mean']);sc=np.array(select['preprocessing']['scale']);gamma=np.array(model['coefficient_standardized']);b=np.array(model['coefficient_raw']);left=select['preprocessing']['y_mean']+(V-mu)/sc@gamma;right=model['intercept_raw']+V@b
        axes[0].bar(np.arange(5)-.17,gamma[:5],.34,label='标准化γ',color=COLORS[0]);axes[0].bar(np.arange(5)+.17,b[:5],.34,label='原单位β',color=COLORS[1]);axes[0].set(xticks=range(5),xticklabels=['x01','x02','x03','x04','x05'],title='同一模型，两种坐标');axes[0].legend()
        axes[1].scatter(left,right,color=COLORS[0],s=24);lo=min(left.min(),right.min());hi=max(left.max(),right.max());axes[1].plot([lo,hi],[lo,hi],'--',color=COLORS[1]);axes[1].set(xlabel='标准化空间预测',ylabel='原单位预测',title=f'60验证行最大差 {np.max(np.abs(left-right)):.2g}')
    elif number==9:
        fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
        for family,color in zip(e.FAMILIES,COLORS):
            path=report['paths'][family];l=[r['lambda'] for r in path];b=np.array([r['beta'] for r in path])
            for j,ls in enumerate(('-','--')):axes[0].plot(l,b[:,j],ls,color=color,label=family+f' β{j+1}')
        axes[0].set_xscale('symlog',linthresh=.001);axes[0].set(xlabel='手算原单位λ',ylabel='系数',title='四行：预声明离散路径');axes[0].legend(fontsize=8,ncol=2)
        records=report['selection']['candidates']
        for family,color in zip(e.FAMILIES,COLORS):
            ratios=c['selection_l1_ratios'] if family=='elastic' else [0 if family=='ridge' else 1]
            for rho in ratios:
                vals=[r for r in records if r['family']==family and r['l1_ratio']==rho];label=family+(f' ρ={rho}' if family=='elastic' else '')
                axes[1].plot([r['lambda'] for r in vals],[r['validation_mse'] for r in vals],'-o',markersize=3,label=label,color=color,alpha=.5+rho*.4)
        best=report['selection']['chosen'];axes[1].plot(best['lambda'],best['validation_mse'],'*',ms=15,color='#1C2732',label='锁定赢家');axes[1].set(xscale='log',yscale='log',xlabel='训练标准化空间λ',ylabel='验证MSE',title='340行实验：测试不进入选择图');axes[1].legend(fontsize=8)
    else:raise ValueError('figure index must be 1..9')
    return fig


def main():
    e.verify_fixture();X,y,s,c=e.load_inputs();r=e.main_report(X,y,s,c);out=e.ROOT/'figures';out.mkdir(exist_ok=True)
    for i,name in enumerate(NAMES,1):
        f=make_figure(i,r,X,y,s,c);f.savefig(out/(name+'.png'),bbox_inches='tight',metadata={'Software':'ML040 original computed illustration'});plt.close(f)
    print('9 original figures rendered from actual experiment')

if __name__=='__main__':main()
