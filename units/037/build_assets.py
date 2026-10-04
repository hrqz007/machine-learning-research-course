"""Original explanatory figures. Every graph is recomputed from validated inputs."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/unit037-mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from experiment import load_inputs,validate,run_experiment,soft
COLORS=['#136f8a','#e68a2e','#864da0','#278666']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
NAMES=['01_kink_subgradients','02_proximal_threshold','03_two_step_ledger','04_objectives_and_sparsity','05_three_residuals','06_box_projection','07_cyclic_vs_simultaneous','08_subgradient_failure','09_feature_scaling']
def path_arrays(p):
    fw=[p['initial_forward']]+[t['new_forward'] for t in p['trace']]
    dg=[p['initial_diagnostics']]+[t['diagnostics'] for t in p['trace']]
    return np.array([v['theta'] for v in fw]),np.array([v['objective'] for v in fw]),dg

def make_figure(number,data,s):
    data,s=validate(data,s)
    if type(number) is not int or not 1<=number<=9:raise ValueError('figure number')
    report=run_experiment(data,s)
    fig,axes=plt.subplots(1,2,figsize=(10.2,3.6),layout='constrained');a,b=axes
    x=np.linspace(-2,2,401)
    if number==1:
        a.plot(x,np.abs(x),lw=3,color=COLORS[0],label='absolute value')
        for q in [-1,-.5,0,.5,1]:a.plot(x,q*x,lw=1,alpha=.55,label=f's = {q:g}')
        a.set(xlabel='u',ylabel='value',title='Supporting lines at the kink');a.legend(fontsize=7,ncol=2)
        b.plot([-1,0],[ -1,-1],lw=3,color=COLORS[0]);b.plot([0,1],[1,1],lw=3,color=COLORS[0]);b.plot([0,0],[-1,1],lw=5,color=COLORS[1]);b.set(xlabel='u',ylabel='subgradient s',title='At zero: a whole interval [-1, 1]',xlim=(-1.2,1.2),ylim=(-1.3,1.3))
    elif number==2:
        tau=.25;a.plot(x,x,'--',color='gray',label='identity');a.plot(x,soft(x,tau),lw=3,color=COLORS[0],label='soft threshold, tau=1/4');a.axvspan(-tau,tau,alpha=.16,color=COLORS[1]);a.set(xlabel='smooth trial z',ylabel='new coefficient',title='An interval becomes exactly zero');a.legend(fontsize=8)
        u=np.linspace(-.6,.9,400)
        for z,c in [(.1875,COLORS[0]),(.375,COLORS[1])]:
            v=.5*(u-z)**2+tau*np.abs(u);b.plot(u,v,color=c,label=f'z={z:g}');m=float(soft(np.array([z]),tau)[0]);b.scatter(m,.5*(m-z)**2+tau*abs(m),color=c,s=35,zorder=4)
        b.set(xlabel='candidate u',ylabel='proximal objective',title='Keep the L1 corner intact');b.legend(fontsize=8)
    elif number==3:
        for ax,tr in zip(axes,report['exact_first_two']):
            vals=np.array([[float(__import__('fractions').Fraction(v[k])) for k in ('prediction','residual','half_square')] + [float(__import__('fractions').Fraction(z)) for z in v['gradient_contribution']] for v in tr['old']['rows']])
            ax.axis('off');tbl=ax.table(cellText=[[f'{v:.4g}' for v in row] for row in vals],rowLabels=['s1','s2','s3','s4'],colLabels=['pred','resid','loss','x1*r/n','x2*r/n'],loc='center');tbl.auto_set_font_size(False);tbl.set_fontsize(9);tbl.scale(1,1.7)
            for (r,c),cell in tbl.get_celld().items():
                cell.set_edgecolor('#d9e3e8')
                if r==0:cell.set_facecolor('#dcecf0')
            ax.set_title('Old theta = '+str(tr['old']['theta']),pad=18)
            ax.text(.5,.06,'g = '+str(tr['old']['gradient'])+'\nz = '+str(tr['smooth_trial'])+' -> theta = '+str(tr['new']['theta']),transform=ax.transAxes,ha='center',fontsize=9)
    elif number==4:
        for method,c in zip(('proximal','subgradient','coordinate'),COLORS):
            th,ob,dg=path_arrays(report['paths'][method]);a.plot(ob,color=c,label=method);b.step(np.arange(len(th)),np.sum(th==0,axis=1),where='post',color=c,label=method)
        a.axhline(33/32,color='gray',ls=':',label='default J*');a.set(xlabel='update (CD: sweep)',ylabel='J',title='Same main data and lambda');a.legend(fontsize=8)
        b.set(xlabel='update (CD: sweep)',ylabel='exact zero count',title='Zero means stored value == 0',yticks=[0,1,2]);b.legend(fontsize=8)
    elif number==5:
        th,ob,dg=path_arrays(report['paths']['proximal'])
        for k,c in zip(('smooth_gradient_inf','lasso_kkt_inf','proximal_mapping_inf'),COLORS):a.semilogy(np.maximum([d[k] for d in dg],1e-16),label=k.replace('_inf',''),color=c)
        a.set(xlabel='proximal update',ylabel='infinity norm (floor 1e-16)',title='Smooth gradient need not vanish');a.legend(fontsize=8)
        b.semilogy(np.maximum([d['default_objective_gap'] for d in dg],1e-32),color=COLORS[0]);b.set(xlabel='proximal update',ylabel='J - J* (stable expression)',title='A known optimum is needed for this gap')
    elif number==6:
        p=report['paths']['projected'];th,_,dg=path_arrays(p);a.add_patch(Rectangle((0,0),1,1,facecolor='#e0f0ea',edgecolor=COLORS[3]));a.plot(th[:,0],th[:,1],'o-',color=COLORS[0],ms=3,label='projected path');a.scatter([2],[.25],marker='*',s=100,color=COLORS[1],label='unconstrained minimum');a.set(xlabel='theta1',ylabel='theta2',title='Box [0,1]^2; minimize smooth F',xlim=(-.15,2.2),ylim=(-.1,1.1));a.legend(fontsize=8)
        for k,c in zip(('smooth_gradient_inf','projected_mapping_inf','box_violation'),COLORS):b.semilogy(np.maximum([d[k] for d in dg],1e-16),label=k,color=c)
        b.set(xlabel='projected update',ylabel='residual (floor 1e-16)',title='At a boundary, gradient can be nonzero');b.legend(fontsize=7)
    elif number==7:
        for block,ax,label in [(report,a,'Orthogonal main design'),(report['stress'],b,'Explicit correlated stress design')]:
            for method,c in zip(('proximal','coordinate'),COLORS):
                th,ob,dg=path_arrays(block['paths'][method]);ax.plot(th[:,0],th[:,1],'o-',ms=2,color=c,label=method)
            ax.set(xlabel='theta1',ylabel='theta2',title=label);ax.legend(fontsize=8)
    elif number==8:
        for method,c in zip(('proximal','subgradient'),COLORS):
            th,ob,dg=path_arrays(report['paths'][method]);a.plot(th[:25,1],'o-',ms=3,color=c,label=method);b.plot(np.array([d['lasso_kkt_inf'] for d in dg])[:25],color=c,label=method)
        a.axhline(0,color='gray',ls=':');a.set(xlabel='update',ylabel='theta2',title='sign(0)=0 is a choice, not an ordinary derivative');a.legend(fontsize=8)
        b.set(xlabel='update',ylabel='Lasso KKT residual',title='Fixed subgradient steps may keep oscillating');b.legend(fontsize=8)
    else:
        lam=np.linspace(0,2.5,201)
        for j,c in enumerate(COLORS[:2]):a.plot(lam,np.maximum([2,.25][j]-lam,0),color=c,label=f'theta{j+1}')
        a.set(xlabel='lambda',ylabel='orthogonal optimum',title='Sparsity path; no test-set claim');a.legend(fontsize=8)
        scale=np.array([.5,1,2,4]);b.plot(scale,np.maximum(2-.5/scale,0),'o-',color=COLORS[0],label='effective x1 coefficient');b.axhline(1.5,ls=':',color=COLORS[1],label='original coordinate penalty');b.set(xlabel='feature x1 multiplier c',ylabel='c * fitted coefficient',title='Unadjusted L1 changes under feature scaling');b.legend(fontsize=8)
    fig.suptitle(f'037  |  {NAMES[number-1][3:].replace("_"," ")}',fontsize=13,color='#103c4b')
    return fig
if __name__=='__main__':
    data,s=load_inputs()
    from pathlib import Path
    out=Path(__file__).resolve().parent/'figures'
    for i,name in enumerate(NAMES,1):
        fig=make_figure(i,data,s);fig.savefig(out/(name+'.png'));plt.close(fig)
