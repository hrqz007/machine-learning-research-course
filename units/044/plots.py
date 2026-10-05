"""Rebuild eight original figures from archived data and actual scientific results."""
from pathlib import Path
import argparse, json, os, io, tempfile
os.environ.setdefault('MPLCONFIGDIR',str(Path.home()/'.cache/matplotlib'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.stats import poisson
import experiment as ex
COLORS=['#2364aa','#e58c27','#13856a','#b04874']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.labelsize':10,
                     'figure.dpi':150,'savefig.dpi':180,'axes.spines.top':False,'axes.spines.right':False})


def make_figures(report, directory):
    dest=Path(directory).resolve()
    if dest.is_relative_to(ex.ROOT) and not dest.is_relative_to(ex.ROOT/'outputs'):
        raise ValueError('Generate figures in outputs/ or an external new directory; delivered figures are protected')
    dest.mkdir(parents=True,exist_ok=True); paths=[]
    def save(fig,name):
        target=dest/name
        if target.is_symlink() or (target.exists() and (not target.is_file() or target.stat().st_nlink>1)):
            plt.close(fig);raise ValueError('unsafe figure output')
        buffer=io.BytesIO();fig.savefig(buffer,format='png',bbox_inches='tight',facecolor='white');plt.close(fig)
        fd,tmp=tempfile.mkstemp(prefix='.'+name+'.',dir=dest)
        try:
            with os.fdopen(fd,'wb') as stream:stream.write(buffer.getvalue())
            os.replace(tmp,target)
        finally:
            if os.path.exists(tmp):os.unlink(tmp)
        paths.append(str(target))
    z=np.linspace(-3,3,301)
    fig,ax=plt.subplots(1,3,figsize=(11,3.2),layout='constrained')
    for a,values,title,label in zip(ax,[z,expit(z),np.exp(z)],['Gaussian + identity','Bernoulli + logit','Poisson + log'],['real mean','probability','positive count mean']):
        a.plot(z,values,color=COLORS[0],lw=2.5);a.axvline(0,color='#bbbbbb',lw=.7)
        a.set(xlabel='Linear predictor eta',ylabel='Mean mu',title=title);a.text(.04,.95,label,transform=a.transAxes,va='top',fontsize=9)
    save(fig,'01_response_functions.png')
    fig,ax=plt.subplots(1,3,figsize=(11,3.2),layout='constrained');k=np.arange(17)
    for a,m,c in zip(ax,[.5,2.,5.],COLORS):
        a.bar(k,poisson.pmf(k,m),color=c,width=.8);a.set(title=f'Poisson mean = variance = {m:g}',xlabel='Observed integer count y',ylabel='Probability',xlim=(-.6,16.6));a.set_xticks([0,4,8,12,16])
        a.text(.97,.94,f'P(y > 16) = {poisson.sf(16,m):.2g}',ha='right',va='top',transform=a.transAxes,fontsize=9)
    save(fig,'02_poisson_distributions.png')
    fig,ax=plt.subplots(1,2,figsize=(10.5,3.5),layout='constrained');x=np.array([-1,0,1]);ys=np.array([0,1,3])
    ax[0].scatter(x,ys,color='black',s=55,label='observed count',zorder=5)
    for s,c in zip(report['tiny_ledger'],COLORS):ax[0].plot(x,s['mu'],'o-',color=c,label=f"state {s['update']}")
    ax[0].set(xlabel='Feature x',ylabel='Count / mean count',title='All rows share one old parameter');ax[0].set_xticks(x);ax[0].legend(fontsize=8)
    losses=[s['mean_loss'] for s in report['tiny_ledger']];ax[1].plot([0,1,2],losses,'o-',color=COLORS[0]);ax[1].set(xlabel='Completed synchronous updates',ylabel='Mean full negative log likelihood',title='Two updates, three forward passes');ax[1].set_xticks([0,1,2]);ax[1].set_ylim(1.1,1.7)
    for k,loss in enumerate(losses):ax[1].annotate(f'{loss:.6f}',(k,loss),xytext=(0,10),textcoords='offset points',ha='center')
    save(fig,'03_two_updates.png')
    fig,ax=plt.subplots(figsize=(7.7,3.5),layout='constrained');mu=np.linspace(.04,6,400)
    for yy,c in zip([0,1,3],COLORS):
        ax.plot(mu,ex.deviance_rows(np.full(len(mu),yy),mu),label=f'y = {yy}',color=c,lw=2)
        ax.scatter([yy],[0],color=c,s=35,zorder=5)
    ax.set(xlabel='Candidate positive mean mu',ylabel='Single-row Poisson deviance',title='Saturated reference: mu = y',ylim=(-.5,21));ax.legend()
    ax.annotate('y = 0: boundary limit at mu = 0',xy=(0,0),xytext=(1.2,18),arrowprops={'arrowstyle':'->','color':'#666666'},fontsize=9)
    save(fig,'04_deviance.png')
    rows,X,e,truth,train=ex.read_data();xx=np.linspace(-2,2,300);Z=np.column_stack([np.ones(len(xx)),xx])
    fig,ax=plt.subplots(1,2,figsize=(11,3.8),layout='constrained')
    for a,label,field,title in zip(ax,['poisson','overdispersed'],['poisson_y','overdispersed_y'],['Poisson draw','Gamma-Poisson draw']):
        c=report['cases'][label];y=np.array([float(r[field]) for r in rows]);a.scatter(X[~train,1],y[~train]/e[~train],s=9,color='#778899',alpha=.35,label='test count / exposure')
        a.plot(xx,np.exp(Z@np.array(report['protocol']['true_beta'])),color='black',ls='--',label='true rate')
        a.plot(xx,np.exp(Z@np.array(c['poisson_fit']['state']['beta'])),color=COLORS[0],lw=2,label='Poisson fitted rate')
        a.plot(xx,Z@np.array(c['linear_beta']),color=COLORS[1],lw=2,label='linear fitted rate')
        a.axhline(0,color='#888888',lw=.8);a.set(xlabel='Feature x',ylabel='Events per exposure unit',title=title,xlim=(-2.05,2.05));a.legend(fontsize=7,loc='upper left')
    save(fig,'05_count_model_comparison.png')
    fig,ax=plt.subplots(1,2,figsize=(10.5,3.4),layout='constrained');ee=np.linspace(.5,2,200)
    for a,label in zip(ax,['poisson','overdispersed']):
        c=report['cases'][label];b=c['poisson_fit']['state']['beta'][0];bn=c['no_offset_fit']['state']['beta'][0]
        a.plot(ee,ee*np.exp(b),lw=2.5,label='with log(exposure) offset',color=COLORS[0]);a.plot(ee,np.full(len(ee),np.exp(bn)),lw=2.5,label='omitted exposure',color=COLORS[1]);a.plot(ee,ee*np.exp(.2),'k--',label='true mean')
        a.set(xlabel='Exposure (units)',ylabel='Predicted count at x = 0',title=label.replace('overdispersed','Gamma-Poisson'));a.legend(fontsize=8)
    save(fig,'06_exposure_offset.png')
    fig,ax=plt.subplots(1,2,figsize=(10.7,3.8),layout='constrained');names=['Poisson','Gamma-Poisson'];ind=np.arange(2)
    raw=[report['cases'][q]['raw_train_variance_ddof1']/report['cases'][q]['raw_train_mean'] for q in ['poisson','overdispersed']];adj=[report['cases'][q]['pearson_dispersion_train'] for q in ['poisson','overdispersed']]
    ax[0].bar(ind-.17,raw,.34,color='#97a5b1',label='raw variance / mean');ax[0].bar(ind+.17,adj,.34,color=COLORS[0],label='adjusted Pearson dispersion');ax[0].axhline(1,color='black',ls='--',lw=1);ax[0].set_xticks(ind,names);ax[0].set(ylabel='Training diagnostic',title='Marginal spread is not conditional spread');ax[0].legend(fontsize=8)
    for i in range(2):
        ax[0].text(i-.17,raw[i]+.15,f'{raw[i]:.2f}',ha='center',fontsize=9);ax[0].text(i+.17,adj[i]+.15,f'{adj[i]:.2f}',ha='center',fontsize=9)
    mm=np.linspace(0,12,200);ax[1].plot(mm,mm,color=COLORS[0],label='Poisson: variance = mu',lw=2);ax[1].plot(mm,mm+mm**2/2,color=COLORS[1],label='Gamma-Poisson: mu + mu^2/2',lw=2);ax[1].set(xlabel='Conditional mean mu',ylabel='Conditional variance',title='True data-generating variance functions');ax[1].legend(fontsize=8)
    save(fig,'07_conditional_dispersion.png')
    fig,ax=plt.subplots(1,2,figsize=(10.6,3.5),layout='constrained')
    for a,label in zip(ax,['poisson','overdispersed']):
        c=report['cases'][label]
        for key,name,col in [('poisson_fit','with offset',COLORS[0]),('no_offset_fit','without offset',COLORS[1])]:
            h=c[key]['history'];a.semilogy([r['update'] for r in h],[r['gradient_l2'] for r in h],'o-',label=name,color=col)
        a.axhline(report['protocol']['newton_gradient_tolerance'],color='black',ls='--',lw=1,label='stopping tolerance')
        a.set(xlabel='Completed Newton updates',ylabel='Gradient L2 norm',title=label.replace('overdispersed','Gamma-Poisson'));a.legend(fontsize=8)
    save(fig,'08_optimization.png')
    return paths

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',required=True);p.add_argument('--directory',default=str(ex.ROOT/'outputs/figures'));a=p.parse_args()
    print(json.dumps(make_figures(json.loads(Path(a.report).read_text()),a.directory),indent=2))
