"""Original mechanism figures, regenerated from the current validated experiment."""
from pathlib import Path
import os
import tempfile
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'unit039-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import experiment as e

COLORS={'good':'#176f91','fault':'#ca4e45','wrong':'#bd8b23','dark':'#26394c','green':'#328269'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':COLORS['dark'],'text.color':COLORS['dark']})


def frame(title,wide=True):
    f=plt.figure(figsize=(10,4.2) if wide else (8,4.2));f.suptitle(title,fontsize=14,fontweight='bold');return f


def chain(report):
    f=frame('One frozen parameter state, four rows, one synchronous update');ax=f.add_subplot();ax.axis('off');ax.set_xlim(0,10);ax.set_ylim(0,4)
    entries=[(.15,2.1,1.6,1.,'Input rows\n(id, x, y)\nshape (4, 3)'),(2.15,2.1,1.6,1.,'Prediction\np = b + w x\nshape (4,)'),(4.15,2.1,1.6,1.,'Paired residual\nr = p - y\nshape (4,)'),(6.15,2.1,1.6,1.,'Local chain\nr [1, x] / 4\nshape (4, 2)'),(8.15,2.1,1.6,1.,'Sum rows\ng = [-.5, -.75]\nshape (2,)')]
    for x,y,w,h,t in entries:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.05',facecolor='#eaf2f5',edgecolor=COLORS['good']));ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=9)
    for x in [1.8,3.8,5.8,7.8]:ax.annotate('',(x+.3,2.6),(x,2.6),arrowprops={'arrowstyle':'->','color':COLORS['good']})
    ax.text(5,.95,'theta(0) = [0, 0]     eta = 1/2     delta = [1/4, 3/8]\ntheta(1) = [1/4, 3/8]     Recompute every row at this NEW state',ha='center',va='center',fontsize=12)
    ax.annotate('',(2.95,2.02),(8.9,1.65),arrowprops={'arrowstyle':'->','connectionstyle':'arc3,rad=-.08','color':COLORS['green'],'lw':2})
    f.tight_layout(rect=(0,0,1,.91));return f


def pairing(report):
    data=report['data'];theta=report['paths']['none']['trace'][0]['new']['theta'];p=np.array([theta[0]+theta[1]*r[1] for r in data]);y=np.array([r[2] for r in data]);cross=p[:,None]-y
    f=frame('Broadcasting changes which prediction is paired with which label');axs=f.subplots(1,2)
    for ax,m,title in [(axs[0],(p-y)[:,None],'Intended residual: (4,) shown as a column'),(axs[1],cross,'Wrong residual: (4, 1) - (4,) = (4, 4)')]:
        ax.imshow(m,cmap='RdBu_r',vmin=-1,vmax=1,aspect='auto');ax.set_title(title,fontsize=10);ax.set_yticks(range(4),['row 1','row 2','row 3','row 4']);ax.set_xticks(range(m.shape[1]),['own label'] if m.shape[1]==1 else ['y1','y2','y3','y4'])
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):ax.text(j,i,f'{m[i,j]:.3f}',ha='center',va='center',fontsize=11,color='white' if abs(m[i,j])>.75 else COLORS['dark'])
    for j in range(4):axs[1].add_patch(plt.Rectangle((j-.48,j-.48),.96,.96,fill=False,edgecolor='#111111',lw=2))
    axs[1].set_xlabel('Black boxes are the only intended pairings')
    f.tight_layout(rect=(0,0,1,.91));return f


def two_rounds(report):
    f=frame('The same four rows before and after two updates');axs=f.subplots(1,2);x=np.array([r[1] for r in report['data']]);y=[r[2] for r in report['data']]
    states=[report['paths']['none']['initial']]+[t['new'] for t in report['paths']['none']['trace'][:2]]
    axs[0].scatter(x,y,c='black',s=55,zorder=4,label='observed y')
    for k,st in enumerate(states):axs[0].plot(x,[r['prediction'] for r in st['rows']],marker='o',label=f'state {k}: F={st["loss"]:.6f}')
    axs[0].set(xlabel='x',ylabel='prediction / target');axs[0].legend(fontsize=8)
    loc=np.arange(3);width=.34
    for j,name in enumerate(['intercept gradient','slope gradient']):axs[1].bar(loc+(j-.5)*width,[st['gradient'][j] for st in states],width,label=name)
    axs[1].axhline(0,color='#555',lw=.8);axs[1].set(xticks=loc,xticklabels=['state 0','state 1','state 2'],ylabel='mean gradient contribution sum');axs[1].legend(fontsize=8)
    f.tight_layout(rect=(0,0,1,.91));return f


def finite_difference(report):
    f=frame('A gradient check tests a formula pair; its step size also matters');axs=f.subplots(1,2);rows=report['gradient_checks']['table'];h=[v['h'] for v in rows]
    for key,label,col in [('paired_error','quadratic (zero truncation term)',COLORS['good']),('logistic_error','smooth logistic',COLORS['green'])]:
        axs[0].loglog(h,[max(v[key] or 0,1e-17) for v in rows],'-o',ms=3,label=label,color=col)
    axs[0].set(xlabel='central difference h',ylabel='normalized gradient discrepancy');axs[0].legend(fontsize=8);axs[0].grid(alpha=.2,which='both')
    mid=next(v for v in rows if v['h']==1e-5);keys=['paired_error','sign_error','broadcast_vs_intended_error','broadcast_self_error']
    axs[1].bar(range(4),[max(mid[k],1e-17) for k in keys],color=[COLORS['good'],COLORS['fault'],COLORS['wrong'],COLORS['wrong']]);axs[1].set_yscale('log');axs[1].set_xticks(range(4),['correct\nvs intended','sign bug\nvs intended','broadcast\nvs intended','broadcast\nvs itself'],fontsize=8);axs[1].set(ylabel='discrepancy at h = 1e-5');axs[1].set_ylim(1e-13,10);axs[1].grid(axis='y',alpha=.2)
    f.tight_layout(rect=(0,0,1,.91));return f


def paths(report):
    f=frame('A falling loss can still optimize the wrong data pairing');axs=f.subplots(1,2)
    for name,col in [('none',COLORS['good']),('broadcast',COLORS['wrong'])]:
        p=report['paths'][name];states=[p['initial']]+[t['new'] for t in p['trace'] if t['accepted']];axs[0].plot([st['loss'] for st in states],'-o',ms=2,color=col,label=name+' objective')
        axs[1].semilogy([max(e.norm(st['gradient']),1e-16) for st in states],'-o',ms=2,color=col,label=name)
    for name,marker in [('sign','x'),('large_step','s')]:
        p=report['paths'][name];axs[0].scatter([1],[p['trace'][0]['new']['loss']],color=COLORS['fault'],marker=marker,s=60,label=name+' rejected trial')
    axs[0].set_yscale('log');axs[0].set(xlabel='attempt / committed state index',ylabel='its own objective');axs[0].legend(fontsize=8)
    axs[1].axhline(report['configuration']['gradient_tolerance'],color='#555',ls=':',label='gradient tolerance');axs[1].set(xlabel='committed updates',ylabel='its own gradient norm');axs[1].legend(fontsize=8)
    f.tight_layout(rect=(0,0,1,.91));return f


def stability(report):
    f=frame('Equivalent real formulas can have different floating-point failures');axs=f.subplots(1,2)
    t=np.linspace(-1000,1000,501);stable=np.array([e.softplus_scalar(float(v)) for v in t])
    with np.errstate(over='ignore',divide='ignore',under='ignore'):naive=np.log(1+np.exp(t))
    axs[0].plot(t,stable,label='max(t,0) + log1p(exp(-|t|))',color=COLORS['good']);mask=np.isfinite(naive);axs[0].plot(t[mask],naive[mask],'--',label='log(1 + exp(t)), finite portion',color=COLORS['fault']);axs[0].axvspan(710,1000,color=COLORS['fault'],alpha=.12);axs[0].text(720,180,'overflow\nregion',fontsize=9);axs[0].set(xlabel='t',ylabel='softplus(t)');axs[0].legend(fontsize=8)
    z=np.arange(1.,61.);tail=np.array([e.softplus_scalar(-float(v)) for v in z]);cancel=np.logaddexp(0.,z)-z
    axs[1].semilogy(z,tail,label='signed softplus(-z), y=1',color=COLORS['good']);nz=cancel>0;axs[1].semilogy(z[nz],cancel[nz],'x',label='softplus(z)-z, positive results',color=COLORS['fault']);axs[1].axvspan(34,60,color=COLORS['fault'],alpha=.1);axs[1].text(36,1e-7,'subtraction can\nround to zero',fontsize=9);axs[1].set(xlabel='positive logit z for target y=1',ylabel='per-row binary log loss');axs[1].legend(fontsize=8)
    f.tight_layout(rect=(0,0,1,.91));return f


def diagnostics(report):
    f=frame('Small movement and a descent direction are separate diagnostics');axs=f.subplots(1,2)
    names=['none','tiny_step','sign','large_step'];short=['correct','tiny step','wrong sign','large step'];tr=[report['paths'][n]['trace'][0] for n in names]
    axs[0].bar(range(4),[t['relative_update'] for t in tr],color=[COLORS['good'],COLORS['wrong'],COLORS['fault'],COLORS['fault']]);axs[0].set_yscale('log');axs[0].set_xticks(range(4),short,rotation=10);axs[0].set_ylabel('relative update, first attempt')
    axs[1].bar(range(4),[t['directional_derivative'] for t in tr],color=[COLORS['good'],COLORS['wrong'],COLORS['fault'],COLORS['fault']]);axs[1].axhline(0,color='#555',lw=1);axs[1].set_xticks(range(4),short,rotation=10);axs[1].set_ylabel('g dot actual proposed update');axs[1].text(.03,.08,'Large-step direction is downhill locally,\nbut the evaluated finite step raises F.',transform=axs[1].transAxes,fontsize=9)
    f.tight_layout(rect=(0,0,1,.91));return f


def kink(report):
    f=frame('A symmetric quotient at a kink is not evidence of differentiability');axs=f.subplots(1,2);x=np.linspace(-1,1,300)
    for ax in axs:ax.plot(x,abs(x),color=COLORS['good'],lw=2);ax.set(xlabel='u',ylabel='|u|');ax.scatter([0],[0],color='black',zorder=5)
    axs[0].plot([-.7,.7],[.7,.7],'--o',color=COLORS['fault']);axs[0].text(-.85,.85,'central quotient = 0',fontsize=10);axs[0].set_title('Same function values on both sides',fontsize=10)
    axs[1].plot([-.7,0],[.7,0],color=COLORS['wrong'],lw=4,label='left slope = -1');axs[1].plot([0,.7],[0,.7],color=COLORS['green'],lw=4,label='right slope = +1');axs[1].legend(fontsize=9);axs[1].set_title('No unique tangent at zero',fontsize=10)
    f.tight_layout(rect=(0,0,1,.91));return f


FIGURES=[('01_frozen_chain',chain),('02_broadcast_pairing',pairing),('03_two_rounds',two_rounds),('04_gradient_checks',finite_difference),('05_fault_paths',paths),('06_stable_logits',stability),('07_update_diagnostics',diagnostics),('08_nonsmooth_boundary',kink)]


def main():
    data,s=e.load_inputs();report=e.run_experiment(data,s)
    out=Path(__file__).resolve().parent/'figures';out.mkdir(exist_ok=True)
    for name,fun in FIGURES:
        fig=fun(report);fig.savefig(out/(name+'.png'),dpi=165,bbox_inches='tight');plt.close(fig)
    print('Generated',len(FIGURES),'original figures from validated inputs.')


if __name__=='__main__':main()
