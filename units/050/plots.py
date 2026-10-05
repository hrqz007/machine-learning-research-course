"""Ten figures from fixed source data and actual computed scores."""
from pathlib import Path
import argparse,json,os,tempfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.patches import Patch
import experiment as e
ROOT=Path(__file__).resolve().parent
NAMES=['01_split_maps','02_label_availability','03_hand_step','04_main_scores','05_repeated_panels','06_shift_and_residual','07_device_errors','08_bootstrap_units','09_weighting_target','10_window_support']
COLORS=['#276888','#d76b35','#517948'];LABELS=['Pooled','Personalized','With time trend'];SL=['Random rows','Held-out devices','Future same devices','Future new devices']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','figure.dpi':150,'savefig.dpi':150,'axes.labelsize':10,'axes.titlesize':11})

def render_all(result,directory):
    directory=Path(directory);e.safe_output(directory/'probe.png');directory.mkdir(parents=True,exist_ok=True)
    c,d,s=e.load_data();p=e.panel(c,d,0)
    def save(fig,i):
        path=e.safe_output(directory/(NAMES[i]+'.png'))
        with tempfile.NamedTemporaryFile(dir=directory,suffix='.png',delete=False) as f:tmp=f.name
        try:fig.savefig(tmp,bbox_inches='tight',facecolor='white',metadata={'Software':'ML050 deterministic source plot'});os.replace(tmp,path)
        finally:
            plt.close(fig)
            if os.path.exists(tmp):os.unlink(tmp)
    fig,axes=plt.subplots(2,2,figsize=(11.8,6.7),layout='constrained');cmap=ListedColormap(['#e3e6e9',COLORS[0],COLORS[1]])
    for ax,k,title in zip(axes.flat,e.SPLITS,SL):
        mat=np.zeros(2880,dtype=int);tr,te=s[k];mat[tr]=1;mat[te]=2
        ax.imshow(mat.reshape(60,48),aspect='auto',interpolation='nearest',cmap=cmap,vmin=-.5,vmax=2.5,extent=[-.5,47.5,59.5,-.5]);ax.set(xlabel='Prediction day',ylabel='Device ID',title=f'{title}: {len(tr)} train / {len(te)} test',xticks=[0,15,30,33,47],yticks=[0,15,30,45,59])
    fig.legend(handles=[Patch(color='#e3e6e9',label='Unused'),Patch(color=COLORS[0],label='Train'),Patch(color=COLORS[1],label='Test')],loc='outside lower center',ncol=3,frameon=False);save(fig,0)
    fig,ax=plt.subplots(figsize=(11.8,4.5),layout='constrained')
    days=np.arange(28,36)
    for i,t in enumerate(days):
        col=COLORS[0] if t+2<=32 else COLORS[1]
        ax.plot([t,t+2],[i,i],c=col,lw=3);ax.scatter([t],[i],s=65,c=col,marker='o');ax.scatter([t+2],[i],s=65,c=col,marker='s')
    ax.axvline(32,c='#333',ls='--',label='Model fit cutoff = day 32');ax.set(xlim=(27.5,37.5),xticks=np.arange(28,38),yticks=np.arange(8),yticklabels=[f'Row at day {t}' for t in days],xlabel='Calendar day',title='A row can precede the cutoff while its label is still unavailable');ax.invert_yaxis()
    ax.legend(handles=[plt.Line2D([],[],marker='o',color='#333',ls='',label='x and device at t'),plt.Line2D([],[],marker='s',color='#333',ls='',label='y at t + 2'),Patch(color=COLORS[0],label='Label available by 32'),Patch(color=COLORS[1],label='Label arrives after 32'),plt.Line2D([],[],ls='--',c='#333',label='Fit cutoff')],loc='upper center',bbox_to_anchor=(.5,-.14),ncol=3,frameon=False);save(fig,1)
    fig,ax=plt.subplots(1,2,figsize=(11.8,4.2),layout='constrained');h=result['hand'];xx=np.arange(4)
    ax[0].plot(xx,[v['y'] for v in h['rows']],'ko-',label='Observed training y');ax[0].plot(xx,[0]*4,'s--',c=COLORS[0],label='Initial prediction');ax[0].plot(xx,[v['next_prediction'] for v in h['rows']],'o-',c=COLORS[1],label='After one synchronized step');ax[0].set(xticks=xx,xticklabels=['H1 A x=0','H2 A x=1','H3 B x=0','H4 B x=1'],ylabel='Target / prediction',title='Same four training records')
    vals=[h['initial_objective'],h['next_objective']];bars=ax[1].bar(['Initial','One GD step'],vals,color=COLORS[:2]);ax[1].bar_label(bars,fmt='%.7f');ax[1].set(ylim=(0,4.3),ylabel='Ridge objective J',title='Mean half loss plus penalty / (2n)');ax[0].legend(loc='upper center',bbox_to_anchor=(.5,-.17),ncol=1,frameon=False);save(fig,2)
    fig,ax=plt.subplots(figsize=(11.8,4.4),layout='constrained');x=np.arange(4);w=.23
    for j,m in enumerate(e.MODELS):
        vals=[result['main'][k]['models'][m]['mse'] for k in e.SPLITS];b=ax.bar(x+(j-1)*w,vals,w,color=COLORS[j],label=LABELS[j]);ax.bar_label(b,fmt='%.3f',fontsize=9,padding=3)
    ax.set(xticks=x,xticklabels=SL,ylabel='Test MSE',ylim=(0,3.55),title='Panel 0: same raw panel, different evaluation targets');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.12),ncol=3,frameon=False);save(fig,3)
    fig,axes=plt.subplots(1,2,figsize=(11.8,4.5),layout='constrained')
    for ax,which in zip(axes,[['random','time'],['group','group_time']]):
        for kidx,k in enumerate(which):
            for j,m in enumerate(e.MODELS):
                vals=np.array([r['results'][k]['models'][m]['mse'] for r in result['replications']]);pos=kidx+(j-1)*.22;jitter=.055*np.sin(np.arange(len(vals))*2.4)
                ax.scatter(pos+jitter,vals,s=9,alpha=.35,color=COLORS[j]);q=result['summary'][k][m];ax.plot([pos,pos],[q['q10'],q['q90']],color=COLORS[j],lw=3);ax.plot(pos,q['mean_mse'],'D',color=COLORS[j],ms=6)
        ax.set(xticks=[0,1],xticklabels=[SL[e.SPLITS.index(k)] for k in which],ylabel='Test MSE',title='80 independently regenerated panels')
    fig.legend(handles=[plt.Line2D([],[],marker='D',c=c,label=l) for c,l in zip(COLORS,LABELS)],loc='outside lower center',ncol=3,frameon=False);save(fig,4)
    fig,ax=plt.subplots(1,2,figsize=(11.8,4.4),layout='constrained');tr,te=s['time'];bins=np.linspace(p['x'].min(),p['x'].max(),24)
    ax[0].hist(p['x'][tr],bins,density=True,histtype='step',lw=2,color=COLORS[0],label='Train days 0-30');ax[0].hist(p['x'][te],bins,density=True,histtype='step',lw=2,color=COLORS[1],label='Test days 33-47');ax[0].set(xlabel='Observed sensor x',ylabel='Density',title='Later inputs have shifted');ax[0].legend(frameon=False)
    for j,m in enumerate(e.MODELS):
        pred=np.array(result['main']['time']['models'][m]['predictions']);res=p['y'][te]-pred;days=np.unique(p['day'][te]);ax[1].plot(days,[res[p['day'][te]==t].mean() for t in days],'--s' if j==0 else '-o',ms=4 if j==0 else 3,c=COLORS[j],label=LABELS[j])
    ax[1].axhline(0,c='#333',ls='--',lw=1);ax[1].set(xlabel='Test day',ylabel='Mean residual y - prediction',title='Chronological drift remains without time term');ax[1].legend(loc='upper center',bbox_to_anchor=(.5,-.15),ncol=2,frameon=False);save(fig,5)
    fig,ax=plt.subplots(1,2,figsize=(11.8,4.3),layout='constrained');te=s['group'][1];g=p['group'][te];groups=np.unique(g);pred=np.array(result['main']['group']['models']['personalized']['predictions']);loss=(pred-p['y'][te])**2;means=np.array([loss[g==gg].mean() for gg in groups]);res=np.array([(p['y'][te]-pred)[g==gg].mean() for gg in groups])
    ax[0].bar(np.arange(15),means,color=COLORS[0]);ax[0].set(xticks=np.arange(15),xticklabels=groups,xlabel='Held-out device ID',ylabel='Mean test squared error',title='48 test rows per held-out device');ax[1].scatter(d['a'][0,groups],res,c=COLORS[1]);
    ax[1].axhline(0,c='#888',lw=.8);ax[1].set(xlabel='Simulator latent device offset (not a model input)',ylabel='Mean residual y - prediction',title='Unknown device offsets remain unlearned');save(fig,6)
    fig,ax=plt.subplots(figsize=(11.8,4.3),layout='constrained');b=result['bootstrap'];both=b['row_draws']+b['group_draws'];bins=np.linspace(min(both),max(both),35)
    ax.hist(b['row_draws'],bins,density=True,alpha=.45,color=COLORS[0],label=f'720 independent rows assumed; SD {b["row_sd"]:.3f}');ax.hist(b['group_draws'],bins,density=True,alpha=.45,color=COLORS[1],label=f'15 whole devices resampled; SD {b["group_sd"]:.3f}')
    ax.axvline(result['main']['group']['models']['personalized']['mse'],c='#333',ls='--',label='Original test MSE');ax.set(xlabel='Bootstrap test MSE (no refitting)',ylabel='Density',title='Same fitted personalized model; 800 resamples per method');ax.legend(frameon=False);save(fig,7)
    fig,ax=plt.subplots(1,2,figsize=(11.8,4.2),layout='constrained');x=np.arange(2)
    ax[0].bar(x-.16,[1,9],.32,color=COLORS[0],label='Model A');ax[0].bar(x+.16,[4,4],.32,color=COLORS[1],label='Model B');ax[0].set(xticks=x,xticklabels=['Device U: 8 rows','Device V: 2 rows'],ylabel='Mean loss within device',title='Fixed illustrative losses; no refitting')
    ax[1].bar(x-.16,[2.6,5],.32,color=COLORS[0],label='Model A');ax[1].bar(x+.16,[4,4],.32,color=COLORS[1],label='Model B');ax[1].set(xticks=x,xticklabels=['Uniform record','Uniform device'],ylabel='Weighted MSE',title='A wins per record; B wins per device')
    for aa in ax:
        for container in aa.containers:aa.bar_label(container,fmt='%.1f',padding=2)
        aa.margins(y=.18)
    fig.legend(handles=[Patch(color=COLORS[0],label='Model A'),Patch(color=COLORS[1],label='Model B')],loc='outside lower center',ncol=2,frameon=False);save(fig,8)
    fig,ax=plt.subplots(figsize=(11.8,4.7),layout='constrained');ts=[27,28,30,33,34]
    for i,t in enumerate(ts):
        ax.broken_barh([(t-3,3)],(i-.22,.44),facecolors=COLORS[0]);ax.plot(t,i,'o',c=COLORS[0]);ax.broken_barh([(t+1,1)],(i-.22,.44),facecolors=COLORS[1]);ax.scatter([t+1,t+2],[i,i],s=15,c=COLORS[1]);ax.plot([t,t+1],[i,i],c='#888',lw=1)
    ax.axvline(32,c='#333',ls='--');ax.axvspan(30,32,color='#f1ca57',alpha=.25);ax.set(yticks=np.arange(5),yticklabels=['Train t=27 (disjoint)','Train t=28 (touches)','Train t=30 (overlap)','Test t=33','Test t=34'],xlabel='Raw observation day',xticks=np.arange(24,37),xlim=(23.5,36.5),title='Separate window example: x uses [t-3,t], target uses days t+1,t+2');ax.invert_yaxis();ax.legend(handles=[Patch(color=COLORS[0],label='Historical features'),Patch(color=COLORS[1],label='Target days'),Patch(color='#f1ca57',alpha=.5,label='First test history shared with training support'),plt.Line2D([],[],c='#333',ls='--',label='Fit cutoff 32')],loc='upper center',bbox_to_anchor=(.5,-.14),ncol=2,frameon=False);save(fig,9)
    return [str(directory/(n+'.png')) for n in NAMES]
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',default=str(ROOT/'experiment-result.json'));p.add_argument('--directory',default=str(ROOT/'outputs/figures'));a=p.parse_args();print('\n'.join(render_all(json.loads(Path(a.report).read_text()),a.directory)))
