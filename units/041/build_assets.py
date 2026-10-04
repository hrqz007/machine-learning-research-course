"""Original explanation figures rendered from actual result objects."""
from pathlib import Path
import argparse
import json
import os
import tempfile
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'unit041-mpl'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import experiment as e

COL={'gd':'#2563eb','diagonal_gd':'#0d9488','minibatch':'#d97706','adam':'#8b5cf6','newton':'#1f2937'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.22,'figure.facecolor':'white','axes.titleweight':'bold','savefig.dpi':160})

def _save(fig,out,name):
    fig.tight_layout(pad=1.4,rect=(0,0,1,.89) if getattr(fig,'_top_legend',False) else (0,0,1,1));fig.savefig(out/name,bbox_inches='tight');plt.close(fig);return out/name

def _box(ax,x,y,w,h,text,color):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.025',facecolor=color,edgecolor='white'));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10)

def build_figures(r,timing,out):
    out=Path(out)
    if out.resolve()==(e.ROOT/'figures').resolve():raise ValueError('render to a separate output directory; shipped figures protected')
    out.mkdir(parents=True,exist_ok=True)
    names=['01_evidence_chain.png','02_hand_chain.png','03_exact_and_iterative.png','04_gradient_and_error.png','05_gradient_checks.png','06_cost_convergence.png','07_measured_cost.png','08_selection_lock.png','09_interval_objects.png','10_coverage_failure.png','11_preconditioner.png','12_failed_attempt.png']
    for name in names:e.check_destination(out/name)
    paths=[]
    fig,ax=plt.subplots(figsize=(10,3.5));ax.axis('off');ax.set_xlim(0,10);ax.set_ylim(0,3)
    items=[('Same rows\nSame objective','#dbeafe'),('Exact arithmetic\nIndependent solver','#ccfbf1'),('Residual + gap\nFailure + cost','#fef3c7'),('Validation lock\nOne test report','#ede9fe')]
    for i,(text,col) in enumerate(items):
        _box(ax,.1+i*2.5,1.35,2.1,1.15,text,col)
        if i<3:ax.annotate('',(.1+(i+1)*2.5,1.92),(2.25+i*2.5,1.92),arrowprops={'arrowstyle':'->','lw':2,'color':'#64748b'})
    ax.text(5,.8,'Numerical correctness   /   Predictive evidence   /   Causal identification',ha='center',fontsize=12,color='#b91c1c');ax.text(5,.32,'Each needs its own assumptions. No arrow proves the next claim by itself.',ha='center',fontsize=10)
    paths.append(_save(fig,out,names[0]))
    fig,axes=plt.subplots(1,2,figsize=(10,3.9));st=r['hand']['states'];pos=np.arange(4);width=.23
    for k,a in enumerate(st):axes[0].bar(pos+(k-1)*width,[z['prediction'] for z in a['rows']],width,label='state '+str(k))
    axes[0].plot(pos,[z['y'] for z in st[0]['rows']],'ko',label='observed y');axes[0].set_xticks(pos,['row 1','row 2','row 3','row 4']);axes[0].set_ylabel('Prediction and label');axes[0].legend(ncol=2,fontsize=8);axes[0].set_title('The same four rows after each update')
    axes[1].plot(range(3),[a['data_loss'] for a in st],'o-',label='data loss');axes[1].plot(range(3),[a['penalty'] for a in st],'s-',label='penalty');axes[1].plot(range(3),[a['objective'] for a in st],'D-',label='total J');axes[1].set_xticks(range(3));axes[1].set_xlabel('Completed updates');axes[1].legend();axes[1].set_title('Every state has a new forward pass')
    paths.append(_save(fig,out,names[1]))
    fig,axes=plt.subplots(1,2,figsize=(9.7,4));u,v=np.meshgrid(np.linspace(-.2,1.3,160),np.linspace(-.2,1.5,160));H=np.array(r['hand']['H']);c=np.array(r['hand']['c'])
    for ax,lam,key in zip(axes,[0,.5],['ols','ridge']):
        J=1.75+.5*((H[0,0]+lam)*u*u+2*H[0,1]*u*v+(H[1,1]+lam)*v*v)-c[0]*u-c[1]*v
        ref=r['hand']['references'][key];ax.contour(u,v,J,levels=ref['objective']+np.array([.01,.04,.12,.3,.7,1.3]),colors='#94a3b8');ax.plot(*ref['w'],'*',ms=14,color='#dc2626',label='analytic reference')
        w=np.array([x['w'] for x in st]);ax.plot(w[:,0],w[:,1],'o-',label='Ridge hand path' if key=='ridge' else 'same Ridge path (different target)',color='#2563eb');ax.set(xlabel='w1',ylabel='w2',title=key.upper()+' objective');ax.legend(fontsize=8)
    paths.append(_save(fig,out,names[2]))
    fig,axes=plt.subplots(1,2,figsize=(9.7,3.8));q=np.linspace(-8,8,201)
    for mu,col in [(1.,'#2563eb'),(1e-4,'#d97706')]:
        axes[0].plot(q,.5*mu*q*q,label='curvature = '+str(mu),color=col);axes[1].plot(q,mu*q,label='curvature = '+str(mu),color=col)
    axes[0].set(xlabel='Parameter error along one eigenvector',ylabel='Objective gap',title='Same error; different objective gaps')
    axes[0].set_yscale('symlog',linthresh=1e-3)
    axes[1].set(xlabel='Parameter error along one eigenvector',ylabel='Gradient component',title='Small slope can hide a large error')
    axes[1].set_yscale('symlog',linthresh=1e-4)
    for ax in axes:ax.legend(fontsize=9)
    paths.append(_save(fig,out,names[3]))
    fig,ax=plt.subplots(figsize=(9,3.7));checks=r['hand']['gradient_checks']
    for point in [checks[j]['point'] for j in [0,5,10]]:
        rows=[z for z in checks if z['point']==point];ax.loglog([z['h'] for z in rows],[z['relative_error'] for z in rows],'o-',label='correct at '+str(point))
    rows=checks[:5];ax.loglog([z['h'] for z in rows],[z['wrong_sign_error'] for z in rows],'x--',color='#dc2626',label='wrong sign at first point');ax.set(xlabel='Central difference h (not learning rate)',ylabel='Normalized discrepancy',title='Three nonstationary points, one intentional sign failure');ax.legend(ncol=2,fontsize=8)
    paths.append(_save(fig,out,names[4]))
    fig,axes=plt.subplots(2,3,figsize=(10,6.4))
    for j,case in enumerate(e.CASES):
        for run in r['optimizers']['runs']:
            if run['case']!=case or run['lambda']!=0.:continue
            tr=run['trace'];x=[v['gradient_rows']/run['n'] for v in tr]
            for row,key in [(0,'quadratic_gap'),(1,'relative_gradient')]:
                yy=[max(v[key],1e-18) for v in tr];axes[row,j].loglog(x,yy,label=run['method'],color=COL[run['method']]);axes[row,j].plot(x[-1],yy[-1], 'x' if run['status']!='gradient_tolerance' else 'o',color=COL[run['method']],ms=6)
        axes[0,j].set_title(case+' / OLS');axes[1,j].axhline(r['configuration']['gradient_tolerance'],ls='--',color='#dc2626',lw=.9);axes[1,j].set_xlabel('Gradient row-equivalents / n')
    axes[0,0].set_ylabel('Stable quadratic gap');axes[1,0].set_ylabel('Relative full gradient');fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.945),ncol=5,fontsize=10);fig._top_legend=True
    for ax in axes.flat:
        ax.tick_params(labelsize=10);ax.xaxis.label.set_size(11);ax.yaxis.label.set_size(11);ax.title.set_size(12)
    fig.suptitle('o: gradient criterion reached     x: budget exhausted     lower floor for display: 1e-18',fontsize=10)
    paths.append(_save(fig,out,names[5]))
    fig,axes=plt.subplots(1,2,figsize=(10,4.4));runs=[v for v in r['optimizers']['runs'] if v['case']=='scaled' and v['lambda']==.1];idx=np.arange(5)
    records=[next(z for z in timing['records'] if z['case']=='scaled' and z['lambda']==.1 and z['method']==v['method']) for v in runs]
    for j,(run,z) in enumerate(zip(runs,records)):
        col=COL[run['method']];axes[0].barh(j,run['cost']['attempted_updates'],color=col);axes[1].barh(j,z['median_ns']/1e6,color=col)
        axes[1].plot([z['min_ns']/1e6,z['max_ns']/1e6],[j,j],'k-',lw=2);axes[1].text(z['max_ns']/1e6*1.12,j,'FAIL' if run['status']!='gradient_tolerance' else 'pass',va='center',fontsize=8)
    for ax in axes:ax.set_yticks(idx,[v['method'] for v in runs]);ax.set_xscale('log')
    axes[0].set(xlabel='Attempted updates (log scale)',title='An update is not an equal-cost unit');axes[1].set(xlabel='Milliseconds (median; line min-max)',title='Measured diagnostic solver call');axes[1].set_xlim(right=max(z['max_ns'] for z in records)/1e6*2.2)
    paths.append(_save(fig,out,names[6]))
    fig,axes=plt.subplots(1,2,figsize=(10,3.8));lock=r['selection']['lock'];cand=lock['candidates'];axes[0].plot(range(len(cand)),[z['validation_mse'] for z in cand],'o-',color='#2563eb');j=next(i for i,z in enumerate(cand) if z['lambda']==lock['selected_lambda']);axes[0].plot(j,cand[j]['validation_mse'],'*',ms=17,color='#dc2626');axes[0].set_xticks(range(len(cand)),[str(z['lambda']) for z in cand]);axes[0].set(xlabel='Predeclared lambda candidates',ylabel='Validation MSE',title='Validation chooses; the test does not')
    ax=axes[1];ax.axis('off');ax.set_xlim(0,5);ax.set_ylim(0,4)
    _box(ax,.2,2.9,1.8,.8,'Train 72\nfit means + scales','#dbeafe');_box(ax,2.6,2.9,2,.8,'Validation 48\nchoose lambda','#ccfbf1');_box(ax,1.4,1.5,2.2,.8,'Serialize choice\nSHA-256 lock','#fef3c7');_box(ax,1.4,.1,2.2,.8,'Test 120\nevaluate once','#ede9fe')
    ax.annotate('',(2.3,2.3),(1.2,2.9),arrowprops={'arrowstyle':'->'});ax.annotate('',(2.7,2.3),(3.6,2.9),arrowprops={'arrowstyle':'->'});ax.annotate('',(2.5,.9),(2.5,1.5),arrowprops={'arrowstyle':'->'})
    paths.append(_save(fig,out,names[7]))
    cov=r['coverage'];z=cov['rows'][:24];fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,key,title in [(axes[0],'mean_half','Interval for the conditional mean'),(axes[1],'prediction_half','Interval for one new observation')]:
        for i,a in enumerate(z):
            target=cov['true_mean'] if key=='mean_half' else a['new_y'];ok=abs(a['fitted']-target)<=a[key];color='#0d9488' if ok else '#dc2626'
            ax.plot([a['fitted']-a[key],a['fitted']+a[key]],[i,i],color=color);ax.plot(target,i,'k.',ms=3)
        ax.set(xlabel='Response scale; black dot = target',ylabel='Replication (first 24, unselected)',title=title)
    paths.append(_save(fig,out,names[8]))
    fig,axes=plt.subplots(1,2,figsize=(10,3.8));keys=['mean_covered','prediction_covered','wrong_mean_as_prediction','shifted_prediction_covered'];rates=[cov['summary'][k]['rate'] for k in keys];labels=['mean CI\ntrue mean','prediction PI\nIID future','mean CI\nIID future','prediction PI\nnoise shift'];axes[0].bar(range(4),rates,color=['#2563eb','#0d9488','#f59e0b','#dc2626']);axes[0].axhline(.95,color='black',ls='--',label='nominal 0.95');axes[0].set_xticks(range(4),labels,fontsize=8);axes[0].set(ylim=(0,1.1),ylabel='Coverage proportion',title='600 independent training replications')
    for i,val in enumerate(rates):axes[0].text(i,val+.02,f'{val:.3f}',ha='center')
    axes[0].legend(fontsize=8);xx=np.linspace(-5,5,300)
    for sd,col in [(.5,'#0d9488'),(1.5,'#dc2626')]:axes[1].plot(xx,np.exp(-xx**2/(2*sd*sd))/(sd*np.sqrt(2*np.pi)),label='future noise SD '+str(sd),color=col)
    axes[1].set(xlabel='New measurement noise',ylabel='Density',title='Training noise model: SD = 0.5');axes[1].legend(fontsize=9)
    paths.append(_save(fig,out,names[9]))
    fig,axes=plt.subplots(1,2,figsize=(10,3.8));runs=[z for z in r['optimizers']['runs'] if z['case']=='scaled' and z['lambda']==.1 and z['method'] in ('gd','diagonal_gd')]
    for z in runs:
        axes[0].semilogy([v['epoch'] for v in z['trace']],[max(v['quadratic_gap'],1e-18) for v in z['trace']],label=z['method'],color=COL[z['method']])
    axes[0].legend();axes[0].set(xlabel='Full-gradient updates',ylabel='Same original Ridge objective gap',title='Change the path; preserve the optimum')
    ax=axes[1];ax.axis('off');ax.text(.04,.85,r'$D=\mathrm{diag}(H_\lambda)$',fontsize=16);ax.text(.04,.59,r'$w^+=w-D^{-1}g/L_D$',fontsize=17);ax.text(.04,.32,'All gradients include the original penalty.\nNo new lambda, target, or feature data.',fontsize=11);ax.text(.04,.09,'Unadjusted standardization + isotropic Ridge\nwould generally change the objective.',fontsize=10,color='#b91c1c')
    paths.append(_save(fig,out,names[10]))
    failure=r['optimizers']['injected_failure'];fig,axes=plt.subplots(1,2,figsize=(10,3.6));axes[0].bar(['initial retained','candidate rejected'],[failure['final']['objective'],failure['failure']['objective']],color=['#2563eb','#dc2626']);axes[0].set(ylabel='Original Ridge objective',title='Unsafe GD: eta = 3 / L')
    cost=failure['cost'];labels=['attempted\nupdates','committed\nupdates','full objective\ncalls','full gradient\ncalls'];vals=[cost[k] for k in ['attempted_updates','committed_updates','objective_calls','full_gradient_calls']];axes[1].bar(labels,vals,color=['#d97706','#0d9488','#2563eb','#8b5cf6']);axes[1].set(ylim=(0,4),title='A rejected candidate still costs work')
    for i,v in enumerate(vals):axes[1].text(i,v+.1,str(v),ha='center')
    paths.append(_save(fig,out,names[11]));return paths


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    r=e.strict_json((e.ROOT/'experiment-result.json').read_text());timing=e.strict_json((e.ROOT/'benchmark-result.json').read_text());print(len(build_figures(r,timing,a.output_dir)),'figures built')
if __name__=='__main__':main()
