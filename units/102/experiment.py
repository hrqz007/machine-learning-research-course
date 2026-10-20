from lm_common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    parts=corpus();tok=Tokenizer(parts['train'],24);tok.save();runs=[]
    # Full factorial width x unique-data size; same steps and batch document draws.
    for seed in [7,19,31]:
        for width in [12,24,48]:
            for count in [120,1800]:
                model,r=train(parts['train'][:count],parts['validation'],tok,width,80,seed)
                r['compute_proxy_6ND']=6*r['parameters']*r['processed_target_tokens'];runs.append(r)
    # Validation chooses the setting. Test is opened only for that fixed setting.
    settings=[(w,d) for w in [12,24,48] for d in [120,1800]]
    means={f'{w}-{d}':float(np.mean([r['validation']['nll'] for r in runs if (r['width'],r['unique_train_docs'])==(w,d)])) for w,d in settings}
    best=min(settings,key=lambda z:means[f'{z[0]}-{z[1]}'])
    model,selected=train(parts['train'][:best[1]],parts['validation'],tok,best[0],80,7)
    selected['test']=evaluate(model,parts['test'],tok)
    xs=[];ys=[]
    for w in [12,24,48]:
        group=[r for r in runs if r['width']==w and r['unique_train_docs']==1800];xs.append(group[0]['parameters']);ys.append(np.mean([r['validation']['nll'] for r in group]))
    slope,intercept=np.polyfit(np.log(xs),np.log(ys),1)
    result={'lesson':'102','device':'cpu','seeds':[7,19,31],'split_sizes':{k:len(v) for k,v in parts.items()},'runs':runs,'setting_mean_validation_nll':means,'selected_setting':{'width':best[0],'unique_docs':best[1]},'selected_seed7_test':selected['test'],'descriptive_log_fit':{'slope':float(slope),'intercept':float(intercept),'points':3,'irreducible_floor_assumed':0,'warning':'Descriptive interpolation only; not a scaling-law replication.'},'compute_proxy_warning':'6ND is a rough dense-training proxy, not measured GRU FLOPs or wall time.'}
    (ROOT/'outputs/results.json').write_text(json.dumps(result,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    for count,color in [(120,'#bf7829'),(1800,'#247eac')]:
        means=[];stds=[]
        for w in [12,24,48]:
            group=[r['validation']['nll'] for r in runs if r['width']==w and r['unique_train_docs']==count];means.append(np.mean(group));stds.append(np.std(group,ddof=1))
        axes[0].errorbar(xs,means,yerr=stds,marker='o',capsize=4,label=f'{count} unique documents',color=color)
    axes[0].set(xlabel='Trainable parameters',ylabel='Validation nats/token',title='80 steps; mean ± sample SD, 3 seeds');axes[0].legend()
    for r in runs:
        if r['seed']==7 and r['unique_train_docs']==1800:axes[1].plot([h['tokens'] for h in r['history']],[h['validation_nll'] for h in r['history']],'-o',label=f"width={r['width']}")
    axes[1].set(xlabel='Processed target tokens',ylabel='Validation nats/token',title='Learning curves: seed 7, full training data');axes[1].legend();fig.tight_layout();fig.savefig(ROOT/'figures/results.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,3));ax.axis('off')
    labels=['Unique documents\nD_unique','Repeated sampling\nD_processed','Model width\nN parameters','Loss + time\nmeasured outcomes']
    for i,label in enumerate(labels):
        ax.text(.12+i*.25,.5,label,ha='center',va='center',bbox={'boxstyle':'round','facecolor':['#e6f1ff','#fff0d8','#e6f1ff','#e3f6e9'][i]})
        if i<3:ax.annotate('',xy=(.26+i*.25,.5),xytext=(.21+i*.25,.5),arrowprops={'arrowstyle':'->'})
    ax.set_title('Budget audit: same steps are NOT the same compute budget');fig.tight_layout();fig.savefig(ROOT/'figures/mechanism.png',dpi=160);plt.close(fig)
    print(json.dumps({'runs':len(runs),'mean_validation_nll':result['setting_mean_validation_nll'],'selected_setting':result['selected_setting'],'selected_test':selected['test'],'fit':result['descriptive_log_fit']},indent=2))
if __name__=='__main__':
    main()
    import runpy
    runpy.run_path(str(ROOT/'draw_mechanism.py'))
