"""Ten actual-data illustrations; no cherry-picked seeds or omitted negative scores."""
from pathlib import Path
import argparse,json,os,tempfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import experiment as e
ROOT=Path(__file__).resolve().parent
NAMES=['01_fit_boundaries','02_numeric_processing','03_hand_preprocessing','04_hand_gradients','05_scaling_differences','06_affine_control','07_feature_selection','08_encoder_hand','09_encoding_scores','10_inner_provenance']
C=['#206b8a','#d36b32','#567d46','#8061a2']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','figure.dpi':150,'savefig.dpi':150})

def render_all(r,directory):
    directory=Path(directory);e.safe_output(directory/'probe.png');directory.mkdir(parents=True,exist_ok=True);cfg,d,s=e.load_data()
    def save(fig,i):
        path=e.safe_output(directory/(NAMES[i]+'.png'))
        with tempfile.NamedTemporaryFile(dir=directory,suffix='.png',delete=False) as f:tmp=f.name
        try:fig.savefig(tmp,bbox_inches='tight',facecolor='white',metadata={'Software':'ML052 deterministic plot'});os.replace(tmp,path)
        finally:
            plt.close(fig)
            if os.path.exists(tmp):os.unlink(tmp)
    fig,ax=plt.subplots(1,2,figsize=(11.8,4.5),layout='constrained');mat=np.zeros((4,48),int)
    for k in range(4):mat[k,np.unique(d['group'][s[f'group_{k}_valid']])]=1
    ax[0].imshow(mat,aspect='auto',cmap=ListedColormap(C[:2]),interpolation='nearest',vmin=0,vmax=1);ax[0].set(xlabel='Device ID: all 6 rows move together',ylabel='Outer fold',yticks=range(4),title='48 devices: 36 train / 12 validate');ax[1].axis('off');ax[1].set_title('Inside each outer training boundary')
    lines=['216 training rows -> median imputer','-> fitted 5% / 95% clipping bounds','-> fitted mean / population scale','+ raw missing flags + train one-hot','-> Ridge fit on training y','','72 held-out rows -> transform only','-> predict -> compare to held-out y']
    for i,line in enumerate(lines):ax[1].text(.02,.93-i*.12,line,transform=ax[1].transAxes,fontsize=11,color=C[0] if i<5 else C[1])
    fig.legend(handles=[Patch(color=C[0],label='Training devices'),Patch(color=C[1],label='Held-out devices')],loc='outside lower center',ncol=2,frameon=False);save(fig,0)
    fig,ax=plt.subplots(1,2,figsize=(11.8,4.3),layout='constrained');tr=s['group_0_train'];raw=d['numeric'][0,tr,0];st=r['replications'][0]['group'][0]['preprocessing_state'];im=np.where(np.isnan(raw),st['medians'][0],raw);cl=np.clip(im,st['clip_lower'][0],st['clip_upper'][0]);bins=np.linspace(np.nanmin(raw)-.5,np.nanmax(raw)+.5,30)
    ax[0].hist(raw[~np.isnan(raw)],bins,alpha=.42,color=C[0],label='Observed nonmissing');ax[0].hist(cl,bins,histtype='step',lw=2,color=C[1],label='Imputed then clipped')
    for bound in [st['clip_lower'][0],st['clip_upper'][0]]:ax[0].axvline(bound,c=C[2],ls='--')
    ax[0].set(xlabel='Sensor 1 (raw units)',ylabel='Training row count',title='Dataset 0 / fold 0; no rows dropped');ax[0].legend(frameon=False);v=[np.isnan(d['numeric'][0,tr,j]).sum() for j in range(2)];bars=ax[1].bar(['Sensor 1','Sensor 2'],v,color=C[:2]);ax[1].bar_label(bars);ax[1].set(ylabel='Missing rows among 216 training rows',title='Flags are built from each raw row',ylim=(0,max(v)+10));save(fig,1)
    fig,ax=plt.subplots(1,2,figsize=(11.8,4.2),layout='constrained');h=r['hand'];x=np.arange(4);v=[q['raw_x'] if q['raw_x'] is not None else np.nan for q in h['rows']];ax[0].plot(x,v,'o',ms=9,c=C[0],label='Raw observed');ax[0].plot(x,[q['imputed'] for q in h['rows']],'x--',ms=10,c=C[1],label='Train median imputed');ax[0].annotate('H2 missing -> 3',(1,3),xytext=(.5,4.3),arrowprops={'arrowstyle':'->'});ax[0].axhline(3,c='#777',ls=':');ax[0].set(xticks=x,xticklabels=['H1','H2','H3','H4'],ylabel='Raw / imputed x',title='Median 3; clipping bounds [1, 5]');ax[0].legend(loc='lower right',frameon=False)
    bars=ax[1].bar(x,[q['scaled'] for q in h['rows']],color=C[0]);ax[1].axhline(0,c='#777');ax[1].bar_label(bars,fmt='%.4f',padding=3);ax[1].set(xticks=x,xticklabels=['H1','H2','H3','H4'],ylabel='z = (clipped x - 3) / sqrt(2)',title='Train mean 3; variance denominator 4',ylim=(-1.85,1.85));save(fig,2)
    fig,ax=plt.subplots(1,2,figsize=(11.8,4.3),layout='constrained');loc=np.array([q['local_derivatives'] for q in h['rows']]);im=ax[0].imshow(loc,cmap='RdBu_r',vmin=-8,vmax=8,aspect='auto')
    for i in range(4):
        for j in range(5):ax[0].text(j,i,f'{loc[i,j]:.3f}',ha='center',va='center',color='white' if abs(loc[i,j])>4 else 'black',fontsize=9)
    ax[0].set(xticks=range(5),xticklabels=['b','w','v_A','v_B','u_missing'],yticks=range(4),yticklabels=['H1','H2','H3','H4'],title='Local derivatives before averaging');fig.colorbar(im,ax=ax[0],shrink=.75);ax[1].plot(x,[1,2,2,5],'ko-',label='y');ax[1].plot(x,[q['next_prediction'] for q in h['rows']],'o-',c=C[1],label='One synchronous step');ax[1].set(xticks=x,xticklabels=['H1','H2','H3','H4'],ylabel='Target / prediction',title='Mean half loss: 4.25 -> 2.18875');ax[1].legend(frameon=False);save(fig,3)
    fig,ax=plt.subplots(1,2,figsize=(11.8,4.2),layout='constrained');su=r['summary']['group'];a=np.array(su['correct_mse']['per_dataset']);b=np.array(su['global_scale_mse']['per_dataset']);x=np.arange(12);ax[0].plot(x,a,'o-',c=C[0],label='Train-only scaler');ax[0].plot(x,b,'x--',c=C[1],label='Forbidden global-X scaler');ax[0].set(xlabel='Independent dataset',ylabel='Four-fold MSE',title='Nearly equal scores; different fitted scope',xticks=x);ax[0].legend(frameon=False);dif=b-a;ax[1].bar(x,dif,color=[C[1] if v<0 else C[0] for v in dif]);ax[1].axhline(0,c='#555');ax[1].set(xlabel='Independent dataset',ylabel='Global minus train-only MSE',title='8 lower, 4 higher: no universal direction',xticks=x);save(fig,4)
    fig,ax=plt.subplots(1,2,figsize=(11.8,4.2),layout='constrained');q=r['affine_control'];ax[0].scatter(q['train_x'],q['train_y'],c=C[0],label='Training observations');ax[0].plot(q['valid_x'],q['predictions'][0],'o-',c=C[1],label='Train-only scale OLS');ax[0].plot(q['valid_x'],q['predictions'][1],'x--',ms=10,c=C[2],label='Global scale OLS');ax[0].set(xlabel='Raw x',ylabel='y / prediction',title='Same predictions: affine coordinates');ax[0].legend(frameon=False);st=q['scaler_states'];ax[1].bar(np.arange(2)-.17,[v['mean'][0] for v in st],.34,color=C[0],label='Learned mean');ax[1].bar(np.arange(2)+.17,[v['scale'][0] for v in st],.34,color=C[1],label='Learned scale');ax[1].set(xticks=[0,1],xticklabels=['Train only n=4','Train + held-out n=6'],ylabel='Raw units',title='Different state; equal OLS function');ax[1].axhline(0,c='#888');ax[1].legend(frameon=False);save(fig,5)
    fig,ax=plt.subplots(1,2,figsize=(11.8,4.2),layout='constrained');su=r['summary']['selection']
    for j,key in enumerate(['correct_mse','global_selection_mse','train_mean_baseline_mse']):ax[0].plot(range(12),su[key]['per_dataset'],'o-',ms=4,c=C[j],label=['Fold-local top 5','Global-label top 5','Training mean'][j])
    ax[0].set(xlabel='Independent noise dataset',ylabel='Four-fold MSE',title='300 noise features; choose 5 by correlation');ax[0].legend(frameon=False);X=d['selection_X'][0];y=d['selection_y'][0];tr=s['selection_0_train'];va=s['selection_0_valid'];top=r['replications'][0]['selection'][0]['global_features'];x=np.arange(5)
    for j,(ids,label) in enumerate([(tr,'Training rows'),(va,'Held-out rows')]):ax[1].bar(x+(j-.5)*.34,[abs(np.corrcoef(X[ids,k],y[ids])[0,1]) for k in top],.34,color=C[j],label=label)
    ax[1].set(xticks=x,xticklabels=top,xlabel='Globally selected feature ID',ylabel='Absolute observed correlation',title='Held-out y helped select these features');ax[1].legend(frameon=False);save(fig,6)
    fig,ax=plt.subplots(1,2,figsize=(11.8,4.4),layout='constrained');eh=r['encoder_hand'];x=np.arange(8);ax[0].plot(x,eh['y'],'ko:',label='Training labels');ax[0].plot(x,eh['crossfit_training'],'s-',c=C[0],label='fit_transform: inner-held-out');ax[0].plot(x,eh['full_training_transform'],'x--',c=C[1],label='fit then transform: own target included');ax[0].set(xticks=x,xticklabels=[f'{i+1}:{v}' for i,v in enumerate(eh['categories'])],xlabel='Training row : category',ylabel='Encoded value / y',title='Same data, different training features');ax[0].legend(loc='upper center',bbox_to_anchor=(.5,-.18),fontsize=9,frameon=False);bars=ax[1].bar(eh['valid_categories'],eh['valid_transform'],color=[C[0],C[0],C[2]]);ax[1].bar_label(bars,fmt='%.2f');ax[1].axhline(4.5,c='#555',ls='--',label='Full training mean 4.5');ax[1].set(ylabel='Validation encoding',title='Validation uses all 8 training labels',ylim=(0,6.3));ax[1].legend(frameon=False);save(fig,7)
    fig,ax=plt.subplots(1,2,figsize=(11.8,4.5),layout='constrained');su=r['summary']['encoding'];keys=['correct_mse','self_encoded_mse','global_target_mse','train_mean_baseline_mse'];labels=['Inner cross-fit','Self-encoded train','Forbidden all-label encoding','Training mean']
    for j,key in enumerate(keys):ax[0].plot(range(12),su[key]['per_dataset'],'o-',ms=4,c=C[j],label=labels[j])
    ax[0].set(xlabel='Independent categorical dataset',ylabel='Four-fold validation MSE',title='Category has no population signal');ax[0].legend(loc='upper center',bbox_to_anchor=(.5,-.16),ncol=2,fontsize=9,frameon=False);blocks=r['replications'][0]['encoding'];means=[np.mean([b[k] for b in blocks]) for k in ['correct_training_mse','correct_mse','self_encoded_training_mse','self_encoded_mse']];bars=ax[1].bar([0,1,3,4],means,color=[C[0],C[0],C[1],C[1]]);ax[1].bar_label(bars,fmt='%.3f');ax[1].set(xticks=[0,1,3,4],xticklabels=['CF train','CF valid','Self train','Self valid'],ylabel='MSE',title='Dataset 0: train optimism vs validation',ylim=(0,max(means)*1.2));save(fig,8)
    fig,ax=plt.subplots(figsize=(11.8,4.1),layout='constrained');b=r['replications'][0]['encoding'][0];tr=b['train_row_ids'];matrix=np.zeros((4,len(tr)),int)
    for k,p in enumerate(b['inner_provenance']):matrix[k]=[1 if i in p['fit_row_ids'] else 2 for i in tr]
    matrix[3]=1;ax.imshow(matrix,aspect='auto',interpolation='nearest',cmap=ListedColormap(['#eee',C[0],C[1]]),vmin=0,vmax=2);ax.set(yticks=range(4),yticklabels=['Inner fold 0','Inner fold 1','Inner fold 2','Validation transform'],xlabel='Position among 270 outer-training rows (outer-validation rows absent)',title='Labels supplying the maps: 180 per inner fit; 270 for validation',xticks=[0,45,90,135,180,225,269]);fig.legend(handles=[Patch(color=C[0],label='Label allowed in this fitted map'),Patch(color=C[1],label='Row encoded; its label excluded from this map')],loc='outside lower center',ncol=2,frameon=False);save(fig,9)
    return [str(directory/(n+'.png')) for n in NAMES]
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',default=str(ROOT/'experiment-result.json'));p.add_argument('--directory',default=str(ROOT/'outputs/figures'));a=p.parse_args();print('\n'.join(render_all(json.loads(Path(a.report).read_text()),a.directory)))
