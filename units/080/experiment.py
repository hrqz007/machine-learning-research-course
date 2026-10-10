"""冻结协议：三种标注比例×三种噪声×五个种子；测试仅作最终审计。
比例/噪声/阈值均预先写定，不选择测试集上表现最好的设置。
"""
from pathlib import Path
import argparse,json
import numpy as np
from generate_data import SEEDS
from supervision import *
ROOT=Path(__file__).resolve().parent

def select_labels(y,ratio,noise,seed):
    y=labels(y,len(y));rng=np.random.default_rng(seed+1000)
    if not 0<ratio<=1 or not 0<=noise<=1:raise ValueError('ratio or noise out of range')
    # 嵌套抽样：每类最少2条，先各自打乱，增大比例只增加标注。
    ids=np.concatenate([rng.permutation(np.flatnonzero(y==c))[:max(2,int(round(len(y)*ratio/2)))] for c in [0,1]])
    observed=y[ids].copy();flip=rng.random(len(ids))<noise;observed[flip]=1-observed[flip]
    # 不重新抽样以制造好结果：如果噪声让单类消失，状态必须报告。
    return ids,observed,flip

def weak_experiment(X,y):
    X=features(X);y=labels(y,len(X));rule1=np.where(np.abs(X[:,0])>.6,(X[:,0]>0).astype(int),-1)
    rule2=np.where(np.abs(X[:,0]+.7*X[:,1])>.6,(X[:,0]+.7*X[:,1]>0).astype(int),-1)
    rule3=np.where(np.abs(X[:,1])>1.,(X[:,1]>0).astype(int),-1)
    L=np.column_stack([rule1,rule2,rule3]);vote=majority_vote(L);covered=vote!=-1
    base=independent_label_probability(np.column_stack([rule1,rule2]),[.8,.8]);dup=independent_label_probability(np.column_stack([rule1,rule1,rule2]),[.8,.8,.8])
    disagreement=(rule1!=-1)&(rule2!=-1)&(rule1!=rule2)
    return {'coverage':float(np.mean(covered)),'abstention_rate':float(np.mean(~covered)),'covered_accuracy':float(np.mean(vote[covered]==y[covered])),'conflict_rate':float(np.mean(((L==0).any(axis=1))&((L==1).any(axis=1)))),'rule_coverage':[float(np.mean(L[:,j]!=-1)) for j in range(3)],'rule_accuracy':[float(np.mean(L[L[:,j]!=-1,j]==y[L[:,j]!=-1])) for j in range(3)],'duplicate_conflict_rows':int(disagreement.sum()),'independent_conflict_mean_confidence':float(np.mean(np.maximum(base[disagreement],1-base[disagreement]))),'duplicate_conflict_mean_confidence':float(np.mean(np.maximum(dup[disagreement],1-dup[disagreement]))),'assumed_accuracy':.8,'assumption':'given symmetric class accuracy; conditional independence; ignorable abstention; not learned from hidden labels'}

def stress_experiment():
    # 刻意翻转两个种子标签；审计真值始终与x符号一致。
    Xl=np.array([[-1.2],[1.2]]);yl=np.array([1,0]);u=np.r_[np.linspace(-3,-1.5,60),np.linspace(1.5,3,60)][:,None];truth=(u[:,0]>0).astype(int)
    fit=self_train(Xl,yl,u,threshold=.8,pseudo_weight=.5,rounds=5,l2=.05)
    rounds=[]
    for i,model in enumerate(fit['models']):
        prob=predict_proba(model,u);wrong=(prob>=.5)!=truth
        rounds.append({'stage':i,'accuracy':float(np.mean(~wrong)),'mean_wrong_confidence':float(np.mean(np.maximum(prob,1-prob)[wrong])),'coef':model['coef'].tolist()})
    return {'description':'deliberately inverted seed labels; same sign-based underlying target','accepted':int(fit['accepted'].sum()),'wrong_accepted':int(np.sum(fit['pseudo_labels'][fit['accepted']]!=truth[fit['accepted']])),'stages':rounds}

def run():
    rows=[];ratios=[.02,.10,.30];noises=[0.,.25,.5]
    for seed in SEEDS:
        visible=json.loads((ROOT/f'data/seed-{seed}-features.json').read_text());audit=json.loads((ROOT/f'data/seed-{seed}-audit.json').read_text());X=np.array(visible['pool_X']);truth=np.array(audit['pool_y']);test=np.array(visible['test_X']);testy=np.array(audit['test_y'])
        for ratio in ratios:
            for noise in noises:
                ids,y,flips=select_labels(truth,ratio,noise,seed);mask=np.ones(len(X),bool);mask[ids]=False;U=X[mask];uy=truth[mask]
                if len(np.unique(y))<2:
                    rows.append({'seed':seed,'ratio':ratio,'noise':noise,'status':'single_observed_class','label_count':len(ids)});continue
                fit=self_train(X[ids],y,U,threshold=.8,pseudo_weight=.5,rounds=5,l2=.05)
                base=evaluate(fit['models'][0],test,testy);final=evaluate(fit['model'],test,testy);accepted=fit['accepted'];pseudo=fit['pseudo_labels'];n=int(accepted.sum());wrong=int(np.sum(pseudo[accepted]!=uy[accepted]));audit_history=[]
                for h in fit['history']:
                    new=np.array(h['new_ids'],dtype=int);labs=np.array(h['new_labels'],dtype=int)
                    audit_history.append({'round':h['round'],'accepted_total':h['accepted_total'],'new_count':len(new),'wrong_new':int(np.sum(labs!=uy[new])),'reason':h['reason']})
                rows.append({'seed':seed,'ratio':ratio,'noise':noise,'status':'ok','label_count':len(ids),'actual_noise_fraction':float(np.mean(flips)),'supervised':base,'pseudo':final,'accuracy_change':final['accuracy']-base['accuracy'],'accepted':n,'wrong_accepted':wrong,'pseudo_error_rate':wrong/n if n else None,'history':audit_history})
    summaries=[]
    for ratio in ratios:
        for noise in noises:
            group=[r for r in rows if r['ratio']==ratio and r['noise']==noise and r['status']=='ok'];a=np.array([r['supervised']['accuracy'] for r in group]);b=np.array([r['pseudo']['accuracy'] for r in group]);delta=b-a
            summaries.append({'ratio':ratio,'noise':noise,'valid_seeds':len(group),'supervised_mean_accuracy':float(a.mean()),'pseudo_mean_accuracy':float(b.mean()),'mean_accuracy_change':float(delta.mean()),'change_std_ddof1':float(delta.std(ddof=1)) if len(group)>1 else None,'mean_pseudo_error_rate':float(np.mean([r['pseudo_error_rate'] for r in group if r['pseudo_error_rate'] is not None])) if any(r['pseudo_error_rate'] is not None for r in group) else None})
    first=json.loads((ROOT/'data/seed-7-features.json').read_text());truth=json.loads((ROOT/'data/seed-7-audit.json').read_text())
    return {'protocol':{'ratios':ratios,'nominal_noise_probabilities':noises,'seeds':SEEDS,'threshold':.8,'pseudo_weight':.5,'rounds':5,'l2':.05,'test_selection':'none; all predetermined conditions reported','validation_usage':'reserved; no tuning in frozen experiment'},'summaries':summaries,'runs':rows,'stress':stress_experiment(),'weak_labels':weak_experiment(first['pool_X'],truth['pool_y'])}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(ROOT/'outputs/result.json'));a=p.parse_args();r=run();out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)+'\n');print(json.dumps({'summaries':r['summaries'],'stress':r['stress'],'weak_labels':r['weak_labels']},ensure_ascii=False,indent=2))
