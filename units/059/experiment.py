"""Train-only pruning path, validation-only choice, test once choices are frozen."""
from pathlib import Path
import argparse,hashlib
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from common import ROOT,write_json
from pruning import export_tree,weakest_link_path,predict_probability,select_validation

def load(name):
    z=np.genfromtxt(ROOT/'data'/f'{name}.csv',delimiter=',',names=True)
    return np.column_stack([z[f'x{k}'] for k in range(4)]),z['y'].astype(int)
def score(y,p):return {'accuracy':float(np.mean((p>.5)==y)),'brier':float(np.mean((p-y)**2))}
def fit(X,y,**kw):return DecisionTreeClassifier(random_state=59059,**kw).fit(X,y)
def run():
    X,y=load('train');V,v=load('validation')
    grown=fit(X,y);tree=export_tree(grown);path=weakest_link_path(tree)
    for row in path:
        p=predict_probability(tree,V,row['leaves']);row['validation_brier']=score(v,p)['brier']
        row['train_brier']=score(y,predict_probability(tree,X,row['leaves']))['brier']
    post_index=select_validation(path);post=path[post_index]
    grid=[];models=[]
    for depth in [1,2,3,4,6,None]:
        for leaf in [1,5,15]:
            m=fit(X,y,max_depth=depth,min_samples_leaf=leaf);models.append(m)
            grid.append({'max_depth':depth,'min_samples_leaf':leaf,'depth':m.get_depth(),'n_leaves':m.get_n_leaves(),
                         'validation_brier':score(v,m.predict_proba(V)[:,1])['brier']})
    pre_index=select_validation(grid);pre=grid[pre_index]
    choices={'pre_index':pre_index,'post_index':post_index,'pre':pre,'post_alpha':post['alpha'],'selection':'validation Brier; exact ties fewer leaves, lower depth, earlier row','refit':False}
    # Choices above do not have access to the next read or any test labels.
    T,t=load('test');evaluation={}
    for name,p,d,l in [('unrestricted',grown.predict_proba(T)[:,1],grown.get_depth(),grown.get_n_leaves()),
                        ('pre',models[pre_index].predict_proba(T)[:,1],pre['depth'],pre['n_leaves']),
                        ('post',predict_probability(tree,T,post['leaves']),post['depth'],post['n_leaves'])]:
        evaluation[name]={**score(t,p),'depth':d,'n_leaves':l,'probability':p.tolist()}
    evaluation['constant']={**score(t,np.full(len(t),y.mean())),'probability':float(y.mean())}
    # This probe has no labels, and never participates in model selection.
    axis=np.linspace(-2,2,51);a,b=np.meshgrid(axis,axis);probe=np.column_stack([a.ravel(),b.ravel(),a.ravel(),np.zeros(a.size)])
    perturb={};rng=np.random.default_rng(59159);sets=[np.sort(rng.choice(len(y),size=427,replace=False)) for _ in range(25)]
    for name,kw in [('unrestricted',{}),('pre',{'max_depth':pre['max_depth'],'min_samples_leaf':pre['min_samples_leaf']}),('post',{'ccp_alpha':post['alpha']})]:
        ps=[];roots=[];leaves=[];depths=[]
        for ids in sets:
            m=fit(X[ids],y[ids],**kw);ps.append(m.predict_proba(probe)[:,1]);roots.append(int(m.tree_.feature[0]));leaves.append(m.get_n_leaves());depths.append(m.get_depth())
        ps=np.array(ps);h=ps>.5;freq=h.mean(axis=0)
        perturb[name]={'probability_variance_mean':float(ps.var(axis=0).mean()),'pair_disagreement':float((2*freq*(1-freq)*25/24).mean()),'root_features':roots,'n_leaves':leaves,'depths':depths,'mean_probability':ps.mean(axis=0).tolist(),'sd_probability':ps.std(axis=0).tolist()}
    library=grown.cost_complexity_pruning_path(X,y)
    # Compare interior alphas, avoiding ambiguous exactly-at-breakpoint ties.
    checks=[]
    for k,r in enumerate(path):
        alpha=(r['alpha']+path[k+1]['alpha'])/2 if k+1<len(path) else r['alpha']+.01
        m=fit(X,y,ccp_alpha=alpha);p=m.predict_proba(T)[:,1]
        checks.append({'alpha':alpha,'n_leaves':m.get_n_leaves(),'max_prediction_difference':float(np.max(np.abs(p-predict_probability(tree,T,r['leaves']))))})
    return {'lesson':'059','seed':59059,'sizes':{'train':450,'validation':225,'test':225},'path':path,'pre_grid':grid,'choices':choices,'evaluation':evaluation,'stability':perturb,'perturbation':{'replicates':25,'kept_rows':427,'row_sets':[q.tolist() for q in sets],'probe_axis':axis.tolist(),'hyperparameters':'frozen from original training/validation selection'},'library':{'alphas':library.ccp_alphas.tolist(),'risks':library.impurities.tolist(),'interior_comparisons':checks},'hand':{'risks':[.14,.32,.50],'leaves':[4,2,1],'alphas':[0,.09,.18]},'data_sha256':{n:hashlib.sha256((ROOT/'data'/f'{n}.csv').read_bytes()).hexdigest() for n in ['train','validation','test']}}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='outputs/result.json');a=p.parse_args();write_json(a.out,run())
