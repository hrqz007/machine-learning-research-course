"""Fixed-protocol shallow-tree demonstration; no test-set model selection."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from common import ROOT, write_json
from tree import ShallowTree, candidates

def load(name):
    a=np.loadtxt(ROOT/'data'/(name+'.csv'), delimiter=',', skiprows=1)
    return a[:,:-1],a[:,-1]

def hand_calculations():
    X=np.arange(1,7,dtype=float)[:,None]
    classification=np.array([0,0,1,0,1,1],dtype=float)
    regression=np.array([1,2,2,8,9,8],dtype=float)
    return {'x':X[:,0].tolist(),'classification_y':classification.tolist(),'regression_y':regression.tolist(),
            'gini':candidates(X,classification,'gini'),'entropy':candidates(X,classification,'entropy'),
            'squared_error':candidates(X,regression,'squared_error')}

def run():
    X,y=load('classification_train'); rx,ry=load('regression_train')
    models={c:ShallowTree(c,2,12).fit(X,y) for c in ['gini','entropy']}
    models['squared_error']=ShallowTree('squared_error',2,12).fit(rx,ry)
    libraries={c:DecisionTreeClassifier(criterion=c,max_depth=2,min_samples_leaf=12,random_state=58).fit(X,y) for c in ['gini','entropy']}
    libraries['squared_error']=DecisionTreeRegressor(max_depth=2,min_samples_leaf=12,random_state=58).fit(rx,ry)
    frozen={'protocol':{'max_depth':2,'min_samples_leaf':12,'classification_criteria':['gini','entropy'],
                        'regression_criterion':'squared_error','tie_tolerance':1e-12,
                        'test_use':'one fixed illustrative evaluation; no tuning','seed':58058},
            'models':{k:v.state() for k,v in models.items()},'baseline_class':int(y.mean()>.5),'baseline_regression':float(ry.mean())}
    frozen_hash=hashlib.sha256(json.dumps(frozen,sort_keys=True,allow_nan=False).encode()).hexdigest()
    # Test files are read only after all fitted states and protocol are frozen.
    tx,ty=load('classification_test'); rtx,rty=load('regression_test')
    results={}
    for criterion,model in models.items():
        a,b=(rtx,rty) if criterion=='squared_error' else (tx,ty)
        pred=model.predict(a); ref=libraries[criterion].predict(a)
        entry={'n_test':len(b),'prediction':pred.tolist(),'target':b.tolist(),'leaf_ids':model.apply(a).tolist(),
               'library_prediction':ref.tolist(),'max_prediction_difference':float(np.max(np.abs(pred-ref))),
               'library_root_feature':int(libraries[criterion].tree_.feature[0]),
               'library_root_threshold':float(libraries[criterion].tree_.threshold[0])}
        if criterion=='squared_error':
            entry.update(mse=float(np.mean((pred-b)**2)),baseline_mse=float(np.mean((ry.mean()-b)**2)))
        else:
            p=model.predict_proba(a)[:,1]
            entry.update(accuracy=float(np.mean(pred==b)),brier=float(np.mean((p-b)**2)),probability=p.tolist(),
                         baseline_accuracy=float(np.mean(frozen['baseline_class']==b)),
                         library_probability_max_difference=float(np.max(np.abs(p-libraries[criterion].predict_proba(a)[:,1]))))
        results[criterion]=entry
    # XOR shows a limitation of this strictly-positive one-step rule.
    xor_X=np.array([[0,0],[0,1],[1,0],[1,1]],float); xor_y=np.array([0,1,1,0])
    xor=ShallowTree('gini',2,1).fit(xor_X,xor_y)
    return {'unit':'058','title':'决策树的分裂机制','versions':{'numpy':np.__version__,'sklearn':sklearn.__version__},
            **frozen,'frozen_sha256_before_test_read':frozen_hash,'test':results,'hand':hand_calculations(),
            'xor':{'X':xor_X.tolist(),'y':xor_y.tolist(),'candidates':candidates(xor_X,xor_y),
                   'greedy_state':xor.state(),'greedy_accuracy':float(np.mean(xor.predict(xor_X)==xor_y)),
                   'depth2_constructed_prediction':xor_y.tolist(),'constructed_rule':'split x0 at .5, then x1 at .5 in both children'}}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(ROOT/'outputs/result.json'));args=p.parse_args()
    result=run();write_json(args.out,result)
    print(json.dumps({'status':'passed','test':{k:{a:b for a,b in v.items() if a in ['accuracy','mse','baseline_mse','baseline_accuracy','max_prediction_difference']} for k,v in result['test'].items()}},ensure_ascii=False))
