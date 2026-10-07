"""Compare each raw-score stage against sklearn with matching loss conventions."""
import argparse,platform
import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingRegressor,GradientBoostingClassifier
from gbdt import GBDT,loss
from common import ROOT,load_data,write_json,safe_output
CONFIG={'n_estimators':40,'learning_rate':.1,'max_depth':1,'min_samples_leaf':3}

def run():
    data=load_data();X={k:v['x'][:,None] for k,v in data.items()};results={}
    for kind,column in [('squared','y_reg'),('logistic','y_class')]:
        y=data['train'][column];model=GBDT(kind,**CONFIG).fit(X['train'],y)
        cls=GradientBoostingRegressor if kind=='squared' else GradientBoostingClassifier
        library=cls(**CONFIG,criterion='squared_error',random_state=62).fit(X['train'],y)
        reference=list(library.staged_predict(X['train'])) if kind=='squared' else [v.ravel() for v in library.staged_decision_function(X['train'])]
        own=list(model.staged_decision_function(X['train']))
        stage_difference=[float(np.max(np.abs(a-b))) for a,b in zip(own,reference)]
        result={'init':model.init_,'training_loss':model.losses_,'history':model.history_,'max_stage_difference':max(stage_difference),'stage_difference':stage_difference,'metrics':{},'curves':{}}
        for split in data:
            target=data[split][column];F=model.decision_function(X[split])
            ref=library.predict(X[split]) if kind=='squared' else library.decision_function(X[split])
            result['metrics'][split]={'from_zero_loss':loss(target,F,kind),'sklearn_loss':loss(target,ref,kind),'max_raw_score_difference':float(np.max(np.abs(F-ref)))}
            if kind=='logistic':result['metrics'][split]['accuracy']=float(np.mean(model.predict(X[split])==target))
            result['curves'][split]=[loss(target,np.full(len(target),model.init_),kind)]+[loss(target,F,kind) for F in model.staged_decision_function(X[split])]
        results[kind]=result
    toyX=np.arange(1,5)[:,None]
    toy={kind:GBDT(kind,n_estimators=2,learning_rate=.5,min_samples_leaf=1).fit(toyX,y).history_ for kind,y in [('squared',np.array([1,1,3,3.])),('logistic',np.array([0,0,1,1.]))]}
    return {'lesson':'062','config':CONFIG,'versions':{'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__},'toy':toy,'results':results,'comparison_note':'Squared loss is HALF MSE. Logistic raw score is full logit. Matching Newton leaf update, learning rate, depth and squared-error tree criterion.'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(ROOT/'outputs/result.json'));a=p.parse_args();out=safe_output(a.out);report=run();write_json(out,report)
    print({k:{'test':v['metrics']['test'],'stage_difference':v['max_stage_difference']} for k,v in report['results'].items()})
if __name__=='__main__':main()
