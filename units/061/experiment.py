"""Frozen configurations: no test-driven selection and no external data access."""
from pathlib import Path
import argparse,platform
import numpy as np
import sklearn
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from adaboost import AdaBoost
from common import ROOT,load_data,write_json,safe_output
CONFIG={'rounds':40,'learning_rate':1.,'noise_fraction':.1,'sklearn_random_state':61}

def run():
    data=load_data();X={k:np.column_stack([v['x0'],v['x1']]) for k,v in data.items()}
    results={}
    for mode in ['clean','noisy']:
        y=data['train']['y_'+mode];model=AdaBoost(40).fit(X['train'],y)
        library=AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1),n_estimators=40,learning_rate=1.,random_state=61).fit(X['train'],y)
        result={'stop_reason':model.stop_reason_,'rounds':len(model.stumps_),'history':model.history_,'metrics':{},'curves':{}}
        for split in data:
            truth=y if split=='train' else data[split]['y_clean']
            pred=model.predict(X[split]);lib=library.predict(X[split])
            result['metrics'][split]={'from_zero_accuracy':float(np.mean(pred==truth)),'sklearn_accuracy':float(np.mean(lib==truth)),'prediction_agreement':float(np.mean(pred==lib))}
            result['curves'][split]=[float(np.mean(np.where(F>=0,1,-1)!=truth)) for F in model.staged_decision_function(X[split])]
        result['flipped_weight_mass']=[float(np.array(h['weights_after'])[data['train']['flipped']==1].sum()) for h in model.history_]
        result['effective_sample_size']=[float(1/np.sum(np.array(h['weights_after'])**2)) for h in model.history_]
        results[mode]=result
    toy=AdaBoost(2).fit(np.arange(1,7).reshape(-1,1),[-1,-1,1,-1,1,1])
    return {'lesson':'061','config':CONFIG,'versions':{'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__},'toy':toy.history_,'results':results,'comparison_note':'Own stump minimizes weighted 0-1 error; sklearn stump minimizes weighted Gini. SAMME alpha is twice binary half-log alpha when the same error is used; raw scores need not match.'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(ROOT/'outputs/result.json'));a=p.parse_args();out=safe_output(a.out)
    report=run();write_json(out,report)
    print({k:v['metrics']['test'] for k,v in report['results'].items()})
if __name__=='__main__':main()
