"""固定方案：拟合→干净校准→一次评价；每一步数据职责可审计。"""
import argparse,json,platform
from pathlib import Path
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from anomaly import make_models,calibrate,evaluate
from generate_data import make_data
ROOT=Path(__file__).resolve().parent

def run(return_artifacts=False):
    data=make_data();result={'seed':data['seed'],'alpha':.05,'protocol':'fixed hyperparameters; clean calibration only; strict score > threshold','versions':{'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__}};artifacts={}
    for name,source in [('synthetic',data['synthetic']),('digits',data['digits'])]:
        arrays={key:np.asarray(value['x']) for key,value in source.items()}
        scaler=StandardScaler().fit(arrays['train']) if name=='synthetic' else None
        x={key:(scaler.transform(value) if scaler else value) for key,value in arrays.items()}
        rows={};saved={}
        for method,model in make_models().items():
            model.fit(x['train']);scores={key:model.score(x[key]) for key in ('calibration','normal_test','anomaly_test')}
            cal=calibrate(scores['calibration'],.05);row=evaluate(scores['normal_test'],scores['anomaly_test'],cal)
            row.update({'threshold':cal.threshold,'calibration_n':cal.n,'order_rank':cal.rank,'marginal_fpr_bound':cal.marginal_bound,'calibration_false_alarms':int(cal.alarms(scores['calibration']).sum())})
            # 正常子群误报可揭示总误报掩盖的差异；不用于调参。
            if name=='synthetic':
                groups=np.asarray(source['normal_test']['group']);kinds=np.asarray(source['anomaly_test']['kind'])
                row['normal_group_fpr']={str(g):float(cal.alarms(scores['normal_test'][groups==g]).mean()) for g in (0,1)}
                row['recall_by_kind']={kind:float(cal.alarms(scores['anomaly_test'][kinds==kind]).mean()) for kind in ('bridge','far')}
            # 每万条、0.1%真实异常的情景推算，非实际部署实测。
            row['scenario_per_10000_at_prevalence_0.001']={'false_alarms':9990*row['fpr'],'true_alarms':10*row['recall']}
            rows[method]=row;saved[method]={'model':model,'scores':scores,'calibration':cal}
        result[name]={'split_sizes':{k:len(v) for k,v in arrays.items()},'methods':rows}
        artifacts[name]={'x':x,'raw':source,'scaler':scaler,'methods':saved}
    # 同一预声明方法受污染前后对照。校准集依旧干净，模型变了必须重新打分校准。
    a=artifacts['synthetic'];dirty=np.vstack([a['x']['train'],a['x']['contaminants']]);rows={}
    for method,model in make_models().items():
        model.fit(dirty);cal=calibrate(model.score(a['x']['calibration']),.05)
        rows[method]=evaluate(model.score(a['x']['normal_test']),model.score(a['x']['anomaly_test']),cal)
    result['contaminated_training']={'contamination_fraction':67/667,'methods':rows,'note':'additional contaminant observations, not test rows; clean calibration reused for a separate frozen comparison'}
    if return_artifacts:return result,artifacts
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(ROOT/'outputs/result.json'));a=p.parse_args();r=run();out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)+'\n');print(json.dumps(r,ensure_ascii=False,indent=2))
