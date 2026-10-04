"""第007讲：完整、可重跑的合成基线实验。CPU/offline，需NumPy。

python experiment.py --output results_run1
python experiment.py --output results_run2
python experiment.py --self-test
默认文件相对于本脚本；所有结果明确标注合成数据边界。
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import platform
import tempfile
import shutil
import numpy as np

BASE=Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_materials(data_dir, config_path):
    config=json.loads(Path(config_path).read_text(encoding='utf-8'))
    if config.get('tie_rule')!='first_in_config_order':
        raise ValueError('不支持的 tie_rule；本实现只允许 first_in_config_order')
    def checked_number(value, name):
        if isinstance(value,bool) or not isinstance(value,(int,float)):
            raise ValueError(name+' 必须是有限数值，不接受布尔或数字文本')
        try:
            converted=float(value)
        except (ValueError,OverflowError) as error:
            raise ValueError(name+' 超出本教学浮点范围') from error
        if not np.isfinite(converted):raise ValueError(name+' 必须有限')
        return converted
    config['tolerance']=checked_number(config.get('tolerance'),'tolerance')
    if config['tolerance']<0:raise ValueError('tolerance 必须非负')
    for field in ['candidate_weights','candidate_biases']:
        values=config.get(field)
        if not isinstance(values,list) or not values:
            raise ValueError(field+' 必须是非空数值列表')
        config[field]=[checked_number(value,field) for value in values]
    counts=config.get('expected_counts')
    if not isinstance(counts,dict) or set(counts)!={'train','validation','test'}:
        raise ValueError('expected_counts 必须明确三个数据角色')
    if any(isinstance(value,bool) or not isinstance(value,int) or value<=0 for value in counts.values()):
        raise ValueError('每个数据角色的预期数量必须为正整数')
    with (Path(data_dir)/'houses.csv').open(encoding='utf-8',newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames != ['house_id','area_m2','price_wan']:
            raise ValueError('houses.csv 字段与约定不符')
        houses=list(reader)
    with (Path(data_dir)/'split.csv').open(encoding='utf-8',newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames != ['house_id','role']:
            raise ValueError('split.csv 字段与约定不符')
        split=list(reader)
    ids=[r['house_id'] for r in houses]
    split_ids=[r['house_id'] for r in split]
    if not ids or any(not value for value in ids) or len(ids)!=len(set(ids)):
        raise ValueError('数据编号必须非空且唯一')
    if len(split_ids)!=len(set(split_ids)) or set(ids)!=set(split_ids):
        raise ValueError('划分编号必须唯一，且与数据编号完全一致')
    roles={r['house_id']:r['role'] for r in split}
    if set(roles.values())!={'train','validation','test'}:
        raise ValueError('划分角色不符合 train/validation/test 约定')
    groups={role:[] for role in ['train','validation','test']}
    for row in houses:
        area=float(row['area_m2']);price=float(row['price_wan'])
        if not np.isfinite(area) or not np.isfinite(price) or area<=0:
            raise ValueError('面积必须为正且数值必须有限')
        groups[roles[row['house_id']]].append({'house_id':row['house_id'],'area_m2':area,'price_wan':price})
    if {k:len(v) for k,v in groups.items()}!=config['expected_counts']:
        raise ValueError('划分数量与配置不一致')
    if config['metric']!='mae_wan' or config['model_order']!=['constant','linear_grid']:
        raise ValueError('当前教学实现只支持约定的指标和模型顺序')
    return groups,config


def arrays(records):
    x=np.asarray([r['area_m2'] for r in records],dtype=np.float64)
    y=np.asarray([r['price_wan'] for r in records],dtype=np.float64)
    return x,y


def mae(predictions, labels):
    predictions=np.asarray(predictions,dtype=np.float64)
    labels=np.asarray(labels,dtype=np.float64)
    if predictions.ndim!=1 or predictions.shape!=labels.shape or predictions.size==0:
        raise ValueError('MAE 要求同形状、非空的一维预测与标签')
    if not np.all(np.isfinite(predictions)) or not np.all(np.isfinite(labels)):
        raise ValueError('MAE 输入必须有限')
    value=float(np.mean(np.abs(predictions-labels)))
    if not np.isfinite(value):raise ValueError('MAE 结果溢出')
    return value


def predict(model, x):
    x=np.asarray(x,dtype=np.float64)
    if model['name']=='constant':return np.full(x.shape,model['value_wan'])
    return model['weight']*x+model['bias']


def fit_models(training, config):
    """只接收训练记录；不能根据验证或测试标签拟合参数。"""
    x,y=arrays(training)
    constant={'name':'constant','value_wan':float(np.mean(y))}
    grid=[];best=None
    for weight in config['candidate_weights']:
        for bias in config['candidate_biases']:
            item={'name':'linear_grid','weight':float(weight),'bias':float(bias)}
            score=mae(predict(item,x),y)
            grid.append({'weight':float(weight),'bias':float(bias),'training_mae_wan':score})
            if best is None or score<best['score']-config['tolerance']:
                best={'model':item,'score':score}
    if best is None:raise ValueError('候选网格不能为空')
    return [constant,best['model']],{'grid_scores':grid,'constant_training_mae_wan':mae(predict(constant,x),y)}


def select_model(models, validation, tolerance):
    """用验证比较预定的已拟合方案，分数相同保留先前方案。"""
    x,y=arrays(validation);scores=[];best_index=0
    for model in models:scores.append(mae(predict(model,x),y))
    for i in range(1,len(models)):
        if scores[i]<scores[best_index]-tolerance:best_index=i
    return dict(models[best_index]),[{'name':m['name'],'validation_mae_wan':s} for m,s in zip(models,scores)]


def score_records(model, records):
    x,y=arrays(records);predictions=predict(model,x)
    rows=[{'house_id':r['house_id'],'prediction_wan':float(p),'label_wan':float(t),'absolute_error_wan':float(abs(p-t))} for r,p,t in zip(records,predictions,y)]
    return rows,mae(predictions,y)


def run(data_dir=BASE/'data',config_path=BASE/'config.json',output=None):
    groups,config=load_materials(data_dir,config_path)
    models,training_record=fit_models(groups['train'],config)
    selected,validation_scores=select_model(models,groups['validation'],config['tolerance'])
    training_record['fitted_models']=models
    training_record['fitted_training_predictions']={model['name']:score_records(model,groups['train'])[0] for model in models}
    validation_records=[{**summary,'predictions':score_records(model,groups['validation'])[0]} for model,summary in zip(models,validation_scores)]
    # 到这里模型已冻结。测试信息没有传入上面的拟合和选择函数。
    test_rows,test_mae=score_records(selected,groups['test'])
    baseline_rows,baseline_mae=score_records(models[0],groups['test'])
    record={'task':config['task'],'data_kind':'original_synthetic_teaching_data',
            'counts':{k:len(v) for k,v in groups.items()},
            'split_ids':{k:[r['house_id'] for r in v] for k,v in groups.items()},
            'configuration':config,'training':training_record,'validation':validation_records,
            'frozen_model':selected,'test_mae_wan':test_mae,'baseline_test_mae_wan':baseline_mae,
            'test_predictions':test_rows,'baseline_test_predictions':baseline_rows,
            'inputs_sha256':{'houses.csv':digest(Path(data_dir)/'houses.csv'),'split.csv':digest(Path(data_dir)/'split.csv'),'config.json':digest(config_path)},
            'code_sha256':digest(Path(__file__)),
            'environment':{'python':platform.python_version(),'numpy':np.__version__,'device':'CPU'},
            'scope':'Fixed public synthetic dataset; not evidence of real-market accuracy or broad robustness.'}
    if output is not None:
        output=Path(output);output.mkdir(parents=True,exist_ok=True);target=output/'result.json'
        if target.exists():
            old=json.loads(target.read_text())
            if old['inputs_sha256']!=record['inputs_sha256'] or old['code_sha256']!=record['code_sha256']:
                raise RuntimeError('输出目录已有不同输入或代码版本，请使用新的输出目录保留历史')
        target.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return record


def check_reference(record):
    ref=json.loads((BASE/'expected_result.json').read_text())
    if record['frozen_model']!=ref['frozen_model']:raise AssertionError('选中模型与固定参考不同')
    for key in ['test_mae_wan','baseline_test_mae_wan']:
        if abs(record[key]-ref[key])>ref['tolerance']:raise AssertionError(key+' 与固定参考不同')
    return True


def self_test():
    record=run();check_reference(record)
    assert mae([1,4],[2,2])==1.5
    assert mae([101,104],[102,102])==1.5
    assert record['counts']=={'train':36,'validation':12,'test':12}
    sets=[set(v) for v in record['split_ids'].values()]
    assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
    groups,config=load_materials(BASE/'data',BASE/'config.json')
    models,details=fit_models(groups['train'],config)
    selected,scores=select_model(models,groups['validation'],config['tolerance'])
    for row in groups['test']:row['price_wan']+=10000
    models2,details2=fit_models(groups['train'],config)
    selected2,scores2=select_model(models2,groups['validation'],config['tolerance'])
    assert (models,details,selected,scores)==(models2,details2,selected2,scores2)
    with tempfile.TemporaryDirectory() as temp:
        tmp=Path(temp);shutil.copytree(BASE/'data',tmp/'data')
        split=tmp/'data/split.csv';lines=split.read_text().splitlines();split.write_text('\n'.join(lines[:-1])+'\n')
        try:run(tmp/'data',BASE/'config.json')
        except ValueError:pass
        else:raise AssertionError('遗漏划分没有被拒绝')
    with tempfile.TemporaryDirectory() as temp:
        config_file=Path(temp)/'bad-config.json'
        for field,value in [('tie_rule','last'),('tolerance',float('nan')),('tolerance',-1),('tolerance',True),('candidate_weights',[]),('candidate_weights',['2']),('candidate_biases',[float('inf')])]:
            altered=dict(config);altered[field]=value
            config_file.write_text(json.dumps(altered),encoding='utf-8')
            try:load_materials(BASE/'data',config_file)
            except ValueError:pass
            else:raise AssertionError('无效配置没有被拒绝：'+field)
    for p,y in [([],[]),([1,2],[1]),([[1],[2]],[1,2]),([float('nan')],[1])]:
        try:mae(p,y)
        except ValueError:pass
        else:raise AssertionError('错误输入没有被拒绝')
    print('PASS: reference, split, translation, test-isolation, config-contract, missing-ID and shape/finite checks')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data',type=Path,default=BASE/'data');parser.add_argument('--config',type=Path,default=BASE/'config.json');parser.add_argument('--output',type=Path,default=BASE/'results');parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    if args.self_test:self_test();return
    record=run(args.data,args.config,args.output)
    print('Selected:',record['frozen_model']);print('Test MAE:',record['test_mae_wan']);print('Baseline test MAE:',record['baseline_test_mae_wan']);print('Synthetic teaching data only; see result.json for full evidence.')

if __name__=='__main__':main()
