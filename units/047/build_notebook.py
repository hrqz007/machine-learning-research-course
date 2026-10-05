"""Build an executable, first-cell-contract tutorial Notebook."""
from pathlib import Path
import argparse,hashlib,json
import nbformat as nbf
ROOT=Path(__file__).resolve().parent
CONTRACT=['data/protocol.json','data/classification.csv','data/regression_train.csv','data/regression_calibration.csv','data/regression_evaluation.csv','data_integrity.json','numeric.py','protocol.py','calibration.py','conformal.py','generate_data.py','experiment.py','audit.py','plots.py','requirements.txt','environment.yml']

def build():
    expected={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in CONTRACT};cells=[]
    def code(s):cells.append(nbf.v4.new_code_cell(s))
    def md(s):cells.append(nbf.v4.new_markdown_cell(s))
    code('''# First physical cell: standard-library integrity checks before teaching imports.
from pathlib import Path
import hashlib, json
BASE = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / 'data_integrity.json').is_file() and (p / 'experiment.py').is_file()), None)
if BASE is None:
    raise ValueError('Run within the complete ML047 directory or its outputs subdirectory')
EXPECTED = '''+repr(expected)+'''
for relative, wanted in EXPECTED.items():
    if hashlib.sha256((BASE / relative).read_bytes()).hexdigest() != wanted:
        raise ValueError('Teaching contract changed: ' + relative)
print('All', len(EXPECTED), 'data, code, and environment contracts verified.')''')
    md('# 第047讲 校准与预测不确定性\n\n两个独立实验：固定分类分数的温度校准，以及独立残差构造的回归预测区间。最高置信度ECE和正类ECE分开计算。全部为合成数据，不声称训练网络或现实部署效果。')
    code('''import os, sys, io, math
sys.path.insert(0, str(BASE))
os.environ.setdefault('MPLCONFIGDIR', str(BASE / 'outputs/mpl'))
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image, display
import experiment as e
import calibration as cb
import conformal as cf
import plots
from numeric import canonical_bytes, safe_write
classification, regression, cfg = e.load_inputs()
report = e.main_report(classification, regression, cfg)
def show(i):
    fig = plots.figure(i, report)
    stream = io.BytesIO()
    fig.savefig(stream, format='png', dpi=160, bbox_inches='tight')
    display(Image(data=stream.getvalue(), format='png'))
    plt.close(fig)
for name in ['initial_metrics', 'optimum_metrics']:
    value = report['hand'][name]
    print(name, 'Brier', value['binary_Brier'], 'accuracy', value['accuracy'],
          'positive ECE', value['positive_class_reliability']['ece'],
          'top ECE', value['top_label_reliability']['ece'])
show(1)''')
    md('## 同一四行：三次前向、两次同步更新\n每行显示损失、局部导数、平均梯度和Hessian贡献。第1行损失上升并不阻止平均损失下降。')
    code('''for state in report['hand']['states']:
    print('STAGE', state['stage'], 'a=', state['inverse_temperature'], 'T=', state['temperature'])
    for row in state['rows']:
        print(json.dumps(row, sort_keys=True))
    print('Mean loss, gradient, Hessian:', state['mean_log_loss'], state['gradient'], state['hessian'])
    if state['stage'] < 2:
        print('Synchronous next a:', state['inverse_temperature'] - .5*state['gradient'])
show(2)''')
    code('''from decimal import Decimal, localcontext
analytic = report['hand']['analytic_certificate']
print('Independent analytic certificate:', analytic)
print('Numerical optimum:', {k:v for k,v in report['hand']['fitted'].items() if k != 'bracket_trace'})
eta = .75
for p in [.1, .25, .5, .75, .9]:
    direct = eta*(p-1)**2 + (1-eta)*p**2
    decomposed = (p-eta)**2 + eta*(1-eta)
    if not math.isclose(direct, decomposed, abs_tol=1e-15):
        raise RuntimeError('Brier expectation decomposition failed')
    print('p, expected Brier:', p, direct)
show(3)''')
    md('## 只用400行校准选择，4000行评价不反馈\n选择记录在计算评价指标前冻结。oracle列不进入温度拟合。')
    code('''temp = report['temperature']
fit = temp['frozen_choice']['fit']
print('Status, bounds, iterations:', fit['status'], fit['bounds'], fit['iterations'])
print('Chosen a,T,NLL:', fit['final']['inverse_temperature'], fit['final']['temperature'], fit['final']['mean_log_loss'])
print('Projected gradient residual:', fit['projected_gradient_residual'])
print('Choice SHA256:', temp['choice_sha256'])
print('Independent SciPy scalar fit:', temp['scipy_reference'])
print('First and last bisection records:', fit['bracket_trace'][:2], fit['bracket_trace'][-2:])
show(4)''')
    code('''for role, stages in temp['metrics'].items():
    for timing, values in stages.items():
        print(role, timing, {k:values[k] for k in ['n','mean_log_loss','binary_Brier','accuracy','library_binary_Brier','library_log_loss']})
        print('Positive-class bins:', values['positive_class_reliability'])
        print('Top-label bins:', values['top_label_reliability'])
show(5)
import copy
changed = copy.deepcopy(classification)
changed['evaluation']['y'] = 1 - changed['evaluation']['y']
changed['evaluation']['true_probability'] = 1 - changed['evaluation']['true_probability']
changed_result = e.temperature_report(changed, cfg)
if canonical_bytes(changed_result['frozen_choice']) != canonical_bytes(temp['frozen_choice']):
    raise RuntimeError('Evaluation information changed selection')
print('Evaluation-label mutation leaves choice exactly unchanged.')''')
    md('## 新回归实验：60训练、99校准、400评价\n重复训练的预测方差不是所有认知不确定性的总和；区间目标是新的响应Y，不只是条件均值。')
    code('''main = report['conformal']['main']
print('Training fit:', main['fit'])
print('Noise mechanism:', {k:cfg[k] for k in cfg if k.startswith('regression_noise')})
u = report['conformal']['uncertainty_components']
for i in [0,30,60]:
    print('x, noise variance, training-prediction variance:', u['x'][i], u['known_conditional_noise_variance'][i], u['repeated_training_prediction_variance_ddof1'][i])
print('Scope:', u['scope'])
show(6)''')
    code('''for i, (x,y,pred,score) in enumerate(zip(regression['calibration']['x'], regression['calibration']['y'], main['calibration_prediction'], main['calibration_scores']),1):
    print('Calibration row:', i, 'x=', float(x), 'y=', float(y), 'prediction=', pred, 'abs residual=', score)
print('Finite-sample quantile:', main['quantile'])
ordered = sorted(main['calibration_scores'])
print('Sorted values at ranks 89,90,91:', ordered[88:91])
show(7)''')
    code('''for i in range(cfg['interval_display_count']):
    print('Evaluation row:', i+1, 'x=', float(regression['evaluation']['x'][i]),
          'prediction=', main['test_prediction'][i],
          'interval=', [main['baseline']['interval_lower'][i], main['baseline']['interval_upper'][i]],
          'Y=', float(regression['evaluation']['y'][i]), 'covered=', main['baseline']['covered'][i],
          'shifted Y=', float(regression['evaluation']['shifted_y'][i]), 'shift covered=', main['noise_shift']['covered'][i])
print('All 400 baseline / shifted coverages:', main['baseline']['coverage'], main['noise_shift']['coverage'])
print('Groups:', main['groups'])
show(8)''')
    md('## 保留整体、分组与噪声改变结果\n120次重复的独立单位是整套训练/校准/评价实验；±2 MCSE仅是数值实验均值的近似误差描述。')
    code('''for row in report['conformal']['repetitions']:
    print(json.dumps(row, sort_keys=True))
print('Summary:', json.dumps(report['conformal']['summary'], sort_keys=True))
print('MC scope:', report['conformal']['mc_scope'])
show(9)''')
    code('''for alpha in cfg['miscoverage_boundary_examples']:
    print('Nine calibration scores:', cf.finite_sample_quantile(list(range(1,10)), alpha))
print('99 scores, alpha=.29:', cf.finite_sample_quantile(list(range(99)), .29))
print('Direct rank vs higher recipe:', cf.finite_sample_quantile(list(range(1,11)), .2)['quantile'],
      float(np.quantile(np.arange(1,11), .9, method='higher')))
print('Invalid reused 1NN quantile:', main['invalid_calibration_reuse']['quantile'])
print('Invalid reused 1NN coverage:', main['invalid_calibration_reuse']['evaluation']['coverage'])
show(10)''')
    code('''for z,y in [([1,2],[0,0]), ([0,0],[0,1]), ([1,2],[1,1])]:
    result = cb.fit_temperature(z,y)
    print('Boundary example:', z,y,result['status'],result['final']['inverse_temperature'],result['final']['temperature'])
infinite = cf.finite_sample_quantile([1,2], .1)
print('Infinite interval:', cf.interval_report([0,1], [100,-100], infinite))
for name, operation in [
    ('bool logit', lambda: cb.temperature_state([0, True], [0,1], 1)),
    ('invalid label', lambda: cb.temperature_state([0,1], [0,.5], 1)),
    ('rank deficient fit', lambda: cf.fit_line([1,1], [0,1])),
    ('negative score', lambda: cf.finite_sample_quantile([1,-1], .1))]:
    try: operation()
    except ValueError as ex: print('Expected rejection:', name, str(ex))
    else: raise RuntimeError('Invalid input accepted: '+name)''')
    md('## 独立参照与可恢复科学结果\nDecimal、标量OLS、实际SciPy/sklearn、秩枚举与定向泄漏检测并用。测试不能代替可交换性假设。')
    code('''import audit
audit_result = audit.audit()
print(json.dumps(audit_result, ensure_ascii=False, sort_keys=True))
science_hash = hashlib.sha256(canonical_bytes(report)).hexdigest()
if science_hash != audit_result['core_sha256']:
    raise RuntimeError('Notebook main report differs from audited report')
safe_write(BASE / 'outputs/notebook-result.json', report)
print('Full report saved. Core SHA256:', science_hash)''')
    for i,cell in enumerate(cells):cell['id']='ml047-'+str(i).zfill(3)
    return nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(ROOT/'calibration_uncertainty.ipynb'));a=p.parse_args();path=Path(a.out).absolute()
    if path.suffix!='.ipynb':raise ValueError('Notebook output extension required')
    for part in (path,*path.parents):
        if part.is_symlink():raise ValueError('symlink notebook output')
    if path.exists() and (not path.is_file() or path.stat().st_nlink>1):raise ValueError('unsafe notebook output')
    path.parent.mkdir(parents=True,exist_ok=True);nbf.write(build(),path);print(path)
if __name__=='__main__':main()
