"""Build a real Notebook for the supplied-probability decision experiment."""
from pathlib import Path
import hashlib
import nbformat as nbf
ROOT=Path(__file__).resolve().parent
CONTRACT=['data/posteriors.csv','data/label_shift.csv','data/decision_spec.json','data_integrity.json',
          'experiment.py','audit.py','plots.py','requirements.txt','environment.yml']

def main():
    expected={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in CONTRACT};cells=[]
    def code(s):cells.append(nbf.v4.new_code_cell(s))
    def md(s):cells.append(nbf.v4.new_markdown_cell(s))
    code("""# First physical cell verifies complete byte contracts before imports or output.
from pathlib import Path
import hashlib, json
BASE = Path.cwd()
EXPECTED = """+repr(expected)+"""
for relative, wanted in EXPECTED.items():
    if hashlib.sha256((BASE / relative).read_bytes()).hexdigest() != wanted:
        raise ValueError('Teaching contract changed: ' + relative)
print('All nine input/source/environment contracts verified.')""")
    md('# 第045讲 从概率到决策\n\n概率是固定合成机制的已知输入，不训练模型，也不将这份机制表当成未见测试数据。逐个真实状态算损失贡献，再选择行动。')
    code("""import os, io
os.environ.setdefault('MPLCONFIGDIR', str(BASE / 'outputs/mpl'))
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Image
import experiment as e
import plots
p, w, ids, q0, q1, shift_ids, cfg = e.load_inputs()
report = e.main_report(p, w, ids, q0, q1, shift_ids, cfg)
P = np.column_stack([1-p, p])
def show(number):
    fig = plots.figure(number, report)
    stream = io.BytesIO()
    fig.savefig(stream, format='png', dpi=160, bbox_inches='tight')
    display(Image(data=stream.getvalue(), format='png'))
    plt.close(fig)
print('Groups:', list(zip(ids, p.tolist(), w.tolist())))
print('Fixed config:', cfg)
show(1)""")
    md('## 一行的完整计算\nD4每个真实状态的损失贡献是什么？为什么p小于1/2却选1？')
    code("""one = e.risk_table([[5/8, 3/8]], [[0, 3], [1, 0]])
print('D4 full decision chain:', one['rows'][0])
print('Cost-derived threshold:', report['binary_threshold'])
print('Action order and tie rule:', report['decision']['tie_rule'])
print('Risk derivatives with respect to p:', report['risk_derivatives_wrt_positive_probability'])
show(2)""")
    md('## 相同概率下的三个完整阶段\n各阶段都重新计算逐状态贡献、风险、并列集合和行动。改变成本不是更新模型参数。')
    code("""for name in ['accuracy_decision', 'decision', 'rejection']:
    print('STAGE:', name)
    for sid, row in zip(ids, report[name]['rows']):
        print(sid, json.dumps(row, ensure_ascii=False, sort_keys=True))
for name, result in report['evaluation'].items():
    print('POPULATION EVALUATION', name, result)
    if not np.isclose(sum(result['weighted_risk_contribution']), result['expected_cost'], rtol=1e-13, atol=1e-14):
        raise RuntimeError('Population contributions do not sum')
show(3)""")
    code("""print('Accuracy-optimal expected accuracy:', 1-report['evaluation']['accuracy_optimal']['expected_error_rate'])
print('Cost-optimal expected accuracy:', 1-report['evaluation']['cost_optimal']['expected_error_rate'])
print('Accuracy rule actual cost:', report['evaluation']['accuracy_optimal']['expected_cost'])
print('Cost rule actual cost:', report['evaluation']['cost_optimal']['expected_cost'])
show(4)
# Independent vector expression checks the row-ledger calculation.
C = np.array(report['decision']['costs'])
print('P @ C.T:', P @ C.T)
if not np.array_equal(P @ C.T, np.array(report['decision']['risks'])):
    raise RuntimeError('Main dyadic matrix product differs from risk ledger')""")
    code("""for row in report['cost_scenarios']:
    print(row)
print('Both zero:', e.binary_threshold(0, 0))
print('All-zero risk tie:', e.risk_table([[.3, .7]], [[0, 0], [0, 0]]))
print('Swapping FP and FN changes threshold:', e.binary_threshold(3, 1))""")
    code("""print('Reject bounds:', report['rejection_bounds'])
print('With rejection:', report['evaluation']['with_reject'])
print('Reject cost 3/4:', e.reject_bounds(1, 3, .75))
print('Free rejection:', e.risk_table(P, [[0, 3], [1, 0], [0, 0]])['actions'])
show(5)""")
    code("""print('Three-class probabilities:', cfg['multiclass_probabilities'])
print('Action-by-state costs:', cfg['multiclass_costs'])
print('Full three-class decision:', report['multiclass'])
print('Rectangular 2-action, 3-class example:', e.risk_table([[.5, .25, .25]], [[0, 2, 4], [1, 1, 0]]))
show(6)""")
    md('## 先验变化逐格更新\n明确P(X|Y)保持不变，再比较直接Bayes与后验重加权；目标评价也换用目标边缘。')
    code("""shift = report['label_shift']
print('Fixed P(X|Y=0):', q0, 'P(X|Y=1):', q1)
for population in ['source', 'target']:
    print(population, shift[population])
print('Reweighted posterior:', shift['corrected_positive_posterior'])
print('Stale target evaluation:', shift['stale_expected_target_cost'])
print('Corrected target evaluation:', shift['corrected_expected_target_cost'])
print('Posterior endpoints:', e.correct_prior([0, 1], .25, .75))
if not np.allclose(shift['corrected_positive_posterior'], shift['target']['positive_posterior'], atol=2e-15, rtol=2e-15):
    raise RuntimeError('Prior correction disagrees with direct Bayes')
show(7)""")
    md('## 一次实际模拟和精确总体\n三个政策共享2048个标签，不改种子追结果。每组固定样本数，因此标准误使用分层公式。')
    code("""print('Sampling design:', report['simulation']['sampling'])
for name, result in report['simulation']['results'].items():
    print(name, json.dumps(result, ensure_ascii=False, sort_keys=True))
show(8)
print('The ±2 standard-error bars illustrate scale; no exact 95% interval claim.')""")
    code("""for row in report['threshold_sweep']:
    print(row)
print('No observed-label threshold search occurs in this experiment.')
show(9)
eta, estimated = 3/8, 7/40
true = e.risk_table([[1-eta, eta]], [[0, 3], [1, 0]])
wrong = e.risk_table([[1-estimated, estimated]], [[0, 3], [1, 0]])
chosen = wrong['actions'][0]
excess = true['risks'][0][chosen] - min(true['risks'][0])
print('Probability-error example:', 'excess risk', excess, 'upper bound', 4*abs(eta-estimated))""")
    code("""for label, call in [
    ('bad probability sum', lambda: e.risk_table([[.2, .2]], [[0, 3], [1, 0]])),
    ('bad last cost bool', lambda: e.risk_table([[.5, .5]], [[0, 3], [1, True]])),
    ('undefined source prior correction', lambda: e.correct_prior([.5], 0, .5)),
    ('undefined empty-marginal bin', lambda: e.posterior_from_likelihoods([1, 0], [1, 0], .5))]:
    try:
        call()
    except ValueError as exc:
        print(label, 'rejected:', str(exc))
    else:
        raise RuntimeError('Invalid input was accepted: ' + label)
nearby = np.array([np.nextafter(.25, 0), .25, np.nextafter(.25, 1)])
print('Adjacent float posteriors:', nearby)
print('Threshold actions:', e.threshold_actions(nearby, .25))
print('Computed-risk actions:', e.risk_table(np.column_stack([1-nearby, nearby]), [[0, 3], [1, 0]])['actions'])""")
    code("""import audit
checked = audit.audit()
print('Exact policy enumeration and independent checks:', checked)
core = hashlib.sha256(e.canonical_bytes(report)).hexdigest()
if core != checked['core_sha256']:
    raise RuntimeError('Complete Notebook scientific result differs from independent audit run')
e.safe_write(BASE / 'outputs/notebook-result.json', report)
print('Complete scientific SHA256:', core)
print('Limitations:', report['limitations'])""")
    md('## 写出自己的边界说明\n区分已知总体、一次模拟、实际估计概率和真正独立测试。按answers.pdf完成十八题，说明哪种成本与哪种信息支持了自己的行动。')
    for i,cell in enumerate(cells):cell['id']=hashlib.sha256((str(i)+cell['source']).encode()).hexdigest()[:12]
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
    nbf.write(nb,ROOT/'experiment.ipynb');print(ROOT/'experiment.ipynb')
if __name__=='__main__':main()
