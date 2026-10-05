"""Build a real, initially unexecuted Notebook with an explicit input contract."""
from pathlib import Path
import hashlib,json
import nbformat as nbf
ROOT=Path(__file__).resolve().parent
CONTRACT=['data/observations.csv','data/model_spec.json','data_integrity.json',
          'experiment.py','audit.py','plots.py','requirements.txt','environment.yml']

def main():
    expected={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in CONTRACT}
    cells=[]
    def md(s):cells.append(nbf.v4.new_markdown_cell(s))
    def code(s):cells.append(nbf.v4.new_code_cell(s))
    code("""# First physical cell: standard library only, before numerical import or output.
from pathlib import Path
import hashlib, json
BASE = Path.cwd()
EXPECTED = """+repr(expected)+"""
for relative, wanted in EXPECTED.items():
    if hashlib.sha256((BASE / relative).read_bytes()).hexdigest() != wanted:
        raise ValueError('Teaching contract changed: ' + relative)
print('All eight teaching contracts verified before numerical work.')""")
    md('# 第043讲 多分类与信息论损失\n\n先阅读lecture.pdf与lab.pdf，再运行。八行是机制例；没有现实泛化或校准结论。下面将真正重新计算数值与图，不读取缓存报告。')
    code("""import os, io
os.environ.setdefault('MPLCONFIGDIR', str(BASE / 'outputs/mpl-cache'))
import numpy as np
import experiment as e
import plots
import matplotlib.pyplot as plt
from IPython.display import display, Image
X, y, ids, cfg = e.load_inputs()
report = e.main_report(X, y, ids, cfg)
def show(number):
    fig = plots.make_figure(number, report)
    stream = io.BytesIO()
    fig.savefig(stream, format='png', dpi=160, bbox_inches='tight')
    display(Image(data=stream.getvalue(), format='png'))
    plt.close(fig)
print('Data:', list(zip(ids, X.tolist(), y.tolist())))
print('Configuration:', cfg)
show(1)""")
    md('## 三类共同归一化\n预测得分(log2,0,log3)的概率与真实类乙的损失，再核对轴和平移。')
    code("""z = np.array([[np.log(2), 0., np.log(3)]])
p, logp = e.stable_softmax(z)
print('Probabilities:', p, 'sum:', p.sum(axis=1))
print('True class 1 loss:', -logp[0, 1])
from scipy.special import softmax
batch = np.array([[2., 1., 0.], [0., 1., 2.]])
print('Wrong axis=None row sums:', softmax(batch).sum(axis=1))
print('Correct axis=1 row sums:', softmax(batch, axis=1).sum(axis=1))
for item in report['diagnostics']['common_shifts']:
    print(item)
show(2)
show(3)""")
    md('## 独立手动推进两次同步更新\n所有行先用同一个旧参数，再将平均梯度一次提交。第三个状态是第二次更新后的新前向。')
    code("""theta = np.array(cfg['initial_theta'])
hand = []
for k in range(3):
    state = e.row_ledger(X, y, theta)
    hand.append(state)
    print('STATE', k, 'Theta', state['theta'], 'F', state['objective'], 'G', state['gradient'])
    if k < 2:
        theta = theta - cfg['learning_rate'] * np.array(state['gradient'])
if e.canonical_bytes(hand) != e.canonical_bytes(report['hand_states']):
    raise RuntimeError('Manual staged calculation does not match fit states')""")
    code("""# All 24 full row chains, including local matrices, not just the final loss.
for k, state in enumerate(hand):
    print('\\nSTATE', k, 'FULL EIGHT-ROW CHAIN')
    for sid, row in zip(ids, state['rows']):
        print(sid, json.dumps(row, ensure_ascii=False, sort_keys=True))
    summed = np.sum([row['gradient_contribution'] for row in state['rows']], axis=0)
    if not np.allclose(summed, state['gradient'], rtol=3e-13, atol=3e-14):
        raise RuntimeError('Rows did not sum to total gradient')
    print('Row-summed gradient:', summed)
show(4)""")
    code("""J = np.array(hand[0]['rows'][0]['local_softmax_jacobian'])
print('Initial softmax Jacobian:', J)
print('J @ ones:', J @ np.ones(3))
print('Eigenvalues:', np.linalg.eigvalsh(J))
print('Common-column shift changes coefficients but not probabilities:')
shifted_theta = np.array(hand[2]['theta']) + np.array([[.5], [-.25]])
shifted = e.row_ledger(X, y, shifted_theta)
print('Max probability difference:', np.max(np.abs(np.array(shifted['probabilities']) - hand[2]['probabilities'])))
show(5)""")
    md('## 数值拟合与独立解析证书\n注意BFGS的success=False会保留。训练正确率不是解析证书的替代。')
    code("""print('Analytic certificate:', report['certificate'])
print('GD:', report['fit']['status'], report['fit']['updates'], report['fit']['final']['gradient_norm'])
print('GD final Theta:', report['fit']['final']['theta'])
print('Reference optimizer:', report['reference_optimizer'])
print('Predictions:', report['training_predictions'], 'accuracy:', report['training_accuracy'])
show(6)""")
    md('## 信息论分解与零支持\n手算每一项，再比较两个方向；null加infinite标记代表正无穷，不代表缺失观测。')
    code("""print('q to p:', report['information'])
print('p to q:', report['reverse_information'])
print('Missing model support:', report['support_failure'])
print('Identical deterministic distributions:', e.information([1, 0, 0], [1, 0, 0]))
show(7)
show(8)""")
    md('## 数值失败要成为可解释证据\n普通log(sum(exp))在赢者尾项上可能过早得到0；概率下溢与log概率有限可以同时成立。')
    code("""for row in report['diagnostics']['extremes']:
    print(row)
with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
    bad = np.exp(np.array([1000., 0., -1000.]))
    print('Naive exponentials (intentional failure):', bad)
    print('Naive normalization:', bad / bad.sum())
p, logp = e.stable_softmax([[1000., 0., -1000.]])
with np.errstate(divide='ignore'):
    print('log(probabilities):', np.log(p))
print('Direct log probabilities:', logp)
print('Double-softmax:', e.stable_softmax([[.8, .1, .1]])[0])
for row in report['diagnostics']['finite_difference']:
    print(row)
show(9)""")
    code("""for label, kwargs in [('zero budget', {'max_updates': 0}),
                      ('short budget', {'max_updates': 1}),
                      ('initial stationary', {'initial': report['certificate']['theta']}),
                      ('rejected step', {'learning_rate': 100})]:
    args = {'initial': cfg['initial_theta'], **kwargs}
    result = e.fit(X, y, **args)
    print(label, result['status'], 'updates', result['updates'],
          'attempts', result['state_attempts'], 'successes', result['state_successes'])
    print('Second state:', result['trace'][2]['theta'] if len(result['trace']) > 2 else None)
try:
    bad_X = X.tolist(); bad_X[-1][-1] = True
    e.fit(bad_X, y, cfg['initial_theta'])
except ValueError as exc:
    print('Bad last field rejected:', str(exc))
else:
    raise RuntimeError('Invalid bool accepted')""")
    code("""import audit
verified = audit.audit()
print('Independent exact and Decimal checks:', verified)
core = hashlib.sha256(e.canonical_bytes(report)).hexdigest()
if core != verified['core_sha256']:
    raise RuntimeError('Whole Notebook scientific result differs from audit')
e.safe_write(BASE / 'outputs/notebook-result.json', report)
print('Complete scientific SHA256:', core)
print('Limitations:', report['limitations'])""")
    md('## 完成练习\n按answers.pdf核对十八题；尤其要解释同一x的冲突标签、有限最优点、KL支持集及保留的不成功状态。运行成功不是现实部署或研究新颖性的证据。')
    for i,cell in enumerate(cells):cell['id']=hashlib.sha256((str(i)+cell['source']).encode()).hexdigest()[:12]
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
    nbf.write(nb,ROOT/'experiment.ipynb');print(ROOT/'experiment.ipynb')
if __name__=='__main__':main()
