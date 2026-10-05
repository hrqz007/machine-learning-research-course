"""Generate the first-cell-contract learning-theory tutorial notebook."""
from pathlib import Path
import argparse,hashlib
import nbformat as nbf
ROOT=Path(__file__).resolve().parent
CONTRACT=['data/protocol.json','data/population.csv','data/hypotheses.csv','data/hand_sample.csv','data_integrity.json','numeric.py','protocol.py','bounds.py','finite_class.py','selection.py','experiment.py','audit.py','plots.py','requirements.txt','environment.yml']
def build():
    expected={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in CONTRACT};cells=[]
    def md(s):cells.append(nbf.v4.new_markdown_cell(s))
    def code(s):cells.append(nbf.v4.new_code_cell(s))
    code('''# First physical cell: only standard-library checks before teaching imports.
from pathlib import Path
import hashlib, json
BASE = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / 'data_integrity.json').is_file() and (p / 'experiment.py').is_file()), None)
if BASE is None: raise ValueError('Run within the complete ML049 directory')
EXPECTED = '''+repr(expected)+'''
for relative, wanted in EXPECTED.items():
    if hashlib.sha256((BASE / relative).read_bytes()).hexdigest() != wanted:
        raise ValueError('Teaching contract changed: ' + relative)
print('All', len(EXPECTED), 'input, source, and environment byte contracts verified.')''')
    md('# 第049讲 学习理论的第一组保证\n\n固定规则、同时控制、数据依赖选择和独立留出各有不同对象。下面所有真实风险来自已知合成总体；不会用训练分数冒充未知总体。')
    code('''import os,sys,io,math
sys.path.insert(0,str(BASE))
os.environ.setdefault('MPLCONFIGDIR',str(BASE/'outputs/mpl'))
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display,Image
from fractions import Fraction
import experiment as e
import finite_class as fc
import selection as sel
import bounds,plots
from numeric import canonical_bytes,safe_write
pop,hyp,sample,cfg=e.load_inputs()
report=e.main_report(pop,hyp,sample,cfg)
def show(i):
    f=plots.figure(i,report);s=io.BytesIO()
    f.savefig(s,format='png',dpi=160,bbox_inches='tight')
    display(Image(data=s.getvalue(),format='png'));plt.close(f)
print('Full finite population:',pop)
print('Predeclared hypotheses:',hyp)
print('Population risks:',report['population'])
show(1)''')
    md('## 整批前向→逐行损失→汇总→选择→下一前向\n离散枚举没有梯度步骤。新增两行明确属于训练数据，不能把它们说成封存测试。')
    code('''for stage in ['initial','after_batch']:
    value=report['hand'][stage]
    print('STAGE',stage)
    for row in value['rows']: print(json.dumps(row,sort_keys=True))
    for key in ['sample_size','loss_counts','empirical_risks','population_risk_fractions','tie_indices','selected_index','selected_id']:
        print(key,value[key])
print('Next forward:',report['hand']['next_forward_on_all_four_inputs'])
print('Optimization definition:',report['hand']['initial']['optimization'])
show(2)''')
    md('## 公式不是无条件承诺\nHoeffding需要观测独立、有界损失；union bound不需要候选模型独立。')
    code('''for n,M in [(8,5),(20,1),(20,512),(100,512)]:
    print('n,M,delta,radius:',n,M,.05,bounds.radius(n,.05,M))
print('Exact tail bound example:',bounds.tail(8,.5,5))
for label,operation in [
    ('bool n',lambda:bounds.radius(True,.05)),
    ('delta0',lambda:bounds.radius(20,0)),
    ('complex delta',lambda:bounds.radius(20,.05+0j)),
    ('invalid unused candidate column',lambda:sel.choose([[0,21]],20,1))]:
    try:operation()
    except ValueError as ex:print('Expected rejection:',label,str(ex))
    else:raise RuntimeError('Invalid input accepted: '+label)''')
    code('''for block in report['exact']:
    print('EXACT ENUMERATION',json.dumps(block,ensure_ascii=False,sort_keys=True))
print('n=4 ordered sequences:',8**4,'compositions:',math.comb(11,7))
print('n=8 ordered sequences:',8**8,'compositions:',math.comb(15,7))
show(3)''')
    md('## 14000次IID主重复\n完整计数留在report和最后输出JSON；这里展示每个n的前3次与完整汇总，避免将大量重复文本当作教学深度。')
    code('''for block in report['main_repetitions']:
    print('N=',block['n'],'uniform radius=',block['uniform_radius'])
    print('First3 actual records:',block['records'][:3])
    print('All summaries:',json.dumps(block['summary'],sort_keys=True))
    print('Violation counts:',json.dumps(block['violations'],sort_keys=True))
    if len(block['records'])!=cfg['main_repetitions']:raise RuntimeError('Missing repeated records')
show(4)''')
    md('## 纯噪声筛选：每个规则真实风险都为1/2\n候选间独立是此特定反例的构造，用于解析分布计算；不是union bound额外要求。')
    code('''for block in report['selection_noise']:
    print('NOISE SAMPLE SIZE',block['n'])
    for scenario in block['scenarios']:
        print('M=',scenario['M'],'mean train risk=',scenario['mean_training_risk'])
        print('Misused fixed radius events:',scenario['fixed_radius_violations'])
        print('Correct uniform radius events:',scenario['uniform_radius_violations'])
        print('Analytic probabilities:',{k:v for k,v in scenario['analytic'].items() if k!='minimum_pmf'})
        print('First8 chosen indices/counts:',scenario['selection']['selected_indices'][:8],scenario['selection']['selected_error_counts'][:8])
q=Fraction(sum(math.comb(20,k) for k in range(4)),2**20)
print('Exact single lower-tail q=',str(q),'float=',float(q))
show(5)''')
    code('''for scenario in report['selection_noise'][0]['scenarios']:
    print('M=',scenario['M'],'analytic minimum PMF:',scenario['analytic']['minimum_pmf'])
    print('Actual minimum count histogram:',scenario['selected_error_count_histogram'])
    if sum(scenario['selected_error_count_histogram'])!=cfg['noise_repetitions']:
        raise RuntimeError('Histogram count mismatch')
show(6)''')
    md('## PAC样本量公式：两个不同前提\n右图可实现一致学习器的结论不能直接用在有噪声主总体。')
    code('''for row in report['sample_complexity_table']:print(json.dumps(row,sort_keys=True))
print('Main population minimum risk:',report['population']['minimum_risk'])
print('Realizable zero-risk premise holds?',report['population']['minimum_risk']==0)
show(7)''')
    code('''print('DEPENDENCE COUNTEREXAMPLE:',json.dumps(report['dependence_counterexample'],sort_keys=True))
for bit in [0,1]:
    copied=np.repeat(bit,cfg['dependent_repeat_count'])
    print('Single bit, copied mean, absolute gap:',bit,float(copied.mean()),abs(float(copied.mean())-.5))
show(8)''')
    md('## 独立留出：先冻结选择再生成评价结果\n训练侧和留出侧按各自样本数算半径，不能把不同数据角色混成一个分数。')
    code('''for block in report['selection_noise']:
    holdout=block['independent_holdout']
    print('Training n=',block['n'],'holdout:',{k:v for k,v in holdout.items() if k!='error_counts'})
    print('First10 holdout error counts:',holdout['error_counts'][:10])
    frozen=block['scenarios'][-1]['selection']
    if hashlib.sha256(canonical_bytes(frozen)).hexdigest()!=holdout['training_choice_sha256']:
        raise RuntimeError('Choice hash differs')
print('Standalone choice API has no holdout argument:',sel.choose([[3,2,5],[1,1,4]],20,3))
show(9)''')
    code('''for row in report['vc']['ordered_distinct_point_patterns']:
    print('POINT COUNT',row['points'],'counts',row['threshold_count'],row['interval_count'],row['all_labelings'])
    if row['points']<=3:
        print('Threshold patterns/missing:',row['threshold_patterns'],row['threshold_missing'])
        print('Interval patterns/missing:',row['interval_patterns'],row['interval_missing'])
print('Scope:',report['vc']['scope'])
show(10)''')
    md('## 独立核验与全部结果\n4096条有序序列、精确二项尾概率、Decimal、实际SciPy分布与随机流重建。Monte Carlo偏离解析值记录为诊断，不调整种子或用随意门槛筛结果。')
    code('''import audit
checks=audit.audit()
print(json.dumps(checks,ensure_ascii=False,sort_keys=True))
science_hash=hashlib.sha256(canonical_bytes(report)).hexdigest()
if checks['core_sha256']!=science_hash:raise RuntimeError('Notebook and script differ')
safe_write(BASE/'outputs/notebook-result.json',report)
print('Saved every repeated record. Scientific SHA256:',science_hash)''')
    for i,cell in enumerate(cells):cell['id']='ml049-'+str(i).zfill(3)
    return nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(ROOT/'learning_guarantees.ipynb'));a=p.parse_args();out=Path(a.out).absolute()
    if out.suffix!='.ipynb':raise ValueError('ipynb output required')
    for part in (out,*out.parents):
        if part.is_symlink():raise ValueError('symlink notebook output')
    if out.exists() and (not out.is_file() or out.stat().st_nlink>1):raise ValueError('unsafe notebook output')
    out.parent.mkdir(parents=True,exist_ok=True);nbf.write(build(),out);print(out)
if __name__=='__main__':main()
