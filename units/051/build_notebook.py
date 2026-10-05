"""Create a real nested-CV notebook with a first-cell byte contract."""
from pathlib import Path
import argparse,hashlib
import nbformat as nbf
ROOT=Path(__file__).resolve().parent
CONTRACT=['data/protocol.json','data/development.csv','data/repeated_development.csv','data_integrity.json','numeric.py','protocol.py','models.py','folds.py','cross_validation.py','hand_example.py','generate_data.py','experiment.py','reference.py','audit.py','plots.py','requirements.txt','environment.yml']
def build():
    expected={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in CONTRACT};cells=[]
    def code(s):cells.append(nbf.v4.new_code_cell(s))
    def md(s):cells.append(nbf.v4.new_markdown_cell(s))
    code('''# First physical cell verifies all bytes before teaching-code imports.
from pathlib import Path
import hashlib,json
BASE=next((p for p in [Path.cwd(),*Path.cwd().parents]
           if (p/'data_integrity.json').is_file() and (p/'experiment.py').is_file()),None)
if BASE is None:raise ValueError('Run within the complete ML051 folder')
EXPECTED='''+repr(expected)+'''
for relative,wanted in EXPECTED.items():
    if hashlib.sha256((BASE/relative).read_bytes()).hexdigest()!=wanted:
        raise ValueError('Teaching contract changed: '+relative)
print('All',len(EXPECTED),'data/source/environment contracts verified.')''')
    md('# 第051讲 交叉验证与模型选择\n\n先锁定候选和数据角色，再评价包含内层选参的完整过程。所有数据为合成，oracle只作诊断。')
    code('''import os,sys,io,math
sys.path.insert(0,str(BASE))
os.environ.setdefault('MPLCONFIGDIR',str(BASE/'outputs/mpl'))
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image,display
import experiment as e
import models,folds,cross_validation as cv,plots
from numeric import canonical_bytes,safe_write
main,repeated,cfg=e.load_inputs()
report=e.main_report(main,repeated,cfg)
def show(i):
    f=plots.figure(i,report);b=io.BytesIO()
    f.savefig(b,format='png',dpi=160,bbox_inches='tight')
    display(Image(data=b.getvalue(),format='png'));plt.close(f)
print('Protocol:',cfg)
print('Primary outer splits:',report['primary']['outer_splits'])
show(1)''')
    md('## 四行纸笔内层选择\n每个候选都接受两折评价，选出λ后用全部外层训练行重拟合。')
    code('''h=report['hand']
for candidate in h['candidates']:
    print('PENALTY',candidate['penalty'])
    for inner in candidate['folds']:
        print('Fit:',inner['fit'],'train',inner['training_indices'],'validation',inner['validation_indices'])
        for row in inner['validation_rows']:print(row)
    print('Pooled CV MSE:',candidate['pooled_CV_MSE'])
print('Selected penalty:',h['selected_penalty'])
print('Refit all four:',h['outer_training_refit'])
print('Outer test rows and MSE:',h['outer_test_rows'],h['outer_test_MSE'])
show(2)''')
    code('''for state in h['gradient_trace']:
    print('STAGE',state['stage'],'weight',state['weight'])
    for row in state['rows']:print(json.dumps(row,sort_keys=True))
    print({k:v for k,v in state.items() if k!='rows'})
    if state['stage']<2:print('Next synchronous weight:',state['weight']-cfg['hand_learning_rate']*state['total_gradient'])
print('Fixed-trace closed-form optimum:',h['trace_closed_form_optimum'])
show(3)''')
    md('## 固定基函数与惩罚缩放\nLegendre基不是从全数据fit得到的。λ与sklearn alpha需按训练行数换算。')
    code('''print('All17 candidates:',cfg['candidates'])
for n in [48,64,80]:print('n,lambda,sklearn alpha:',n,.01,n*.01)
from reference import ReferenceRidge
fit=models.fit(main['x'],main['y'],{'degree':3,'penalty':.01})
reference=ReferenceRidge(degree=3,penalty=.01).fit(main['x'][:,None],main['y'])
print('Augmented least-squares coefficients:',fit['coefficients'])
print('Actual sklearn Ridge coefficients:',reference.coefficients_.tolist())
if not np.allclose(fit['coefficients'],reference.coefficients_,rtol=1e-10,atol=1e-12):raise RuntimeError('Ridge objective mismatch')
print('Unequal fold example pooled versus unweighted:',(3*0+2*1+2*4)/7,(0+1+4)/3)''')
    code('''primary=report['primary']
for outer in primary['nested']['folds']:
    print('OUTER FOLD',outer['outer_fold'])
    print('Outer train/validation:',outer['training_indices'],outer['validation_indices'])
    s=outer['choice']['selection']
    for inner in s['folds']:
        print('Inner global train/validation:',inner['training_indices_global'],inner['validation_indices_global'])
        print('All candidate MSE:',inner['candidate_MSE'])
    print('Pooled scores and selection:',s['candidate_scores'],s['selected_index'],s['selected_candidate'])
    print('Choice hash before outer score:',outer['choice_sha256'])
print('Flat reused selection:',primary['flat']['selection']['selected_candidate'])
show(4)''')
    code('''for outer in primary['nested']['folds']:
    print('Outer fold:',outer['outer_fold'],'predictions:',outer['validation_predictions'])
    print('Squared errors:',outer['validation_squared_errors'])
    print('MSE and same-model oracle:',outer['validation_MSE'],outer['oracle_risk_same_fitted_model'])
print('Flat CV/oracle:',primary['flat']['minimum_reused_CV_MSE'],primary['flat']['mean_oracle_risk_same_selected_fold_models'])
print('Nested CV/oracle:',primary['nested']['pooled_outer_MSE'],primary['nested']['mean_oracle_risk_same_outer_models'])
show(5)''')
    md('## 80个独立数据集 不把五折当独立实验\n完整索引与候选分数在report中保存，下面打印每份数据的关键配对结果。')
    code('''for row in report['repeated']:
    result=row['result'];print(row['repetition'],
       'flat CV/oracle/gap',result['flat']['minimum_reused_CV_MSE'],result['flat']['mean_oracle_risk_same_selected_fold_models'],result['flat']['oracle_minus_CV'],
       'nested CV/oracle/gap',result['nested']['pooled_outer_MSE'],result['nested']['mean_oracle_risk_same_outer_models'],result['nested']['oracle_minus_CV'])
print('Summary:',json.dumps(report['repeated_summary'],sort_keys=True))
show(6)''')
    code('''differences=np.array([row['result']['flat']['minimum_reused_CV_MSE']-row['result']['nested']['pooled_outer_MSE'] for row in report['repeated']])
print('Flat score lower/equal/higher:',int((differences<0).sum()),int((differences==0).sum()),int((differences>0).sum()))
print('Final selected indices:',[row['result']['final_choice']['selected_index'] for row in report['repeated']])
print('Negative optimism counts:',report['repeated_summary']['flat_optimism']['negative_count'],report['repeated_summary']['nested_optimism']['negative_count'])
show(7)''')
    code('''demo=report['splitter_demo']
print('Labels:',demo['binary_labels'],'groups:',demo['groups'])
for name,rows in demo['splitters'].items():
    print('SPLITTER',name)
    for row in rows:print(json.dumps(row,sort_keys=True))
print('Scope:',demo['scope'])
show(8)''')
    code('''training=[set(row['training_indices']) for row in primary['outer_splits']]
overlap=np.array([[len(a&b)/len(a) for b in training] for a in training])
print('Training-row overlap fraction, NOT correlation:',overlap)
print('Fold MSEs:',[row['validation_MSE'] for row in primary['nested']['folds']])
print('No independent-fold t test is performed.')
show(9)''')
    md('## 最终80行模型与独立4000行测试\n不从五个外折模型挑最好者，最终模型按预定流程在全部开发数据选择与拟合。')
    code('''print('Final choice:',primary['final_choice'])
print('Final choice SHA256:',primary['final_choice_sha256'])
test=report['final_test']
print('Choice recorded before test:',test['choice_sha256_before_test'])
print('Final test MSE/conditional SE:',test['MSE'],test['conditional_test_mean_loss_SE'])
print('Same final model oracle:',primary['final_model_oracle_risk'])
for i in range(5):print('Test row',test['test_data']['id'][i],test['test_data']['x'][i],test['test_data']['y'][i],test['predictions'][i],test['squared_losses'][i])
show(10)''')
    md('## 独立数学与库参照\nFraction链、不同基函数的QR、实际Ridge/GridSearchCV和独立求积；信息流变换与普通输入检查。')
    code('''import audit
checks=audit.audit(report)
print(json.dumps(checks,ensure_ascii=False,sort_keys=True))
scientific_hash=hashlib.sha256(canonical_bytes(report)).hexdigest()
if checks['core_sha256']!=scientific_hash:raise RuntimeError('Notebook report and audit differ')
safe_write(BASE/'outputs/notebook-result.json',report)
print('Saved full scientific report. SHA256:',scientific_hash)''')
    for i,cell in enumerate(cells):cell['id']='ml051-'+str(i).zfill(3)
    return nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(ROOT/'nested_cross_validation.ipynb'));a=p.parse_args();out=Path(a.out).absolute()
    if out.suffix!='.ipynb':raise ValueError('ipynb output required')
    for part in (out,*out.parents):
        if part.is_symlink():raise ValueError('symlink output')
    if out.exists() and (not out.is_file() or out.stat().st_nlink>1):raise ValueError('unsafe notebook output')
    out.parent.mkdir(parents=True,exist_ok=True);nbf.write(build(),out);print(out)
if __name__=='__main__':main()
