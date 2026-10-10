"""新Python进程内IPython真实顺序执行，无网络内核套接字。"""
from pathlib import Path
import argparse,json,sys
import nbformat
ROOT=Path(__file__).resolve().parent
CELLS = [('markdown', '# 第082课 概率与隐结构阶段项目\n\n主项目：合成GMM真值、真实Wine边界、无混合负对照。另加解析共轭标量后验预测基准，与主GMM固定参数复制清楚分离。'), ('code', "import json, numpy as np\nfrom pathlib import Path\nfrom latent_project import *\nfrom generate_data import make_data\nfrom experiment import run\nfrom IPython.display import display, Image\ndata=make_data()\nprint('真实特征:',data['wine_features'])\nprint('Wine划分:',{k:len(v['x']) for k,v in data['wine'].items()})\nprint('责任度手算:',np.array([.4*.2,.6*.1])/(.4*.2+.6*.1))"), ('markdown', '## 1 按验证密度选择，测试只评价'), ('code', "result,artifacts=run(True)\nfor name in ['synthetic','wine','negative']:\n    print(name,'候选:',result[name]['candidates'])\n    print('选K',result[name]['selected_k'],'测试密度',result[name]['test_log_density'])\ndisplay(Image(filename='figures/02_model_selection.png'))"), ('code', "s=result['synthetic']\nprint('合成ARI:',s['truth_ari'],'均值恢复:',s['mean_recovery'])\nprint('真参数测试密度:',s['oracle_test_log_density'])\ndisplay(Image(filename='figures/01_truth_recovery.png'))"), ('markdown', '## 2 精确离散责任度与硬近似的可验缺口'), ('code', "for name in ['synthetic','wine']:\n    row=result[name]\n    print(name,'责任度误差',row['responsibility_max_error'],'log密度误差',row['log_density_max_error'])\n    print('硬q平均ELBO缺口',row['hard_q_mean_ELBO_gap'],'平均熵',row['mean_component_entropy'])\nprint('0.5/0.5硬化缺口:',hard_variational_gap([[-3,-3]])[0])\ndisplay(Image(filename='figures/03_responsibility_gap.png'))"), ('code', "for name in ['synthetic','wine']:\n    for kind in ['restart_stability','bootstrap_stability']:\n        values=[r['ari_against_reference'] for r in result[name][kind]]\n        print(name,kind,'ARI范围',min(values),max(values))\ndisplay(Image(filename='figures/04_stability.png'))"), ('markdown', '## 3 主GMM：固定参数复制，不是完整参数后验预测'), ('code', "for name in ['synthetic','wine']:\n    print(name,json.dumps(result[name]['fixed_parameter_predictive_check'],indent=2))\ndisplay(Image(filename='figures/05_predictive_checks.png'))\nprint('真实外部ARI:',result['wine']['external_cultivar_ari'],'分量占用:',result['wine']['test_component_counts'])"), ('markdown', '## 4 独立共轭基准：每份数据共享一次后验参数抽样\n\n原始单高斯负对照第一维；噪声方差1已知，θ先验N(0,4)。此基准真正对参数后验积分，但不替代主GMM贝叶斯推断。'), ('code', "c=result['conjugate_posterior_predictive']\nprint(json.dumps(c,indent=2))\nprint('手算后验方差:',1/600.25)\nprint('复制均值方差：完整',1/600.25+1/400,'固定',1/400)\nprint('不能为同一份复制中的每个观测独立重抽theta。')"), ('code', "import unittest\nfrom test_experiment import Checks\nchecks=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))\nif not checks.wasSuccessful():\n    raise RuntimeError('unit tests failed')\nprint('实现与预声明实验验收完成，不代表真实隐结构已获证明。')"), ('markdown', '## 复盘\n\n保留合成相关检查冲突、真实重采样不稳定与品种ARI1三个结果。它们回答不同问题，不能互相替代。')]

def main():
    import os
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    parser=argparse.ArgumentParser();parser.add_argument('--out',default=str(ROOT/'outputs/experiment.ipynb'));args=parser.parse_args();output_path=Path(args.out).absolute()
    notebook=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(text) if kind=='markdown' else nbformat.v4.new_code_cell(text) for kind,text in CELLS])
    notebook.metadata.kernelspec={'display_name':'Python 3 (course)','language':'python','name':'python3'}
    notebook.metadata.execution={'engine':'fresh Python process, in-process IPython InteractiveShell; no socket kernel','order':'top-to-bottom','errors_allowed':False}
    shell=InteractiveShell.instance();os.chdir(ROOT);count=0
    for cell in notebook.cells:
        if cell.cell_type!='code':continue
        count+=1
        with capture_output(stdout=True,stderr=True,display=True) as captured:
            result=shell.run_cell(cell.source,store_history=True)
        if result.error_before_exec or result.error_in_exec:
            raise RuntimeError('Notebook cell failed: '+str(result.error_before_exec or result.error_in_exec))
        cell.execution_count=count;cell.outputs=[]
        if captured.stdout:cell.outputs.append(nbformat.v4.new_output('stream',name='stdout',text=captured.stdout))
        if captured.stderr:cell.outputs.append(nbformat.v4.new_output('stream',name='stderr',text=captured.stderr))
        for output in captured.outputs:
            cell.outputs.append(nbformat.v4.new_output('display_data',data=output.data,metadata=output.metadata))
    out=output_path;out.parent.mkdir(parents=True,exist_ok=True);nbformat.validate(notebook);nbformat.write(notebook,out)
    print(f'Executed {count} code cells sequentially in fresh-process IPython: {out}')
if __name__=='__main__':main()
