"""在新Python进程内以IPython逐单元真实执行；不需要本地套接字。"""
from pathlib import Path
import argparse,json,sys,tempfile
import nbformat
from nbclient import NotebookClient
from jupyter_client.kernelspec import KernelSpecManager
from jupyter_client import AsyncKernelManager
ROOT=Path(__file__).resolve().parent
CELLS = [('markdown', '# 第078课 Monte Carlo与MCMC\n先看解析真值，再看随机误差和失败对照。'), ('code', "from pathlib import Path\nimport json, numpy as np\nfrom samplers import *\nfrom experiment import run\nfrom IPython.display import display, Image\nmodel=json.loads(Path('data/model.json').read_text())\ny=np.array(model['observations'])\nm,v=analytic_posterior(y)\nprint('解析均值/方差/标准差:',m,v,v**.5)"), ('markdown', '## 独立Monte Carlo与重要性采样'), ('code', "result,chains=run(True)\nprint('面积估计:',result['monte_carlo_integral'])\nprint('独立后验样本:',result['iid'])\nprint('重要性采样:',result['importance_sampling'])\ndisplay(Image(filename='figures/02_importance.png'))"), ('markdown', '## 四链三步长：相同预算，不同探索'), ('code', "for name,row in result['mh'].items():\n    print(name,json.dumps(row,ensure_ascii=False,indent=2))\ndisplay(Image(filename='figures/04_trace.png'))"), ('code', "for name,a in chains.items():\n    print(name,'链1前10个ACF:',autocorrelation(a[0],10))\n    print('重复相邻状态比例:',float(np.mean(np.diff(a[0])==0)))\ndisplay(Image(filename='figures/05_diagnostics.png'))\nprint('注意tiny链不平稳，MCSE没有可靠解释。')"), ('markdown', '## 条件更新与相关后验的几何'), ('code', "print(json.dumps(result['gibbs'],indent=2))\nprint('rho=.95条件标准差:',np.sqrt(1-.95**2))\ntry:\n    gibbs_gaussian(1,100,0)\nexcept ValueError as error:\n    print('正确拒绝奇异条件:',error)"), ('code', "import unittest\nfrom test_experiment import Checks\nr=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))\nif not r.wasSuccessful():\n    raise RuntimeError('unit tests failed')\nprint('全部单元测试通过；这不等于所有设置都混合。')"), ('markdown', '## 结论\n样本数不是有效信息量。比较解析目标、轨迹、相关性与多链诊断，再决定哪些摘要有资格被解释。')]

def main():
    import os
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    parser=argparse.ArgumentParser();parser.add_argument('--out',default=str(ROOT/'outputs/experiment.ipynb'));args=parser.parse_args()
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
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True);nbformat.validate(notebook);nbformat.write(notebook,out)
    print(f'Executed {count} code cells sequentially in fresh-process IPython: {out}')
if __name__=='__main__':main()
