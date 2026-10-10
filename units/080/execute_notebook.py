"""新Python进程内IPython逐单元执行并捕获真实输出，无socket内核。"""
from pathlib import Path
import argparse,json
import nbformat
ROOT=Path(__file__).resolve().parent
CELLS = [('markdown', '# 第080课 半监督与弱监督\n固定预算与噪声，保留所有配置，不用测试标签选择参数。'), ('code', "import json, numpy as np\nfrom experiment import run,stress_experiment\nfrom supervision import *\nfrom IPython.display import display, Image\nr=run()\nprint(json.dumps(r['protocol'],ensure_ascii=False,indent=2))\ndisplay(Image(filename='figures/01_information.png'))"), ('markdown', '## 1 全部预定配置与不可训练条件'), ('code', "for row in r['summaries']: print(row)\nprint('不可训练:',[v for v in r['runs'] if v['status']!='ok'])\ndisplay(Image(filename='figures/03_results.png'))"), ('code', "example=next(v for v in r['runs'] if v['seed']==7 and v['ratio']==.1 and v['noise']==.25)\nprint('逐轮审计:',json.dumps(example,ensure_ascii=False,indent=2))\ndisplay(Image(filename='figures/02_cycle.png'))"), ('markdown', '## 2 120个全错伪标签与置信度'), ('code', "print(json.dumps(r['stress'],ensure_ascii=False,indent=2))\ndisplay(Image(filename='figures/04_reinforcement.png'))"), ('markdown', '## 3 弃权、冲突和复制规则'), ('code', "print('多数票:',majority_vote([[-1,-1],[0,1],[1,1],[0,-1]]))\nprint('两票相抵:',independent_label_probability([[1,0]],[.8,.8]))\nprint('复制正票:',independent_label_probability([[1,1,0]],[.8,.8,.8]))\nprint('弱标签审计:',r['weak_labels'])\ndisplay(Image(filename='figures/05_weak_labels.png'))"), ('code', "import unittest\nfrom test_experiment import Checks\ntests=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))\nif not tests.wasSuccessful(): raise RuntimeError('tests failed')\nprint('全部测试通过；不代表无标签数据一定有帮助。')"), ('markdown', '## 结论\n训练与审计的信息分开。高置信度不是可靠性证明，重复规则不是新证据。')]

def main():
    import os
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    parser=argparse.ArgumentParser();parser.add_argument('--out',default=str(ROOT/'outputs/experiment.ipynb'));args=parser.parse_args()
    notebook=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(text) if kind=='markdown' else nbformat.v4.new_code_cell(text) for kind,text in CELLS])
    notebook.metadata.kernelspec={'display_name':'Python 3 (course)','language':'python','name':'python3'}
    notebook.metadata.execution={'engine':'fresh Python process, in-process IPython InteractiveShell; no socket kernel','order':'top-to-bottom','errors_allowed':False}
    os.environ.setdefault('IPYTHONDIR',str(ROOT/'outputs/ipython'))
    Path(os.environ['IPYTHONDIR']).mkdir(parents=True,exist_ok=True)
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
