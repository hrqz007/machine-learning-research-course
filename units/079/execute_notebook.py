"""新Python进程内IPython逐单元执行并捕获真实输出，无socket内核。"""
from pathlib import Path
import argparse,json
import nbformat
ROOT=Path(__file__).resolve().parent
CELLS = [('markdown', '# 第079课 变分推断与ELBO\n从可解析后验检查每一步更新和最终偏差。'), ('code', "from pathlib import Path\nimport json, numpy as np\nfrom variational import *\nfrom experiment import run\nfrom IPython.display import display, Image\nmodel=json.loads(Path('data/correlated.json').read_text())\np=posterior(**model)\nprint('解析均值:',p['mean'])\nprint('解析协方差:',p['covariance'])\nprint('证据:',p['log_evidence'])\ndisplay(Image(filename='figures/01_geometry.png'))"), ('markdown', '## 1 两轮手算与坐标轨迹'), ('code', "short=coordinate_mean_field(p['precision'],p['natural_mean'],max_sweeps=2)\nfor row in short['trace']: print(row)\nprint('是否已收敛:',short['converged'])\ndisplay(Image(filename='figures/03_coordinates.png'))"), ('code', "r=run()\nfor key in ['q_mean','q_variance','variance_ratio','elbo','kl_q_p','max_mean_error','max_decomposition_error']: print(key,r[key])\ndisplay(Image(filename='figures/02_elbo.png'))"), ('markdown', '## 2 方向与最终统计量'), ('code', "print('另一方向最优:',r['forward_kl_optimum'])\nprint('线性组合:',json.dumps(r['linear_functionals'],indent=2))\nprint('独立对照:',r['independent_control'])\ndisplay(Image(filename='figures/04_direction.png'))\ndisplay(Image(filename='figures/05_noise.png'))"), ('code', "try:\n    coordinate_mean_field([[1.,2.],[2.,1.]],[0.,0.])\nexcept ValueError as error:\n    print('正确拒绝非正定矩阵:',error)\nelse:\n    raise RuntimeError('invalid matrix accepted')\nimport unittest\nfrom test_experiment import Checks\ntests=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))\nif not tests.wasSuccessful(): raise RuntimeError('tests failed')"), ('markdown', '## 结论\n本例均值恢复，但方差和相关性没有恢复。不同线性组合的方差偏差甚至方向相反。')]

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
