"""从可信模板创建并在当前Python的新内核中真实顺序执行Notebook。"""
from pathlib import Path
import argparse,json,sys,tempfile
import nbformat
from nbclient import NotebookClient
from jupyter_client.kernelspec import KernelSpecManager
from jupyter_client import AsyncKernelManager
ROOT=Path(__file__).resolve().parent
CELLS = [('markdown', '# 第077课 精确概率推断\n\n给定参数，计算未知状态的边缘与后验。穷举、变量消元与两遍消息传递独立实现后相互比较。'), ('code', "import json\nfrom pathlib import Path\nfrom exact_inference import *\nfrom experiment import run\nfrom IPython.display import display, Image\nparams=json.loads(Path('data/chain.json').read_text())\nfactors=chain_factors(**params)\nprint(params)\nprint('各因子范围:',[f.scope for f in factors])"), ('markdown', '## 1 将末端观测固定，再消去隐藏状态'), ('code', "ve,z,trace=eliminate(factors,'X0',{'E':1},['X3','X2','X1'])\nprint('X0的后验:',ve.table)\nprint('证据概率:',z)\nfor row in trace:\n    print(row)  # 清楚显示每次临时范围和条目数。\nbrute,bz=enumerate_query(factors,'X0',{'E':1})\nprint('穷举对照:',brute.table,'差:',abs(brute.table[(1,)]-ve.table[(1,)]))"), ('code', "marginals,mass,forward,backward=chain_sum_product(**params)\nfor i,(a,b,m) in enumerate(zip(forward,backward,marginals)):\n    print('X'+str(i),'前向',a,'后向',b,'归一化结果',m)\nprint('证据概率:',mass)\ndisplay(Image(filename='figures/04_posteriors.png'))"), ('markdown', '## 2 相同精确答案可能需要不同大小的中间表'), ('code', "star=star_factors()\nfor order in (['L0','L1','L2','L3','L4','C'],['C','L0','L1','L2','L3','L4']):\n    value,mass,trace=eliminate(star,'L5',order=order)\n    print('顺序:',order)\n    print('P(L5=1):',value.table[(1,)],'Z:',mass,'峰值条目:',max(t['entries'] for t in trace))\ndisplay(Image(filename='figures/05_order.png'))"), ('code', "impossible=chain_factors([.6,.4],[[.85,.15],[.25,.75]],2,[0.,0.])\ntry:\n    eliminate(impossible,'X0',{'E':1})\nexcept ValueError as error:\n    print('正确拒绝零概率证据:',str(error))\nelse:\n    raise RuntimeError('impossible evidence was not rejected')\nprint('完整实验:',run()['max_method_error'])"), ('code', "import unittest\nfrom test_experiment import Checks\ntests=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))\nif not tests.wasSuccessful():\n    raise RuntimeError('unit tests failed')\nprint('本Notebook所有计算和测试完成。')"), ('markdown', '## 复盘\n\n消元改变计算顺序，不改变目标分布。树上两遍sum-product为精确推断；一般有环图直接迭代消息不享有同样保证。')]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',default=str(ROOT/'outputs/experiment.ipynb'));args=parser.parse_args()
    notebook=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(text) if kind=='markdown' else nbformat.v4.new_code_cell(text) for kind,text in CELLS])
    notebook.metadata.kernelspec={'display_name':'Python 3 (course)','language':'python','name':'python3'}
    with tempfile.TemporaryDirectory(prefix='course-kernel-') as temporary:
        kernel=Path(temporary)/'course-current';kernel.mkdir()
        (kernel/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Course current Python','language':'python'}))
        manager=KernelSpecManager(kernel_dirs=[temporary])
        kernel_manager=AsyncKernelManager(kernel_name='course-current',kernel_spec_manager=manager)
        client=NotebookClient(notebook,timeout=120,km=kernel_manager,resources={'metadata':{'path':str(ROOT)}},allow_errors=False)
        client.execute()
    for cell in notebook.cells:
        if cell.cell_type=='code':
            if cell.execution_count is None or any(output.output_type=='error' for output in cell.outputs):
                raise RuntimeError('Notebook execution incomplete')
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True);nbformat.write(notebook,out)
    print(f'Executed {sum(cell.cell_type=="code" for cell in notebook.cells)} code cells: {out}')
if __name__=='__main__':main()
