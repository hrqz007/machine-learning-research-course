"""从可信模板创建并在当前Python的新内核中真实顺序执行Notebook。"""
from pathlib import Path
import argparse,json,sys,tempfile
import nbformat
from nbclient import NotebookClient
from jupyter_client.kernelspec import KernelSpecManager
from jupyter_client import AsyncKernelManager
ROOT=Path(__file__).resolve().parent
CELLS = [('markdown', '# 第076课 概率图模型的语言\n\n先枚举联合表，再分别检查概率独立与图分离。所有参数原创，执行不下载数据。'), ('code', "from generate_data import model  # 读取人工参数。\nfrom graph_models import *  # 读取本课小图算法。\nfrom experiment import run  # 运行可重建的整体实验。\nfrom IPython.display import display, Image  # 内嵌显示图片。\nmodels = model()\nchain = joint_from_bn(**models['chain'])\nfor row in chain:\n    print(row)  # 八行完整状态，每行含联合概率。"), ('markdown', '## 1 条件独立不是边缘独立\n\n下面先打印联合总和，再比较两个不同条件集合的最大独立残差。'), ('code', "print('总和:', probability(chain, {}))  # 空事件包含全部状态。\nprint('A,C边缘残差:', ci_residual(chain, 'A', 'C'))\nprint('A,C给定B残差:', ci_residual(chain, 'A', 'C', ['B']))\nprint('P(A=1,B=1,C=0):', probability(chain, {'A':1,'B':1,'C':0}))"), ('code', "wet = joint_from_bn(**models['collider'])\nfor evidence in ({}, {'W':1}, {'W':1,'S':1}, {'W':1,'S':0}):\n    print(evidence, conditional(wet, {'R':1}, evidence))  # 条件集合每次明确显示。\ndisplay(Image(filename='figures/03_explaining_away.png'))"), ('markdown', '## 2 图允许的关系与具体参数分开检验'), ('code', "nodes=['R','S','W','D']\nedges=[('R','W'),('S','W'),('W','D')]\nprint('无证据时d分离:', d_separated(nodes,edges,'R','S'))\nprint('观测后代D时d分离:',d_separated(nodes,edges,'R','S',['D']))\nextra=joint_from_bn(['A','B'],[('A','B')],{'A':[.4],'B':[.3,.3]})\nprint('有边但额外独立的残差:',ci_residual(extra,'A','B'))"), ('code', "mrf,z=undirected_chain()\nprint('Z:',z,'总概率:',probability(mrf,{}))\nprint('P(A=1,C=1|B=1):',conditional(mrf,{'A':1,'C':1},{'B':1}))\nresult=run()\nprint('反向链最大差:',result['reverse_max_error'])\nprint('Markov等价:',result['markov_equivalent'])\ndisplay(Image(filename='figures/04_equivalence.png'))"), ('code', "import unittest  # 显式测试不受python -O去除assert影响。\nfrom test_experiment import Checks\ntests=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))\nif not tests.wasSuccessful():\n    raise RuntimeError('unit tests failed')\nprint('本Notebook所有计算和测试完成。')"), ('markdown', '## 复盘\n\n数值独立是固定概率表的性质；d分离是因子化分布家族的结构保证。测试和枚举不能自动转化为现实因果结论。')]

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
