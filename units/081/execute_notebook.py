"""新Python进程内IPython真实顺序执行，无网络内核套接字。"""
from pathlib import Path
import argparse,json,sys
import nbformat
ROOT=Path(__file__).resolve().parent
CELLS = [('markdown', '# 第081课 异常检测与密度估计\n\n训练只用正常参考；阈值只用干净校准。先手算，再看机制与真实负例。'), ('code', "import json, numpy as np\nfrom pathlib import Path\nfrom anomaly import *\nfrom generate_data import make_data\nfrom experiment import run\nfrom IPython.display import display, Image\ndata=make_data()\nprint('种子:',data['seed'])\nprint('真实各集合大小:',{k:len(v['x']) for k,v in data['digits'].items()})"), ('markdown', '## 1 邻居半径与秩阈值手算'), ('code', "print('第二近邻距离:',KNNRadius(2).fit([[0],[2],[5]]).score([[1],[4]]))\nc=calibrate([1,2,3,4,5],.34)\nprint('校准:',c,'分数4/5报警:',c.alarms([4,5]))\nprint('样本太少:',calibrate([1,2],.05))"), ('markdown', '## 2 冻结流程执行全部数据条件'), ('code', "result,artifacts=run(True)\nfor method,row in result['synthetic']['methods'].items():\n    print(method,'FP',row['false_positives'],'/',row['normal_n'],'召回',row['recall'],'分机制',row['recall_by_kind'])\ndisplay(Image(filename='figures/01_mechanisms.png'))\ndisplay(Image(filename='figures/02_boundaries.png'))"), ('code', "for method,row in result['synthetic']['methods'].items():\n    print(method,'阈值',row['threshold'],'边际界',row['marginal_fpr_bound'],'正常子群',row['normal_group_fpr'])\ndisplay(Image(filename='figures/03_calibration.png'))\nprint('边际保证不是本次校准集的条件5%保证，也不是子群保证。')"), ('code', "N,pi,fpr,recall=10000,.001,.05,.9\nfp=N*(1-pi)*fpr;tp=N*pi*recall\nprint('情景推算真报警/误报/精确率:',tp,fp,tp/(tp+fp))\ndisplay(Image(filename='figures/04_budget.png'))"), ('markdown', '## 3 污染单独研究，真实负例不得删除'), ('code', "for method,row in result['contaminated_training']['methods'].items():\n    print('污染训练',method,'误报',row['fpr'],'召回',row['recall'])\nfor method,row in result['digits']['methods'].items():\n    print('真实Digits',method,'FP',row['false_positives'],'/',row['normal_n'],'区间',row['fpr_wilson95'],'TP',row['true_positives'],'/',row['anomaly_n'])\ndisplay(Image(filename='figures/05_real_negatives.png'))"), ('code', "try:\n    calibrate([1,np.nan])\nexcept ValueError as error:\n    print('正确拒绝非法分数:',error)\nelse:\n    raise RuntimeError('NaN was not rejected')"), ('code', "import unittest\nfrom test_experiment import Checks\nchecks=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))\nif not checks.wasSuccessful():\n    raise RuntimeError('unit tests failed')\nprint('全部测试通过；真实部署与漂移未被本实验验证。')"), ('markdown', '## 复盘\n\n高分不是有害概率；有限误报预算必须与基率、复核容量和正常子群一起解释。')]

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
