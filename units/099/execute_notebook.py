import os, tempfile
_CACHE = tempfile.mkdtemp(prefix='course-cache-')
os.environ.setdefault('MPLCONFIGDIR', _CACHE)
os.environ.setdefault('XDG_CACHE_HOME', _CACHE)
os.environ.setdefault('IPYTHONDIR', _CACHE)
"""Create and genuinely execute the notebook with the same Python environment."""
from pathlib import Path
import json,sys,time
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
import os
ROOT=Path(__file__).resolve().parent
nb=nbformat.v4.new_notebook()
nb.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}
nb.cells=[nbformat.v4.new_markdown_cell('# 第099讲 CPU可复现实验\n先读lecture.md与lab.md。本笔记本实际运行相同脚本、测试并显示图像；输出不是手工粘贴。'),
nbformat.v4.new_code_cell("import sys, pathlib, subprocess, json\nprint(sys.version)\nprint('Experiment file present:', pathlib.Path('experiment.py').is_file())"),
nbformat.v4.new_code_cell("result=subprocess.run([sys.executable,'experiment.py'],capture_output=True,text=True,check=True)\nprint(result.stdout)\nprint('Script exit:',result.returncode)"),
nbformat.v4.new_code_cell("result=subprocess.run([sys.executable,'-m','unittest','-v','test_experiment.py'],capture_output=True,text=True,check=True)\nprint(result.stdout+'\\n'.join(line for line in result.stderr.splitlines() if line.strip() != 'Fontconfig error: No writable cache directories'))"),
nbformat.v4.new_code_cell("data=json.loads(pathlib.Path('outputs/results.json').read_text())\nprint('Result fields:', list(data))"),
nbformat.v4.new_code_cell("from IPython.display import display, Image\nfor p in sorted(pathlib.Path('figures').glob('*.png')):\n    print(p.name)\n    display(Image(filename=str(p)))"),
nbformat.v4.new_markdown_cell('解释任务：逐项说明图中的控制变量、预算口径、失败对照和不能支持的结论。详细答案见answers.md。')]
# Every code cell has its own execution_count and real kernel outputs.
started=time.time()
os.chdir(ROOT)
nb.metadata.execution={'engine':'fresh process, in-process IPython; no socket kernel','order':'top-to-bottom','errors_allowed':False}
shell=InteractiveShell.instance();count=0
for cell in nb.cells:
    if cell.cell_type!='code':continue
    count+=1
    with capture_output(stdout=True,stderr=True,display=True) as captured:
        result=shell.run_cell(cell.source,store_history=True)
    if result.error_before_exec or result.error_in_exec:
        raise RuntimeError(str(result.error_before_exec or result.error_in_exec))
    cell.execution_count=count;cell.outputs=[]
    if captured.stdout:cell.outputs.append(nbformat.v4.new_output('stream',name='stdout',text=captured.stdout))
    if captured.stderr:cell.outputs.append(nbformat.v4.new_output('stream',name='stderr',text=captured.stderr))
    for display in captured.outputs:
        cell.outputs.append(nbformat.v4.new_output('display_data',data=display.data,metadata=display.metadata))
nbformat.validate(nb)
nbformat.write(nb,ROOT/'experiment.ipynb')
report={'executed':True,'engine':'fresh-process in-process IPython; not a socket kernel','code_cells':sum(c.cell_type=='code' for c in nb.cells),'errors':sum(o.output_type=='error' for c in nb.cells if c.cell_type=='code' for o in c.outputs),'seconds':time.time()-started,'python':sys.version}
(ROOT/'outputs/notebook-execution.json').write_text(json.dumps(report,indent=2))
print(report)
