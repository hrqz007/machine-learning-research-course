"""Execute every teaching cell in a fresh process using in-process IPython.
This is real execution, but does not test Jupyter socket kernels or browser UI.
"""
from pathlib import Path
import argparse,json,os
import nbformat
ROOT=Path(__file__).resolve().parent

def main():
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(ROOT/'outputs/experiment.ipynb'));a=p.parse_args()
    out=Path(a.out).absolute();out.parent.mkdir(parents=True,exist_ok=True)
    cells=json.loads((ROOT/'notebook_cells.json').read_text())
    notebook=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(c['source']) if c['kind']=='markdown' else nbformat.v4.new_code_cell(c['source']) for c in cells])
    notebook.metadata.kernelspec={'display_name':'Python 3 (course)','language':'python','name':'python3'}
    notebook.metadata.execution={'engine':'fresh Python process, in-process IPython InteractiveShell; no socket kernel','order':'top-to-bottom','errors_allowed':False}
    os.chdir(ROOT);shell=InteractiveShell.instance();count=0
    for cell in notebook.cells:
        if cell.cell_type!='code':continue
        count+=1
        with capture_output(stdout=True,stderr=True,display=True) as captured:
            result=shell.run_cell(cell.source,store_history=True)
        if result.error_before_exec or result.error_in_exec:raise RuntimeError(str(result.error_before_exec or result.error_in_exec))
        cell.execution_count=count;cell.outputs=[]
        if captured.stdout:cell.outputs.append(nbformat.v4.new_output('stream',name='stdout',text=captured.stdout))
        if captured.stderr:cell.outputs.append(nbformat.v4.new_output('stream',name='stderr',text=captured.stderr))
        for display in captured.outputs:cell.outputs.append(nbformat.v4.new_output('display_data',data=display.data,metadata=display.metadata))
    nbformat.validate(notebook);nbformat.write(notebook,out)
    print(json.dumps({'code_cells':count,'output':str(out),'engine':'fresh-process in-process IPython'}))
if __name__=='__main__':main()
