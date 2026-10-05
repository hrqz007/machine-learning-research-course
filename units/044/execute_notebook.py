"""Execute the delivered notebook in a fresh in-process IPython kernel.

Start this script in a NEW Python process. It does not test browser/socket I/O.
"""
from pathlib import Path
import argparse, json, os, sys, tempfile
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(ROOT/'outputs/executed.ipynb'));a=p.parse_args()
    dest=Path(a.out).resolve()
    if (dest==ROOT/'experiment.ipynb' or dest.suffix!='.ipynb' or
        (dest.exists() and os.path.samefile(dest,ROOT/'experiment.ipynb'))):
        raise ValueError('choose a new .ipynb output; source notebook is protected')
    if dest.is_relative_to(ROOT) and not dest.is_relative_to(ROOT/'outputs'):
        raise ValueError('inside this unit, notebook output must go under outputs/')
    import nbformat
    from ipykernel.inprocess.manager import InProcessKernelManager
    nb=nbformat.read(ROOT/'experiment.ipynb',as_version=4)
    os.chdir(ROOT);sys.path.insert(0,str(ROOT));dest.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='ml044-kernel-') as td:
        os.environ.setdefault('IPYTHONDIR',str(Path(td)/'ipython'))
        os.environ.setdefault('MPLCONFIGDIR',str(Path(td)/'mpl'))
        os.environ.setdefault('XDG_CACHE_HOME',str(Path(td)/'cache'))
        from traitlets.config import Config
        config=Config();config.HistoryManager.hist_file=':memory:'
        manager=InProcessKernelManager();manager.start_kernel(config=config);client=manager.client();client.start_channels()
        count=0
        try:
            for cell in nb.cells:
                if cell.cell_type!='code':continue
                count+=1;cell.outputs=[];cell.execution_count=count
                mid=client.execute(cell.source,store_history=True)
                while True:
                    msg=client.get_iopub_msg(timeout=120)
                    if msg.get('parent_header',{}).get('msg_id')!=mid:continue
                    kind=msg['msg_type'];content=msg['content']
                    if kind=='status' and content['execution_state']=='idle':break
                    if kind in ['stream','display_data','execute_result','error']:
                        cell.outputs.append(nbformat.v4.output_from_msg(msg))
                    if kind=='error':raise RuntimeError(f"Cell {count}: {content['ename']}: {content['evalue']}")
            nbformat.validate(nb)
            nbformat.write(nb,dest)
        finally:
            # Close history while its private directory still exists.
            manager.kernel.shell._atexit_once()
            client.stop_channels();manager.shutdown_kernel()
    print(json.dumps({'notebook':str(dest),'code_cells':count,'status':'passed',
        'execution':'real InProcessKernel in a new Python process; all code cells in order',
        'scope':'browser Jupyter and socket transport not tested'},ensure_ascii=False))
if __name__=='__main__':main()
