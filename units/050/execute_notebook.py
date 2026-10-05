"""Execute in a fresh true IPython kernel; preserve source unless author copies output."""
from pathlib import Path
import argparse,json,os,sys,time
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('IPYTHONDIR',str(ROOT/'outputs/ipython'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/mpl'))
import nbformat
from ipykernel.inprocess.manager import InProcessKernelManager
from traitlets.config import Config

def main():
    p=argparse.ArgumentParser();p.add_argument('--notebook',default=str(ROOT/'experiment.ipynb'));p.add_argument('--out',default=str(ROOT/'outputs/fresh-kernel.ipynb'));a=p.parse_args()
    path=Path(a.notebook).resolve();out=Path(a.out).absolute();sys.path.insert(0,str(ROOT));import experiment as e;e.safe_output(out)
    os.chdir(path.parent);sys.path.insert(0,str(path.parent));nb=nbformat.read(path,as_version=4)
    config=Config();config.HistoryManager.hist_file=':memory:';manager=InProcessKernelManager(config=config);manager.start_kernel();client=manager.client();client.start_channels();start=time.monotonic();count=0
    try:
        for cell in nb.cells:
            if cell.cell_type!='code':continue
            count+=1;cell.outputs=[];cell.execution_count=count
            mid=client.execute(cell.source,store_history=True)
            while True:
                msg=client.get_iopub_msg(timeout=60);kind=msg['msg_type'];content=msg['content']
                if kind=='status' and content['execution_state']=='idle':break
                if kind in ['stream','display_data','execute_result','error']:
                    if kind=='execute_result':content['execution_count']=count
                    cell.outputs.append(nbformat.v4.output_from_msg(msg))
                if kind=='error':raise RuntimeError(f"Code cell {count}: {content['ename']}: {content['evalue']}")
    finally:
        client.stop_channels();manager.shutdown_kernel();out.parent.mkdir(parents=True,exist_ok=True);nbformat.write(nb,out)
    print(json.dumps({'status':'passed','code_cells':count,'saved':str(out),'seconds':round(time.monotonic()-start,3),'kernel':'ipykernel InProcessKernel in a fresh Python process','not_tested':['Jupyter browser UI','external socket kernel transport'],'python':sys.version.split()[0]},ensure_ascii=False))
if __name__=='__main__':main()
