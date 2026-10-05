"""Run each notebook cell in a fresh in-process IPython kernel (no sockets).

Use for CPU/offline teaching notebooks when the build sandbox disallows sockets.
The process is new for each notebook. All code cells run in order; outputs are
saved. This does not claim to test a browser Jupyter UI or out-of-process transport.
"""
from pathlib import Path
import sys,os,json,time,queue
ROOT=Path(__file__).resolve().parent
if (ROOT/'.deps').exists():sys.path.insert(0,str(ROOT/'.deps'))
os.environ.setdefault('IPYTHONDIR',str(ROOT/'outputs/ipython'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/mpl'))
import nbformat
from ipykernel.inprocess.manager import InProcessKernelManager
from traitlets.config import Config
path=Path(sys.argv[1]).resolve();os.chdir(path.parent);sys.path.insert(0,str(path.parent))
nb=nbformat.read(path,as_version=4);config=Config();config.HistoryManager.hist_file=':memory:';manager=InProcessKernelManager(config=config);manager.start_kernel();client=manager.client();client.start_channels()
start=time.monotonic();count=0
for cell in nb.cells:
 if cell.cell_type!='code':continue
 count+=1;cell.outputs=[];cell.execution_count=count
 mid=client.execute(cell.source,store_history=True)
 while True:
  msg=client.get_iopub_msg(timeout=1)
  kind=msg['msg_type'];content=msg['content']
  if kind=='status' and content['execution_state']=='idle':break
  if kind in ['stream','display_data','execute_result','error']:
   if kind=='execute_result':content['execution_count']=count
   cell.outputs.append(nbformat.v4.output_from_msg(msg))
  if kind=='error':
   nbformat.write(nb,path)
   raise RuntimeError(f"Cell {count}: {content['ename']}: {content['evalue']}")
client.stop_channels();manager.shutdown_kernel()
nb.metadata['kernelspec']={'display_name':'Python 3','language':'python','name':'python3'}
nbformat.write(nb,path)
print(json.dumps({'notebook':path.name,'status':'passed','seconds':round(time.monotonic()-start,3),'code_cells':count,'kernel':'ipykernel InProcessKernel, fresh Python process; sequential execution','transport_note':'in-process kernel chosen for execution; browser UI and external transport not tested','python':sys.version.split()[0]},ensure_ascii=False))
