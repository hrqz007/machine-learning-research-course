"""Compare scientific content, not machine-dependent elapsed times or ZIP timestamps."""
from pathlib import Path
import json,nbformat
import numpy as np
from experiment import ROOT,dump

def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items() if not (k.endswith('seconds') or k in ['python','error'])}
    if isinstance(x,list):return [clean(v) for v in x]
    return x

def compare(a,b,path='root'):
    if isinstance(a,dict):
        if set(a)!=set(b):raise AssertionError(path+' keys')
        for k in a:compare(a[k],b[k],path+'/'+k)
    elif isinstance(a,list):
        if len(a)!=len(b):raise AssertionError(path+' length')
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'/'+str(i))
    elif isinstance(a,(float,int)) and not isinstance(a,bool):
        if not np.isclose(a,b,rtol=1e-10,atol=1e-10):raise AssertionError(path+' number')
    elif a!=b:raise AssertionError(path+' value')
if __name__=='__main__':
    base=ROOT/'outputs/rebuild'
    compare(clean(json.loads((ROOT/'experiment-result.json').read_text())),clean(json.loads((base/'result.json').read_text())))
    nb=nbformat.read(base/'executed.ipynb',as_version=4)
    cells=[c for c in nb.cells if c.cell_type=='code']
    if any(c.execution_count is None or any(o.output_type=='error' for o in c.outputs) for c in cells):raise AssertionError('Notebook execution')
    figs=sum('image/png' in o.get('data',{}) for c in cells for o in c.outputs)
    out={'status':'passed','scientific_content_matches':True,'code_cells':len(cells),'embedded_pngs':figs,'timing_fields_excluded':True,'exception_wording_excluded':True}
    dump(out,base/'rebuild-verification.json');print(json.dumps(out))
