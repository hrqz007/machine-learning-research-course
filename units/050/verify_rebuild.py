"""Verify deterministic numerical/data/figure rebuild, independent of cwd."""
from pathlib import Path
import hashlib,json,tempfile
import nbformat
import generate_data
ROOT=Path(__file__).resolve().parent

def need(ok,msg):
    if not ok:raise RuntimeError(msg)

def main():
    out=ROOT/'outputs/rebuild'
    actual=json.loads((out/'result.json').read_text());reference=json.loads((ROOT/'experiment-result.json').read_text())
    need(actual==reference,'numeric result changed')
    need(actual==json.loads((out/'result-O.json').read_text()),'optimized result changed')
    for path in (ROOT/'figures').glob('*.png'):need(path.read_bytes()==(out/'figures'/path.name).read_bytes(),'figure bytes changed: '+path.name)
    nb=nbformat.read(out/'executed.ipynb',as_version=4);cells=[c for c in nb.cells if c.cell_type=='code'];need(all(c.execution_count is not None for c in cells),'unexecuted notebook cell');need(not any(o.output_type=='error' for c in cells for o in c.outputs),'notebook error');need(sum('image/png' in o.get('data',{}) for c in cells for o in c.outputs)==10,'notebook image count')
    with tempfile.TemporaryDirectory(prefix='ml050-data-',dir=out) as td:
        config=json.loads((ROOT/'data/protocol.json').read_text());generate_data.make(config,td)
        for rel,digest in json.loads((ROOT/'data_integrity.json').read_text()).items():need(hashlib.sha256((Path(td)/Path(rel).name).read_bytes()).hexdigest()==digest,'regenerated data differs: '+rel)
    for name in ['lecture','lab','answers']:need((out/'pdf'/f'{name}.pdf').stat().st_size>10000,'missing PDF')
    report={'numeric_same_as_reference':True,'ordinary_equals_optimized':True,'all_ten_figures_byte_identical':True,'all_fixed_data_byte_identical':True,'fresh_notebook_code_cells':len(cells),'fresh_notebook_images':10,'all_three_PDFs_rebuilt':True,'PDF_byte_identity_required':False}
    (out/'rebuild-verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
