"""Verify independent-directory whole-unit rebuild without changing shipped files."""
from pathlib import Path
import hashlib,json,tempfile
import nbformat
import generate_data
ROOT=Path(__file__).resolve().parent

def need(ok,msg):
    if not ok:raise RuntimeError(msg)

def main():
    out=ROOT/'outputs/rebuild';actual=json.loads((out/'result.json').read_text());ref=json.loads((ROOT/'experiment-result.json').read_text());need(actual==ref,'numeric reference differs');need(actual==json.loads((out/'result-O.json').read_text()),'-O differs')
    for path in (ROOT/'figures').glob('*.png'):need(path.read_bytes()==(out/'figures'/path.name).read_bytes(),'figure differs '+path.name)
    nb=nbformat.read(out/'executed.ipynb',as_version=4);cells=[v for v in nb.cells if v.cell_type=='code'];need(all(v.execution_count is not None for v in cells),'unexecuted cell');need(not any(o.output_type=='error' for v in cells for o in v.outputs),'notebook error');images=sum('image/png' in o.get('data',{}) for v in cells for o in v.outputs);need(images==10,'notebook image count')
    with tempfile.TemporaryDirectory(prefix='ml052-data-',dir=out) as td:
        generate_data.make(json.loads((ROOT/'data/protocol.json').read_text()),td)
        for rel,h in json.loads((ROOT/'data_integrity.json').read_text()).items():need(hashlib.sha256((Path(td)/Path(rel).name).read_bytes()).hexdigest()==h,'data regeneration differs '+rel)
    for name in ['lecture','lab','answers']:need((out/'pdf'/f'{name}.pdf').stat().st_size>10000,'missing rebuilt PDF')
    report={'numeric_reference_equal':True,'ordinary_equals_optimized':True,'ten_figures_byte_equal':True,'fixed_data_byte_equal':True,'fresh_notebook_code_cells':len(cells),'fresh_notebook_images':images,'three_PDFs_rebuilt':True,'PDF_byte_identity_required':False};(out/'rebuild-verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
