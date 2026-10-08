"""Small public helpers shared by this self-contained lesson."""
from pathlib import Path
import json, os
import numpy as np
ROOT=Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/"outputs/mpl"))
os.environ.setdefault("XDG_CACHE_HOME",str(ROOT/"outputs/cache"))
for _directory in [os.environ["MPLCONFIGDIR"],os.environ["XDG_CACHE_HOME"]]:
    Path(_directory).mkdir(parents=True,exist_ok=True)

def write_json(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
def require(condition,message):
    if not condition: raise ValueError(message)
def finite_matrix(X):
    X=np.asarray(X,dtype=float)
    require(X.ndim==2 and X.shape[0]>0 and X.shape[1]>0,"X must be a nonempty 2D matrix")
    require(np.isfinite(X).all(),"X must contain finite numbers; impute missing values first")
    return X
def figure_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":140,"savefig.dpi":160})
    return plt

def safe_output(path):
    """Reject output symlinks and hardlinks before creating a report."""
    path=Path(path).absolute()
    for part in (path,*path.parents):
        if part.is_symlink():raise ValueError('symlink output component')
    if path.exists() and (not path.is_file() or path.stat().st_nlink>1):
        raise ValueError('unsafe output file')
    path.parent.mkdir(parents=True,exist_ok=True)
    return path

def load_data(directory=None):
    """Verify both the recorded digest and deterministic source-of-truth bytes."""
    import csv,hashlib
    from generate_data import generated
    directory=Path(directory or ROOT/'data')
    manifest=json.loads((directory/'generation.json').read_text())
    expected=generated();result={};seen=set()
    require(set(manifest['sha256'])==set(expected),'unexpected data file set')
    for name,original in expected.items():
        raw=(directory/name).read_bytes()
        digest=hashlib.sha256(raw).hexdigest()
        require(digest==manifest['sha256'][name],'data digest mismatch: '+name)
        require(raw==original,'data do not match the frozen generator: '+name)
        rows=list(csv.DictReader(raw.decode().splitlines()))
        ids=[row['id'] for row in rows]
        require(len(set(ids))==len(ids) and not (seen&set(ids)),'duplicate or overlapping ids')
        seen.update(ids)
        result[name[:-4]]={k:np.array([r[k] for r in rows],dtype=str if k=='id' else float) for k in rows[0]}
    return result
