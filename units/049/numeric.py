"""Shared validation and strict atomic JSON utilities, local to ML049."""
from pathlib import Path
from decimal import Decimal
import json,math,os,tempfile
import numpy as np
ROOT=Path(__file__).resolve().parent

def scalar(x,name,lo=-1000,hi=1000,integer=False):
    if isinstance(x,(bool,np.bool_)) or not isinstance(x,(int,float,np.integer,np.floating)):raise ValueError(name+' requires a real scalar, not bool/string/complex')
    if integer and not isinstance(x,(int,np.integer)):raise ValueError(name+' requires integer type')
    try:v=float(x)
    except (ValueError,OverflowError) as ex:raise ValueError(name+' not representable') from ex
    if not math.isfinite(v) or not lo<=v<=hi or (x!=0 and (v==0 or abs(v)<1e-100)):raise ValueError(name+' outside supported finite raw range')
    return int(x) if integer else v

def vector(x,name,lo=-1000,hi=1000,size=None):
    if not isinstance(x,(list,tuple,np.ndarray)):raise ValueError(name+' requires vector')
    a=np.asarray(x,dtype=object)
    if a.ndim!=1 or not 1<=len(a)<=20000 or (size is not None and len(a)!=size):raise ValueError(name+' shape/length mismatch')
    return np.array([scalar(v,name,lo,hi) for v in a])

def labels(y,size=None):
    v=vector(y,'binary labels',0,1,size)
    if np.any((v!=0)&(v!=1)):raise ValueError('labels must be 0/1')
    return v.astype(int)

def pairs(items):
    d={}
    for k,v in items:
        if k in d:raise ValueError('duplicate JSON key')
        d[k]=v
    return d

def token(s):
    d=Decimal(s);v=float(d)
    if not math.isfinite(v) or (d!=0 and (v==0 or abs(v)<1e-100)):raise ValueError('unsupported numeric token')
    return v

def read_json(path):return json.loads(Path(path).read_text(),parse_float=token,object_pairs_hook=pairs,parse_constant=lambda s:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
def canonical_bytes(x):return (json.dumps(x,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()

def safe_write(path,x,extra_protected=()):
    raw=canonical_bytes(x);p=Path(path).absolute()
    if p.suffix!='.json':raise ValueError('output must be JSON')
    for part in (p,*p.parents):
        if part.is_symlink():raise ValueError('symlink output component')
    if p.exists() and (not p.is_file() or p.stat().st_nlink>1):raise ValueError('unsafe output file')
    protected=[v for v in ROOT.rglob('*') if v.is_file() and 'outputs' not in v.relative_to(ROOT).parts]+[Path(v) for v in extra_protected]
    if any(p.resolve()==v.resolve() or (p.exists() and v.exists() and os.path.samefile(p,v)) for v in protected):raise ValueError('cannot overwrite teaching/input asset')
    p.parent.mkdir(parents=True,exist_ok=True);temporary=None
    try:
        fd,temporary=tempfile.mkstemp(prefix='.ml049-',dir=p.parent)
        with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        os.replace(temporary,p);temporary=None
    finally:
        if temporary is not None:Path(temporary).unlink(missing_ok=True)
