"""Small public helpers shared by this self-contained lesson."""
from pathlib import Path
import json, os
import numpy as np
ROOT=Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/"outputs/mpl"))
os.environ.setdefault("XDG_CACHE_HOME",str(ROOT/"outputs/cache"))
def write_json(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x: x.tolist() if isinstance(x,np.ndarray) else x.item() if isinstance(x,np.generic) else (_ for _ in ()).throw(TypeError(type(x).__name__)))+"\n")
def require(condition,message):
    if not condition: raise ValueError(message)
def finite_matrix(X):
    X=np.asarray(X,dtype=float)
    require(X.ndim==2 and X.shape[0]>0 and X.shape[1]>0,"X must be a nonempty 2D matrix")
    require(np.isfinite(X).all(),"X must contain finite numbers; impute missing values first")
    return X
def wilson(success,total,z=1.959963984540054):
    if total==0:return None
    p=success/total;den=1+z*z/total
    center=(p+z*z/(2*total))/den
    half=z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return [float(center-half),float(center+half)]
def figure_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":140,"savefig.dpi":160})
    return plt
