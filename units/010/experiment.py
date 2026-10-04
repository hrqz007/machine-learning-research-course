"""Unit 010: matrix actions and axes. Synthetic, deterministic, CPU/offline.
All geometric coordinates and sample features are dimensionless.
No model is fitted, no performance is evaluated. Raw data are read only.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import platform
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ATOL = 1e-12


def as_matrix(value, name):
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must have exactly two axes")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain finite numbers")
    return arr


def matrix_product_loop(left, right):
    """Explicit row/column matching; independent of NumPy matmul."""
    left, right = as_matrix(left, "left"), as_matrix(right, "right")
    m, d = left.shape
    if d != right.shape[0]:
        raise ValueError("left columns must equal right rows")
    k = right.shape[1]
    output = np.zeros((m, k), dtype=float)
    for i in range(m):
        for j in range(k):
            total = 0.0
            for ell in range(d):
                total += left[i, ell] * right[ell, j]
            output[i, j] = total
    return output


def apply_rows(X, A):
    """Rows are samples: X(n,d), A(k,d), result Y(n,k) = X @ A.T."""
    X, A = as_matrix(X, "X"), as_matrix(A, "A")
    if X.shape[1] != A.shape[1]:
        raise ValueError("X feature axis must match A input-coordinate axis")
    return X @ A.T


def predict_rows(X, w, bias=0.0):
    """Scalar output per sample; require w(d,), preserve n even when n=1."""
    X = as_matrix(X, "X")
    w = np.asarray(w, dtype=float)
    if w.ndim != 1 or len(w) != X.shape[1]:
        raise ValueError("w must be a one-dimensional vector of feature length")
    if not np.isfinite(w).all():
        raise ValueError("w must be finite")
    if np.ndim(bias) != 0 or not np.isfinite(bias):
        raise ValueError("bias must be a finite scalar")
    return X @ w + bias


def rotation(theta_radians):
    """Counterclockwise rotation of column vectors in dimensionless xy-plane."""
    if np.ndim(theta_radians) != 0 or not np.isfinite(theta_radians):
        raise ValueError("rotation angle must be one finite scalar in radians")
    c, s = np.cos(theta_radians), np.sin(theta_radians)
    return np.array([[c, -s], [s, c]])


def coordinates_in_diagonal_basis(x):
    """x = c1*(1,1)+c2*(1,-1), solved by adding/subtracting two equations."""
    x = np.asarray(x, dtype=float)
    if x.shape != (2,):
        raise ValueError("x must have shape (2,)")
    return np.array([(x[0]+x[1])/2, (x[0]-x[1])/2])


def load_data():
    with (DATA / "samples.csv").open(encoding="utf-8", newline="") as f:
        records = list(csv.DictReader(f))
    ids = [r["sample_id"] for r in records]
    X = np.array([[float(r[c]) for c in ["x1","x2","x3"]] for r in records])
    config = json.loads((DATA / "matrices.json").read_text(encoding="utf-8"))
    return ids, X, config


def input_hashes():
    return {name:hashlib.sha256((DATA/name).read_bytes()).hexdigest()
            for name in ["samples.csv","matrices.json"]}


def run(output_dir=None):
    out = Path(output_dir) if output_dir else HERE / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    before = input_hashes()
    ids, X, config = load_data()
    A = np.array(config["A_rect"], dtype=float)
    B = np.array(config["B_rect"], dtype=float)
    H = np.array(config["shear"], dtype=float)
    D = np.array(config["stretch"], dtype=float)
    P = np.array(config["collapse"], dtype=float)
    C = np.array(config["basis"], dtype=float)
    w = np.array(config["weights"], dtype=float)
    y = predict_rows(X, w)
    y_affine = predict_rows(X, w, config["bias"])
    mapped = apply_rows(X, A)
    checks = []
    def check(name, condition):
        if not bool(condition):
            raise AssertionError(name)
        checks.append({"name":name,"passed":True})
    def close(name, actual, expected):
        check(name, np.shape(actual)==np.shape(expected) and np.allclose(actual, expected, atol=ATOL, rtol=ATOL))
    def reject(name, func):
        try:
            func()
        except ValueError:
            check(name, True)
        else:
            raise AssertionError(name + " did not raise ValueError")
    check("X axes n=4 d=3", X.shape==(4,3))
    check("A axes k=2 d=3", A.shape==(2,3))
    check("B axes d=3 k=2", B.shape==(3,2))
    close("AB hand calculation", A@B, [[5,-1],[-4,6]])
    close("BA hand calculation", B@A, [[2,3,1],[1,2,-1],[-1,-4,7]])
    close("explicit loops equal rectangular AB", matrix_product_loop(A,B), A@B)
    close("explicit loops equal rectangular BA", matrix_product_loop(B,A), B@A)
    close("rectangular transpose product identity", (A@B).T, B.T@A.T)
    close("transpose twice", A.T.T, A)
    close("left identity", np.eye(2)@A, A)
    close("right identity", A@np.eye(3), A)
    close("batch mapping hand calculation", mapped, [[5,-2],[-1,8],[-1,4],[1,6]])
    close("batch loops", mapped, matrix_product_loop(X,A.T))
    for i in range(len(X)):
        close(f"sample {ids[i]} column action", mapped[i], A@X[i])
    close("batch scalar prediction", y, [0,8,8,12])
    close("batch affine prediction", y_affine, [.5,8.5,8.5,12.5])
    close("scalar prediction loops", y, matrix_product_loop(X,w[:,None])[:,0])
    close("single sample mapping stays 2D", apply_rows(X[:1],A), [[5,-2]])
    close("single sample prediction stays length one", predict_rows(X[:1],w), [0])
    check("single row does not equal vector shape", X[:1].shape==(1,3) and X[0].shape==(3,))
    check("1D transpose changes no shape", X[0].T.shape==(3,))
    check("explicit column shape", X[0][:,None].shape==(3,1))
    close("elementwise distinct from product", H*D, [[2,0],[0,1]])
    close("H then D matrix", D@H, [[2,2],[0,1]])
    close("D then H matrix", H@D, [[2,1],[0,1]])
    check("square product not commutative", not np.allclose(H@D,D@H))
    close("D then H applied to x", (H@D)@np.array([1,1]), [3,1])
    close("H then D applied to x", (D@H)@np.array([1,1]), [4,1])
    close("compatible associativity", (X@B)@A, X@(B@A))
    close("matrix action distributivity", A@(X[0]+X[1]), A@X[0]+A@X[1])
    close("matrix action homogeneity", A@(-2*X[0]), -2*(A@X[0]))
    close("matrix action sends zero to zero", A@np.zeros(3), np.zeros(2))
    close("nonstandard basis coordinates", coordinates_in_diagonal_basis([3,1]), [2,1])
    close("basis reconstruction", C@coordinates_in_diagonal_basis([3,1]), [3,1])
    for point in [[0,0],[-3,2],[.5,-1.5]]:
        close(f"basis round trip {point}", C@coordinates_in_diagonal_basis(point), point)
    close("collapse distinct points coincide", P@np.array([1,2]), P@np.array([1,-1]))
    close("collapse hand output", P@np.array([1,2]), [1,0])
    x,z=np.array([2.,1.]),np.array([1.,3.])
    R=rotation(np.deg2rad(45))
    close("degree radian conversion", np.array(np.deg2rad(180)), np.array(np.pi))
    close("45 degree rotation entries", R, np.sqrt(.5)*np.array([[1,-1],[1,1]]))
    close("rotation RT R identity", R.T@R, np.eye(2))
    close("rotation preserves inner product", np.array((R@x)@(R@z)), np.array(x@z))
    close("rotation preserves squared length", np.array((R@x)@(R@x)), np.array(x@x))
    close("rotation preserves distance", np.array(np.linalg.norm(R@x-R@z)), np.array(np.linalg.norm(x-z)))
    close("rotation reverse using transpose", R.T@(R@x), x)
    for degrees in [0,30,90,-45,180,360]:
        Q=rotation(np.deg2rad(degrees))
        close(f"rotation orthogonality {degrees} degrees", Q.T@Q, np.eye(2))
    check("ordinary scaling changes norm", not np.isclose(np.linalg.norm(D@x),np.linalg.norm(x)))
    bias=np.array([1.,-1.])
    f=lambda v:H@v+bias
    close("affine origin is bias", f(np.zeros(2)),bias)
    check("nonzero bias breaks additivity", not np.allclose(f(x+z),f(x)+f(z)))
    close("affine additivity discrepancy", f(x)+f(z)-f(x+z),bias)
    X_aug=np.column_stack([X,np.ones(len(X))]); w_aug=np.append(w,config["bias"])
    close("augmented coordinate prediction", X_aug@w_aug,y_affine)
    reject("incompatible rectangular product",lambda:matrix_product_loop(A,A))
    reject("wrong batch input axis",lambda:apply_rows(np.zeros((4,2)),A))
    reject("single vector is not a batch",lambda:apply_rows(X[0],A))
    reject("column weight not silently accepted",lambda:predict_rows(X,w[:,None]))
    reject("wrong weight length",lambda:predict_rows(X,[1,2]))
    reject("nonfinite coordinate rejected",lambda:apply_rows([[np.nan,0,1]],A))
    reject("nonscalar bias rejected",lambda:predict_rows(X,w,[1,2,3,4]))
    check("input data unchanged",before==input_hashes())
    with (out/"batch_results.csv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.writer(f);writer.writerow(["sample_id","x1","x2","x3","mapped_1","mapped_2","linear_score","affine_score"])
        for sid,row,coords,pred,pred_a in zip(ids,X,mapped,y,y_affine):writer.writerow([sid,*row,*coords,pred,pred_a])
    summary={"synthetic_data":True,"dimensionless_coordinates":True,"models_fitted":0,
             "X_shape":list(X.shape),"A_shape":list(A.shape),"B_shape":list(B.shape),
             "AB":(A@B).tolist(),"BA":(B@A).tolist(),"mapped_rows":mapped.tolist(),
             "linear_predictions":y.tolist(),"affine_predictions":y_affine.tolist(),
             "D_then_H":(H@D).tolist(),"H_then_D":(D@H).tolist(),
             "basis_coordinates_of_3_1":coordinates_in_diagonal_basis([3,1]).tolist(),
             "rotation_angle_degrees":45,"rotation_angle_radians":float(np.deg2rad(45)),
             "rotation_inner_product_before":float(x@z),"rotation_inner_product_after":float((R@x)@(R@z)),
             "rotation_RT_R_max_abs_error":float(np.max(np.abs(R.T@R-np.eye(2)))),
             "raw_files_unchanged":before==input_hashes()}
    tests={"status":"passed","unit":"010","checked_on":"2026-10-04","test_count":len(checks),
           "tests":checks,"tolerance":{"atol":ATOL,"rtol":ATOL},
           "versions":{"python":platform.python_version(),"numpy":np.__version__},
           "scope":"Finite deterministic implementation checks; algebraic proofs are in lecture.md.",
           "limitations":["Anaconda installation, browser UI and socket-based Jupyter transport not tested.","No fitted models or performance claims."]}
    for name,value in [("summary.json",summary),("test-result.json",tests),("input_hashes.json",before)]:
        (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return {"X":X,"A":A,"B":B,"summary":summary,"tests":tests,"mapped":mapped,"predictions":y}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=HERE/"outputs")
    args=parser.parse_args()
    result=run(args.output_dir)
    print(json.dumps({"status":"passed","test_count":result["tests"]["test_count"],"summary":result["summary"]},ensure_ascii=False,indent=2))
