"""029 多元线性回归：有界离线教学求解器和精确存储值参照。

浮点入口拒绝 bool、文本、复数、Fraction/Decimal、扩展精度和不支持的尺度。
Fraction 仅用于明确命名的 exact_* 接口及精确 CSV/存储值核对。
"""
from pathlib import Path
from fractions import Fraction
import argparse, csv, json, math, re, platform
import numpy as np
from sklearn.linear_model import LinearRegression
import sklearn
ROOT = Path(__file__).resolve().parent
TINY = np.finfo(np.float64).tiny


def require(condition, message):
    if not condition:
        raise ValueError(message)


def real(value, name='value', low=-1e6, high=1e6):
    supported = type(value) in (int, float) or isinstance(value, (np.integer, np.floating))
    require(supported and not isinstance(value, (bool, np.bool_)), name+' requires a supported real scalar')
    if isinstance(value, np.floating):
        require(value.dtype.itemsize <= 8, name+' extended precision unsupported')
    require(low <= value <= high, name+' outside finite supported range')
    out=float(value)
    require(math.isfinite(out) and (value == 0 or abs(value) >= 1e-100), name+' nonfinite or below nonzero floor')
    require(value == 0 or out != 0, name+' nonzero conversion collapsed')
    if isinstance(value,(int,np.integer)):
        require(int(out)==value,name+' inexact integer conversion')
    return out


def integer(value,name,low,high):
    require((type(value) is int or isinstance(value,np.integer)) and not isinstance(value,(bool,np.bool_)),name+' requires integer')
    require(low<=value<=high,name+' outside range')
    return int(value)


def vector(values,name='vector',max_length=128):
    require(isinstance(values,(list,tuple,np.ndarray)),name+' requires sequence')
    if isinstance(values,np.ndarray): require(values.ndim==1,name+' must have shape (n,)')
    require(1<=len(values)<=max_length,name+' unsupported length')
    return np.array([real(v,name+' element') for v in values],dtype=float)


def matrix(values,name='matrix'):
    require(isinstance(values,(list,tuple,np.ndarray)),name+' requires rows')
    if isinstance(values,np.ndarray):require(values.ndim==2,name+' must have shape (n,p)')
    require(1<=len(values)<=128,name+' unsupported row count')
    rows=[vector(row,name+' row',16) for row in values]
    require(all(len(r)==len(rows[0]) for r in rows),name+' ragged rows')
    return np.array(rows,dtype=float)


def validate_problem(A,y):
    A=matrix(A,'A');y=vector(y,'y')
    require(len(y)==len(A),'A and y row mismatch')
    return A,y


def validate_rcond(rcond):
    return None if rcond is None else real(rcond,'rcond',0,.1)


def _norm_squared(values,name):
    terms=[]
    for v in np.asarray(values).ravel():
        q=float(v)
        require(math.isfinite(q) and abs(q)<=1e100,name+' unsupported computed scale')
        t=q*q
        require(q==0 or t>=TINY,name+' positive square subnormal or zero; rescale')
        terms.append(t)
    return math.fsum(terms)


def _solution_report(A,y,beta,method,rank,singular,cutoff,raw_residuals=None):
    require(beta.shape==(A.shape[1],) and np.all(np.isfinite(beta)),'invalid solver coefficient output')
    prediction=A@beta;residual=y-prediction
    require(np.all(np.isfinite(prediction)),'nonfinite prediction')
    sse=_norm_squared(residual,'residual')
    gradient=A.T@(-residual)/len(y)
    full_rank=rank==A.shape[1]
    # JSON uses null and an explicit reason instead of Infinity for a singular matrix.
    condition=float(singular[0]/singular[-1]) if singular[-1]>0 else None
    require(condition is None or math.isfinite(condition),'condition estimate overflow; rescale')
    return {'status':'ok','method':method,'n':len(y),'p':A.shape[1],
            'beta':beta.tolist(),'prediction':prediction.tolist(),'residual':residual.tolist(),
            'sse':sse,'mse':sse/len(y),'J':sse/(2*len(y)),
            'gradient':gradient.tolist(),'gradient_norm':math.sqrt(_norm_squared(gradient,'gradient')),
            'rank':int(rank),'numerical_full_column_rank':bool(full_rank),'singular_values':singular.tolist(),
            'absolute_cutoff':float(cutoff),'condition_estimate':condition,
            'condition_note':'floating singular-value ratio; mathematical rank is separately checked',
            'raw_lstsq_residuals':None if raw_residuals is None else np.asarray(raw_residuals).tolist()}


def solve_ols(A,y,method='lstsq',rcond=None):
    """Organize lstsq, SVD or thin-QR routes. A includes the intercept if desired.

    QR is only used when n>=p and the declared numerical rank is p.
    normal is a comparison route: solve(A.T@A,A.T@y), never explicit inversion.
    All inputs are checked before any factorization; no silent y flattening.
    """
    A,y=validate_problem(A,y);rcond=validate_rcond(rcond)
    require(type(method) is str and method in ['lstsq','svd','qr','normal'],'unknown solver method')
    U,s,Vh=np.linalg.svd(A,full_matrices=False)
    cutoff=(np.finfo(float).eps*max(A.shape) if rcond is None else rcond)*s[0]
    rank=int(np.count_nonzero(s>cutoff));raw=None
    if method=='lstsq':
        beta,raw,rank,s=np.linalg.lstsq(A,y,rcond=rcond)
        cutoff=(np.finfo(float).eps*max(A.shape) if rcond is None else rcond)*s[0]
    elif method=='svd':
        keep=s>cutoff
        beta=Vh[keep,:].T@((U[:,keep].T@y)/s[keep]) if np.any(keep) else np.zeros(A.shape[1])
    elif method=='qr':
        if A.shape[0]<A.shape[1] or rank<A.shape[1]:
            return {'status':'not_applicable','method':method,'reason':'thin full-column-rank QR condition fails','rank':rank}
        Q,R=np.linalg.qr(A,mode='reduced');beta=np.linalg.solve(R,Q.T@y)
    else:
        try:beta=np.linalg.solve(A.T@A,A.T@y)
        except np.linalg.LinAlgError:
            return {'status':'linear_algebra_failure','method':method,'reason':'computed Gram matrix is singular','rank':rank}
    return _solution_report(A,y,beta,method,rank,s,cutoff,raw)


def exact_rank(rows):
    require(type(rows) is list and 1<=len(rows)<=128,'exact rows required')
    require(all(type(r) is list for r in rows) and 1<=len(rows[0])<=16,'exact row shape')
    p=len(rows[0]);require(all(len(r)==p for r in rows),'exact ragged rows')
    require(all(type(v) is Fraction for r in rows for v in r),'exact API requires Fraction entries')
    a=[r[:] for r in rows];lead=0
    for j in range(p):
        pivot=next((i for i in range(lead,len(a)) if a[i][j]),None)
        if pivot is None:continue
        a[lead],a[pivot]=a[pivot],a[lead];d=a[lead][j];a[lead]=[v/d for v in a[lead]]
        for i in range(lead+1,len(a)):
            q=a[i][j];a[i]=[v-q*w for v,w in zip(a[i],a[lead])]
        lead+=1
        if lead==len(a):break
    return lead


def _exact_square_solve(G,q):
    n=len(q);rows=[G[i][:]+[q[i]] for i in range(n)]
    for j in range(n):
        pivot=next((i for i in range(j,n) if rows[i][j]),None)
        require(pivot is not None,'exact normal matrix singular')
        rows[j],rows[pivot]=rows[pivot],rows[j];d=rows[j][j];rows[j]=[v/d for v in rows[j]]
        for i in range(n):
            if i!=j:
                t=rows[i][j];rows[i]=[v-t*w for v,w in zip(rows[i],rows[j])]
    return [r[-1] for r in rows]


def exact_ols(A,y):
    """Small exact Fraction reference; not a recommended large numerical solver."""
    rank=exact_rank(A);n=len(A);p=len(A[0])
    require(type(y) is list and len(y)==n and all(type(v) is Fraction for v in y),'exact response requires matching Fractions')
    require(rank==p,'exact coefficient is not unique')
    G=[[sum((row[j]*row[k] for row in A),Fraction(0)) for k in range(p)] for j in range(p)]
    q=[sum((row[j]*v for row,v in zip(A,y)),Fraction(0)) for j in range(p)]
    beta=_exact_square_solve(G,q)
    prediction=[sum((a*b for a,b in zip(row,beta)),Fraction(0)) for row in A]
    residual=[v-pr for v,pr in zip(y,prediction)];sse=sum((v*v for v in residual),Fraction(0))
    return {'gram':[[str(v) for v in row] for row in G],'rhs':[str(v) for v in q],
            'rank':rank,'beta':[str(v) for v in beta],'prediction':[str(v) for v in prediction],
            'residual':[str(v) for v in residual],'sse':str(sse),'mse':str(sse/n),'J':str(sse/(2*n))}


def stored_fractions(A):
    A=matrix(A,'stored matrix')
    return [[Fraction.from_float(float(v)) for v in row] for row in A]


def _csv_number(token):
    require(type(token) is str and len(token)<=40 and re.fullmatch(r'-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?',token) is not None,'CSV requires bounded plain decimal number')
    q=Fraction(token);v=real(float(q),'CSV numeric value')
    require(Fraction.from_float(v)==q,'CSV decimal is not exactly representable in binary64')
    return v


def _read_csv(path,header):
    with Path(path).open(newline='',encoding='utf-8') as f:
        r=csv.DictReader(f);require(r.fieldnames==header,'CSV header mismatch');rows=list(r)
    require(1<=len(rows)<=128,'CSV row count outside budget')
    ids=set();numeric=[]
    for row in rows:
        require(set(row)==set(header) and all(v is not None for v in row.values()),'ragged CSV row')
        identifier=row['id'];require(re.fullmatch(r'[A-Z][1-9][0-9]{0,2}',identifier) is not None and identifier not in ids,'invalid or repeated id')
        ids.add(identifier);numeric.append([_csv_number(row[k]) for k in header[1:]])
    return np.array(numeric,dtype=float)


def load_hand(path=None):
    t=_read_csv(path or ROOT/'data/hand_regression.csv',['id','x1','x2','y'])
    require(len(t)>=3,'hand example needs >=3 rows')
    A=np.column_stack([np.ones(len(t)),t[:,:2]]);y=t[:,2]
    require(exact_rank(stored_fractions(A))==3,'hand example must have exact full column rank')
    return A,y


def load_vectors(path=None):
    t=_read_csv(path or ROOT/'data/orthogonal_vectors.csv',['id','x','z','w'])
    expected=np.array([[-3,1,1],[-2,0,-2],[-1,-1,1],[0,0,0],[1,-1,1],[2,0,-2],[3,1,1]],dtype=float)
    require(t.shape==expected.shape and np.array_equal(t,expected),'basis CSV must match the declared seven-row exact model')
    return t


SPEC_KEYS={'kind','near_collinear_powers','noise_scale','response_perturbation','rcond_values','feature_rescaling'}


def validate_spec(spec):
    require(type(spec) is dict and set(spec)==SPEC_KEYS,'spec keys must exactly match schema')
    require(spec['kind']=='synthetic_multivariate_least_squares_demo','wrong model kind')
    powers=spec['near_collinear_powers'];require(type(powers) is list and 1<=len(powers)<=10,'powers list budget')
    powers=[integer(k,'power',0,48) for k in powers];require(len(set(powers))==len(powers),'duplicate power')
    noise=real(spec['noise_scale'],'noise_scale',2**-12,1)
    eps=real(spec['response_perturbation'],'response_perturbation',2**-30,2**-4)
    require(math.frexp(noise)[0]==.5 and math.frexp(eps)[0]==.5,'noise and perturbation must be powers of two')
    rs=spec['rcond_values'];require(type(rs) is list and 1<=len(rs)<=8,'rcond_values list budget')
    rs=[real(r,'rcond',0,.1) for r in rs];require(len(set(rs))==len(rs),'duplicate rcond')
    scale=real(spec['feature_rescaling'],'feature_rescaling',1,10000)
    return {'kind':spec['kind'],'near_collinear_powers':powers,'noise_scale':noise,
            'response_perturbation':eps,'rcond_values':rs,'feature_rescaling':scale}


def unique_object(pairs):
    out={}
    for k,v in pairs:require(k not in out,'duplicate JSON key: '+k);out[k]=v
    return out


def decimal_token(token):
    v=float(token);require(math.isfinite(v),'JSON nonfinite or overflow')
    if v==0:require(not any(c in '123456789' for c in token.lower().split('e')[0]),'JSON nonzero token underflow')
    return real(v,'JSON number')


def reject_constant(token):raise ValueError('nonstandard JSON constant: '+token)


def load_spec(path=None):
    return validate_spec(json.loads(Path(path or ROOT/'data/model_spec.json').read_text(),parse_float=decimal_token,parse_constant=reject_constant,object_pairs_hook=unique_object))


def construct_family(vectors,power,noise_scale=.0625,epsilon=0):
    t=matrix(vectors,'basis');require(t.shape==(7,3),'basis requires seven rows/three columns')
    expected=load_vectors();require(np.array_equal(t,expected),'unsupported basis')
    k=integer(power,'power',0,48);a=real(noise_scale,'noise',2**-12,1);eps=real(epsilon,'epsilon',0,2**-4)
    require(math.frexp(a)[0]==.5 and (eps==0 or math.frexp(eps)[0]==.5),'dyadic power parameters required')
    x,z,w=t.T;delta=2.0**(-k);A=np.column_stack([np.ones(7),x,x+delta*z]);y=1+3*x+delta*z+a*w+eps*z
    # Check *actual stored* matrix/response against the exact construction, not merely its formula.
    F=Fraction;fd=F(1,2**k);fa=F.from_float(a);fe=F.from_float(eps)
    for i in range(7):
        fx,fz,fw=map(lambda v:F.from_float(float(v)),t[i])
        require(F.from_float(float(A[i,2]))==fx+fd*fz,'matrix construction lost a dyadic term')
        require(F.from_float(float(y[i]))==1+3*fx+fd*fz+fa*fw+fe*fz,'response construction lost a dyadic term')
    return A,y


def _add_reference_metrics(report,reference,A):
    if report['status']!='ok':return report
    beta=np.array(report['beta']);truth=np.array([float(Fraction(v)) for v in reference['beta']])
    report['coefficient_error_norm']=math.sqrt(_norm_squared(beta-truth,'coefficient error'))
    report['reference_prediction_error_norm']=math.sqrt(_norm_squared(A@beta-np.array([float(Fraction(v)) for v in reference['prediction']]),'prediction error'))
    report['reference_sse']=float(Fraction(reference['sse']))
    return report


def compare_sklearn(A,y):
    A,y=validate_problem(A,y);require(np.all(A[:,0]==1) and A.shape[1]>=2,'sklearn comparison needs explicit intercept first')
    raw=LinearRegression(fit_intercept=True).fit(A[:,1:],y)
    augmented=LinearRegression(fit_intercept=False).fit(A,y)
    tol_large=LinearRegression(fit_intercept=True,tol=.1).fit(A[:,1:],y)
    return {'version':sklearn.__version__,'raw_features':{'beta':[float(raw.intercept_)]+raw.coef_.tolist(),'prediction':raw.predict(A[:,1:]).tolist(),'rank_centered_X':int(raw.rank_), 'singular_centered_X':raw.singular_.tolist()},
            'augmented':{'beta':augmented.coef_.tolist(),'prediction':augmented.predict(A).tolist(),'rank_A':int(augmented.rank_)},
            'dense_tol_predictions_identical':bool(np.array_equal(raw.predict(A[:,1:]),tol_large.predict(A[:,1:])))}


def build_report(A,y,vectors,spec):
    # Validate all inputs/configuration before the first numerical solver.
    A,y=validate_problem(A,y);s=validate_spec(spec);t=matrix(vectors,'basis');require(np.array_equal(t,load_vectors()),'unsupported basis')
    hand_exact=exact_ols(stored_fractions(A),[Fraction.from_float(float(v)) for v in y])
    hand={m:solve_ols(A,y,m) for m in ['lstsq','svd','qr','normal']}
    x,z,w=t.T;duplicate=np.column_stack([np.ones(7),x,x]);yd=1+3*x+s['noise_scale']*w
    rank_deficient=solve_ols(duplicate,yd)
    alternatives=[[1,3,0],[1,1.5,1.5],[1,0,3]]
    scale=s['feature_rescaling'];rescaled=np.column_stack([np.ones(7),x,scale*x])
    near=[]
    for k in s['near_collinear_powers']:
        B,v=construct_family(t,k,s['noise_scale']);ref=exact_ols(stored_fractions(B),[Fraction.from_float(float(q)) for q in v])
        Bp,vp=construct_family(t,k,s['noise_scale'],s['response_perturbation']);pref=exact_ols(stored_fractions(Bp),[Fraction.from_float(float(q)) for q in vp])
        methods={m:_add_reference_metrics(solve_ols(B,v,m),ref,B) for m in ['lstsq','svd','qr','normal']}
        pert=_add_reference_metrics(solve_ols(Bp,vp),pref,Bp)
        near.append({'power':k,'delta':2.0**(-k),'mathematical_rank':3,'stored_exact_rank':ref['rank'],
                     'exact':ref,'methods':methods,'perturbed_exact':pref,'perturbed_lstsq':pert,
                     'exact_coefficient_change_norm':math.sqrt(2)*s['response_perturbation']*2**k,
                     'exact_prediction_change_norm':2*s['response_perturbation'],
                     'rcond_sweep':[solve_ols(B,v,rcond=rc)|{'rcond':rc} for rc in s['rcond_values']]})
    interaction=np.column_stack([A,A[:,1]*A[:,2]])
    return {'environment':{'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__},'spec':s,
            'hand_exact':hand_exact,'hand_methods':hand,'sklearn':compare_sklearn(A,y),
            'rank_deficient':rank_deficient,'rank_deficient_exact_rank':exact_rank(stored_fractions(duplicate)),
            'equivalent_beta':alternatives,'equivalent_training_predictions':[(duplicate@v).tolist() for v in alternatives],
            'off_relation_row':[1,0,1],'off_relation_predictions':[v[0]+v[2] for v in alternatives],
            'rescaled':solve_ols(rescaled,yd),'rescaled_exact_beta':['1',str(Fraction(3)/(1+Fraction.from_float(scale)**2)),str(3*Fraction.from_float(scale)/(1+Fraction.from_float(scale)**2))],
            'interaction':solve_ols(interaction,y),'near_collinear':near,
            'scope':'Small deterministic synthetic CPU matrices. Solver errors, data sensitivity and predictive validity are different questions.'}


def run(input_path=None,vectors_path=None,spec_path=None,output_dir=None):
    target=Path(output_dir or ROOT/'outputs');require(not target.exists() or target.is_dir(),'output target must be directory')
    A,y=load_hand(input_path);t=load_vectors(vectors_path);s=load_spec(spec_path)
    report=build_report(A,y,t,s)
    encoded=json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)
    target.mkdir(parents=True,exist_ok=True);tmp=target/'report.json.tmp';tmp.write_text(encoded);tmp.replace(target/'report.json')
    return report


def self_test():
    A=np.array([[1,0,0],[1,1,0],[1,0,1],[1,1,1]],dtype=float);y=np.array([1,3,0,4],dtype=float);ref=exact_ols(stored_fractions(A),[Fraction.from_float(float(v)) for v in y])
    require(ref['beta']==['1/2','3','0'] and ref['sse']=='1','exact hand anchor')
    for method in ['lstsq','svd','qr','normal']:
        q=solve_ols(A,y,method);require(np.allclose(q['beta'],[.5,3,0],rtol=0,atol=1e-12) and abs(q['sse']-1)<1e-12,'hand solver')
    t=load_vectors();B,v=construct_family(t,48);q=exact_ols(stored_fractions(B),[Fraction.from_float(float(x)) for x in v])
    require(q['beta']==['1','2','1'] and q['sse']=='3/64','dyadic exact anchor')
    q=solve_ols([[1,0,0],[1,1,1],[1,2,2]],[0,2,0]);require(q['raw_lstsq_residuals']==[] and q['sse']>0,'empty residuals trap')
    require(solve_ols([[1,0,0],[1,1,1]],[1,2],'qr')['status']=='not_applicable','QR shape failure')
    return {'core_check_groups':8,'status':'passed'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input');p.add_argument('--vectors');p.add_argument('--spec');p.add_argument('--output-dir');args=p.parse_args()
    report=run(args.input,args.vectors,args.spec,args.output_dir);tests=self_test()
    print(json.dumps({'tests':tests,'hand_exact':report['hand_exact'],'ranks':[r['methods']['lstsq']['rank'] for r in report['near_collinear']],
                     'coefficient_errors':[r['methods']['lstsq']['coefficient_error_norm'] for r in report['near_collinear']]},ensure_ascii=False,indent=2))
