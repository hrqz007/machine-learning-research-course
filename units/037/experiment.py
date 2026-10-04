"""Two-feature convex composite regression: explicit row traces and guarded IO.

High-level APIs validate all raw fields before any numerical kernel executes.
Low-level forward/soft/diagnostics/one_step accept already validated arrays.
The orthogonal closed form is a diagnostic only, never a solver update.
"""
from pathlib import Path
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import argparse, copy, csv, hashlib, json, math, os, tempfile
import numpy as np
ROOT=Path(__file__).resolve().parent
FIELDS={'kind','initial_parameters','steps','learning_rate','lambda','tolerance','lower_bounds','upper_bounds','coordinate_order','stress_rho'}
METHODS=('proximal','subgradient','projected','coordinate')
NUMPY_TYPES={'int8','int16','int32','int64','uint8','uint16','uint32','uint64','float16','float32','float64'}

def real(x,name='number',limit=100.,floor=1e-100):
    t=type(x)
    if t not in (int,float) and not (t.__module__=='numpy' and t.__name__ in NUMPY_TYPES):raise ValueError(name+' requires a real scalar, never bool/string/extended precision')
    try:v=float(x)
    except (OverflowError,ValueError) as e:raise ValueError(name+' conversion') from e
    if not math.isfinite(v) or abs(v)>limit or (v!=0 and abs(v)<floor):raise ValueError(name+' outside supported finite range')
    if isinstance(x,(int,np.integer)) and int(v)!=int(x):raise ValueError(name+' integer precision loss')
    return v

def integer(x,name,lo,hi):
    if type(x) is not int or not lo<=x<=hi:raise ValueError(name+' requires bounded Python int')
    return x

def vector(x,name,length=2,limit=100.):
    if not isinstance(x,(list,tuple,np.ndarray)) or (isinstance(x,np.ndarray) and x.ndim!=1) or len(x)!=length:raise ValueError(name+' shape')
    return np.array([real(v,name,limit) for v in x],dtype=float)

def validate(data,spec):
    if not isinstance(data,(list,tuple,np.ndarray)) or (isinstance(data,np.ndarray) and data.ndim!=2) or not 2<=len(data)<=64:raise ValueError('data needs 2..64 rows')
    data=np.array([vector(row,'data row',3) for row in data])
    if type(spec) is not dict or set(spec)!=FIELDS:raise ValueError('config fields mismatch')
    s=copy.deepcopy(spec)
    if type(s['kind']) is not str or s['kind']!='synthetic_lasso_two_features':raise ValueError('kind mismatch')
    for k in ('initial_parameters','lower_bounds','upper_bounds'):s[k]=vector(s[k],k).tolist()
    s['steps']=integer(s['steps'],'steps',2,200)
    for k in ('learning_rate','lambda','tolerance','stress_rho'):s[k]=real(s[k],k)
    if not 1e-6<=s['learning_rate']<=10 or not 0<=s['lambda']<=10 or not 1e-12<=s['tolerance']<=1e-3 or not 0<=s['stress_rho']<=.99:raise ValueError('hyperparameter range')
    if any(a>=b for a,b in zip(s['lower_bounds'],s['upper_bounds'])):raise ValueError('box bounds need lower < upper')
    order=s['coordinate_order']
    if type(order) is not list or len(order)!=2 or any(type(j) is not int for j in order) or sorted(order)!=[0,1]:raise ValueError('coordinate_order must permute Python integers [0,1]')
    return data,s

def token(text):
    try:d=Decimal(text)
    except (InvalidOperation,TypeError) as e:raise ValueError('bad numeric token') from e
    if not d.is_finite():raise ValueError('nonfinite numeric token')
    v=float(d)
    if d!=0 and v==0:raise ValueError('numeric token underflow')
    return real(v)

def pairs(items):
    d={}
    for k,v in items:
        if k in d:raise ValueError('duplicate JSON key: '+k)
        d[k]=v
    return d

def load_inputs(data=ROOT/'data/regression.csv',config=ROOT/'data/model_spec.json'):
    with Path(data).open(newline='') as f:
        r=csv.DictReader(f)
        if r.fieldnames!=['id','x1','x2','y']:raise ValueError('CSV header mismatch')
        rows=list(r)
    if any(set(row)!= {'id','x1','x2','y'} or None in row.values() or not row['id'] or len(row['id'])>40 for row in rows):raise ValueError('CSV row mismatch')
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('duplicate sample ID')
    a=[[token(r[k]) for k in ('x1','x2','y')] for r in rows]
    s=json.loads(Path(config).read_text(),parse_float=token,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite JSON')),object_pairs_hook=pairs)
    return validate(a,s)

def finite(x,name):
    a=np.asarray(x,dtype=float)
    if not np.all(np.isfinite(a)) or np.max(np.abs(a),initial=0)>1e120:raise FloatingPointError(name+' exceeds arithmetic range')
    return a

def multiply(a,b,name):
    z=finite(np.asarray(a)*np.asarray(b),name)
    if np.any((np.asarray(a)!=0)&(np.asarray(b)!=0)&(z==0)):raise FloatingPointError(name+' multiplication underflow')
    return z

def soft(z,tau):
    """Exact zero branch, including both equality boundaries."""
    return np.where(z>tau,z-tau,np.where(z < -tau,z+tau,0.))

def forward(data,theta,lam):
    X,y=data[:,:2],data[:,2];n=len(data)
    products=multiply(X,theta,'row prediction products');pred=finite(np.sum(products,axis=1),'predictions')
    r=finite(pred-y,'residual');squares=multiply(r,r,'residual squares');losses=squares/2
    contributions=multiply(X,r[:,None],'row derivatives')/n
    g=finite(np.sum(contributions,axis=0),'gradient')
    f=float(np.sum(losses)/n);pen=float(lam*np.sum(np.abs(theta)));finite([f,pen,f+pen],'objectives')
    rows=[{'id':i+1,'x':X[i].tolist(),'y':float(y[i]),'parameter_products':products[i].tolist(),'prediction':float(pred[i]),'residual':float(r[i]),'half_square':float(losses[i]),'d_loss_d_residual':float(r[i]),'d_residual_d_prediction':1.,'d_prediction_d_theta':X[i].tolist(),'mean_gradient_contribution':contributions[i].tolist()} for i in range(n)]
    return {'theta':theta.tolist(),'rows':rows,'gradient':g.tolist(),'data_loss':f,'l1_penalty':pen,'objective':f+pen}

def diagnostics(data,theta,s,method):
    fw=forward(data,theta,s['lambda']);g=np.array(fw['gradient']);eta=s['learning_rate'];lam=s['lambda']
    v=finite(theta-multiply(eta,g,'diagnostic step'),'diagnostic z')
    gm=finite((theta-soft(v,eta*lam))/eta,'gradient mapping')
    # Minimal absolute coordinate in g + lambda * partial |theta|.
    kkt=np.where(theta>0,g+lam,np.where(theta<0,g-lam,np.sign(g)*np.maximum(np.abs(g)-lam,0.)))
    projected=np.clip(v,s['lower_bounds'],s['upper_bounds']);pg=(theta-projected)/eta
    feasibility=float(np.max(np.maximum(np.array(s['lower_bounds'])-theta,np.maximum(theta-np.array(s['upper_bounds']),0.))))
    result={'smooth_gradient_inf':float(np.max(np.abs(g))),'lasso_kkt_vector':kkt.tolist(),'lasso_kkt_inf':float(np.max(np.abs(kkt))),'proximal_mapping':gm.tolist(),'proximal_mapping_inf':float(np.max(np.abs(gm))),'projected_mapping':pg.tolist(),'projected_mapping_inf':float(np.max(np.abs(pg))),'box_violation':feasibility,'exact_zero_count':int(np.sum(theta==0))}
    default=np.array_equal(data,np.array([[-1.,-1.,-1.75],[-1.,1.,-2.25],[1.,-1.,1.25],[1.,1.,2.75]]))
    if default:
        optimum=soft(np.array([2.,.25]),lam);optfw=forward(data,optimum,lam)
        # Stable expansion around the exact orthogonal optimum avoids subtraction.
        d=theta-optimum; c=np.array([2.,.25]); gap=.5*float(np.dot(d,d))
        for j in range(2):
            if optimum[j]>0:gap+=2*lam*max(-theta[j],0.)
            elif optimum[j]<0:gap+=2*lam*max(theta[j],0.)
            else:gap+=lam*abs(theta[j])-c[j]*theta[j]
        result.update(default_optimum=optimum.tolist(),default_minimum=optfw['objective'],default_objective_gap=gap)
    else:result.update(default_optimum=None,default_minimum=None,default_objective_gap=None)
    result['stopping_residual']=max(result['projected_mapping_inf'],feasibility) if method=='projected' else result['lasso_kkt_inf']
    return result

def one_step(data,theta,s,method):
    old=forward(data,theta,s['lambda']);g=np.array(old['gradient']);eta=s['learning_rate'];lam=s['lambda']
    z=finite(theta-multiply(eta,g,'smooth step'),'smooth trial');tau=eta*lam
    details=[]
    if method=='proximal':new=soft(z,tau)
    elif method=='subgradient':
        sub=g+lam*np.sign(theta);new=finite(theta-multiply(eta,sub,'subgradient step'),'subgradient new')
        details={'chosen_l1_subgradient':np.sign(theta).tolist(),'total_subgradient':sub.tolist()}
    elif method=='projected':new=np.clip(z,s['lower_bounds'],s['upper_bounds'])
    else:
        new=theta.copy();X,y=data[:,:2],data[:,2];n=len(data)
        for j in s['coordinate_order']:
            before=new.copy();partial=finite(y-np.sum(multiply(X,new,'coordinate products'),axis=1)+X[:,j]*new[j],'partial response')
            q=float(np.sum(multiply(X[:,j],X[:,j],'coordinate squares'))/n)
            c=float(np.sum(multiply(X[:,j],partial,'coordinate correlation'))/n)
            new[j]=0. if q==0 else float(soft(np.array([c]),lam)[0]/q)
            finite(new,'coordinate update')
            details.append({'coordinate':j,'old_theta':before.tolist(),'partial_response':partial.tolist(),'q':q,'c':c,'new_theta':new.tolist()})
    finite(new,'new theta');newfw=forward(data,new,lam);diag=diagnostics(data,new,s,method)
    return new,{'old_forward':old,'smooth_trial':z.tolist(),'threshold':tau,'coordinate_branches':['positive' if v>tau else 'negative' if v < -tau else 'zero' for v in z],'details':details,'new_forward':newfw,'diagnostics':diag}

def run_path(data,spec,method):
    data,s=validate(data,spec)
    if type(method) is not str or method not in METHODS:raise ValueError('unknown method')
    theta=np.array(s['initial_parameters']);projection_of_initial=None
    if method=='projected':projection_of_initial=theta.tolist();theta=np.clip(theta,s['lower_bounds'],s['upper_bounds'])
    first=forward(data,theta,s['lambda']);firstdiag=diagnostics(data,theta,s,method);diag=firstdiag;trace=[];attempts=0;status='fixed_budget_completed'
    if diag['stopping_residual']<=s['tolerance']:status='tolerance_satisfied'
    else:
        for t in range(s['steps']):
            attempts+=1
            try:new,tr=one_step(data,theta,s,method)
            except (FloatingPointError,OverflowError) as e:status='arithmetic_range_stop: '+str(e);break
            theta=new;tr['update']=t+1;trace.append(tr);diag=tr['diagnostics']
            if diag['stopping_residual']<=s['tolerance']:status='tolerance_satisfied';break
    return {'method':method,'parameters':s,'projection_of_initial':projection_of_initial,'initial_forward':first,'initial_diagnostics':firstdiag,'trace':trace,'final_theta':theta.tolist(),'final_diagnostics':diag,'status':status,'completed_updates':len(trace),'attempted_updates':attempts,'update_unit':'full cyclic sweep' if method=='coordinate' else 'simultaneous full batch update','objective_name':'data_loss over box' if method=='projected' else 'data_loss + lambda L1'}

def exact_hand(data,spec):
    data,s=validate(data,spec);Q=Fraction;rows=[[Q(float(v)) for v in row] for row in data];theta=[Q(v) for v in s['initial_parameters']];eta=Q(s['learning_rate']);lam=Q(s['lambda']);n=len(rows)
    def fwd(th):
        out=[];g=[Q(0),Q(0)]
        for x1,x2,y in rows:
            p=x1*th[0]+x2*th[1];r=p-y;c=[r*x1/n,r*x2/n];g=[a+b for a,b in zip(g,c)]
            out.append({'prediction':str(p),'residual':str(r),'half_square':str(r*r/2),'local_derivatives':[str(r),'1',str(x1),str(x2)],'gradient_contribution':list(map(str,c))})
        f=sum(Q(v['half_square']) for v in out)/n;return {'theta':list(map(str,th)),'rows':out,'gradient':list(map(str,g)),'data_loss':str(f),'objective':str(f+lam*sum(map(abs,th)))}
    result=[]
    for _ in range(2):
        old=fwd(theta);g=list(map(Q,old['gradient']));z=[a-eta*b for a,b in zip(theta,g)];tau=eta*lam
        theta=[v-tau if v>tau else v+tau if v < -tau else Q(0) for v in z]
        result.append({'old':old,'smooth_trial':list(map(str,z)),'threshold':str(tau),'new':fwd(theta)})
    return result

def run_experiment(data,spec):
    data,s=validate(data,spec) # Every field, including the last, before ANY solver.
    main={method:run_path(data,s,method) for method in METHODS}
    stress=data.copy();stress[:,1]=s['stress_rho']*data[:,0]+(1-s['stress_rho'])*data[:,1]
    stress_s=copy.deepcopy(s);L=float(np.linalg.eigvalsh(stress[:,:2].T@stress[:,:2]/len(data))[-1]);stress_s['learning_rate']=min(s['learning_rate'],1/L) if L>0 else s['learning_rate']
    # Internal derived step may be below the public bound; do not run narrower API.
    stress_result=None
    stress_safe=stress_s['learning_rate']>=1e-6 and not np.any((stress!=0)&(np.abs(stress)<1e-100))
    if stress_safe:stress_result={m:run_path(stress,stress_s,m) for m in ('proximal','coordinate')}
    return {'schema':'unit037.v1','config':s,'data':data.tolist(),'paths':main,'exact_first_two':exact_hand(data,s),'stress':{'label':'explicit transformed correlated design, not the main experiment','data':stress.tolist(),'L':L,'status':'run' if stress_result is not None else 'derived_input_outside_public_range','paths':stress_result},'limits':'Finite synthetic optimization experiment; no generalization or universal convergence claim.'}

def protect_output(path,inputs=()):
    p=Path(os.path.abspath(path));cursor=p
    while True:
        if cursor.is_symlink():raise ValueError('output path contains symbolic link')
        if cursor==cursor.parent:break
        cursor=cursor.parent
    if p.exists() and (not p.is_file() or p.stat().st_nlink>1):raise ValueError('output is not an unlinked regular file')
    assets=[v for v in ROOT.iterdir() if v.is_file()]+list((ROOT/'data').rglob('*'))+list((ROOT/'figures').rglob('*'))
    for q in [*map(Path,inputs),*assets]:
        if p.resolve()==q.resolve() or (p.exists() and q.exists() and q.is_file() and os.path.samefile(p,q)):raise ValueError('output would replace input or package asset')
    return p

def write_report(report,path,inputs=()):
    payload=(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    p=protect_output(path,inputs) # No directory is made until validation and serialization finish.
    missing=[];cursor=p.parent
    while not cursor.exists():missing.append(cursor);cursor=cursor.parent
    tmp=None
    try:
        p.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=p.parent,prefix='.report-',delete=False) as f:tmp=f.name;f.write(payload);f.flush();os.fsync(f.fileno())
        os.replace(tmp,p)
    except BaseException:
        if tmp and os.path.exists(tmp):os.unlink(tmp);tmp=None
        for directory in missing:
            if directory.exists():directory.rmdir()
        raise
    finally:
        if tmp and os.path.exists(tmp):os.unlink(tmp)
    return {'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=ROOT/'data/regression.csv');p.add_argument('--config',type=Path,default=ROOT/'data/model_spec.json');p.add_argument('--output-dir',type=Path,default=Path('outputs'));a=p.parse_args()
    data,s=load_inputs(a.data,a.config);dest=protect_output(a.output_dir/'report.json',(a.data,a.config));report=run_experiment(data,s);meta=write_report(report,dest,(a.data,a.config));print(json.dumps({'report':str(dest),'sha256':meta['sha256'],'statuses':{k:v['status'] for k,v in report['paths'].items()}},ensure_ascii=False))
if __name__=='__main__':main()
