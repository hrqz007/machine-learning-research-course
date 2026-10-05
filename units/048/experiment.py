"""048: risks, exact fixed-design reference, repeated random-design learning.

No resampling seed search, clipping, failed-fit exclusion, explicit inverse or
iteration-count story. Pure least squares is solved by NumPy's SVD route.
"""
from pathlib import Path
import argparse, csv, hashlib, itertools, json, os, tempfile
import numpy as np
from numpy.polynomial.legendre import legvander, leggauss
ROOT=Path(__file__).resolve().parent

def check(condition, message):
    if not condition: raise ValueError(message)

def basis(x,p):
    x=np.asarray(x,dtype=float)
    check(x.ndim==1 and 1<=len(x)<=100000,'x must be a nonempty vector')
    check(np.isfinite(x).all() and np.all(np.abs(x)<=1),'x must be finite and in [-1,1]')
    check(type(p) is int and 1<=p<=9,'p must be an integer in 1..9')
    return legvander(x,p-1)*np.sqrt(2*np.arange(p)+1)

def fit(x,y,p):
    A=basis(x,p); y=np.asarray(y,dtype=float)
    check(y.shape==(len(A),) and np.isfinite(y).all(),'y must be a finite matching vector')
    check(len(y)>=p,'at least p observations required')
    coef,_,rank,s=np.linalg.lstsq(A,y,rcond=None)
    check(rank==p,'rank deficient design; no silent fit exclusion')
    pred=A@coef; resid=pred-y
    return coef, {'train':float(np.mean(resid**2)), 'condition':float(s[0]/s[-1]), 'normal_residual':float(np.max(np.abs(A.T@resid))/len(y)), 'rank':int(rank)}

def atomic_json(path,obj):
    path=Path(path).absolute()
    for item in (path,*path.parents):check(not item.is_symlink(),'symlink output is unsupported')
    check(path.suffix=='.json','output must have .json suffix')
    protected={p.resolve() for p in ROOT.glob('*') if p.is_file()}
    protected|={p.resolve() for p in (ROOT/'data').glob('*') if p.is_file()}
    check(path.resolve() not in protected,'cannot overwrite shipped teaching files')
    check(not path.exists() or (path.is_file() and path.stat().st_nlink==1),'unsafe output')
    payload=json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix='.ml048-')
    try:
        with os.fdopen(fd,'w') as f:f.write(payload)
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)

def load_data():
    hashes=json.loads((ROOT/'data_integrity.json').read_text())
    for name,digest in hashes.items():
        check(hashlib.sha256((ROOT/'data'/name).read_bytes()).hexdigest()==digest,'data hash mismatch: '+name)
    cfg=json.loads((ROOT/'data/config.json').read_text())
    with np.load(ROOT/'data/draws.npz',allow_pickle=False) as z: data={k:z[k] for k in z.files}
    check(set(data)=={'x','eps','xt','et','eps_fixed'},'unexpected archive keys')
    for k,a in data.items():
        expected=(400,40) if k=='eps_fixed' else (400,256)
        check(a.shape==expected and np.isfinite(a).all(),'invalid draws: '+k)
    return cfg,data

def hand_reference():
    with (ROOT/'data/hand.csv').open() as fh: raw=list(csv.DictReader(fh))
    x=np.array([float(row['x']) for row in raw]); y=np.array([float(row['y']) for row in raw])
    check(np.array_equal(x,[-1.,0.,1.]) and np.array_equal(y,[1.,-1.,1.]),'paper example contract')
    with (ROOT/'data/enumeration.csv').open() as fh: combos=[tuple(float(row[k]) for k in ('e1','e2','e3')) for row in csv.DictReader(fh)]
    check(len(combos)==8 and set(combos)==set(itertools.product([-1.,1.],repeat=3)),'eight-state enumeration contract')
    q,w=leggauss(32);w=w/2
    rows=[]; summary=[]
    for p in (1,2,3):
        A=np.vander(x,p,increasing=True);coef=np.linalg.lstsq(A,y,rcond=None)[0];pred=A@coef
        integ=float(np.dot(w,(np.vander(q,p,increasing=True)@coef)**2))
        summary.append({'p':p,'monomial_coef':coef.tolist(),'train_mse':float(np.mean((pred-y)**2)),'signal_error':integ,'risk':integ+1})
        for i in range(3):rows.append({'p':p,'row':i+1,'x':float(x[i]),'y':float(y[i]),'prediction':float(pred[i]),'residual_pred_minus_y':float(pred[i]-y[i]),'squared_loss':float((pred[i]-y[i])**2),'loss_div_n':float((pred[i]-y[i])**2/3)})
    enum=[]
    for p in (1,2,3):
        predictions=[];tr=[]
        for eps in combos:
            A=np.vander(x,p,increasing=True);c=np.linalg.lstsq(A,np.array(eps),rcond=None)[0]
            predictions.append(np.vander(q,p,increasing=True)@c);tr.append(float(np.mean((A@c-eps)**2)))
        predictions=np.array(predictions);mean=predictions.mean(axis=0)
        enum.append({'p':p,'bias2':float(np.dot(w,mean**2)), 'variance':float(np.dot(w,np.var(predictions,axis=0,ddof=0))), 'noise':1.0,'risk':float(np.dot(w,np.mean(predictions**2,axis=0)))+1,'mean_train':float(np.mean(tr))})
    return {'realized':summary,'rows':rows,'all_eight_equiprobable_designs':enum}

def evaluate_block(x,y,xt,yt,p,beta,sigma):
    R=len(x); coef=np.zeros((R,len(beta))); rows=[]
    for r in range(R):
        c,diag=fit(x[r],y[r],p);coef[r,:p]=c
        risk=float(np.sum((coef[r]-beta)**2)+sigma**2)
        diag.update({'rep':r,'risk':risk,'test':float(np.mean((basis(xt[r],p)@c-yt[r])**2))})
        rows.append(diag)
    mean=coef.mean(axis=0);bias2=float(np.sum((mean-beta)**2));var0=float(np.var(coef,axis=0,ddof=0).sum());var1=float(np.var(coef,axis=0,ddof=1).sum())
    vals=np.array([a['risk'] for a in rows]);train=np.array([a['train'] for a in rows]);test=np.array([a['test'] for a in rows])
    summary={'p':p,'n':len(x[0]),'repetitions':R,'bias2_mc':bias2,'variance_ddof0':var0,'variance_ddof1':var1,'bias2_unbiased_estimate':bias2-var1/R,'noise':sigma**2,'approximation':float(np.sum(beta[p:]**2)), 'risk_mean':float(vals.mean()), 'risk_sd_ddof1':float(vals.std(ddof=1)), 'risk_mcse':float(vals.std(ddof=1)/np.sqrt(R)), 'risk_q10':float(np.quantile(vals,.1)), 'risk_q90':float(np.quantile(vals,.9)), 'train_mean':float(train.mean()), 'test_mean':float(test.mean()), 'test_mcse':float(test.std(ddof=1)/np.sqrt(R)), 'max_condition':float(max(a['condition'] for a in rows)), 'max_normal_residual':float(max(a['normal_residual'] for a in rows)), 'decomposition_residual':float(vals.mean()-bias2-var0-sigma**2)}
    return {'summary':summary,'runs':rows,'coefficients':coef.tolist()}

def run():
    cfg,d=load_data();beta=np.array(cfg['beta']);sig=cfg['sigma'];R=cfg['repetitions']
    f=lambda a: np.stack([basis(row,9)@beta for row in a])
    y=f(d['x'])+d['eps'];yt=f(d['xt'])+d['et']
    complexity=[]
    for p in cfg['parameters']:
        complexity.append(evaluate_block(d['x'][:,:40],y[:,:40],d['xt'],yt,p,beta,sig))
    learning=[]
    for p in cfg['learning_p']:
        for n in cfg['learning_n']:learning.append(evaluate_block(d['x'][:,:n],y[:,:n],d['xt'],yt,p,beta,sig))
    xf=np.tile(np.linspace(-.95,.95,40),(R,1));yf=f(xf)+d['eps_fixed']
    fixed=[evaluate_block(xf,yf,d['xt'],yt,p,beta,sig) for p in (2,4,8)]
    q,w=leggauss(32);w=w/2; sample=np.array(complexity[-1]['coefficients'][0]); integ=float(np.dot(w,(basis(q,9)@(sample-beta))**2)+sig**2)
    from sklearn.linear_model import LinearRegression
    A=basis(d['x'][0,:40],9); m=LinearRegression(fit_intercept=False).fit(A,y[0,:40]);c=np.array(complexity[-1]['coefficients'][0])
    alignment={'numpy_route':'lstsq rcond=None','sklearn_route':'LinearRegression(fit_intercept=False), same orthonormal columns','max_coef_difference':float(np.max(np.abs(m.coef_-c))), 'max_prediction_difference':float(np.max(np.abs(m.predict(A)-A@c))), 'quadrature_risk':integ,'coefficient_risk':complexity[-1]['runs'][0]['risk'],'quadrature_difference':abs(integ-complexity[-1]['runs'][0]['risk'])}
    failures=[]
    for name,xx,yy,p in [('rank_deficient',[0,0,0],[1,2,3],3),('n_less_p',[0,1],[1,2],3),('nonfinite',[0,np.nan],[1,2],1),('outside_domain',[-2,0],[1,2],1)]:
        try:fit(xx,yy,p)
        except ValueError as e:failures.append({'case':name,'status':'rejected','message':str(e)})
        else:raise RuntimeError('invalid case accepted: '+name)
    return {'schema':1,'protocol':cfg,'hand':hand_reference(),'complexity':complexity,'learning':learning,'fixed_design':fixed,'alignment':alignment,'negative_tests':failures,'limitations':['synthetic mechanism study, not empirical evidence for a real application','400 repetitions do not identify the exact infinite-repetition expected risk','cross-n and cross-p points are paired; no independent-point confidence claims','test outcomes used for illustration only, not model selection','no iterative optimizer; normal equation residual is a numerical check, not a statistical guarantee']}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default=str(ROOT/'outputs/result.json'));args=ap.parse_args(); result=run();atomic_json(args.out,result)
    print(json.dumps({'out':str(Path(args.out).absolute()),'alignment':result['alignment'],'complexity':[a['summary'] for a in result['complexity']]},ensure_ascii=False,allow_nan=False))
if __name__=='__main__':main()
