"""Independent scientific and common-failure checks; never disabled by python -O."""
from decimal import Decimal as D, localcontext
from fractions import Fraction as Q
from pathlib import Path
import argparse, copy, hashlib, json, os, tempfile
from unittest.mock import patch
import numpy as np
import experiment as e


def check(condition,label):
    if not condition:raise RuntimeError(label)


def near(actual,expected,label,atol=3e-14,rtol=4e-13):
    check(np.allclose(actual,expected,atol=atol,rtol=rtol),label)


def reject(fn,label):
    try:fn()
    except (ValueError,TypeError,OverflowError):return
    raise RuntimeError('accepted invalid input: '+label)


def dec(v):return D.from_float(float(v))


def decimal_state(X,y,theta,precision=100):
    """Direct exponent/normalization in high precision; no call to core functions."""
    with localcontext() as c:
        c.prec=precision;n=len(X);K=len(theta[0]);d=len(theta)
        T=[[dec(v) for v in row] for row in theta];g=[[D(0)]*K for _ in range(d)]
        loss=[];rows=[]
        for xx,yy in zip(X,y):
            a=[D(1),*[dec(v) for v in xx]]
            z=[sum(a[j]*T[j][k] for j in range(d)) for k in range(K)]
            ex=[v.exp() for v in z];den=sum(ex);p=[v/den for v in ex]
            l=-(p[int(yy)].ln());r=[v-D(int(k==yy)) for k,v in enumerate(p)]
            J=[[p[k]*(D(int(k==j))-p[j]) for j in range(K)] for k in range(K)]
            contrib=[[a[j]*r[k]/D(n) for k in range(K)] for j in range(d)]
            for j in range(d):
                for k in range(K):g[j][k]+=contrib[j][k]
            rows.append({'probabilities':list(map(float,p)),'loss':float(l),
                         'stable_dloss_dz':list(map(float,r)),
                         'local_softmax_jacobian':[[float(v) for v in row] for row in J],
                         'gradient_contribution':[[float(v) for v in row] for row in contrib]})
            loss.append(l)
        return {'rows':rows,'objective':float(sum(loss)/D(n)),'gradient':[[float(v) for v in row] for row in g]}


def science(X,y,c,r):
    n=len(X);G=[[sum((Q(1,3)-Q(int(yy==k)))*Q(int(xx[0]) if j else 1)/n for xx,yy in zip(X,y)) for k in range(3)] for j in range(2)]
    expected=[[Q(-1,24),Q(1,12),Q(-1,24)],[Q(1,8),Q(0),Q(-1,8)]]
    check(G==expected,'exact Fraction initial aggregate')
    near(r['hand_states'][0]['gradient'],[[float(v) for v in row] for row in G],'exact initial chain')
    near(r['hand_states'][1]['theta'],[[.025,-.05,.025],[-.075,0,.075]],'simultaneous first update')
    # Every saved GD state, row contribution and local Jacobian has independent reference.
    for state in r['fit']['trace']:
        ref=decimal_state(X,y,state['theta'])
        for key in ['objective','gradient']:near(state[key],ref[key],'Decimal trajectory '+key)
        for row,rr in zip(state['rows'],ref['rows']):
            for key in rr:near(row[key],rr[key],'Decimal row '+key)
            near(np.sum(row['probabilities']),1,'probability sum')
            near(np.sum(row['local_softmax_jacobian'],axis=1),0,'Jacobian null common-shift direction')
            local=np.array(row['local_dloss_dp']);J=np.array(row['local_softmax_jacobian'])
            near(local@J,row['stable_dloss_dz'],'explicit local chain')
        near(np.sum([row['gradient_contribution'] for row in state['rows']],axis=0),state['gradient'],'row aggregate')
    cert=r['certificate'];near(r['fit']['final']['theta'],cert['theta'],'analytic finite optimum',atol=8e-10)
    near(r['fit']['final']['objective'],cert['minimum_objective'],'analytic minimum')
    check(r['training_accuracy']==.5,'retained limited training accuracy')
    lib=r['reference_optimizer'];near(lib['state']['theta'],cert['theta'],'BFGS reference probabilities/coefficients',atol=1e-8)
    check(lib['state']['gradient_norm']<1e-8,'BFGS recomputed residual independent of success flag')
    # Non-square, one-row and varying class counts are inside the declared API domain.
    rng=np.random.default_rng(4305)
    dimensions=[(1,1,2),(4,3,5),(7,2,3),(2,5,2),(9,1,7)]
    for n,d,K in dimensions:
        A=rng.integers(-4,5,size=(n,d))/4;Y=rng.integers(0,K,size=n);T=rng.integers(-4,5,size=(d+1,K))/4
        st=e.row_ledger(A,Y,T);ref=decimal_state(A,Y,T)
        near(st['objective'],ref['objective'],'non-square Decimal objective');near(st['gradient'],ref['gradient'],'non-square Decimal gradient')
        perm=rng.permutation(K);inverse=np.argsort(perm)
        permuted=e.row_ledger(A,inverse[Y],T[:,perm]);near(st['objective'],permuted['objective'],'class permutation loss');near(np.asarray(st['gradient'])[:,perm],permuted['gradient'],'class permutation gradient')
    # Extreme logits require enough digits to retain exp(-1000) before float conversion.
    with localcontext() as ctx:
        ctx.prec=1000
        for z in [[0,-40,-40],[1000,0,-1000],[-1000,-1000,-1000],[0,0,-40],[0,-740,-741]]:
            pp,ll=e.stable_softmax([z]);zz=[D(v) for v in z];ex=[a.exp() for a in zz];den=sum(ex);rp=[v/den for v in ex];rl=[v.ln() for v in rp]
            near(pp[0],list(map(float,rp)),'1000-digit probabilities',atol=0,rtol=5e-13)
            near(ll[0],list(map(float,rl)),'1000-digit log probabilities',atol=0,rtol=5e-13)
            for yy in range(3):
                st=e.row_ledger([[0]],[yy],[z,[0,0,0]]) if max(map(abs,z))<=100 else None
                if st:
                    rr=[float(a-D(int(k==yy))) for k,a in enumerate(rp)]
                    near(st['rows'][0]['stable_dloss_dz'],rr,'1000-digit stable class derivative',atol=0,rtol=5e-13)
    check(e.stable_softmax([[0,-40,-40]])[1][0,0]<0,'preserve representable winning log tail')
    for value in r['diagnostics']['common_shifts']:
        check(value['probability_max_error']<8e-14 and value['log_probability_max_error']<2e-13,'float shift invariant tolerance')
    for scale in [1.,-.7]:
        check(min(v['maximum_gradient_error'] for v in r['diagnostics']['finite_difference'] if v['scale']==scale)<1e-9,'nonstationary difference')
    from scipy.special import softmax,log_softmax
    z=rng.normal(size=(30,5))*3;p,lp=e.stable_softmax(z)
    near(p,softmax(z,axis=1),'SciPy ordinary probability');near(lp,log_softmax(z,axis=1),'SciPy ordinary log probability')
    from scipy.stats import entropy
    I=r['information'];R=r['reverse_information']
    near(I['entropy'],entropy(I['q']),'SciPy entropy');near(I['kl'],entropy(I['q'],I['p']),'SciPy KL')
    near(I['cross_entropy'],I['entropy']+I['kl'],'entropy cross entropy KL decomposition')
    check(abs(I['kl']-R['kl'])>.001,'asymmetry actual example')
    zero=e.information([0,1,0],[0,1,0]);check(zero['entropy']==0 and zero['kl']==0,'zero conventions')
    check(e.information([1,0,0],[0,.5,.5])['infinite_cross_entropy_and_kl'],'missing support infinite, no clipping')
    return {'exact_initial_gradient':[[str(v) for v in row] for row in G],
            'Decimal_checked_full_states':len(r['fit']['trace']),'Decimal_precision':100,'tail_precision':1000,
            'non_square_shapes':dimensions,'reference_optimizer_success':lib['success'],
            'reference_optimizer_message':lib['message'],'reference_recomputed_gradient':lib['state']['gradient_norm']}


def guards(X,y,c):
    initial=c['initial_theta'];cases=0
    for bad in [True,'2',1+0j,np.nan,np.inf,1e200,1e-200,np.longdouble('1e-400')]:
        A=X.tolist();A[-1][-1]=bad;Y=y.tolist();Y[-1]=bad;T=copy.deepcopy(initial);T[-1][-1]=bad
        for fn in [lambda A=A:e.fit(A,y,initial),lambda Y=Y:e.fit(X,Y,initial),lambda T=T:e.fit(X,y,T)]:
            with patch.object(e,'_state',side_effect=RuntimeError('numerics reached before validation')):reject(fn,'bad tail')
            cases+=1
    for key in c:
        cc=copy.deepcopy(c)
        if isinstance(cc[key],list):cc[key][-1]=True
        else:cc[key]=True
        reject(lambda:e.validate_config(cc),'bad config '+key)
    for logits in [[],[1,2],[[1],[2]],[[1,2],[3]],[[True,0]],[[1,float('nan')]]]:reject(lambda:e.stable_softmax(logits),'logit shape/type')
    for q,p in [([1,1],[.5,.5]),([.5,-.1],[.5,.5]),([.5,.5],[.2,.3,.5])]:reject(lambda:e.information(q,p),'invalid distribution')
    short=e.fit(X,y,initial,max_updates=1);check(short['updates']==1,'short budget')
    zero=e.fit(X,y,initial,max_updates=0);check(zero['updates']==0 and len(zero['trace'])==1,'zero budget')
    stat=e.fit(X,y,e.exact_main_certificate()['theta']);check(stat['updates']==0,'initial stationary')
    rejected=e.fit(X,y,initial,learning_rate=100,max_updates=2);check(rejected['status']=='step_rejected' and rejected['updates']==0 and rejected['state_attempts']==2,'oversized learning rate failure')
    original=e._state;called=[]
    def fail_after(*a,**kw):
        out=original(*a,**kw);called.append(True)
        if len(called)==2:raise FloatingPointError('injected post-compute failure')
        return out
    with patch.object(e,'_state',side_effect=fail_after):failed=e.fit(X,y,initial)
    check(failed['state_attempts']==2 and failed['state_successes']==1 and failed['updates']==0,'failed evaluation charged')
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/'config.json'
        for text in ['{"x":1,"x":2}','{"x":NaN}','{"x":1e-400}']:
            p.write_text(text);reject(lambda:e.read_json(p),'strict JSON')
        out=Path(d)/'result.json';out.write_bytes(b'OLD')
        reject(lambda:e.safe_write(out,{'bad':float('nan')}),'strict output');check(out.read_bytes()==b'OLD','serialize first')
        with patch.object(e.os,'replace',side_effect=OSError('simulated full disk')):
            try:e.safe_write(out,{'valid':1})
            except OSError:pass
            else:raise RuntimeError('injected replacement succeeded')
        check(out.read_bytes()==b'OLD' and not list(Path(d).glob('.ml043-*')),'atomic replacement failure')
        reject(lambda:e.safe_write(e.ROOT/'data/model_spec.json',{}),'source preservation')
        sym=Path(d)/'link.json';sym.symlink_to(out);reject(lambda:e.safe_write(sym,{}),'symlink preservation')
        link=Path(d)/'hard.json';os.link(out,link);reject(lambda:e.safe_write(link,{}),'hardlink preservation')
    return {'invalid_numeric_tail_cases':cases,'zero_short_stationary_rejected_states_checked':True,
            'config_all_fields_checked':True,'failure_cost_checked':True,'atomic_output_preservation':True}


def audit():
    X,y,ids,c=e.load_inputs();r=e.main_report(X,y,ids,c)
    return {'unit':'043','science':science(X,y,c,r),'guards':guards(X,y,c),
            'core_sha256':hashlib.sha256(e.canonical_bytes(r)).hexdigest()}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args();result=audit();e.safe_write(args.out,result);print(json.dumps(result,ensure_ascii=False))
