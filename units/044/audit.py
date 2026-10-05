"""Independent numeric references and focused contract checks for ML044.

A passed author audit is evidence about these checks, not independent acceptance.
"""
from pathlib import Path
from decimal import Decimal, localcontext
import argparse, csv, json, math, tempfile
import numpy as np
from scipy.stats import poisson
from sklearn.metrics import mean_poisson_deviance
import experiment as ex


def require(condition, label):
    if not condition: raise RuntimeError('audit failed: ' + label)


def near(a, b, label, atol=2e-11, rtol=2e-11):
    require(np.allclose(a, b, atol=atol, rtol=rtol), label)


def decimal_state(beta):
    """Scalar 80-digit PMF calculation: no NumPy/SciPy or production state call."""
    D=Decimal; xs=[D(-1), D(0), D(1)]; ys=[0,1,3]; n=D(3)
    eta=[beta[0]+beta[1]*x for x in xs]; mu=[z.exp() for z in eta]
    # Compute -log(P(Y=y)) from a factorial PMF, independently of vector loss.
    loss=[-(m**y*(-m).exp()/D(math.factorial(y))).ln() for m,y in zip(mu,ys)]
    grad=[sum(m-D(y) for m,y in zip(mu,ys))/n,
          sum((m-D(y))*x for m,y,x in zip(mu,ys,xs))/n]
    H=[[sum(m*(D(1) if j==0 else x)*(D(1) if k==0 else x) for m,x in zip(mu,xs))/n for k in range(2)] for j in range(2)]
    return eta,mu,loss,grad,H


def audit():
    ex.verify_data(); checks=[]; errors={}
    X=np.array([[1.,-1],[1.,0],[1.,1]]); y=np.array([0.,1.,3.]); e=np.ones(3)
    with localcontext() as ctx:
        ctx.prec=80; D=Decimal; b=[D(0),D(0)]; bf=np.zeros(2); worst=0.
        for k in range(3):
            z,m,l,g,H=decimal_state(b); s=ex.state(bf,X,y,e)
            for name, ref in [('eta',z),('mu',m),('loss',l),('gradient',g),('hessian',H)]:
                arr=np.array(ref,dtype=float); near(s[name],arr,'Decimal '+name)
                worst=max(worst,float(np.max(np.abs(np.array(s[name])-arr))))
            near(s['mean_loss'],float(sum(l)/3),'Decimal mean loss')
            if k<2:
                b=[q-D('0.2')*d for q,d in zip(b,g)]
                bf=bf-.2*np.array(s['gradient'])
        t=(D(3)+D(37).sqrt())/D(2); a=D(4)/(t+D(1)+1/t)
        analytic=np.array([float(a.ln()),float(t.ln())])
        fit=ex.fit_poisson(X,y,e)
        require(fit['status']=='gradient_tolerance','tiny optimizer converged')
        near(fit['state']['beta'],analytic,'analytic MLE',atol=2e-9)
        errors['decimal_three_states_max_abs']=worst
        errors['analytic_mle_beta']=analytic.tolist()
        errors['analytic_mle_max_abs_error']=float(np.max(abs(np.array(fit['state']['beta'])-analytic)))
    checks += ['80-digit scalar probability references for all three hand states', 'analytic finite MLE with positive root of t^2-3t-7']
    fd=[]
    for b in [np.array([0.,0.]),np.array([.2,-.4]),np.array([-.3,.6])]:
        s=ex.state(b,X,y,e); g=np.array(s['gradient']); H=np.array(s['hessian'])
        for h in [1e-3,1e-4,1e-5]:
            ge=[]; He=[]
            for j in range(2):
                d=np.eye(2)[j]*h; plus=ex.state(b+d,X,y,e); minus=ex.state(b-d,X,y,e)
                ge.append((plus['mean_loss']-minus['mean_loss'])/(2*h))
                He.append((np.array(plus['gradient'])-np.array(minus['gradient']))/(2*h))
            errg=float(np.max(abs(g-np.array(ge)))); errH=float(np.max(abs(H-np.array(He).T)))
            require(errg<2e-6 and errH<2e-6,'central differences')
            fd.append({'beta':b.tolist(),'h':h,'gradient_error':errg,'hessian_error':errH})
        v=np.array([.7,-1.3]); near(v@H@v,np.mean(np.array(s['mu'])*(X@v)**2),'Hessian PSD identity')
        near(s['loss'],-poisson.logpmf(y,s['mu']),'scipy pmf')
        near(np.mean(ex.deviance_rows(y,s['mu'])),mean_poisson_deviance(y,s['mu']),'sklearn deviance')
        r=np.array(s['mean_gradient_contributions']); near(r.sum(axis=0),g,'row aggregation')
    checks += ['three nonstationary points and three finite-difference step sizes', 'Hessian quadratic identity', 'SciPy PMF and sklearn deviance']
    s=ex.state([.2,.4],X,y,e); shifted=ex.state([.2-math.log(4),.4],X,y,4*e)
    for k in ['mu','loss','gradient','hessian']:near(s[k],shifted[k],'exposure unit invariance '+k)
    exposure=np.array([.5,1.,2.]); b=np.array([.2,.4]); count=ex.state(b,X,y,exposure)
    rate_grad=X.T@(exposure*(np.exp(X@b)-y/exposure))/exposure.sum()
    near(rate_grad,np.array(count['gradient'])*len(y)/exposure.sum(),'weighted rate objective gradient equivalence')
    near(ex.deviance_rows([0,1,3],[2,1,3]),[4,0,0],'zero and saturated deviance')
    eps=1e-8
    with localcontext() as ctx:
        ctx.prec=80; d=Decimal.from_float(1+eps); ref=float(2*(-d.ln()-1+d))
    near(ex.deviance_rows([1],[1+eps]),[ref],'near-saturation stable deviance',atol=1e-28,rtol=1e-12)
    checks += ['offset exposure-unit invariance', 'weighted-rate gradient equivalence', 'zero-count and saturated deviance', 'near-saturation Decimal deviance']
    failures=[('negative count',lambda:ex.state([0,0],X,[-1,1,3],e)),
              ('fractional count',lambda:ex.state([0,0],X,[0,.5,3],e)),
              ('bool input',lambda:ex.state([0,0],X,[True,False,True],e)),
              ('zero exposure',lambda:ex.state([0,0],X,y,[0,1,1])),
              ('nonfinite input',lambda:ex.state([0,np.nan],X,y,e)),
              ('misaligned input',lambda:ex.state([0,0],X,y[:2],e)),
              ('eta outside scope',lambda:ex.state([31,0],X,y,e)),
              ('rank-deficient fit',lambda:ex.fit_poisson(np.ones((3,2)),y,e)),
              ('all-zero finite-MLE refusal',lambda:ex.fit_poisson(X,np.zeros(3),e)),
              ('negative prediction deviance',lambda:ex.deviance_rows(y,[-1,1,2]))]
    for name, fn in failures:
        try: fn()
        except ValueError: pass
        else: raise RuntimeError('expected ValueError: '+name)
    budget=ex.fit_poisson(X,y,e,max_updates=0)
    require(budget['status']=='max_updates' and budget['updates']==0,'budget reported honestly')
    checks+=['10 focused invalid-input cases remain active under -O', 'zero-update budget is not convergence']
    run=ex.scientific_run()
    for label,c in run['cases'].items():
        for key in ['poisson_fit','no_offset_fit']:
            require(c[key]['status']=='gradient_tolerance',label+' '+key+' converged')
        require(c['sklearn_max_abs_beta_difference']<2e-8,label+' actual sklearn fit')
        require(c['sklearn_gradient_l2_original_count_objective']<2e-8,label+' sklearn original-target gradient')
        require(c['test_metrics']['linear_exposure']['mean_poisson_deviance'] is None,'invalid OLS deviance stays absent')
    checks+=['actual sklearn unregularized fits through exposure-weighted rates', 'both fixed scenarios and all prespecified models preserved']
    from generate_data import generate
    with tempfile.TemporaryDirectory() as tmp:
        regenerated=generate(Path(tmp)/'simulated.csv')
        require(regenerated.read_bytes()==(ex.ROOT/'data/simulated.csv').read_bytes(),'fixed-seed data regeneration')
        protected=Path(tmp)/'prior.json'; protected.write_text('old')
        try: ex.write_json(protected,{'bad':float('nan')})
        except ValueError: pass
        else:raise RuntimeError('NaN JSON accepted')
        require(protected.read_text()=='old','invalid JSON did not replace previous output')
    checks+=['archived CSV equals one fixed-seed regeneration', 'invalid JSON leaves previous output unchanged']
    return {'status':'passed','scope':'author numerical and focused interface checks; independent QA not implied',
            'checks':checks,'reference_errors':errors,'finite_differences':fd,'versions':run['versions']}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(ex.ROOT/'outputs/audit.json'))
    print(ex.write_json(p.parse_args().out,audit()))
