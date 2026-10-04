"""Reproducible independent algebra/numerical/guard checks; no assert removal under -O."""
from fractions import Fraction as Q
from decimal import Decimal as D, localcontext
from pathlib import Path
import copy, hashlib, json, os, subprocess, sys, tempfile
import numpy as np
import experiment as e

COUNT=0
GROUPS={}
def check(v,label):
    global COUNT
    COUNT+=1
    if not bool(v):raise RuntimeError('audit failed: '+label)
def close(a,b,label,atol=2e-11,rtol=2e-11):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    check(a.shape==b.shape and np.allclose(a,b,atol=atol,rtol=rtol),label)
def raises(fn,label):
    try:fn()
    except (ValueError,OSError):check(True,label);return
    raise RuntimeError('missing rejection: '+label)

def rational_rows(x,y,t,model):
    n=len(x);a,b=t; rows=[];g=[Q(0),Q(0)];H=[[Q(0),Q(0)],[Q(0),Q(0)]];F=Q(0)
    for xx,yy in zip(x,y):
        pred=a+(b if model=='linear' else b*b)*xx;r=pred-yy;j=[Q(1),xx if model=='linear' else 2*b*xx]
        rowH=[[j[u]*j[v]/n for v in range(2)]for u in range(2)]
        if model=='square':rowH[1][1]+=2*xx*r/n
        for u in range(2):
            g[u]+=r*j[u]/n
            for v in range(2):H[u][v]+=rowH[u][v]
        F+=r*r/(2*n);rows.append((pred,r,r*r/2,j,[r*z/n for z in j],rowH))
    return F,g,H,rows

def qsolve(H,g):
    det=H[0][0]*H[1][1]-H[0][1]*H[1][0]
    return [(-g[0]*H[1][1]+g[1]*H[0][1])/det,(-g[1]*H[0][0]+g[0]*H[1][0])/det]

def run_audit(cli=True):
    global COUNT
    COUNT=0;x,y,c=e.load_inputs();qx=[Q(v)for v in x];qy=[Q(v)for v in y];rng=np.random.default_rng(38170)
    before=COUNT;t=[Q(0),Q(3,2)];exact=[]
    for k in range(3):
        F,g,H,rows=rational_rows(qx,qy,t,'square');actual=e.RowObjective(x,y,'square').evaluate(np.array(t,dtype=float),True,True,True)
        close(actual['f'],float(F),'main Fraction loss');close(actual['g'],g,'main Fraction g');close(actual['H'],H,'main Fraction H')
        for rr,ar in zip(rows,actual['rows']):
            for val,key in zip(rr,('prediction','residual','half_square','jacobian','gradient_contribution','hessian_contribution')):close(ar[key],val,'main row '+key)
        exact.append({'theta':[str(z)for z in t],'F':str(F),'g':[str(z)for z in g],'H':[[str(z)for z in row]for row in H]})
        if k<2:
            p=qsolve(H,g);old=F;t=[t[i]+p[i]for i in range(2)]
            new=rational_rows(qx,qy,t,'square')[0];check(new<=old+Q(1,10000)*sum(g[i]*p[i]for i in range(2)),'exact full Armijo')
    check(exact[1]['theta']==['1','27/23'],'first exact theta');check(exact[2]['theta']==['1','19683/19067'],'second exact theta')
    check(exact[2]['F']=='141285388452139121/1057351664417112968','second exact F')
    for j in range(120):
        n=int(rng.integers(2,10));xx=[Q(int(z),8)for z in rng.integers(-16,17,size=n)];yy=[Q(int(z),8)for z in rng.integers(-16,17,size=n)];tt=[Q(int(z),8)for z in rng.integers(-12,13,size=2)]
        for model in ('linear','square'):
            F,g,H,rows=rational_rows(xx,yy,tt,model);out=e.RowObjective([float(z)for z in xx],[float(z)for z in yy],model).evaluate(np.array(tt,dtype=float),True,True,True)
            close(out['f'],F,'random Fraction F');close(out['g'],g,'random Fraction g');close(out['H'],H,'random Fraction H')
    GROUPS['fraction_checks']=COUNT-before
    before=COUNT
    # Independent 120-digit Newton trajectory using scalar row derivatives and 2x2 determinant solve.
    main=e.optimize(e.RowObjective(x,y,'square'),c['nonlinear_initial_parameters'],c,'newton')
    with localcontext() as ctx:
        ctx.prec=120;td=[D(0),D('1.5')];xd=[D(int(v))for v in x];yd=[D(str(v))for v in y]
        for k,state in enumerate(main['states']):
            close(state['theta'],[float(v)for v in td],'Decimal Newton parameter')
            F=D(0);g=[D(0),D(0)];h=[[D(0),D(0)],[D(0),D(0)]]
            for xx,yy in zip(xd,yd):
                r=td[0]+td[1]*td[1]*xx-yy;jb=2*td[1]*xx
                F+=r*r/8;g[0]+=r/4;g[1]+=r*jb/4;h[0][0]+=D(1)/4;h[0][1]+=jb/4;h[1][0]+=jb/4;h[1][1]+=(jb*jb+2*xx*r)/4
            close(state['f'],float(F),'Decimal Newton F')
            if k<len(main['trace']):
                rec=main['trace'][k];close(rec['old']['g'],[float(z)for z in g],'Decimal Newton g');close(rec['old']['H'],[[float(z)for z in row]for row in h],'Decimal Newton H')
                det=h[0][0]*h[1][1]-h[0][1]*h[1][0];p=[(-g[0]*h[1][1]+g[1]*h[0][1])/det,(-g[1]*h[0][0]+g[0]*h[1][0])/det]
                close(rec['direction'],[float(z)for z in p],'Decimal Newton p');td=[td[i]+D(str(rec['alpha']))*p[i]for i in range(2)]
    for _ in range(80):
        t=rng.uniform(-2,2,2);obj=e.RowObjective(x,y,'square');o=obj.evaluate(t,True,True)
        for j in range(2):
            h=1e-5;v=np.zeros(2);v[j]=h
            fd=(obj.evaluate(t+v,False)['f']-obj.evaluate(t-v,False)['f'])/(2*h)
            gd=(obj.evaluate(t+v)['g']-obj.evaluate(t-v)['g'])/(2*h)
            close(fd,o['g'][j],'central gradient',atol=5e-8,rtol=5e-8);close(gd,o['H'][:,j],'central Hessian',atol=5e-8,rtol=5e-8)
    GROUPS['decimal_difference_checks']=COUNT-before
    before=COUNT
    expected=np.array([[4417,-12],[-12,1057]],dtype=float)/4225
    got,info=e.inverse_update(np.eye(2),np.array([.5,2.]),np.array([.5,8.]),1e-12);close(got,expected,'exact BFGS C')
    for _ in range(100):
        d=int(rng.integers(2,9));R=rng.normal(size=(d,d));H=R.T@R+np.eye(d);pairs=[]
        for k in range(8):
            s=rng.normal(size=d);dy=H@s;pairs.append((s,dy));window=pairs[-int(rng.integers(1,6)):]
            gamma=float(window[-1][0]@window[-1][1])/float(window[-1][1]@window[-1][1]);C=gamma*np.eye(d)
            for ss,yy in window:
                # Independent expanded rank-two inverse BFGS expression.
                sy=ss@yy;Cy=C@yy
                C=C+(1+(yy@Cy)/sy)*np.outer(ss,ss)/sy-(np.outer(Cy,ss)+np.outer(ss,Cy))/sy
            g=rng.normal(size=d);p,scale=e.lbfgs_direction(g,window)
            close(p,-C@g,'two-loop versus independent same-window rebuild',atol=2e-10,rtol=2e-10)
            check(np.linalg.eigvalsh(C).min()>0,'rebuilt SPD');close(C@window[-1][1],window[-1][0],'latest secant',atol=2e-10,rtol=2e-10)
    GROUPS['bfgs_lbfgs_checks']=COUNT-before
    before=COUNT
    report=e.run_experiment(x,y,c)
    for name,path in {**report['paths'],**{'m'+k:v for k,v in report['lbfgs_windows'].items()}}.items():
        check(path['counts']['line_trials']==sum(len(z['line_trials'])for z in path['trace']),'count line trials')
        for rec in path['trace']:
            check(rec['gTp']<0,'every applied direction descends')
            if rec['accepted']:
                close(np.array(rec['old_theta'])+rec['alpha']*np.array(rec['direction']),rec['new_theta'],'simultaneous update')
                check(rec['line_trials'][-1]['accepted'],'accept last explicit trial')
                check(rec['line_trials'][-1]['f']<=rec['line_trials'][-1]['armijo_rhs'],'actual Armijo')
                if path['method']=='newton':close(np.array(rec['direction_state']['B'])@rec['direction'],-np.array(rec['old']['g']),'linear system')
    raw=report['raw_indefinite_demo'];close(raw['gTp'],Q(225,416),'indefinite raw positive slope');check(raw['new']['f']>raw['old']['f'],'raw full step rises')
    mod=report['paths']['indefinite_modified_newton']['trace'][0];close(mod['direction_state']['shift'],8,'shift eight');close(mod['alpha'],.5,'half-step accepted');close(mod['new_theta'],[1,Q(7,8)],'corrected theta')
    check(report['paths']['stationary_newton']['status']=='stationary_indefinite','stationary is not minimum')
    no_ls=dict(c,max_line_trials=1)
    failed=e.optimize(e.RowObjective(x,y,'linear'),[0,0],no_ls,'bfgs');check(failed['status']=='line_search_exhausted' and failed['theta']==[0.,0.],'failed search preserves parameters')
    no_shift=dict(c,positive_shift_candidates=[0.])
    failed=e.optimize(e.RowObjective(x,y,'square'),[1,.25],no_shift);check(failed['status']=='positive_shift_exhausted' and failed['theta']==[1.,.25],'shift failure preserves parameters')
    # Round sqrt(1/3) is not exactly singular; report its actual Hessian and finite result.
    near=e.optimize(e.RowObjective(x,y,'square'),[1,float(np.sqrt(1/3))],c);check(np.isfinite(near['theta']).all(),'near singular stored state')
    skip=e.inverse_update(np.eye(2),np.array([1.,0.]),np.array([-1.,0.]),1e-12);check(not skip[1]['updated'],'negative curvature skip')
    for method in e.METHODS:
        obj=e.RowObjective(x,y,'square');z=e.optimize(obj,c['nonlinear_initial_parameters'],c,method,trace=False)
        check(z['states'] is None and z['trace'] is None,'timing trace is off')
    # Independent wrappers count every actual callback, solve and Cholesky invocation.
    solve0=np.linalg.solve;chol0=np.linalg.cholesky
    for method in e.METHODS:
        for trace in (False,True):
            calls={'f':0,'g':0,'H':0,'solve':0,'cholesky':0}
            class Counted(e.RowObjective):
                def evaluate(self,theta,gradient=True,hessian=False,rows=False):
                    calls['f']+=1;calls['g']+=int(gradient or rows);calls['H']+=int(hessian or rows)
                    return super().evaluate(theta,gradient,hessian,rows)
            def counted_solve(*a,**kw):calls['solve']+=1;return solve0(*a,**kw)
            def counted_chol(*a,**kw):calls['cholesky']+=1;return chol0(*a,**kw)
            np.linalg.solve=counted_solve;np.linalg.cholesky=counted_chol
            try:z=e.optimize(Counted(x,y,'square'),[1,.25],c,method,trace=trace)
            finally:np.linalg.solve=solve0;np.linalg.cholesky=chol0
            for key,value in calls.items():check(z['counts'][key]==value,'actual wrapped count '+method+key)
    close(main['states'][0]['default_gap'],Q(29,8),'stable default gap')
    custom=e.RowObjective(x,y+.125,'square').evaluate([0,1.5],True,True,True)
    check(custom['default_gap'] is None,'custom data cannot inherit default gap')
    GROUPS['trace_and_failure_checks']=COUNT-before
    before=COUNT
    class Bomb:
        d=2;counts={'f':0,'g':0,'H':0}
        def evaluate(self,*args):raise RuntimeError('callback entered before validation')
    bad=[]
    for key,value in [('max_updates',True),('max_line_trials',0),('gradient_tolerance',float('nan')),('benchmark_seed',False),('benchmark_dimensions',[8,False]),('positive_shift_candidates',[0,1,float('inf')]),('lbfgs_history',1.2),('armijo_c1',1.),('kind','wrong'),('benchmark_repetitions',10)]:
        v=copy.deepcopy(c);v[key]=value;bad.append(v)
    v=copy.deepcopy(c);v['extra']=1;bad.append(v)
    for v in bad:raises(lambda:e.optimize(Bomb(),[0,1],v),'bad cfg before callback')
    for v in ([0,True],[0,'1'],[0,1j],[0,float('inf')],[0,1e-200],[0,101]):raises(lambda:e.optimize(Bomb(),v,c),'bad initial before callback')
    for C in ([[1,0],[0,False]],[[1,0],[0,float('nan')]],[[1,0],[0,-1]],[[1,1],[0,1]]):raises(lambda:e.optimize(Bomb(),[0,1],c,C0=C),'bad C tail')
    histories=[[(np.ones(2),np.ones(2)),([1,0],[0,False])], [([1,0],[-1,0])],[([1,0],[1,0,2])]]
    for h in histories:raises(lambda:e.optimize(Bomb(),[0,1],c,'lbfgs',history=h),'bad last history')
    # A bad final raw field must be refused before Cholesky, not just before callback.
    original=np.linalg.cholesky
    def bomb_la(*a,**kw):raise RuntimeError('linear algebra entered before raw validation')
    np.linalg.cholesky=bomb_la
    try:
        raises(lambda:e.optimize(Bomb(),[0,1],c,'bfgs',C0=[[1,0],[0,'1']]),'raw C precedes factorization')
        raises(lambda:e.optimize(Bomb(),[0,1],c,'lbfgs',history=[([1,0],[1,False])]),'raw history precedes factorization')
    finally:np.linalg.cholesky=original
    GROUPS['precallback_guards']=COUNT-before
    before=COUNT
    if cli:
        with tempfile.TemporaryDirectory(prefix='ml038-audit-') as td:
            td=Path(td);out=td/'out';out.mkdir();target=out/'report.json';target.write_bytes(b'old-good-output')
            config=td/'bad.json';data=td/'data.csv';data.write_bytes((e.ROOT/'data/regression.csv').read_bytes())
            invalid=[]
            for v in bad:invalid.append(json.dumps(v))
            invalid.extend(['{"kind":1,"kind":2}',(e.ROOT/'data/model_spec.json').read_text().replace('0.0001','1e-999'),(e.ROOT/'data/model_spec.json').read_text().replace('0.0001','NaN')])
            for text in invalid:
                config.write_text(text)
                for dest in (out,td/'must-not-exist'):
                    run=subprocess.run([sys.executable,str(e.ROOT/'experiment.py'),'--data',str(data),'--config',str(config),'--output-dir',str(dest)],capture_output=True)
                    check(run.returncode!=0,'invalid CLI fails');check(target.read_bytes()==b'old-good-output' and not (td/'must-not-exist').exists(),'invalid CLI leaves files')
            config.write_bytes((e.ROOT/'data/model_spec.json').read_bytes())
            for text in ['id,x,y\nS1,0,1\nS1,2,3\n','id,x,y\nS1,0,1\nS2,1,1e-999\n','id,x,y\nS1,0,1\nS2,1,nan\n','id,x,y\nS1,0,1\nS2,1,2,3\n']:
                data.write_text(text);run=subprocess.run([sys.executable,str(e.ROOT/'experiment.py'),'--data',str(data),'--config',str(config),'--output-dir',str(out)],capture_output=True)
                check(run.returncode!=0 and target.read_bytes()==b'old-good-output','bad CSV preserves old output')
            protected=td/'input';protected.write_bytes(b'input')
            alias=td/'alias';alias.mkdir();os.link(protected,alias/'report.json');raises(lambda:e.safe_write(alias,'report.json',b'bad',[protected]),'hardlink input protected');check(protected.read_bytes()==b'input','hardlink unchanged')
            link=td/'link';link.symlink_to(out,target_is_directory=True);raises(lambda:e.safe_write(link,'report.json',b'bad'),'linked directory rejected')
            ltarget=td/'link-target';ltarget.mkdir();(ltarget/'report.json').symlink_to(protected);raises(lambda:e.safe_write(ltarget,'report.json',b'bad'),'linked report rejected')
    GROUPS['file_guards']=COUNT-before
    return {'status':'passed','checks':COUNT,'groups':GROUPS,'exact_main_states':exact,
        'report_sha256':hashlib.sha256(e.serialized(report)).hexdigest(),
        'limitations':['finite declared tests, not all possible inputs','high-precision path comparison is default Newton only','independent QA still required for publication']}
if __name__=='__main__':print(json.dumps(run_audit(),ensure_ascii=False,indent=2))
