"""Offline executable evidence for first-order AD. Python standard library only."""
import argparse, csv, hashlib, json, math, platform
from pathlib import Path
from autodiff import Value, Dual, finite_difference
ROOT=Path(__file__).resolve().parent

def composite(x,y): return (x*y+x.sin())*y.exp()
def composite_float(x,y): return (x*y+math.sin(x))*math.exp(y)
def vector_function(x,y): return [x*y,x*x+y]

def generate_data(directory):
    p=Path(directory); p.mkdir(parents=True,exist_ok=True)
    with (p/'gradient-cases.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['case','x','y']); w.writerows([['center',.7,-1.2],['positive',1.1,.3],['negative',-.9,.8],['zero_x',0.,-.5],['mixed',2.,-2.]])
    return {f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(p.glob('*.csv'))}

def run():
    x,y=Value(2.,label='x'),Value(3.,label='y'); t=x*y; t.label='t'; q=t*t; q.label='q'; z=q+t; z.label='z'; order=z.backward()
    shared=dict(x=x.data,y=y.data,t=t.data,q=q.data,z=z.data,dx=x.grad,dy=y.grad,dt=t.grad,topological_nodes=len(order),edges=sum(len(n.edges) for n in order),trace=[dict(label=n.label or n.op,value=n.data,gradient=n.grad) for n in order])
    shared['analytic']={'dx':(2*6+1)*3,'dy':(2*6+1)*2,'dt':13}
    checks=[]
    with (ROOT/'data/gradient-cases.csv').open(encoding='utf-8') as f:
        for r in csv.DictReader(f):
            a,b=float(r['x']),float(r['y']); vx,vy=Value(a),Value(b); out=composite(vx,vy); out.backward()
            reverse=[vx.grad,vy.grad]
            forward=[composite(Dual(a,1),Dual(b,0)).tangent,composite(Dual(a,0),Dual(b,1)).tangent]
            analytic=[(b+math.cos(a))*math.exp(b),(a*b+math.sin(a)+a)*math.exp(b)]
            finite=finite_difference(composite_float,[a,b],1e-5)
            checks.append(dict(case=r['case'],point=[a,b],value=out.data,reverse=reverse,forward=forward,analytic=analytic,finite_difference=finite,max_finite_abs_error=max(abs(u-v) for u,v in zip(reverse,finite)),max_analytic_abs_error=max(abs(u-v) for u,v in zip(reverse,analytic))))
    sweep=[]; base=checks[0]['reverse']
    for power in range(1,16):
        h=10.**(-power); fd=finite_difference(composite_float,[.7,-1.2],h)
        sweep.append(dict(h=h,max_abs_error=max(abs(a-b) for a,b in zip(base,fd))))
    x,y=Value(2),Value(3); outputs=vector_function(x,y); loss=2*outputs[0]-outputs[1]; loss.backward()
    jvp=[o.tangent for o in vector_function(Dual(2,1),Dual(3,-1))]
    kink=Value(0); k=kink.relu(); k.backward(); kink_fd=finite_difference(lambda a:max(0.,a),[0.])[0]
    a=Value(3); repeated=a*a+a; repeated.backward(); first=a.grad; repeated.backward(); second=a.grad; repeated.backward(seed=2); seeded=a.grad
    w1,w2,b=Value(.4),Value(-.8),Value(.1); pre=2*w1-w2+b; pred=pre.relu(); loss=.5*(pred-1.)**2; loss.backward()
    return dict(lesson='084',python=platform.python_version(),shared_node=shared,gradient_checks=checks,finite_difference_sweep=sweep,
        jacobian=dict(point=[2,3],matrix=[[3,2],[4,1]],input_direction=[1,-1],jvp=jvp,output_cotangent=[2,-1],vjp=[x.grad,y.grad]),
        relu_kink=dict(point=0,chosen_backward=kink.grad,symmetric_difference=kink_fd,warning='ReLU is not differentiable at 0; neither number proves a derivative.'),
        repeated_backward=dict(first=first,second=second,seed2=seeded,semantics='Each call resets all reachable gradients; not PyTorch accumulation semantics.'),
        neuron=dict(preactivation=pre.data,prediction=pred.data,loss=loss.data,gradients={'w1':w1.grad,'w2':w2.grad,'b':b.grad}),
        data_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'data').glob('*.csv'))})

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',default='outputs/result.json'); a=p.parse_args(); result=run()
    target=Path(a.out); target.parent.mkdir(parents=True,exist_ok=True); target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps(result,ensure_ascii=False,indent=2))
