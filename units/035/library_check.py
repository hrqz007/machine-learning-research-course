"""Actual CPU PyTorch SGD checks, including momentum-buffer conventions.

Run separately from experiment.py. Full input validation precedes torch import,
Tensor allocation, optimization or creation of an output directory.
"""
from pathlib import Path
import argparse, hashlib, json, math, platform
import experiment as ex
ROOT=Path(__file__).resolve().parent

def run_checks(data,spec):
    data=ex.validate_data(data);s=ex.validate_spec(spec)
    beta=s['hand_momentum'];eta=s['hand_rate'];T=s['library_steps']
    if beta==0:raise ValueError('library Nesterov audit requires hand_momentum > 0')
    if s['initial_velocity']!=[0.0,0.0]:raise ValueError('library audit specifies zero initial velocity')
    import torch
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    X=torch.tensor([r[:2] for r in data],dtype=torch.float64,device='cpu')
    y=torch.tensor([r[2] for r in data],dtype=torch.float64,device='cpu')
    rows=[];maximum=0.0
    def equal(a,b,why):
        nonlocal maximum
        err=max(abs(float(x)-float(z)) for x,z in zip(a,b));maximum=max(maximum,err)
        if err>5e-12*max(1.0,*map(abs,b)):raise RuntimeError(why+' mismatch: '+str(err))
        return err
    # Optimizer construction is intentionally explicit; no weight decay or mixed precision.
    def make(nag=False,damp=0):
        p=torch.tensor(s['initial'],dtype=torch.float64,requires_grad=True)
        opt=torch.optim.SGD([p],lr=eta,momentum=beta,dampening=damp,
                weight_decay=0,nesterov=nag,maximize=False,foreach=False,fused=False,differentiable=False)
        return p,opt
    def step(p,opt,rate):
        opt.param_groups[0]['lr']=rate;opt.zero_grad(set_to_none=True)
        pred=X@p;r=pred-y;loss=(r*r).sum()/(2*len(data));loss.backward()
        grad=p.grad.detach().tolist();old=p.detach().tolist();opt.step()
        return old,grad,p.detach().tolist(),opt.state[p]['momentum_buffer'].tolist()
    for method,nag in [('hb',False),('nag',True)]:
        manual=ex.run_method(data,s['initial'],[0,0],method,eta,beta,T)
        p,opt=make(nag);records=[]
        for t,ref in enumerate(manual['trace']):
            old,g,new,buf=step(p,opt,eta)
            equal(old,ref['evaluation']['theta'],'old parameter equals gradient point')
            equal(g,ref['evaluation']['gradient'],'autograd vs per-row gradient')
            d=[-eta*v for v in buf]
            theta=[new[j]+eta*beta*buf[j] for j in range(2)] if nag else new
            equal(d,ref['new_velocity'],'buffer/displacement');equal(theta,ref['new']['theta'],'reconstructed paper point')
            records.append({'t':t,'torch_old_parameter':old,'autograd_gradient':g,'torch_new_parameter':new,
                            'momentum_buffer':buf,'paper_theta_reconstructed':theta,'paper_velocity':d,
                            'raw_parameter_minus_paper':[new[j]-ref['new']['theta'][j] for j in range(2)],
                            'old_forward':ex._forward(data,old),'new_torch_forward':ex._forward(data,new),
                            'new_paper_forward':ex._forward(data,theta)})
        rows.append({'case':method+'_constant','updates':len(records),'records':records})
    # Dampening: first buffer is g0; only later gradients are multiplied by 1-tau.
    p,opt=make(False,s['dampening']);buf=None;manual=list(s['initial']);recs=[]
    wrong=[0.0,0.0]
    for t in range(T):
        g=ex._forward(data,manual)['gradient']
        buf=list(g) if buf is None else [beta*buf[j]+(1-s['dampening'])*g[j] for j in range(2)]
        manual=[manual[j]-eta*buf[j] for j in range(2)]
        old,tg,new,tb=step(p,opt,eta);equal(new,manual,'dampening params');equal(tb,buf,'dampening buffer')
        if t==0:
            wrong=[(1-s['dampening'])*v for v in g]
            wrong_new=[s['initial'][j]-eta*wrong[j] for j in range(2)]
        recs.append({'t':t,'buffer':tb,'theta':new,'old_forward':ex._forward(data,old),
                     'new_forward':ex._forward(data,new),'actual_gradient':tg})
    rows.append({'case':'dampening_first_buffer','records':recs,'incorrect_zero_buffer_dampened_first_theta':wrong_new})
    # LR scheduling exposes the placement of lr in the state definition.
    p,opt=make();buf=[0.0,0.0];manual=list(s['initial']);paper=list(s['initial']);d=[0.0,0.0];recs=[]
    for t,rate in enumerate(s['schedule_rates']):
        g=ex._forward(data,manual)['gradient'];buf=[beta*buf[j]+g[j] for j in range(2)]
        manual=[manual[j]-rate*buf[j] for j in range(2)]
        gp=ex._forward(data,paper)['gradient'];d=[beta*d[j]-rate*gp[j] for j in range(2)];paper=[paper[j]+d[j] for j in range(2)]
        old,tg,new,tb=step(p,opt,rate);equal(new,manual,'scheduled external-lr buffer')
        recs.append({'t':t,'rate':rate,'torch_theta':new,'fixed_beta_displacement_theta':list(paper),
                     'difference':[new[j]-paper[j] for j in range(2)],'buffer':tb,
                     'old_forward':ex._forward(data,old),'new_torch_forward':ex._forward(data,new),
                     'new_displacement_forward':ex._forward(data,paper),'actual_gradient':tg})
    rows.append({'case':'scheduled_learning_rate','records':recs})
    # Normalized EMA can match unnormalized HB ONLY after learning-rate conversion.
    raw=ex.run_method(data,s['initial'],[0,0],'hb',eta,beta,T)
    theta=list(s['initial']);m=[0.0,0.0];alpha=eta/(1-beta)
    for ref in raw['trace']:
        g=ex._forward(data,theta)['gradient'];m=[beta*m[j]+(1-beta)*g[j] for j in range(2)];theta=[theta[j]-alpha*m[j] for j in range(2)]
        equal(theta,ref['new']['theta'],'EMA rescaled learning rate')
    # A library rejects Nesterov+dampening; the test must actually attempt it.
    rejected=False
    try:make(True,s['dampening'] if s['dampening']>0 else .25)
    except ValueError:rejected=True
    if not rejected:raise RuntimeError('Nesterov+dampening unexpectedly accepted')
    return {'status':'passed','python':platform.python_version(),'torch':torch.__version__,'dtype':'float64',
            'device':'cpu','threads':torch.get_num_threads(),'foreach':False,'fused':False,'weight_decay':0,
            'max_absolute_comparison_error':maximum,'tolerance':'5e-12 * max(1,max(abs(reference)))',
            'cases':rows,'normalized_ema_rate':alpha,'nesterov_nonzero_dampening_rejected':rejected,
            'scope':'Constant lr, constant beta, zero initial buffer. Raw torch Nesterov parameter is paper lookahead, not paper theta.'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path);p.add_argument('--config',type=Path)
    p.add_argument('--output',type=Path,default=ROOT/'results/library.json');a=p.parse_args()
    try:
        data,s=ex.load_inputs(a.data,a.config);out=ex.protect_output(a.output,[ROOT/'library_check.py',*[v for v in [a.data,a.config] if v]])
        result=run_checks(data,s);payload=ex.report_bytes(result);ex.atomic_write(out,payload)
    except (ValueError,ArithmeticError,OSError,RuntimeError) as err:p.exit(2,'error: '+str(err)+'\n')
    print('PyTorch actual comparison passed; maximum error',result['max_absolute_comparison_error'])
    print('report SHA256',hashlib.sha256(payload).hexdigest())
if __name__=='__main__':main()
