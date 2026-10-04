"""Optional real PyTorch 2.7.1 CPU float64 optimizer/state parity checks.

Run this file with the documented optional PyTorch environment. No download occurs.
"""
import copy, json, math
import numpy as np
import torch
import experiment as e

def main():
    if torch.__version__!='2.7.1+cpu':raise RuntimeError('This recorded parity contract targets torch 2.7.1+cpu exactly; use another version only as a new experiment.')
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    data,s=e.load_inputs();x=torch.tensor(data[:,0]);y=torch.tensor(data[:,1]);checks=0;max_error=0.;cases=[]
    def compare(got,want,label):
        nonlocal checks,max_error
        ga=np.asarray(got,dtype=float);wa=np.asarray(want,dtype=float);err=float(np.max(np.abs(ga-wa),initial=0));max_error=max(max_error,err);checks+=int(ga.size)
        if not np.allclose(ga,wa,rtol=2e-11,atol=2e-12):raise RuntimeError(label+' parity failure')
    for method in ['AdaGrad','RMSProp','Adam','AdamL2','AdamW']:
        for setting in range(3):
            sp=copy.deepcopy(s);sp['steps']=30;sp['resume_split']=7
            if setting==1:sp.update(learning_rate=.05,epsilon=1e-8,beta1=.9,beta2=.99,rmsprop_alpha=.9,decay_mask=[1,1])
            if setting==2:sp.update(learning_rate=.01,epsilon=.1,beta1=0.,beta2=0.,rmsprop_alpha=0.,decay_mask=[1,0])
            path=e.run_path(data,sp,method);initial=path['start_state']['theta'];params=[torch.tensor(z,requires_grad=True) for z in initial]
            groups=[{'params':[p],'weight_decay':sp['weight_decay']*sp['decay_mask'][j] if method in ('AdamL2','AdamW') else 0.} for j,p in enumerate(params)]
            kw={'lr':sp['learning_rate'],'eps':sp['epsilon'],'foreach':False}
            if method=='AdaGrad':optimizer=torch.optim.Adagrad(groups,lr_decay=0.,initial_accumulator_value=0.,fused=False,**kw)
            elif method=='RMSProp':optimizer=torch.optim.RMSprop(groups,alpha=sp['rmsprop_alpha'],momentum=0.,centered=False,**kw)
            else:
                cls=torch.optim.AdamW if method=='AdamW' else torch.optim.Adam
                optimizer=cls(groups,betas=(sp['beta1'],sp['beta2']),amsgrad=False,fused=False,**kw)
            for tr in path['trace']:
                optimizer.zero_grad(set_to_none=True);prediction=params[0]+params[1]*x;loss=((prediction-y)**2).mean()/2;loss.backward()
                compare([p.grad.item() for p in params],tr['data_gradient'],'raw row gradient');optimizer.step()
                compare([p.item() for p in params],tr['new_state']['theta'],'parameters')
                for j,p in enumerate(params):
                    state=optimizer.state[p];compare(state['step'].item(),tr['update'],'step')
                    keys=[('sum','s')] if method=='AdaGrad' else [('square_avg','v')] if method=='RMSProp' else [('exp_avg','m'),('exp_avg_sq','v')]
                    for tk,mk in keys:compare(state[tk].item(),tr['new_state'][mk][j],tk)
            cases.append({'method':method,'setting':setting,'steps':30,'mask':sp['decay_mask'],'final_parameters':[p.item() for p in params]})
    # Fresh AdamW: a real zero gradient updates the step and decay; None skips.
    zero_none={}
    for kind in ['zero','none']:
        p=torch.tensor(-.5,requires_grad=True);optimizer=torch.optim.AdamW([p],lr=.25,betas=(.5,.75),eps=1.,weight_decay=.1,foreach=False,fused=False)
        p.grad=torch.zeros_like(p) if kind=='zero' else None;optimizer.step();st=optimizer.state.get(p,{})
        zero_none[kind]={'parameter':p.item(),'step':st['step'].item() if 'step' in st else None,'state_created':bool(st)}
    compare(zero_none['zero']['parameter'],-.4875,'fresh zero decay')
    if zero_none['none']!={'parameter':-.5,'step':None,'state_created':False}:raise RuntimeError('None should skip')
    # Real zero CURRENT gradient after one nonzero history step still moves.
    p=torch.tensor(.5,requires_grad=True);optimizer=torch.optim.Adam([p],lr=.25,betas=(.5,.75),eps=1.,weight_decay=0.,foreach=False,fused=False)
    p.grad=torch.tensor(2.);optimizer.step();before=p.item();p.grad=torch.zeros_like(p);optimizer.step();after=p.item();st=optimizer.state[p]
    if before==after or st['step'].item()!=2:raise RuntimeError('history should still move under zero current gradient')
    history={'before_zero_gradient':before,'after_zero_gradient':after,'m':st['exp_avg'].item(),'v':st['exp_avg_sq'].item(),'step':st['step'].item()}
    return {'status':'passed','torch':torch.__version__,'numpy':np.__version__,'device':'CPU','dtype':'float64','threads':torch.get_num_threads(),'foreach':False,'fused':'False where supported; RMSProp has no fused argument','checks':checks,'maximum_absolute_difference':max_error,'tolerance':{'rtol':2e-11,'atol':2e-12},'cases':cases,'zero_vs_none':zero_none,'zero_current_gradient_with_history':history,'scope':'Finite deterministic cases; equivalent arithmetic order may differ by ulps. Not a universal optimizer guarantee.'}
if __name__=='__main__':print(json.dumps(main(),ensure_ascii=False,indent=2))
