"""ML088：CPU、单线程、固定数据的优化与精确恢复教学实验。
所有训练索引预生成；checkpoint在更新完成的边界保存，因而无需保存半个
micro-batch的梯度。测试使用真实磁盘序列化而非只复制模型对象。
"""
from pathlib import Path
import copy,json,time,platform
import numpy as np
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent

def setup():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

def data():
    """固定数据生成种子；同一函数多次调用返回相同划分，方便配对对照。"""
    g=torch.Generator().manual_seed(88)
    x=torch.randn(512,6,generator=g)
    y=(x[:,0]*x[:,1]+.7*x[:,2]-.4*x[:,3]**2>0).long()
    return x[:384],y[:384],x[384:],y[384:]

def model(seed=11):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(6,32),nn.Tanh(),nn.Linear(32,2))

def rate(t,total=120):
    # 更新编号0..119。前10次线性预热，随后余弦下降；最后仍有小正步长。
    if t<10:return .02*(t+1)/10
    return float(.002+.018*.5*(1+np.cos(np.pi*(t-10)/(total-10))))

def train(kind='adamw',seed=11,stop=120,state=None,weights_only=False):
    x,y,vx,vy=data();m=model(seed)
    opt=(torch.optim.AdamW(m.parameters(),lr=.02,weight_decay=.01) if kind!='sgd'
         else torch.optim.SGD(m.parameters(),lr=.08,momentum=.9))
    g=torch.Generator().manual_seed(seed+100)
    indices=torch.stack([torch.randperm(len(x),generator=g)[:64] for _ in range(120)])
    start=0;history=[]
    if state is not None:
        m.load_state_dict(state['model']);start=state['step'];history=state['history'].copy()
        if not weights_only:opt.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['rng'])
    tic=time.perf_counter()
    for t in range(start,stop):
        if kind=='warmup':opt.param_groups[0]['lr']=rate(t)
        ids=indices[t];opt.zero_grad(set_to_none=True)
        loss=nn.functional.cross_entropy(m(x[ids]),y[ids]);loss.backward()
        # clip返回裁剪前总范数；不是裁剪后的范数。
        norm=float(nn.utils.clip_grad_norm_(m.parameters(),1.0))
        opt.step()
        with torch.no_grad():
            history.append({'step':t+1,'loss':float(loss),'val_loss':float(nn.functional.cross_entropy(m(vx),vy)),
                            'lr':opt.param_groups[0]['lr'],'grad_norm_before_clip':norm})
    checkpoint={'model':copy.deepcopy(m.state_dict()),'optimizer':copy.deepcopy(opt.state_dict()),
                'step':stop,'rng':torch.get_rng_state(),'history':history}
    return m,checkpoint,time.perf_counter()-tic

def accumulation():
    x,y,_,_=data();a=model();b=copy.deepcopy(a);wrong=copy.deepcopy(a)
    # 不等长微批：[16,16,5]。每个mean loss按样本数/总数加权。
    nn.functional.cross_entropy(a(x[:37]),y[:37]).backward()
    for lo,hi in [(0,16),(16,32),(32,37)]:
        nn.functional.cross_entropy(b(x[lo:hi]),y[lo:hi]).mul((hi-lo)/37).backward()
        nn.functional.cross_entropy(wrong(x[lo:hi]),y[lo:hi]).backward()
    diff=lambda m:max(float((p.grad-q.grad).abs().max()) for p,q in zip(a.parameters(),m.parameters()))
    return {'correct_max_abs_error':diff(b),'unscaled_max_abs_error':diff(wrong)}

def main():
    """执行预先声明的完整协议，并将逐步轨迹与可视化保存为可审计文件。"""
    setup();out=ROOT/'outputs';fig=ROOT/'figures';out.mkdir(exist_ok=True);fig.mkdir(exist_ok=True)
    results={'environment':{'torch':torch.__version__,'python':platform.python_version(),'device':'cpu','threads':1},
             'protocol':{'train_n':384,'validation_n':128,'updates':120,'batch_size':64,'examples_seen':7680,'seeds':[11,23,37]},
             'runs':{},'accumulation':accumulation()}
    for kind in ['sgd','adamw','warmup']:
        results['runs'][kind]=[]
        for seed in [11,23,37]:
            m,c,seconds=train(kind,seed);results['runs'][kind].append({'seed':seed,'seconds':seconds,'history':c['history']})
    full,_,_=train('warmup');_,c,_=train('warmup',stop=57)
    torch.save(c,out/'checkpoint.pt')
    restored=torch.load(out/'checkpoint.pt',weights_only=True)
    resumed,_,_=train('warmup',state=restored);wrong,_,_=train('warmup',state=restored,weights_only=True)
    diff=lambda m:max(float((p.detach()-q.detach()).abs().max()) for p,q in zip(full.parameters(),m.parameters()))
    results['resume']={'checkpoint_step':57,'exact_max_abs_error':diff(resumed),'weights_only_max_abs_error':diff(wrong)}
    (out/'results.json').write_text(json.dumps(results,indent=2))
    f,ax=plt.subplots(1,2,figsize=(10,3.8))
    for kind,runs in results['runs'].items():
        a=np.array([[h['val_loss'] for h in r['history']] for r in runs]);mean=a.mean(0);sd=a.std(0,ddof=1)
        ax[0].plot(np.arange(1,121),mean,label=kind);ax[0].fill_between(np.arange(1,121),mean-sd,mean+sd,alpha=.15)
    ax[0].set(xlabel='Optimizer updates',ylabel='Validation cross entropy');ax[0].legend()
    h=results['runs']['warmup'][0]['history'];ax[1].plot([z['lr'] for z in h]);ax[1].set(xlabel='Update index',ylabel='Learning rate')
    f.tight_layout();f.savefig(fig/'01_budget_schedule.png',dpi=170);plt.close(f)
    f,ax=plt.subplots(figsize=(7,3.8));vals=list(results['accumulation'].values());ax.bar(['Weighted micro-batches','Missing scaling'],vals);ax.set(ylabel='Maximum gradient absolute error');f.tight_layout();f.savefig(fig/'02_accumulation.png',dpi=170);plt.close(f)
    f,ax=plt.subplots(figsize=(7,3.8));ax.bar(['Full checkpoint','Weights only'],[results['resume']['exact_max_abs_error'],results['resume']['weights_only_max_abs_error']]);ax.set(ylabel='Final parameter max absolute difference');f.tight_layout();f.savefig(fig/'03_resume.png',dpi=170);plt.close(f)
    print(json.dumps({'accumulation':results['accumulation'],'resume':results['resume'],'final_val_loss':{k:[r['history'][-1]['val_loss'] for r in v] for k,v in results['runs'].items()}},indent=2))
if __name__=='__main__':main()
