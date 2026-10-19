"""ML090：用最小样本、故障注入、独立验证和配对种子建立诊断证据。"""
from pathlib import Path
import json,time,platform
import numpy as np
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent

def data():
    """固定数据生成种子；同一函数多次调用返回相同划分，方便配对对照。"""
    g=torch.Generator().manual_seed(90);x=torch.randn(280,5,generator=g)
    # 数据机制固定；验证集没有参与调试配置的自动选择。
    score=x[:,0]*x[:,1]+.8*x[:,2]-.3*x[:,3]
    y=(score>0).long();return x[:24],y[:24],x[24:],y[24:],score[24:]

def model(seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(5,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,2))

def run(mode='clean',seed=11):
    """一次只注入一种故障，固定模型和预算；保留失败对照而不是筛选好结果。"""
    x,y,vx,vy,margin=data();m=model(seed);opt=torch.optim.Adam(m.parameters(),lr=.02)
    target=y.clone();g=torch.Generator().manual_seed(991)
    if mode=='shuffled_labels':target=target[torch.randperm(len(y),generator=g)]
    history=[];tic=time.perf_counter()
    for t in range(350):
        opt.zero_grad(set_to_none=True)
        # 错误注入：每一步只打乱X，不同步打乱y，映射随步数变化。
        xx=x[torch.randperm(len(x),generator=g)] if mode=='misaligned_order' else x
        logits=m(xx);loss=nn.functional.cross_entropy(logits,target);loss.backward()
        norm=float(torch.sqrt(sum((p.grad**2).sum() for p in m.parameters())))
        if mode!='no_step':opt.step()  # 故意遗漏更新的失败对照。
        if t%5==0 or t==349:
            with torch.no_grad():
                history.append({'step':t+1,'loss':float(nn.functional.cross_entropy(m(x),target)),
                    'train_acc':float((m(x).argmax(1)==target).float().mean()),'gradient_norm':norm})
    # 验证使用原始机制标签；随机化训练标签不应该顺便污染验证目标。
    with torch.no_grad():
        pred=m(vx).argmax(1);errors=(pred!=vy);group=(margin.abs()<.5)
        groups={name:{'n':int(mask.sum()),'errors':int(errors[mask].sum()),'error_rate':float(errors[mask].float().mean())}
                for name,mask in [('near_boundary',group),('far_boundary',~group)]}
        activation=m[1](m[0](x));saturation=float((activation.abs()>.99).float().mean())
    return {'mode':mode,'seed':seed,'history':history,'validation_acc':float((~errors).float().mean()),'groups':groups,
            'tanh_saturation_fraction':saturation,'seconds':time.perf_counter()-tic}

def integrity_probe():
    x,y,_,_,_=data();ids=torch.arange(len(x));perm=torch.randperm(len(x),generator=torch.Generator().manual_seed(3))
    # 数据ID能检测“值看起来都正常，但配对错了”的错误；不靠观察训练曲线猜。
    pairs_ok=bool(torch.equal(ids[perm],ids[perm]));pairs_bad=bool(torch.equal(ids[perm],ids))
    m=model(11);before=[p.detach().clone() for p in m.parameters()];loss=nn.functional.cross_entropy(m(x),y);loss.backward()
    grad=float(torch.sqrt(sum((p.grad**2).sum() for p in m.parameters())))
    unchanged=max(float((p.detach()-q).abs().max()) for p,q in zip(m.parameters(),before))
    return {'valid_input_finite':bool(torch.isfinite(x).all()),'labels_in_range':bool(((y>=0)&(y<2)).all()),
            'correct_pairing_check':pairs_ok,'wrong_pairing_check':pairs_bad,'gradient_without_step':grad,'parameter_change_without_step':unchanged}

def main():
    """执行预先声明的完整协议，并将逐步轨迹与可视化保存为可审计文件。"""
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    out=ROOT/'outputs';fig=ROOT/'figures';out.mkdir(exist_ok=True);fig.mkdir(exist_ok=True)
    r={'environment':{'torch':torch.__version__,'python':platform.python_version(),'device':'cpu','threads':1},
       'protocol':{'tiny_training_n':24,'validation_n':256,'steps':350,'seeds':[11,23,37],'mode':'full batch'},
       'integrity':integrity_probe(),'runs':{}}
    for mode in ['clean','no_step','misaligned_order','shuffled_labels']:
        r['runs'][mode]=[run(mode,s) for s in [11,23,37]]
    (out/'results.json').write_text(json.dumps(r,indent=2))
    f,ax=plt.subplots(1,2,figsize=(10,3.7))
    for mode,runs in r['runs'].items():
        h=runs[0]['history'];ax[0].plot([v['step'] for v in h],[v['loss'] for v in h],label=mode);ax[1].plot([v['step'] for v in h],[v['gradient_norm'] for v in h],label=mode)
    ax[0].set(xlabel='Update attempt',ylabel='Fixed-set training CE',yscale='log');ax[0].legend(fontsize=8);ax[1].set(xlabel='Update attempt',ylabel='Gradient L2 norm',yscale='log');f.tight_layout();f.savefig(fig/'01_fault_curves.png',dpi=170);plt.close(f)
    f,ax=plt.subplots(figsize=(8,3.8));names=list(r['runs'])
    for j,mode in enumerate(names):
        vals=[q['validation_acc'] for q in r['runs'][mode]];ax.scatter([j]*3,vals,s=45);ax.plot([j-.18,j+.18],[np.mean(vals)]*2,color='black')
    ax.set_xticks(range(4),names);ax.set(ylabel='Validation accuracy',ylim=(0,1),title='Every seed shown; black line = mean');f.tight_layout();f.savefig(fig/'02_all_seeds.png',dpi=170);plt.close(f)
    f,ax=plt.subplots(figsize=(7,3.7));q=r['runs']['clean'][0]['groups'];ax.bar(list(q),[z['error_rate'] for z in q.values()]);ax.set(ylabel='Error rate',title='Seed 11: error groups, not a causal explanation')
    for i,z in enumerate(q.values()):ax.text(i,z['error_rate']+.015,f"{z['errors']}/{z['n']}",ha='center')
    ax.set_ylim(0,1);f.tight_layout();f.savefig(fig/'03_error_groups.png',dpi=170);plt.close(f)
    print(json.dumps({'integrity':r['integrity'],'summary':{k:{'train_loss':[q['history'][-1]['loss'] for q in v],'val_acc':[q['validation_acc'] for q in v]} for k,v in r['runs'].items()}},indent=2))
if __name__=='__main__':main()
