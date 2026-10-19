"""ML089：归一化轴、训练/推理状态、dropout期望和小样本受控比较。"""
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
    g=torch.Generator().manual_seed(89);x=torch.randn(320,8,generator=g)
    y=(x[:,0]+x[:,1]*x[:,2]-.6*x[:,3]>0).long()
    # 刻意少量训练数据；只用训练部分的均值标准差，避免验证泄漏。
    mean=x[:64].mean(0);std=x[:64].std(0,unbiased=False).clamp_min(1e-6)
    x=(x-mean)/std
    return x[:64],y[:64],x[64:],y[64:]

def model(norm='none',drop=0.,seed=11):
    torch.manual_seed(seed)
    layers=[nn.Linear(8,32)]
    if norm=='bn':layers.append(nn.BatchNorm1d(32))
    if norm=='ln':layers.append(nn.LayerNorm(32))
    layers += [nn.ReLU(),nn.Dropout(drop),nn.Linear(32,2)]
    return nn.Sequential(*layers)

def run(norm,drop,seed,batch=16):
    """训练60轮；记录时统一eval，避免dropout掩码把两种loss口径混在一起。"""
    x,y,vx,vy=data();m=model(norm,drop,seed);opt=torch.optim.AdamW(m.parameters(),lr=.01,weight_decay=.01)
    g=torch.Generator().manual_seed(seed+1000);history=[];tic=time.perf_counter()
    for epoch in range(60):
        m.train();ids=torch.randperm(len(x),generator=g)
        for idx in ids.split(batch):
            opt.zero_grad(set_to_none=True);loss=nn.functional.cross_entropy(m(x[idx]),y[idx]);loss.backward();opt.step()
        # eval切换BN/Dropout行为；no_grad另行关闭梯度记录，二者职责不同。
        m.eval()
        with torch.no_grad():
            history.append({'epoch':epoch+1,'train_loss':float(nn.functional.cross_entropy(m(x),y)),
             'val_loss':float(nn.functional.cross_entropy(m(vx),vy)),'val_acc':float((m(vx).argmax(1)==vy).float().mean())})
    # 以验证loss选择epoch是演示，不把它当独立测试分数。
    best=min(history,key=lambda h:h['val_loss'])
    return {'seed':seed,'history':history,'best_validation_epoch':best['epoch'],'seconds':time.perf_counter()-tic}

def axis_probe():
    anchor=torch.tensor([[1.,2.,4.,8.]])
    a=torch.cat([anchor,torch.zeros(7,4)]);b=torch.cat([anchor,torch.ones(7,4)*20])
    bn=nn.BatchNorm1d(4,affine=False,track_running_stats=False);ln=nn.LayerNorm(4,elementwise_affine=False)
    train_bn=float((bn(a)[0]-bn(b)[0]).abs().max());train_ln=float((ln(a)[0]-ln(b)[0]).abs().max())
    frozen=nn.BatchNorm1d(4,affine=False);frozen.train();frozen(a);frozen.eval()
    eval_bn=float((frozen(a)[0]-frozen(b)[0]).abs().max())
    # 同一固定评估统计下改变数据分布，LN对统一正缩放+平移近似不变，BN不是。
    z=torch.tensor([[1.,2.,4.,8.],[2.,3.,5.,9.]])
    return {'bn_train_companion_difference':train_bn,'ln_companion_difference':train_ln,'bn_eval_companion_difference':eval_bn,
            'bn_eval_shift_difference':float((frozen(z)-frozen(z+10)).abs().mean()),
            'ln_shift_difference':float((ln(z)-ln(z+10)).abs().max())}

def dropout_probe():
    torch.manual_seed(89);x=torch.full((200000,),3.);d=nn.Dropout(.25);d.train();y=d(x)
    d.eval();z=d(x)
    return {'p':.25,'input':3.,'train_mean':float(y.mean()),'train_var':float(y.var(unbiased=False)),
            'expected_mean':3.,'expected_variance':3.,'eval_max_error':float((z-x).abs().max())}

def main():
    """执行预先声明的完整协议，并将逐步轨迹与可视化保存为可审计文件。"""
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    out=ROOT/'outputs';fig=ROOT/'figures';out.mkdir(exist_ok=True);fig.mkdir(exist_ok=True)
    r={'environment':{'torch':torch.__version__,'python':platform.python_version(),'device':'cpu','threads':1},
       'protocol':{'training_n':64,'validation_n':256,'epochs':60,'main_batch':16,'main_updates':240,'seeds':[11,23,37]},
       'axis_probe':axis_probe(),'dropout_probe':dropout_probe(),'runs':{},'batch_probe':{}}
    # 一个2x2因子设计：比较LN有无、dropout有无，并保留交互可能。
    for norm in ['none','ln']:
        for drop in [0.,.35]:r['runs'][f'{norm}_drop{drop}']=[run(norm,drop,s) for s in [11,23,37]]
    # 单独检查BN的batch敏感性：匹配epoch/曝光，更新次数不同，明确不作纯因果比较。
    for batch in [4,16,64]:r['batch_probe'][str(batch)]=run('bn',0.,11,batch)
    (out/'results.json').write_text(json.dumps(r,indent=2))
    f,ax=plt.subplots(1,2,figsize=(10,3.7))
    for key,runs in r['runs'].items():
        for a,metric in zip(ax,['train_loss','val_loss']):
            z=np.array([[h[metric] for h in q['history']] for q in runs]);a.plot(np.arange(1,61),z.mean(0),label=key)
    ax[0].set(xlabel='Epoch',ylabel='Training CE in eval mode');ax[1].set(xlabel='Epoch',ylabel='Validation CE');ax[1].legend(fontsize=8)
    f.tight_layout();f.savefig(fig/'01_factorial.png',dpi=170);plt.close(f)
    f,ax=plt.subplots(figsize=(7,3.6));a=r['axis_probe'];ax.bar(['BN train','LN','BN eval'],[a['bn_train_companion_difference'],a['ln_companion_difference'],a['bn_eval_companion_difference']]);ax.set(ylabel='Anchor output max change',title='Change companion samples, keep anchor fixed');f.tight_layout();f.savefig(fig/'02_normalization_axes.png',dpi=170);plt.close(f)
    f,ax=plt.subplots(1,2,figsize=(10,3.7));d=r['dropout_probe'];ax[0].bar(['Expected mean','Measured mean','Expected variance','Measured variance'],[3,d['train_mean'],3,d['train_var']]);ax[0].tick_params(axis='x',rotation=25);ax[0].set(title='Inverted dropout: x=3, p=0.25')
    for key,q in r['batch_probe'].items():ax[1].plot([h['val_loss'] for h in q['history']],label='batch='+key)
    ax[1].set(xlabel='Epoch index',ylabel='Validation CE',title='BN: exposure matched, updates differ');ax[1].legend();f.tight_layout();f.savefig(fig/'03_dropout_batch.png',dpi=170);plt.close(f)
    print(json.dumps({'axis_probe':r['axis_probe'],'dropout_probe':r['dropout_probe'],'validation_last':{k:[q['history'][-1]['val_loss'] for q in v] for k,v in r['runs'].items()}},indent=2))
if __name__=='__main__':main()
