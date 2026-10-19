"""Small CPU CNN comparison; synthetic data, no downloads or remote services."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/tutorial-mpl')
import json,time,platform
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent

def data(n,seed):
    """Horizontal vs vertical short bars at random interior locations + noise."""
    r=np.random.default_rng(seed); y=np.arange(n)%2;r.shuffle(y)
    x=r.normal(0,.15,(n,1,12,12)).astype('float32')
    for i,label in enumerate(y):
        a,b=r.integers(3,9,size=2)
        if label==0:x[i,0,a,b-2:b+3]+=1
        else:x[i,0,a-2:a+3,b]+=1
    return torch.from_numpy(x),torch.from_numpy(y.astype('int64'))

class SmallCNN(nn.Module):
    def __init__(self,residual=False):
        super().__init__();self.residual=residual
        self.stem=nn.Conv2d(1,8,3,padding=1)
        self.conv1=nn.Conv2d(8,8,3,padding=1);self.conv2=nn.Conv2d(8,8,3,padding=1)
        self.head=nn.Linear(8,2)
    def forward(self,x):
        h=F.relu(self.stem(x)); z=self.conv2(F.relu(self.conv1(h)))
        h=F.relu(z+h if self.residual else z)
        return self.head(h.mean(dim=(2,3)))

def count_macs():
    # Each convolution output has Cin * Kh * Kw MACs; FC has 8*2.
    return 12*12*(8*1*9+8*8*9+8*8*9)+8*2

def run(steps=120):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    (ROOT/'outputs').mkdir(exist_ok=True);(ROOT/'figures').mkdir(exist_ok=True)
    train_x,train_y=data(384,920); test_x,test_y=data(256,921)
    records=[];histories={};last=None
    for seed in [0,1,2]:
        # Exactly the same initial tensors and batch ordering in each pair.
        torch.manual_seed(seed); template=SmallCNN().state_dict()
        gen=torch.Generator().manual_seed(1000+seed)
        batches=torch.randint(len(train_x),(steps,32),generator=gen)
        for residual in [False,True]:
            model=SmallCNN(residual);model.load_state_dict(template)
            opt=torch.optim.Adam(model.parameters(),lr=.01)
            history=[];start=time.perf_counter();model.train()
            for ids in batches:
                opt.zero_grad(set_to_none=True)
                loss=F.cross_entropy(model(train_x[ids]),train_y[ids]);loss.backward();opt.step();history.append(float(loss.detach()))
            train_seconds=time.perf_counter()-start
            model.eval()
            with torch.inference_mode():
                logits=model(test_x);acc=float((logits.argmax(1)==test_y).float().mean())
                shifted=torch.zeros_like(test_x);shifted[:,:,:,1:]=test_x[:,:,:,:-1]
                slogits=model(shifted)
                shift_acc=float((slogits.argmax(1)==test_y).float().mean())
                prob_delta=float((logits.softmax(1)-slogits.softmax(1)).abs().mean())
                sample=test_x[:1]
                for _ in range(20):model(sample)
                timings=[]
                for _ in range(100):
                    t=time.perf_counter();model(sample);timings.append((time.perf_counter()-t)*1e3)
            name='residual' if residual else 'plain'
            records.append({'model':name,'seed':seed,'steps':steps,'batch':32,'examples_seen':steps*32,'parameters':sum(p.numel() for p in model.parameters()),'macs_per_example':count_macs(),'approx_conv_linear_flops':2*count_macs(),'residual_additions':8*12*12 if residual else 0,'train_seconds':train_seconds,'latency_batch1_ms_median':float(np.median(timings)),'latency_batch1_ms_p90':float(np.percentile(timings,90)),'test_accuracy':acc,'shifted_accuracy':shift_acc,'mean_probability_shift':prob_delta,'last_batch_loss':history[-1]})
            histories[f'{name}_{seed}']=history
            last=model
    fig,axs=plt.subplots(2,4,figsize=(9,4))
    for i,ax in enumerate(axs.ravel()):ax.imshow(train_x[i,0],cmap='gray');ax.set_title('horizontal' if train_y[i]==0 else 'vertical');ax.axis('off')
    fig.tight_layout();fig.savefig(ROOT/'figures/01_synthetic_shapes.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(9,3))
    for name,col in [('plain','tab:blue'),('residual','tab:orange')]:
        hs=np.array([histories[f'{name}_{s}'] for s in [0,1,2]])
        axs[0].plot(hs.mean(0),label=name,color=col);axs[0].fill_between(np.arange(steps),hs.min(0),hs.max(0),alpha=.15,color=col)
        rows=[r for r in records if r['model']==name]
        axs[1].scatter([r['latency_batch1_ms_median'] for r in rows],[r['test_accuracy'] for r in rows],label=name,color=col)
    axs[0].set(xlabel='Optimizer step',ylabel='Batch cross entropy');axs[0].legend();axs[1].set(xlabel='CPU batch-1 latency (ms)',ylabel='Test accuracy');axs[1].legend()
    fig.tight_layout();fig.savefig(ROOT/'figures/02_training_budget.png',dpi=160);plt.close(fig)
    # Counterexample: stride-2 max pooling need not be invariant to a 1px shift.
    v=torch.tensor([[[[0.,1.,0.,2.,0.,0.]]]])
    p=F.max_pool2d(v,(1,2));ps=F.max_pool2d(torch.roll(v,1,3),(1,2))
    with torch.inference_mode():features=F.relu(last.stem(test_x[:1]))[0]
    fig,axs=plt.subplots(1,4,figsize=(10,3))
    for i in range(3):axs[i].imshow(features[i],cmap='viridis');axs[i].set_title(f'Stem channel {i}');axs[i].axis('off')
    axs[3].plot(p.flatten(),'-o',label='original');axs[3].plot(ps.flatten(),'-s',label='shift 1px');axs[3].set_title('Stride-2 max pooling');axs[3].legend()
    fig.tight_layout();fig.savefig(ROOT/'figures/03_features_pooling.png',dpi=160);plt.close(fig)
    results={'device':'cpu','threads':1,'torch':torch.__version__,'python':platform.python_version(),'data':{'kind':'original synthetic bars','train_seed':920,'test_seed':921,'train_n':384,'test_n':256,'image_shape':[1,12,12]},'budget':'same optimizer updates, batch size, examples, initialization and conv-linear MACs; residual addition is extra','records':records,'pooling_original':p.flatten().tolist(),'pooling_shifted':ps.flatten().tolist(),'limitations':['synthetic single-task evidence only','test used once for fixed predeclared protocol; no hyperparameter selection','latency depends on this CPU and process load','FLOPs omit activations, pooling, bias, loss and training backward']}
    (ROOT/'outputs/results.json').write_text(json.dumps(results,indent=2));(ROOT/'outputs/history.json').write_text(json.dumps(histories))
    print(json.dumps(results,indent=2));return results
if __name__=='__main__':run()
