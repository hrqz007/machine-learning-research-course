"""Real CPU PyTorch: NumPy parity, gradient accumulation, training + validation."""
from pathlib import Path
import argparse, copy, json, platform
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from numpy_core import initialize, loss_and_grad, sgd_step, load_data

DTYPE=torch.float64
DEVICE=torch.device('cpu')

class ArrayDataset(Dataset):
    def __init__(self, X, y):
        self.X=torch.as_tensor(np.asarray(X),dtype=DTYPE,device=DEVICE)
        self.y=torch.as_tensor(np.asarray(y),dtype=torch.long,device=DEVICE)
        if self.X.ndim!=2 or self.X.shape[1]!=2 or self.y.shape!=(len(self.X),):
            raise ValueError('expected X:(N,2), y:(N,)')
    def __len__(self): return len(self.y)
    def __getitem__(self,index): return self.X[index],self.y[index]

class TwoLayer(nn.Module):
    def __init__(self,hidden=8):
        super().__init__()
        self.first=nn.Linear(2,hidden,dtype=DTYPE,device=DEVICE)
        self.second=nn.Linear(hidden,2,dtype=DTYPE,device=DEVICE)
    def forward(self,X):
        return self.second(torch.tanh(self.first(X)))

def set_parameters(model,p):
    with torch.no_grad():
        model.first.weight.copy_(torch.from_numpy(p['W1'].T))
        model.first.bias.copy_(torch.from_numpy(p['b1']))
        model.second.weight.copy_(torch.from_numpy(p['W2'].T))
        model.second.bias.copy_(torch.from_numpy(p['b2']))

def parameters_numpy(model,grad=False):
    def get(t):return (t.grad if grad else t).detach().cpu().numpy().copy()
    return dict(W1=get(model.first.weight).T,b1=get(model.first.bias),
                W2=get(model.second.weight).T,b2=get(model.second.bias))

def parity_audit():
    X=np.array([[.3,-.7],[1.2,.4],[-.5,.9]]);y=np.array([0,1,0])
    p=initialize(8502,3);nl,ng,nc=loss_and_grad(X,y,p)
    model=TwoLayer(3);set_parameters(model,p)
    optimizer=torch.optim.SGD(model.parameters(),lr=.07)
    optimizer.zero_grad(set_to_none=True)
    tx=torch.tensor(X,dtype=DTYPE);ty=torch.tensor(y,dtype=torch.long)
    logits=model(tx);loss=nn.functional.cross_entropy(logits,ty)
    loss.backward();tg=parameters_numpy(model,grad=True)
    np_step=sgd_step(p,ng,.07);optimizer.step();tp=parameters_numpy(model)
    return dict(dtype=str(DTYPE),device=str(DEVICE),torch_version=torch.__version__,
                loss_numpy=nl,loss_torch=loss.item(),loss_abs_error=abs(nl-loss.item()),
                logits_max_abs=float(np.max(np.abs(logits.detach().numpy()-nc['S']))),
                gradient_max_abs={k:float(np.max(np.abs(ng[k]-tg[k]))) for k in p},
                step_max_abs={k:float(np.max(np.abs(np_step[k]-tp[k]))) for k in p},
                checked_scalar_parameters=sum(v.size for v in p.values()))

def accumulation_audit():
    # New forward graph each time; parameters stay fixed so the second derivative equals the first.
    w=torch.tensor(2.,dtype=DTYPE,requires_grad=True)
    (w*3-1).square().backward();first=w.grad.item()
    (w*3-1).square().backward();without_zero=w.grad.item()
    w.grad=None
    (w*3-1).square().backward();with_zero=w.grad.item()
    return dict(first=first,second_without_zero=without_zero,after_reset=with_zero,
                interpretation='30 + 30 = 60 at unchanged w; intentional accumulation requires correct loss scaling')

def state_audit():
    torch.manual_seed(8603);drop=nn.Dropout(.5);x=torch.ones(1000,dtype=DTYPE)
    drop.train();a=drop(x);drop.eval();b=drop(x)
    model=TwoLayer(3);model.eval()
    tracks=model(torch.ones(2,2,dtype=DTYPE)).requires_grad
    with torch.no_grad(): no_tracks=model(torch.ones(2,2,dtype=DTYPE)).requires_grad
    return dict(dropout_train_zero_fraction=float((a==0).double().mean()),
                dropout_eval_all_ones=bool(torch.equal(b,x)),eval_alone_requires_grad=tracks,
                eval_with_no_grad_requires_grad=no_tracks)

def make_loaders(data,batch_size=32):
    gen=torch.Generator().manual_seed(8602)
    return {k:DataLoader(ArrayDataset(*v),batch_size=batch_size,shuffle=(k=='train'),
                          generator=gen if k=='train' else None,num_workers=0,drop_last=False)
            for k,v in data.items()}

def train_epoch(model,loader,optimizer):
    model.train();loss_sum=0.;correct=0;count=0
    for X,y in loader:
        X=X.to(device=DEVICE,dtype=DTYPE);y=y.to(device=DEVICE,dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        logits=model(X);loss=nn.functional.cross_entropy(logits,y,reduction='mean')
        loss.backward();optimizer.step()
        n=y.numel();loss_sum+=loss.item()*n;correct+=(logits.argmax(1)==y).sum().item();count+=n
    if count==0:raise ValueError('empty training loader')
    return dict(loss=loss_sum/count,accuracy=correct/count,examples=count)

@torch.no_grad()
def evaluate(model,loader):
    model.eval();loss_sum=0.;correct=0;count=0
    for X,y in loader:
        X=X.to(device=DEVICE,dtype=DTYPE);y=y.to(device=DEVICE,dtype=torch.long)
        logits=model(X)
        loss_sum+=nn.functional.cross_entropy(logits,y,reduction='sum').item()
        correct+=(logits.argmax(1)==y).sum().item();count+=y.numel()
    if count==0:raise ValueError('empty evaluation loader')
    return dict(loss=loss_sum/count,accuracy=correct/count,examples=count)

def run():
    torch.set_num_threads(1);torch.manual_seed(8601)
    data=load_data();loaders=make_loaders(data)
    # Monitoring must not consume the training shuffle generator.
    train_eval=DataLoader(ArrayDataset(*data['train']),batch_size=32,shuffle=False,num_workers=0)
    model=TwoLayer(8);set_parameters(model,initialize())
    optimizer=torch.optim.SGD(model.parameters(),lr=.15)
    initial=evaluate(model,train_eval);history=[];best_loss=float('inf');best_state=None;best_epoch=0
    for epoch in range(1,181):
        online=train_epoch(model,loaders['train'],optimizer)
        validation=evaluate(model,loaders['validation'])
        if validation['loss']<best_loss:
            best_loss=validation['loss'];best_epoch=epoch;best_state=copy.deepcopy(model.state_dict())
        if epoch==1 or epoch%10==0:
            history.append(dict(epoch=epoch,train_online=online,train_frozen=evaluate(model,train_eval),validation=validation))
    model.load_state_dict(best_state)
    final=dict(train=evaluate(model,train_eval),validation=evaluate(model,loaders['validation']),test=evaluate(model,loaders['test']))
    # Independently check sample-weighted aggregation against a single frozen full validation batch.
    Xv,yv=data['validation'];np_loss=loss_and_grad(Xv,yv,parameters_numpy(model))[0]
    return dict(unit='086',environment=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,device='cpu',dtype='float64',threads=1),
                parity=parity_audit(),accumulation=accumulation_audit(),state=state_audit(),
                training=dict(seed=8601,shuffle_seed=8602,epochs=180,batch_size=32,learning_rate=.15,
                              last_train_batch_size=323%32,last_validation_batch_size=157%32,
                              initial_train=initial,best_epoch=best_epoch,final=final,history=history,
                              weighted_validation_vs_full_numpy_abs=abs(final['validation']['loss']-np_loss)),
                final_parameters={k:v.tolist() for k,v in parameters_numpy(model).items()},
                interpretation='Actual CPU PyTorch run; fixed synthetic example. Validation selects checkpoint; test evaluated once after selection.')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='outputs/result.json');args=ap.parse_args()
    result=run();out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('environment','parity','accumulation','state')},indent=2))
    print(json.dumps({k:v for k,v in result['training'].items() if k!='history'},indent=2))
