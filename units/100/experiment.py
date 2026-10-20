"""Pre-LN Transformer block, reference alignment and order-sensitive CPU task."""
import os,tempfile
_CACHE=tempfile.mkdtemp(prefix='course-cache-')
os.environ.setdefault('MPLCONFIGDIR',_CACHE);os.environ.setdefault('XDG_CACHE_HOME',_CACHE)
from pathlib import Path
import math,json,csv
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
torch.set_num_threads(1)

class Block(nn.Module):
 def __init__(self,d=8,heads=2,ff=16):
  super().__init__();self.d=d;self.heads=heads;self.qkv=nn.Linear(d,3*d);self.proj=nn.Linear(d,d);self.norm1=nn.LayerNorm(d);self.norm2=nn.LayerNorm(d);self.fc1=nn.Linear(d,ff);self.fc2=nn.Linear(ff,d)
 def forward(self,x,allowed=None):
  b,l,d=x.shape;z=self.norm1(x);q,k,v=self.qkv(z).chunk(3,-1)
  q,k,v=[t.reshape(b,l,self.heads,d//self.heads).transpose(1,2) for t in (q,k,v)]
  s=q@k.transpose(-2,-1)/math.sqrt(d//self.heads)
  if allowed is not None:
   if not allowed.any(-1).all():raise ValueError('all-blocked query')
   s=s.masked_fill(~allowed,float('-inf'))
  a=s.softmax(-1);y=(a@v).transpose(1,2).reshape(b,l,d);x=x+self.proj(y)
  return x+self.fc2(F.gelu(self.fc1(self.norm2(x))))

def sinusoidal(length,d):
 if d%2:raise ValueError('even width required')
 position=torch.arange(length).float()[:,None];freq=torch.exp(torch.arange(0,d,2).float()*(-math.log(10000.)/d));p=torch.zeros(length,d);p[:,0::2]=(position*freq).sin();p[:,1::2]=(position*freq).cos();return p

def reference(block):
 ref=nn.TransformerEncoderLayer(block.d,block.heads,block.fc1.out_features,dropout=0.,activation='gelu',batch_first=True,norm_first=True).to(dtype=block.qkv.weight.dtype)
 with torch.no_grad():
  pairs=[(ref.self_attn.in_proj_weight,block.qkv.weight),(ref.self_attn.in_proj_bias,block.qkv.bias),(ref.self_attn.out_proj.weight,block.proj.weight),(ref.self_attn.out_proj.bias,block.proj.bias)]
  for a,b in pairs:a.copy_(b)
  for a,b in [(ref.linear1,block.fc1),(ref.linear2,block.fc2),(ref.norm1,block.norm1),(ref.norm2,block.norm2)]:a.load_state_dict(b.state_dict())
 return ref

def checks():
 torch.manual_seed(100);m=Block().double();ref=reference(m);x=torch.randn(2,4,8,dtype=torch.double,requires_grad=True);xx=x.detach().clone().requires_grad_();allowed=torch.ones(4,4,dtype=torch.bool).tril()
 y=m(x,allowed);r=ref(xx,src_mask=~allowed);up=torch.randn_like(y);(y*up).sum().backward();(r*up).sum().backward()
 permutation=torch.tensor([2,0,3,1]);base=m(x.detach());permuted=m(x.detach()[:,permutation]);p=sinusoidal(4,8).double()
 positional_change=float((m(x.detach()+p).mean(1)-m(x.detach()[:,permutation]+p).mean(1)).abs().max().detach())
 changed=x.detach().clone();changed[:,3]+=torch.arange(8,dtype=torch.double)*5
 future=float((m(changed,allowed)[:,:3]-y.detach()[:,:3]).abs().max().detach())
 param_pairs=[(m.qkv.weight,ref.self_attn.in_proj_weight),(m.qkv.bias,ref.self_attn.in_proj_bias),(m.proj.weight,ref.self_attn.out_proj.weight),(m.proj.bias,ref.self_attn.out_proj.bias),(m.fc1.weight,ref.linear1.weight),(m.fc1.bias,ref.linear1.bias),(m.fc2.weight,ref.linear2.weight),(m.fc2.bias,ref.linear2.bias),(m.norm1.weight,ref.norm1.weight),(m.norm1.bias,ref.norm1.bias),(m.norm2.weight,ref.norm2.weight),(m.norm2.bias,ref.norm2.bias)]
 return {'reference_output_error':float((y-r).abs().max().detach()),'reference_input_gradient_error':float((x.grad-xx.grad).abs().max()),'reference_parameter_gradient_error':max(float((a.grad-b.grad).abs().max()) for a,b in param_pairs),'permutation_equivariance_error':float((permuted-base[:,permutation]).abs().max().detach()),'no_position_pool_change':float((permuted.mean(1)-base.mean(1)).abs().max().detach()),'with_position_pool_change':positional_change,'causal_future_error':future,'parameters_measured':sum(t.numel() for t in m.parameters()),'parameters_formula':4*8*8+2*8*16+9*8+16}

class Classifier(nn.Module):
 def __init__(self,position):
  super().__init__();self.emb=nn.Embedding(2,8);self.block=Block();self.head=nn.Linear(8,2);self.position=position;self.register_buffer('pe',sinusoidal(2,8))
 def forward(self,ids):
  x=self.emb(ids)
  if self.position:x=x+self.pe
  return self.head(self.block(x).mean(1))

def main():
 for d in ('outputs','data','figures'):(ROOT/d).mkdir(exist_ok=True)
 x=torch.tensor([[0,1],[1,0]]*32);y=torch.tensor([0,1]*32);np.savez(ROOT/'data/order_pairs.npz',tokens=x.numpy(),labels=y.numpy())
 curves=[];evaluations=[]
 for seed in (0,1,2):
  for position in (False,True):
   torch.manual_seed(seed);m=Classifier(position);opt=torch.optim.Adam(m.parameters(),lr=.02)
   for step in range(151):
    opt.zero_grad();logits=m(x);loss=F.cross_entropy(logits,y)
    if step%10==0:curves.append({'seed':seed,'position':position,'step':step,'loss':float(loss.detach())})
    if step<150:loss.backward();opt.step()
   with torch.no_grad():evaluations.append({'seed':seed,'position':position,'accuracy':float((m(x).argmax(-1)==y).float().mean()),'loss':float(F.cross_entropy(m(x),y))})
 with (ROOT/'outputs/curves.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(curves[0]));w.writeheader();w.writerows(curves)
 results={'checks':checks(),'order_task':evaluations,'configuration':{'device':'cpu','updates':150,'learning_rate':.02,'seeds':[0,1,2],'d_model':8,'heads':2,'feedforward':16,'dropout':0,'task_scope':'Only two exhaustive order patterns; no held-out generalization claim'}}
 (ROOT/'outputs/results.json').write_text(json.dumps(results,indent=2))
 fig,axs=plt.subplots(1,2,figsize=(10,3.3))
 for pos,color in [(False,'#bc7132'),(True,'#286c9c')]:
  for seed in range(3):
   r=[r for r in curves if r['position']==pos and r['seed']==seed];axs[0].plot([v['step'] for v in r],[v['loss'] for v in r],color=color,alpha=.65,label=('sinusoidal positions' if pos else 'no positions') if seed==0 else None)
 p=sinusoidal(12,8);im=axs[1].imshow(p,aspect='auto',cmap='coolwarm',vmin=-1,vmax=1);axs[1].set(xlabel='Feature index',ylabel='Position',title='Sinusoidal positional vectors');fig.colorbar(im,ax=axs[1]);axs[0].set(xlabel='Updates',ylabel='Cross-entropy',title='Order task: same bag, different label');axs[0].legend();fig.tight_layout();fig.savefig(ROOT/'figures/01_order.png',dpi=170);plt.close(fig)
 fig,ax=plt.subplots(figsize=(10,3));ax.axis('off');xs=[.05,.28,.51,.75,.95];labels=['X','LN -> attention','+ X','LN -> FFN','+ residual']
 for xx,s in zip(xs,labels):ax.text(xx,.55,s,ha='center',va='center',transform=ax.transAxes,bbox={'boxstyle':'round,pad=.5','fc':'#dbeafe' if 'LN' in s else '#dcfce7','ec':'#475569'},fontsize=10)
 for a,b in zip(xs[:-1],xs[1:]):ax.annotate('',(b-.08,.55),(a+.06,.55),xycoords='axes fraction',arrowprops={'arrowstyle':'->','lw':2})
 ax.text(.5,.10,'Pre-LN: normalize the branch input; preserve the residual stream.',ha='center',transform=ax.transAxes,fontsize=11);fig.savefig(ROOT/'figures/02_mechanism.png',dpi=170,bbox_inches='tight');plt.close(fig)
 print(json.dumps(results,indent=2))
if __name__=='__main__':main()
