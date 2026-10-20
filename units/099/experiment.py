import os, tempfile
_CACHE = tempfile.mkdtemp(prefix='course-cache-')
os.environ.setdefault('MPLCONFIGDIR', _CACHE)
os.environ.setdefault('XDG_CACHE_HOME', _CACHE)
os.environ.setdefault('IPYTHONDIR', _CACHE)
"""Double-precision attention, independent manual gradients and masking checks."""
from pathlib import Path
import math,json
import numpy as np
import torch
from torch.nn import functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
torch.set_num_threads(1)

def attention(q,k,v,allowed=None):
 scores=q@k.transpose(-2,-1)/math.sqrt(q.shape[-1])
 if allowed is not None:
  if not allowed.any(-1).all():raise ValueError('every query needs at least one allowed key')
  scores=scores.masked_fill(~allowed,float('-inf'))
 weights=torch.softmax(scores,-1)
 return weights@v,weights

def backward(q,k,v,weights,dy):
 dv=weights.transpose(-2,-1)@dy
 da=dy@v.transpose(-2,-1)
 ds=weights*(da-(da*weights).sum(-1,keepdim=True))
 return ds@k/math.sqrt(q.shape[-1]),ds.transpose(-2,-1)@q/math.sqrt(q.shape[-1]),dv

def run_checks():
 torch.manual_seed(99)
 q=torch.randn(2,2,4,3,dtype=torch.double,requires_grad=True);k=torch.randn_like(q,requires_grad=True);v=torch.randn(2,2,4,2,dtype=torch.double,requires_grad=True)
 allowed=torch.ones(4,4,dtype=torch.bool).tril();y,a=attention(q,k,v,allowed);dy=torch.randn_like(y);loss=(y*dy).sum();loss.backward();grads=backward(q.detach(),k.detach(),v.detach(),a.detach(),dy)
 errors=[float((g-t.grad).abs().max()) for g,t in zip(grads,(q,k,v))]
 ref=F.scaled_dot_product_attention(q.detach(),k.detach(),v.detach(),attn_mask=allowed,dropout_p=0.)
 # Fixed upstream dy gives a scalar objective for central finite differences.
 eps=1e-6;qplus=q.detach().clone();qminus=q.detach().clone();qplus[0,0,2,1]+=eps;qminus[0,0,2,1]-=eps
 fd=float(((attention(qplus,k.detach(),v.detach(),allowed)[0]*dy).sum()-(attention(qminus,k.detach(),v.detach(),allowed)[0]*dy).sum())/(2*eps))
 future_k=k.detach().clone();future_v=v.detach().clone();future_k[:,:,3]+=30;future_v[:,:,3]-=20
 masked_change=float((attention(q.detach(),future_k,future_v,allowed)[0][:,:,:3]-y.detach()[:,:,:3]).abs().max())
 unmasked=attention(q.detach(),k.detach(),v.detach())[0];unmasked_changed=attention(q.detach(),future_k,future_v)[0]
 padk=torch.cat([k.detach(),torch.randn(2,2,2,3,dtype=torch.double)],-2);padv=torch.cat([v.detach(),torch.randn(2,2,2,2,dtype=torch.double)],-2)
 padmask=torch.cat([allowed,torch.zeros(4,2,dtype=torch.bool)],-1)
 return {'framework_output_error':float((y.detach()-ref).abs().max()),'manual_gradient_errors_QKV':errors,'finite_difference_q_error':abs(fd-float(q.grad[0,0,2,1])),'future_perturbation_masked_error':masked_change,'future_perturbation_unmasked_change':float((unmasked[:,:,:3]-unmasked_changed[:,:,:3]).abs().max()),'padding_append_error':float((attention(q.detach(),padk,padv,padmask)[0]-y.detach()).abs().max()),'blocked_weight_max':float(a.detach().masked_select(~allowed).abs().max()),'row_sum_error':float((a.sum(-1)-1).abs().max().detach())}

def main():
 for d in ('outputs','data','figures'):(ROOT/d).mkdir(exist_ok=True)
 q=torch.tensor([[1.,0.],[0.,1.],[1.,1.]],dtype=torch.double);k=q.clone();v=torch.tensor([[10.,0.],[0.,20.],[5.,5.]],dtype=torch.double)
 mask=torch.ones(3,3,dtype=torch.bool).tril();all_y,all_a=attention(q,k,v);causal_y,causal_a=attention(q,k,v,mask)
 np.savez(ROOT/'data/three_tokens.npz',Q=q.numpy(),K=k.numpy(),V=v.numpy(),allowed=mask.numpy())
 results={'checks':run_checks(),'hand_example':{'scores':(q@k.T/math.sqrt(2)).tolist(),'unmasked_weights':all_a.tolist(),'unmasked_output':all_y.tolist(),'causal_weights':causal_a.tolist(),'causal_output':causal_y.tolist()},'configuration':{'dtype':'float64','device':'cpu','seed':99,'dropout':0,'mask_true_means':'allowed'},'complexity':[]}
 for n in (16,32,64,128):results['complexity'].append({'length':n,'heads':4,'score_elements':4*n*n,'float32_score_bytes':4*4*n*n})
 (ROOT/'outputs/results.json').write_text(json.dumps(results,indent=2))
 fig,axs=plt.subplots(1,2,figsize=(9,3.5))
 for ax,a,title in zip(axs,[all_a,causal_a],['Unmasked attention','Causal attention (row=query)']):
  im=ax.imshow(a,vmin=0,vmax=1,cmap='Blues');ax.set(xticks=[0,1,2],yticks=[0,1,2],xlabel='Key position',ylabel='Query position',title=title)
  for i in range(3):
   for j in range(3):ax.text(j,i,f'{a[i,j]:.3f}',ha='center',va='center',color='white' if a[i,j]>.6 else 'black')
 fig.colorbar(im,ax=axs.ravel().tolist(),shrink=.7);fig.savefig(ROOT/'figures/01_weights.png',dpi=170,bbox_inches='tight');plt.close(fig)
 fig,ax=plt.subplots(figsize=(10,2.8));ax.axis('off')
 texts=['Q @ K.T','divide sqrt(dk)','mask then softmax','weights @ V'];xs=[.09,.35,.62,.89]
 for x,s,c in zip(xs,texts,['#dbeafe','#dbeafe','#ffedd5','#dcfce7']):ax.text(x,.55,s,ha='center',va='center',transform=ax.transAxes,bbox={'boxstyle':'round,pad=.7','fc':c,'ec':'#475569'},fontsize=11)
 for a,b in zip(xs[:-1],xs[1:]):ax.annotate('',(b-.10,.55),(a+.10,.55),xycoords='axes fraction',arrowprops={'arrowstyle':'->','lw':2})
 ax.text(.5,.1,'Each row asks one question; blocked keys get zero mass; V supplies content.',ha='center',transform=ax.transAxes,fontsize=11)
 fig.savefig(ROOT/'figures/02_mechanism.png',dpi=170,bbox_inches='tight');plt.close(fig)
 print(json.dumps(results,indent=2))
if __name__=='__main__':main()
