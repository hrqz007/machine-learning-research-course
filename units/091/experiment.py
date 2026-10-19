"""CPU-only convolution audit. Run from any directory; no data downloads."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/tutorial-mpl')
import json, platform
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent

def corr2d(x,w,b,stride=1,padding=0):
    """NCHW input, OIHW weights; explicit loops expose shared parameters."""
    n,c,h,width=x.shape; o,cw,kh,kw=w.shape
    assert c==cw and stride>0 and padding>=0
    oh=(h+2*padding-kh)//stride+1; ow=(width+2*padding-kw)//stride+1
    assert oh>0 and ow>0
    xp=np.pad(x,((0,0),(0,0),(padding,padding),(padding,padding)))
    y=np.zeros((n,o,oh,ow),dtype=np.result_type(x,w,b))
    for a in range(n):
        for q in range(o):
            for i in range(oh):
                for j in range(ow):
                    patch=xp[a,:,i*stride:i*stride+kh,j*stride:j*stride+kw]
                    y[a,q,i,j]=np.sum(patch*w[q])+b[q]
    return y

def backward(x,w,dy,stride=1,padding=0):
    """Overlap means +=, not assignment: several outputs depend on one pixel."""
    xp=np.pad(x,((0,0),(0,0),(padding,padding),(padding,padding)))
    dxp=np.zeros_like(xp); dw=np.zeros_like(w); db=dy.sum(axis=(0,2,3))
    kh,kw=w.shape[-2:]
    for a in range(x.shape[0]):
        for q in range(w.shape[0]):
            for i in range(dy.shape[2]):
                for j in range(dy.shape[3]):
                    rs=slice(i*stride,i*stride+kh); cs=slice(j*stride,j*stride+kw)
                    dw[q]+=dy[a,q,i,j]*xp[a,:,rs,cs]
                    dxp[a,:,rs,cs]+=dy[a,q,i,j]*w[q]
    dx=dxp[:,:,padding:-padding,padding:-padding] if padding else dxp
    return dx,dw,db

def run():
    torch.set_num_threads(1); rng=np.random.default_rng(91)
    (ROOT/'outputs').mkdir(exist_ok=True); (ROOT/'figures').mkdir(exist_ok=True)
    x=np.arange(1,10,dtype=float).reshape(1,1,3,3); w=np.array([[1.,0.],[0.,-1.]]).reshape(1,1,2,2)
    hand=corr2d(x,w,np.zeros(1))[0,0]
    audits=[]
    for stride,padding in [(1,0),(1,1),(2,1)]:
        a=rng.normal(size=(2,2,5,6)); k=rng.normal(size=(3,2,3,2)); b=rng.normal(size=3)
        y=corr2d(a,k,b,stride,padding); dy=rng.normal(size=y.shape)
        dx,dw,db=backward(a,k,dy,stride,padding)
        ta=torch.tensor(a,requires_grad=True); tk=torch.tensor(k,requires_grad=True); tb=torch.tensor(b,requires_grad=True)
        ty=F.conv2d(ta,tk,tb,stride=stride,padding=padding); (ty*torch.tensor(dy)).sum().backward()
        errors={name:float(np.max(np.abs(v-ref))) for name,v,ref in [('forward',y,ty.detach().numpy()),('dx',dx,ta.grad.numpy()),('dw',dw,tk.grad.numpy()),('db',db,tb.grad.numpy())]}
        eps=1e-6; idx=(0,0,1,0); kp=k.copy(); km=k.copy(); kp[idx]+=eps; km[idx]-=eps
        numerical=np.sum((corr2d(a,kp,b,stride,padding)-corr2d(a,km,b,stride,padding))*dy)/(2*eps)
        errors['finite_difference_dw']=float(abs(numerical-dw[idx]))
        audits.append({'stride':stride,'padding':padding,'shape':list(y.shape),'errors':errors})
    fig,axs=plt.subplots(1,3,figsize=(9,3))
    for ax,arr,title in zip(axs,[x[0,0],w[0,0],hand],['Input','Shared kernel','Cross-correlation']):
        ax.imshow(arr,cmap='coolwarm'); ax.set_title(title); ax.set_xticks([]);ax.set_yticks([])
        for (i,j),v in np.ndenumerate(arr): ax.text(j,i,f'{v:g}',ha='center',va='center')
    fig.tight_layout();fig.savefig(ROOT/'figures/01_hand_calculation.png',dpi=160);plt.close(fig)
    # A single output depends on a 3x3 region; two stacked layers on a 5x5 region.
    z=torch.zeros(1,1,11,11,requires_grad=True); k=torch.ones(1,1,3,3)
    one=F.conv2d(z,k,padding=1);two=F.conv2d(one,k,padding=1)
    g1=torch.autograd.grad(one[0,0,5,5],z,retain_graph=True)[0][0,0].numpy()
    g2=torch.autograd.grad(two[0,0,5,5],z)[0][0,0].numpy()
    fig,axs=plt.subplots(1,2,figsize=(7,3))
    for ax,g,title in zip(axs,[g1,g2],['One layer: RF=3','Two layers: RF=5']): ax.imshow(g,cmap='viridis');ax.set_title(title)
    fig.tight_layout();fig.savefig(ROOT/'figures/02_receptive_field.png',dpi=160);plt.close(fig)
    # Circular padding is exactly compatible with circular integer translations.
    a=torch.tensor(rng.normal(size=(1,1,9,9)));k=torch.tensor(rng.normal(size=(1,1,3,3)))
    conv=lambda v:F.conv2d(F.pad(v,(1,1,1,1),mode='circular'),k)
    zero=lambda v:F.conv2d(v,k,padding=1)
    shift=lambda v:torch.roll(v,1,dims=3)
    circ=(conv(shift(a))-shift(conv(a))).abs()[0,0].numpy()
    boundary=(zero(shift(a))-shift(zero(a))).abs()[0,0].numpy()
    fig,axs=plt.subplots(1,2,figsize=(7,3))
    for ax,e,title in zip(axs,[circ,boundary],['Circular padding error','Zero padding + circular shift']): im=ax.imshow(e,cmap='magma',vmin=0,vmax=max(boundary.max(),1e-9));ax.set_title(title);fig.colorbar(im,ax=ax)
    fig.tight_layout();fig.savefig(ROOT/'figures/03_boundary_equivariance.png',dpi=160);plt.close(fig)
    results={'seed':91,'device':'cpu','torch':torch.__version__,'python':platform.python_version(),'hand_output':hand.tolist(),'audits':audits,'circular_equivariance_max_error':float(circ.max()),'zero_padding_circular_shift_max_error':float(boundary.max()),'rf_nonzero_pixels':[int((g1!=0).sum()),int((g2!=0).sum())]}
    (ROOT/'outputs/results.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2));return results
if __name__=='__main__':run()
