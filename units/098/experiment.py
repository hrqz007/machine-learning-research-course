import os, tempfile
_CACHE = tempfile.mkdtemp(prefix='course-cache-')
os.environ.setdefault('MPLCONFIGDIR', _CACHE)
os.environ.setdefault('XDG_CACHE_HOME', _CACHE)
os.environ.setdefault('IPYTHONDIR', _CACHE)
"""CPU delay-memory experiment. All data and every recorded curve are saved."""
from pathlib import Path
import json, csv
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
torch.set_num_threads(1)

def make_data(n,length,seed):
 g=torch.Generator().manual_seed(seed)
 y=torch.arange(n)%2
 x=.10*torch.randn(n,length,1,generator=g)
 x[:,0,0]=y*2.-1
 return x,y

def masked_loss(logits,labels,valid):
 losses=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),labels.reshape(-1),reduction='none').reshape_as(labels)
 if not valid.any():raise ValueError('no valid tokens')
 return (losses*valid).sum()/valid.sum()

class Memory(nn.Module):
 def __init__(self,kind,hidden=12):
  super().__init__(); self.core=getattr(nn,kind)(1,hidden,batch_first=True);self.head=nn.Linear(hidden,2)
 def forward(self,x,lengths=None):
  out,_=self.core(x)
  if lengths is None:return self.head(out[:,-1])
  return self.head(out[torch.arange(len(x)),lengths-1])

def lstm_step(x,h,c,cell):
 gates=F.linear(x,cell.weight_ih,cell.bias_ih)+F.linear(h,cell.weight_hh,cell.bias_hh)
 i,f,g,o=gates.chunk(4,-1); c1=f.sigmoid()*c+i.sigmoid()*g.tanh()
 return o.sigmoid()*c1.tanh(),c1

def checks():
 torch.manual_seed(11);cell=nn.LSTMCell(2,3).double();x=torch.randn(4,2,dtype=torch.double);h=torch.randn(4,3,dtype=torch.double);c=torch.randn_like(h)
 a=lstm_step(x,h,c,cell);b=cell(x,(h,c));step=max(float((u-v).abs().max().detach()) for u,v in zip(a,b))
 logits=torch.randn(2,4,3,dtype=torch.double);labels=torch.randint(3,(2,4));valid=torch.tensor([[1,1,0,0],[1,1,1,0]],dtype=torch.bool)
 l=masked_loss(logits,labels,valid); extended=torch.cat([logits,torch.randn(2,3,3,dtype=torch.double)],1)
 lp=masked_loss(extended,torch.cat([labels,torch.zeros(2,3,dtype=torch.long)],1),torch.cat([valid,torch.zeros(2,3,dtype=torch.bool)],1))
 m=Memory('GRU').double(); xx=torch.randn(2,4,1,dtype=torch.double); lengths=torch.tensor([2,4]);yp=m(xx,lengths);xx2=torch.cat([xx,torch.randn(2,5,1,dtype=torch.double)],1);padding=float((yp-m(xx2,lengths)).abs().max().detach())
 # A deliberately wrong terminal readout is sensitive to appended padding.
 wrong=float((m(xx)-m(xx2)).abs().max().detach())
 return {'lstm_cell_max_error':step,'valid_loss':float(l),'padded_valid_loss':float(lp),'loss_padding_error':float(abs(l-lp)),'valid_readout_padding_error':padding,'wrong_last_readout_change':wrong,'constant_forget_0.9_after_50':.9**50,'constant_forget_0.99_after_50':.99**50}

def main():
 for name in ('data','outputs','figures'):(ROOT/name).mkdir(exist_ok=True)
 datasets={};train=make_data(192,12,198);datasets['train_x']=train[0].numpy();datasets['train_y']=train[1].numpy()
 tests={}
 for length in (12,24,48):
  tests[length]=make_data(256,length,1000+length)
  datasets[f'test_{length}_x']=tests[length][0].numpy();datasets[f'test_{length}_y']=tests[length][1].numpy()
 np.savez_compressed(ROOT/'data/sequences.npz',**datasets)
 curves=[];rows=[]
 for seed in (0,1,2):
  for kind in ('RNN','GRU','LSTM'):
   torch.manual_seed(seed);m=Memory(kind);opt=torch.optim.Adam(m.parameters(),lr=.02)
   for epoch in range(201):
    opt.zero_grad();loss=F.cross_entropy(m(train[0]),train[1])
    if epoch%10==0:curves.append({'seed':seed,'model':kind,'epoch':epoch,'loss':float(loss.detach())})
    if epoch<200:loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
   with torch.no_grad():
    for length,(x,y) in tests.items():rows.append({'seed':seed,'model':kind,'length':length,'accuracy':float((m(x).argmax(-1)==y).double().mean()),'loss':float(F.cross_entropy(m(x),y)),'parameters':sum(p.numel() for p in m.parameters())})
 for name,values in [('curves',curves),('evaluation',rows)]:
  with (ROOT/f'outputs/{name}.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
 results={'checks':checks(),'configuration':{'train_examples':192,'test_examples_per_length':256,'hidden':12,'updates':200,'learning_rate':.02,'seeds':[0,1,2],'train_length':12,'data_seed':198,'device':'cpu'},'evaluation':rows,'scope':'Synthetic first-bit retention only; equal hidden width and updates, unequal parameter count.'}
 (ROOT/'outputs/results.json').write_text(json.dumps(results,indent=2))
 fig,axs=plt.subplots(1,2,figsize=(10,3.6))
 for kind in ('RNN','GRU','LSTM'):
  for seed in range(3):
   cc=[r for r in curves if r['model']==kind and r['seed']==seed];axs[0].plot([r['epoch'] for r in cc],[r['loss'] for r in cc],alpha=.5,label=kind if seed==0 else None)
  data=np.array([[r['accuracy'] for r in rows if r['model']==kind and r['length']==l] for l in (12,24,48)])
  axs[1].errorbar([12,24,48],data.mean(1),yerr=data.std(1),marker='o',capsize=3,label=kind)
 axs[0].set(xlabel='Updates',ylabel='Training CE',title='Three initializations');axs[1].set(xlabel='Test sequence length',ylabel='Accuracy',ylim=(0,1.05),title='Mean +/- population SD (3 seeds)')
 for ax in axs:ax.legend();ax.grid(alpha=.2)
 fig.tight_layout();fig.savefig(ROOT/'figures/01_results.png',dpi=170);plt.close(fig)
 fig,ax=plt.subplots(figsize=(10,3));ax.axis('off')
 boxes=[(.02,.60,'old cell c(t-1)', '#dbeafe'),(.29,.60,'forget f * old', '#dbeafe'),(.56,.60,'+ input i * candidate g','#dcfce7'),(.79,.60,'new cell c(t)', '#dcfce7'),(.53,.10,'output o * tanh(c)', '#ffedd5')]
 for x,y,s,c in boxes:ax.text(x,y,s,transform=ax.transAxes,bbox=dict(boxstyle='round,pad=.6',fc=c,ec='#475569'),fontsize=10,ha='center')
 for a,b in [((.12,.62),(.21,.62)),((.39,.62),(.44,.62)),((.69,.62),(.73,.62)),((.79,.52),(.61,.22))]:ax.annotate('',b,a,xycoords='axes fraction',arrowprops={'arrowstyle':'->','lw':2})
 ax.text(.02,.1,'gates depend on x(t), h(t-1)\ncell highway is additive; gates remain learned',transform=ax.transAxes,fontsize=10)
 fig.tight_layout();fig.savefig(ROOT/'figures/02_mechanism.png',dpi=170,bbox_inches='tight');plt.close(fig)
 print(json.dumps(results,indent=2))
if __name__=='__main__':main()
