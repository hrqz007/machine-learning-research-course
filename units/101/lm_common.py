"""Small, transparent CPU language-learning components; no downloads."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/tutorial-mpl')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/tutorial-cache')
from pathlib import Path
import json, random, hashlib, math, time
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parent
torch.set_num_threads(1)

def corpus():
    docs=[f'the {a} {n} {v} the {b} {m}.' for a in ['red','blue','small','calm'] for n in ['cat','bird','dog','fox','hare','bear','wolf','duck'] for v in ['sees','likes','follows'] for b in ['red','blue','small','calm'] for m in ['cat','bird','dog','fox','hare','bear','wolf','duck'] if (a,n)!=(b,m)]
    random.Random(2026).shuffle(docs)
    parts={'train':docs[:1800],'validation':docs[1800:2200],'test':docs[2200:]}
    (ROOT/'data').mkdir(exist_ok=True)
    for split,rows in parts.items():
        (ROOT/'data'/f'{split}.jsonl').write_text(''.join(json.dumps({'id':hashlib.sha256(d.encode()).hexdigest()[:16],'text':d})+'\n' for d in rows))
    return parts

class Tokenizer:
    def __init__(self, docs, merges=0):
        self.merges=[]; seqs=[list(x) for x in docs]
        self.symbols=['<pad>','<bos>','<eos>','<unk>']+sorted(set(''.join(docs)))
        for _ in range(merges):
            counts={}
            for s in seqs:
                for pair in zip(s,s[1:]):counts[pair]=counts.get(pair,0)+1
            pair=min(counts,key=lambda z:(-counts[z],z))
            self.merges.append(pair); self.symbols.append(''.join(pair))
            seqs=[self.merge(s,pair) for s in seqs]
        self.symbols=list(dict.fromkeys(self.symbols));self.ids={s:i for i,s in enumerate(self.symbols)}
    @staticmethod
    def merge(s,p):
        out=[];i=0
        while i<len(s):
            if i+1<len(s) and (s[i],s[i+1])==p:out.append(''.join(p));i+=2
            else:out.append(s[i]);i+=1
        return out
    def encode(self,text):
        s=list(text)
        for p in self.merges:s=self.merge(s,p)
        return [1]+[self.ids.get(z,3) for z in s]+[2]
    def decode(self,ids):return ''.join(self.symbols[i] for i in ids if i not in [0,1,2])
    def save(self):
        (ROOT/'data/tokenizer.json').write_text(json.dumps({'symbols':self.symbols,'merges':self.merges,'fit_split':'train'},indent=2))

def batch(docs,tok):
    rows=[tok.encode(d) for d in docs];length=max(map(len,rows))-1
    x=torch.zeros(len(rows),length,dtype=torch.long);y=x.clone()
    for i,s in enumerate(rows):x[i,:len(s)-1]=torch.tensor(s[:-1]);y[i,:len(s)-1]=torch.tensor(s[1:])
    return x,y

class LM(nn.Module):
    def __init__(self,vocab,width):
        super().__init__();self.emb=nn.Embedding(vocab,width,padding_idx=0);self.rnn=nn.GRU(width,width,batch_first=True);self.head=nn.Linear(width,vocab)
    def forward(self,x):return self.head(self.rnn(self.emb(x))[0])

def evaluate(model,docs,tok):
    total=0.;count=0
    model.eval()
    with torch.no_grad():
        for start in range(0,len(docs),64):
            x,y=batch(docs[start:start+64],tok);total+=nn.functional.cross_entropy(model(x).flatten(0,1),y.flatten(),ignore_index=0,reduction='sum').item();count+=int((y!=0).sum())
    return {'nll':total/count,'ppl':math.exp(total/count),'target_tokens':count,'nll_sum':total,'bits_per_utf8_byte':total/(sum(len(d.encode()) for d in docs)*math.log(2))}

def train(docs,val,tok,width=24,steps=100,seed=0):
    torch.manual_seed(seed);rng=random.Random(seed);model=LM(len(tok.symbols),width);opt=torch.optim.Adam(model.parameters(),lr=.01)
    history=[];tokens=0;start=time.perf_counter()
    initial=evaluate(model,val,tok)
    history.append({'step':0,'tokens':0,'validation_nll':initial['nll']})
    for step in range(1,steps+1):
        model.train();x,y=batch(rng.choices(docs,k=24),tok);loss=nn.functional.cross_entropy(model(x).flatten(0,1),y.flatten(),ignore_index=0)
        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();tokens+=int((y!=0).sum())
        if step%20==0 or step==steps:history.append({'step':step,'tokens':tokens,'validation_nll':evaluate(model,val,tok)['nll']})
    elapsed=time.perf_counter()-start
    return model,{'seed':seed,'width':width,'parameters':sum(p.numel() for p in model.parameters()),'steps':steps,'processed_target_tokens':tokens,'unique_train_docs':len(docs),'unique_train_target_tokens':sum(len(tok.encode(d))-1 for d in docs),'train_seconds':elapsed,'history':history,'validation':evaluate(model,val,tok),'train':evaluate(model,docs,tok),'initial_validation':initial}
