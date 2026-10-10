"""087: offline initialization diagnostics and real Wine memorization, NumPy only."""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import numpy as np

ROOT = Path(__file__).resolve().parent
SEEDS = (8701, 8702, 8703)

def activate(z, kind):
    if kind == 'relu':
        return np.maximum(z, 0.0)
    if kind == 'tanh':
        return np.tanh(z)
    if kind == 'sigmoid':
        return np.exp(-np.logaddexp(0.0, -z))
    raise ValueError('unknown activation')

def derivative(z, kind):
    if kind == 'relu':
        return (z > 0).astype(float)  # defined convention at zero
    a = activate(z, kind)
    return 1-a*a if kind == 'tanh' else a*(1-a)

def weight_std(fan_in, fan_out, mode):
    if fan_in <= 0 or fan_out <= 0:
        raise ValueError('fans must be positive')
    if mode == 'small': return 0.01
    if mode == 'xavier': return np.sqrt(2.0/(fan_in+fan_out))
    if mode == 'he': return np.sqrt(2.0/fan_in)
    if mode == 'large': return 1.0
    raise ValueError('unknown initialization')

def moments(x):
    x = np.asarray(x, dtype=float)
    return {'mean': float(x.mean()), 'variance': float(x.var()),
            'second_moment': float(np.mean(x*x)), 'rms': float(np.sqrt(np.mean(x*x))),
            'zero_fraction': float(np.mean(x == 0))}

def forward(x, weights, biases, activation):
    a, zs = [x], []
    for w, b in zip(weights, biases):
        z = a[-1] @ w + b
        zs.append(z)
        a.append(activate(z, activation))
    return a, zs

def input_vjp(weights, zs, upstream, activation):
    """Gradient of sum(last_activation*upstream); no batch averaging."""
    g = upstream.copy()
    activation_grads = [None]*(len(weights)+1)
    z_grads = [None]*len(weights)
    activation_grads[-1] = g.copy()
    for i in reversed(range(len(weights))):
        dz = g*derivative(zs[i], activation)
        z_grads[i] = dz.copy()
        g = dz @ weights[i].T
        activation_grads[i] = g.copy()
    return activation_grads, z_grads

def probe(depth=24, width=64, seed=8701, activation='relu', mode='he', batch=128):
    if depth < 1 or width < 1 or batch < 1: raise ValueError('positive dimensions required')
    # Independent data, parameter, and upstream streams; paired configurations.
    x = np.random.default_rng(8700).normal(size=(batch, width))
    rng = np.random.default_rng(seed)
    weights = [rng.normal(size=(width,width))*weight_std(width,width,mode) for _ in range(depth)]
    a, zs = forward(x, weights, [np.zeros(width) for _ in weights], activation)
    upstream = np.random.default_rng(8799).normal(size=a[-1].shape)
    grads, dzs = input_vjp(weights, zs, upstream, activation)
    layers = []
    for i in range(depth):
        m = {'layer': i+1, 'activation': moments(a[i+1]),
             'preactivation': moments(zs[i]), 'grad_activation': moments(grads[i+1]),
             'grad_preactivation': moments(dzs[i]),
             'small_derivative_fraction': float(np.mean(np.abs(derivative(zs[i], activation)) < .01))}
        layers.append(m)
    return {'depth': depth, 'width': width, 'seed': seed, 'activation': activation,
            'initialization': mode, 'batch': batch, 'layers': layers,
            'input_grad_rms': moments(grads[0])['rms'], 'output_grad_rms': moments(upstream)['rms'],
            'gradient_rms_ratio': moments(grads[0])['rms']/moments(upstream)['rms'],
            'final_second_moment': moments(a[-1])['second_moment']}

def probe_samples(depth=24, width=64, seed=8701, mode='he', activation='relu'):
    x = np.random.default_rng(8700).normal(size=(128,width))
    rng = np.random.default_rng(seed)
    w = [rng.normal(size=(width,width))*weight_std(width,width,mode) for _ in range(depth)]
    a,z = forward(x,w,[np.zeros(width) for _ in w],activation)
    up = np.random.default_rng(8799).normal(size=a[-1].shape)
    g,_ = input_vjp(w,z,up,activation)
    return a,g

def residual_vjp(x, weights, upstream, scale=1.0):
    a, z = [x], []
    for w in weights:
        z.append(a[-1]@w)
        a.append(a[-1]+scale*np.tanh(z[-1]))
    g=upstream.copy()
    for w, zi in reversed(list(zip(weights,z))):
        g = g + scale*(g*(1-np.tanh(zi)**2))@w.T
    return a, g

def residual_probe(depth, seed, scale):
    width=64
    rng=np.random.default_rng(seed)
    weights=[rng.normal(size=(width,width))/np.sqrt(width) for _ in range(depth)]
    x=np.random.default_rng(8700).normal(size=(128,width))
    u=np.random.default_rng(8799).normal(size=x.shape)
    a,g=residual_vjp(x,weights,u,scale)
    return {'depth':depth,'seed':seed,'branch_scale':scale,
            'final_second_moment':moments(a[-1])['second_moment'],
            'gradient_rms_ratio':moments(g)['rms']/moments(u)['rms']}

def load_wine():
    d=json.loads((ROOT/'data/wine_tiny.json').read_text())
    x=np.array(d['features'],dtype=float); y=np.array(d['labels'],dtype=int)
    mean=x.mean(0); std=x.std(0)
    if np.any(std == 0): raise ValueError('constant feature')
    return (x-mean)/std, y, d

def initialize_classifier(dims, seed, mode):
    rng=np.random.default_rng(seed)
    # Last layer always uses Xavier: initialization intervention targets hidden layers.
    ws=[rng.normal(size=(a,b))*weight_std(a,b,mode if i<len(dims)-2 else 'xavier')
        for i,(a,b) in enumerate(zip(dims[:-1],dims[1:]))]
    return ws,[np.zeros(b) for b in dims[1:]]

def loss_and_grad(x,y,weights,biases):
    if len(weights)!=len(biases) or len(weights)<1: raise ValueError('invalid parameter lists')
    y=np.asarray(y)
    if x.ndim!=2 or len(y)!=len(x) or y.ndim!=1 or not np.issubdtype(y.dtype,np.integer):
        raise ValueError('x must be a matrix and y integer labels')
    if np.any(y<0) or np.any(y>=weights[-1].shape[1]): raise ValueError('label out of range')
    a,z=forward(x,weights[:-1],biases[:-1],'relu')
    logits=a[-1]@weights[-1]+biases[-1]
    shifted=logits-logits.max(1,keepdims=True)
    logp=shifted-np.log(np.exp(shifted).sum(1,keepdims=True))
    p=np.exp(logp)
    loss=float(-logp[np.arange(len(y)),y].mean())
    delta=p.copy(); delta[np.arange(len(y)),y]-=1; delta/=len(y)
    gw=[None]*len(weights); gb=[None]*len(weights)
    gw[-1]=a[-1].T@delta; gb[-1]=delta.sum(0)
    g=delta@weights[-1].T
    for i in reversed(range(len(weights)-1)):
        dz=g*(z[i]>0)
        gw[i]=a[i].T@dz; gb[i]=dz.sum(0)
        g=dz@weights[i].T
    return loss,gw,gb,float(np.mean(logits.argmax(1)==y))

def train_tiny(seed=8701, mode='he', steps=1200, learning_rate=.05, depth=6, width=48):
    x,y,_=load_wine()
    ws,bs=initialize_classifier([x.shape[1]]+[width]*depth+[3],seed,mode)
    history=[]
    for step in range(steps+1):
        loss,gw,gb,acc=loss_and_grad(x,y,ws,bs)
        if not np.isfinite(loss): raise FloatingPointError('nonfinite training loss')
        if step%50==0 or step==steps:
            history.append({'step':step,'loss':loss,'accuracy':acc,
                            'first_weight_grad_norm':float(np.linalg.norm(gw[0])),
                            'last_weight_grad_norm':float(np.linalg.norm(gw[-1]))})
        if step<steps:
            for w,b,dw,db in zip(ws,bs,gw,gb):
                w-=learning_rate*dw; b-=learning_rate*db
    return {'seed':seed,'initialization':mode,'hidden_depth':depth,'width':width,
            'steps':steps,'learning_rate':learning_rate,'optimizer':'full-batch SGD',
            'history':history,'final_loss':loss,'final_accuracy':acc,
            'memorization_pass':bool(acc==1.0 and loss<.02)}

def run():
    configs=[('relu',m) for m in ('small','xavier','he','large')]+[('tanh','xavier'),('tanh','large'),('sigmoid','xavier'),('sigmoid','large')]
    propagation=[probe(d,64,s,a,m) for d in (2,8,24) for s in SEEDS for a,m in configs]
    residual=[residual_probe(d,s,scale) for d in (2,8,24) for s in SEEDS for scale in (1.0,1/np.sqrt(d))]
    training=[train_tiny(s,m) for s in SEEDS for m in ('small','xavier','he')]
    _,_,data=load_wine()
    return {'lesson':'087','protocol':{'seeds':list(SEEDS),'depths':[2,8,24],'width':64,'batch':128,
              'input_seed':8700,'cotangent_seed':8799,'training_width':48,'training_depth':6,
              'training_steps':1200,'training_lr':.05,'training_rows':30,
              'diagnostic_only':'No generalization estimate; all 30 real observations reused for fitting and evaluation.'},
            'environment':{'python':platform.python_version(),'numpy':np.__version__},
            'data_sha256':hashlib.sha256((ROOT/'data/wine_tiny.json').read_bytes()).hexdigest(),
            'data_row_ids':data['row_ids'],'propagation':propagation,'residual':residual,'training':training,
            'residual_counterexamples':{'cancel_branch_derivative_minus_one':0.0,'expand_branch_derivative_one':2.0**24},
            'all_he_memorization_pass':all(t['memorization_pass'] for t in training if t['initialization']=='he')}

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--out',default='outputs/result.json'); args=parser.parse_args()
    result=run(); out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'out':str(out),'all_he_memorization_pass':result['all_he_memorization_pass'],
      'training':[{k:t[k] for k in ('seed','initialization','final_loss','final_accuracy')} for t in result['training']]},indent=2))
