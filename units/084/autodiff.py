"""Small first-order scalar AD engine, intentionally independent of ML libraries.
Values are read-only; each edge saves its local derivative during forward pass.
backward resets ALL reachable gradients before every call (unlike PyTorch).
"""
import math

class Value:
    def __init__(self, data, edges=(), op='leaf', label=''):
        self._data = float(data)
        if not math.isfinite(self._data): raise ValueError('finite values required')
        self.edges = tuple(edges); self.op = op; self.label = label; self.grad = 0.0
    @property
    def data(self): return self._data
    @staticmethod
    def lift(x): return x if isinstance(x, Value) else Value(x)
    def __add__(self, other):
        other=Value.lift(other); return Value(self.data+other.data,((self,1.),(other,1.)),'+')
    __radd__=__add__
    def __mul__(self, other):
        other=Value.lift(other); return Value(self.data*other.data,((self,other.data),(other,self.data)),'*')
    __rmul__=__mul__
    def __neg__(self): return self * -1.
    def __sub__(self, other): return self + (-Value.lift(other))
    def __rsub__(self, other): return Value.lift(other) + (-self)
    def __pow__(self, n):
        if type(n) is not int or n < 1: raise ValueError('only positive integer powers supported')
        return Value(self.data**n,((self,n*self.data**(n-1)),),f'pow{n}')
    def sin(self): return Value(math.sin(self.data),((self,math.cos(self.data)),),'sin')
    def exp(self):
        value=math.exp(self.data); return Value(value,((self,value),),'exp')
    def log(self):
        if self.data <= 0: raise ValueError('log domain requires positive input')
        return Value(math.log(self.data),((self,1/self.data),),'log')
    def relu(self): return Value(max(0.,self.data),((self,1. if self.data > 0 else 0.),),'relu')
    def topology(self):
        # Iterative postorder: visits NODES once, retains duplicate EDGES.
        order=[]; seen=set(); stack=[(self,False)]
        while stack:
            node,expanded=stack.pop()
            if expanded: order.append(node); continue
            if node in seen: continue
            seen.add(node); stack.append((node,True))
            for parent,_ in reversed(node.edges):
                if parent not in seen: stack.append((parent,False))
        return order
    def backward(self, seed=1.):
        seed=float(seed)
        if not math.isfinite(seed): raise ValueError('finite seed required')
        order=self.topology()
        for node in order: node.grad=0.
        self.grad=seed
        for node in reversed(order):
            for parent,local_derivative in node.edges:
                parent.grad += node.grad*local_derivative
        return order

class Dual:
    """Forward mode: value plus derivative in one chosen input direction."""
    def __init__(self, data, tangent=0.): self.data=float(data); self.tangent=float(tangent)
    @staticmethod
    def lift(x): return x if isinstance(x,Dual) else Dual(x)
    def __add__(self, other):
        other=Dual.lift(other); return Dual(self.data+other.data,self.tangent+other.tangent)
    __radd__=__add__
    def __mul__(self,other):
        other=Dual.lift(other); return Dual(self.data*other.data,self.tangent*other.data+self.data*other.tangent)
    __rmul__=__mul__
    def __neg__(self): return self * -1.
    def __sub__(self,other): return self + (-Dual.lift(other))
    def __rsub__(self,other): return Dual.lift(other) + (-self)
    def __pow__(self,n):
        if type(n) is not int or n < 1: raise ValueError('only positive integer powers supported')
        return Dual(self.data**n,n*self.data**(n-1)*self.tangent)
    def sin(self): return Dual(math.sin(self.data),math.cos(self.data)*self.tangent)
    def exp(self):
        e=math.exp(self.data); return Dual(e,e*self.tangent)
    def log(self):
        if self.data <= 0: raise ValueError('positive input required')
        return Dual(math.log(self.data),self.tangent/self.data)
    def relu(self): return Dual(max(0.,self.data),self.tangent if self.data>0 else 0.)

def finite_difference(f, point, h=1e-5):
    if not math.isfinite(h) or h <= 0: raise ValueError('positive finite step required')
    result=[]
    for i in range(len(point)):
        left=list(point); right=list(point); left[i]-=h; right[i]+=h
        result.append((f(*right)-f(*left))/(2*h))
    return result
