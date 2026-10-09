"""二元离散因子的精确推断；字典实现刻意突出scope和变量对齐。"""
from dataclasses import dataclass  # 把范围与数表放在同一对象，避免把列顺序分离。
from itertools import product  # 枚举当前小因子的状态。
import math  # 检查浮点有效性并进行稳定求和。

@dataclass
class Factor:
    scope: tuple  # scope列出此因子涉及的变量，顺序就是table键的顺序。
    table: dict  # 键是0/1元组，值为非负权重；空scope有唯一键()。
    def __post_init__(self):
        self.scope=tuple(self.scope)
        if len(set(self.scope))!=len(self.scope):raise ValueError('duplicate variable')
        expected=set(product((0,1),repeat=len(self.scope)))
        if set(self.table)!=expected:raise ValueError('incomplete factor table')
        if any(not math.isfinite(v) or v<0 for v in self.table.values()):raise ValueError('invalid weight')
    def value(self,state):
        return self.table[tuple(state[v] for v in self.scope)]  # 按变量名读取，不按调用方字典顺序。

def states(scope):
    for values in product((0,1),repeat=len(scope)):
        yield dict(zip(scope,values))

def multiply(factors):
    scope=tuple(dict.fromkeys(v for f in factors for v in f.scope))  # 保序取并集。
    table={}
    for state in states(scope):
        table[tuple(state[v] for v in scope)]=math.prod(f.value(state) for f in factors)
    return Factor(scope,table)  # 空列表的乘积是常数因子1。

def restrict(factor,evidence):
    scope=tuple(v for v in factor.scope if v not in evidence)  # 固定证据后删除对应轴。
    table={}
    for state in states(scope):
        full=dict(state,**evidence)
        table[tuple(state[v] for v in scope)]=factor.value(full)
    return Factor(scope,table)

def sum_out(factor,variable):
    if variable not in factor.scope:raise ValueError('variable not in factor')
    scope=tuple(v for v in factor.scope if v!=variable)
    table={}
    for state in states(scope):
        table[tuple(state[v] for v in scope)]=math.fsum(factor.value(dict(state,**{variable:x})) for x in (0,1))
    return Factor(scope,table)  # 求和后少一根轴，已消元变量不会重新出现。

def prepare(factors,query,evidence):
    all_vars={v for f in factors for v in f.scope}
    if query not in all_vars or query in evidence:raise ValueError('query must be known and unobserved')
    if any(v not in all_vars or state not in (0,1) for v,state in evidence.items()):raise ValueError('invalid evidence')
    return all_vars,[restrict(f,evidence) for f in factors]

def normalize(factor):
    mass=math.fsum(factor.table.values())
    if not math.isfinite(mass) or mass<=0:raise ValueError('zero or invalid evidence mass')
    return Factor(factor.scope,{k:v/mass for k,v in factor.table.items()}),mass

def eliminate(factors,query,evidence=None,order=None):
    """返回查询分布、未归一化证据质量及每一步临时因子规模。"""
    evidence={} if evidence is None else evidence
    variables,work=prepare(factors,query,evidence)
    hidden=variables-{query}-set(evidence)
    order=sorted(hidden) if order is None else list(order)
    if len(order)!=len(hidden) or set(order)!=hidden:raise ValueError('order must contain each hidden variable exactly once')
    trace=[]
    for variable in order:
        bucket=[f for f in work if variable in f.scope]  # 只收集含此变量的因子。
        work=[f for f in work if variable not in f.scope]  # 无关因子留到以后。
        combined=multiply(bucket)  # 必须先乘，再对variable求和。
        reduced=sum_out(combined,variable)
        trace.append({'variable':variable,'joint_scope':list(combined.scope),'entries':len(combined.table),'result_scope':list(reduced.scope)})
        work.append(reduced)  # 新因子总结已消去变量对其余变量的影响。
    final=multiply(work)
    result,mass=normalize(final)
    return result,mass,trace

def enumerate_query(factors,query,evidence=None):
    """独立穷举基准，不调用变量消元。"""
    evidence={} if evidence is None else evidence
    variables,_=prepare(factors,query,evidence)
    sums={(0,):[],(1,):[]}
    for state in states(sorted(variables)):
        if all(state[k]==v for k,v in evidence.items()):
            sums[(state[query],)].append(math.prod(f.value(state) for f in factors))
    return normalize(Factor((query,),{k:math.fsum(v) for k,v in sums.items()}))

def chain_factors(prior,transition,length,emission):
    """末端观测E：X0 -> X1 -> ... -> Xn -> E。"""
    if length<1:raise ValueError('length must be positive')
    if len(prior)!=2 or any(len(row)!=2 for row in transition) or len(transition)!=2 or len(emission)!=2:raise ValueError('binary parameters required')
    for row in [prior,*transition,emission]:
        if any(not math.isfinite(v) or not 0<=v<=1 for v in row):raise ValueError('invalid probability')
    if abs(sum(prior)-1)>1e-12 or any(abs(sum(row)-1)>1e-12 for row in transition):raise ValueError('rows must sum to one')
    factors=[Factor(('X0',),{(x,):prior[x] for x in (0,1)})]
    for i in range(1,length):
        factors.append(Factor((f'X{i-1}',f'X{i}'),{(a,b):transition[a][b] for a,b in product((0,1),repeat=2)}))
    factors.append(Factor((f'X{length-1}','E'),{(x,e):(emission[x] if e else 1-emission[x]) for x,e in product((0,1),repeat=2)}))
    return factors

def chain_sum_product(prior,transition,length,emission,observed=1):
    """链上的未缩放前向/后向消息；小规模教学，长链需缩放或log域。"""
    chain_factors(prior,transition,length,emission)  # 复用输入校验，不复用推断结果。
    if observed not in (0,1):raise ValueError('observed must be binary')
    likelihood=[v if observed else 1-v for v in emission]
    forward=[list(prior)]
    for _ in range(1,length):
        forward.append([math.fsum(forward[-1][a]*transition[a][b] for a in (0,1)) for b in (0,1)])
    backward=[[0.,0.] for _ in range(length)];backward[-1]=likelihood
    for i in range(length-2,-1,-1):
        backward[i]=[math.fsum(transition[a][b]*backward[i+1][b] for b in (0,1)) for a in (0,1)]
    marginals=[]
    for f,b in zip(forward,backward):
        raw=[f[x]*b[x] for x in (0,1)];mass=math.fsum(raw)
        if mass<=0:raise ValueError('zero-probability evidence')
        marginals.append([v/mass for v in raw])
    evidence_mass=math.fsum(forward[-1][x]*likelihood[x] for x in (0,1))
    return marginals,evidence_mass,forward,backward

def star_factors(leaves=6):
    if leaves<2:raise ValueError('need at least two leaves')
    factors=[Factor(('C',),{(0,):.6,(1,):.4})]
    for i in range(leaves):
        factors.append(Factor(('C',f'L{i}'),{(a,b):(3. if a==b else 1.) for a,b in product((0,1),repeat=2)}))
    return factors
