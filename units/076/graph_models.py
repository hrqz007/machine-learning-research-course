"""二元概率图的教学实现；状态0/1，精确枚举小图，不使用采样。"""
from itertools import product, combinations  # product枚举状态；combinations枚举无序节点对。
import math  # isfinite拒绝NaN及无穷数。


def assignments(nodes):
    """依nodes的顺序返回每个完整状态字典。"""
    for states in product((0, 1), repeat=len(nodes)):  # n个二元节点共有2**n行。
        yield dict(zip(nodes, states))  # 将状态与名称一一配对，避免列顺序猜测。


def validate_dag(nodes, edges):
    """Kahn删除入度零节点；全部删完才无环。"""
    if len(set(nodes)) != len(nodes) or not nodes:
        raise ValueError('nodes must be distinct and nonempty')
    if len(set(map(tuple, edges))) != len(edges):
        raise ValueError('duplicate edge')
    if any(a not in nodes or b not in nodes or a == b for a, b in edges):
        raise ValueError('invalid edge')
    remaining = set(nodes)  # 本地副本，绝不修改调用方输入。
    while remaining:
        ready = {v for v in remaining if not any(b == v and a in remaining for a, b in edges)}
        if not ready:  # 每个剩余节点都有剩余父节点，说明遇到了有向环。
            raise ValueError('directed cycle')
        remaining -= ready  # 同一轮可删除多个互不依赖节点。


def joint_from_bn(nodes, edges, cpds):
    """cpds[v]按父节点在nodes中的顺序，列出各父状态下P(v=1)。"""
    validate_dag(nodes, edges)  # 先确认局部条件表可以构成DAG分解。
    parents = {v: [u for u in nodes if (u, v) in map(tuple, edges)] for v in nodes}
    if set(cpds) != set(nodes):
        raise ValueError('CPD keys must equal nodes')
    for v in nodes:
        row = cpds[v]
        if len(row) != 2 ** len(parents[v]) or any(not math.isfinite(p) or not 0 <= p <= 1 for p in row):
            raise ValueError('invalid CPD')
    result = []  # 每行保存assignment与联合概率probability。
    for state in assignments(nodes):
        probability = 1.0  # 乘法单位元；每个节点贡献一张局部条件表。
        for v in nodes:
            index = 0  # 父状态按二进制转成列表位置。
            for parent in parents[v]:
                index = 2 * index + state[parent]
            p1 = cpds[v][index]  # 取出P(v=1 | parents)。
            probability *= p1 if state[v] else 1 - p1
        result.append({'state': state, 'probability': probability})
    return result


def probability(joint, event):
    """边缘概率：对满足event的所有完整状态求和。"""
    if not joint or any(k not in joint[0]['state'] or v not in (0, 1) for k, v in event.items()):
        raise ValueError('invalid event')
    return math.fsum(row['probability'] for row in joint if all(row['state'][k] == v for k, v in event.items()))


def conditional(joint, event, evidence):
    """条件概率 = 交事件概率 / 证据概率；不为不可能证据造答案。"""
    mass = probability(joint, evidence)
    if mass <= 0:
        raise ValueError('zero-probability evidence')
    if any(k in evidence and evidence[k] != v for k, v in event.items()):
        return 0.0
    return probability(joint, dict(evidence, **event)) / mass


def ci_residual(joint, x, y, given=()):
    """所有正概率条件层中，P(x,y|z)-P(x|z)P(y|z)的最大绝对值。"""
    nodes = set(joint[0]['state'])
    if x == y or x in given or y in given or len(set(given)) != len(given) or not {x, y, *given} <= nodes:
        raise ValueError('CI variables must be known and disjoint')
    residual = 0.0
    for evidence in assignments(given):
        if probability(joint, evidence) == 0:  # 零概率条件层无普通条件概率定义。
            continue
        for a, b in product((0, 1), repeat=2):
            observed = conditional(joint, {x: a, y: b}, evidence)
            independent = conditional(joint, {x: a}, evidence) * conditional(joint, {y: b}, evidence)
            residual = max(residual, abs(observed - independent))
    return residual


def d_separated(nodes, edges, x, y, given=()):
    """枚举无向简单路径，逐个检查非碰撞点与碰撞点；适合教学小图。"""
    validate_dag(nodes, edges)
    if x == y or x in given or y in given or len(set(given)) != len(given) or not {x, y, *given} <= set(nodes):
        raise ValueError('query variables must be known and disjoint')
    edges = set(map(tuple, edges))
    ancestors = set(given)  # 碰撞点自身或后代被观测，相当于碰撞点在证据祖先集合中。
    while True:
        extended = ancestors | {a for a, b in edges if b in ancestors}
        if extended == ancestors:
            break
        ancestors = extended
    neighbors = {v: {u for u in nodes if (u, v) in edges or (v, u) in edges} for v in nodes}
    stack = [[x]]  # 路径搜索不沿箭头限制方向。
    while stack:
        path = stack.pop()
        if path[-1] == y:
            active = True
            for left, middle, right in zip(path, path[1:], path[2:]):
                collider = (left, middle) in edges and (right, middle) in edges
                if (collider and middle not in ancestors) or (not collider and middle in given):
                    active = False
                    break
            if active:
                return False  # 只需找到一条活跃路径，就没有被d分离。
        else:
            stack.extend(path + [v] for v in sorted(neighbors[path[-1]]) if v not in path)
    return True  # 所有简单路径都被阻断。


def equivalence_signature(nodes, edges):
    """DAG的骨架与未屏蔽碰撞三元组；相同签名表示Markov等价。"""
    validate_dag(nodes, edges)
    edges = set(map(tuple, edges))
    skeleton = {tuple(sorted((a, b))) for a, b in edges}
    colliders = set()
    for middle in nodes:
        parents = sorted(a for a, b in edges if b == middle)
        for a, b in combinations(parents, 2):
            if tuple(sorted((a, b))) not in skeleton:
                colliders.add((a, middle, b))
    return sorted(skeleton), sorted(colliders)


def undirected_chain():
    """正势函数AB同态得3，BC同态得2；不同态各得1。"""
    rows = []
    for state in assignments(('A', 'B', 'C')):
        weight = (3 if state['A'] == state['B'] else 1) * (2 if state['B'] == state['C'] else 1)
        rows.append({'state': state, 'weight': weight})
    partition = math.fsum(row['weight'] for row in rows)  # Z是所有状态权重之和。
    return [{'state': row['state'], 'probability': row['weight'] / partition} for row in rows], partition
