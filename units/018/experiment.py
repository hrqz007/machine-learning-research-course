"""第018讲：有限事件、条件概率与离线检测器模拟。

固定表是合成总体的精确分布；模拟是从指定模型独立重采样。
无网络、无付费接口。路径相对于本文件。数值检查不依赖 assert。
"""
from pathlib import Path
from fractions import Fraction
from itertools import product
import csv
import json
import math
import numbers
import sys
import numpy as np

BASE = Path(__file__).resolve().parent
SEED = 20261004
ATOL = 1e-12
MAX_COUNT = 10**12
MAX_DRAW = 10**6

def require(condition, message):
    if not condition:
        raise ValueError(message)

def scalar(value, name="value"):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} 必须是实数标量，不接受布尔、字符串、复数或对象")
    try:
        result = float(value)
    except (OverflowError, ValueError, TypeError) as error:
        raise ValueError(f"{name} 不能表示为有限浮点数") from error
    require(math.isfinite(result), f"{name} 必须有限")
    require(not (value != 0 and result == 0), f"{name} 正负非零输入转换时下溢")
    return result

def integer(value, name="count", minimum=0, maximum=MAX_COUNT):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} 必须是整数类型，不接受布尔、字符串或浮点计数")
    result = int(value)
    require(minimum <= result <= maximum, f"{name} 超出允许范围 [{minimum}, {maximum}]")
    return result

def array(value, name="array"):
    # 先看原始类型，避免 [True, 1] 或 ['0.5', '0.5'] 在转换后混过检查。
    if isinstance(value, np.ndarray):
        require(value.dtype.kind in "iuf", f"{name} 禁止布尔、字符串、复数或 object 数组")
    else:
        require(isinstance(value, (list, tuple)), f"{name} 必须是数组、列表或元组")
        def inspect(item):
            if isinstance(item, (list, tuple)):
                for child in item:
                    inspect(child)
            elif isinstance(item, np.ndarray):
                require(item.dtype.kind in "iuf", f"{name} 禁止布尔、字符串、复数或 object 数组")
                require(np.all(np.isfinite(item)), f"{name} 必须有限")
            else:
                scalar(item, name)
        inspect(value)
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} 不能转换为规则的有限数值数组") from error
    require(raw.dtype.kind in "iuf", f"{name} 禁止布尔、字符串、复数或 object 数组")
    require(np.all(np.isfinite(raw)), f"{name} 必须有限")
    return raw

def probability(value, name="probability"):
    p = scalar(value, name)
    require(0 <= value <= 1 and 0 <= p <= 1, f"{name} 必须在 [0,1]")
    return p

def probability_vector(values, name="probabilities"):
    raw = array(values, name)
    require(raw.ndim == 1 and raw.size > 0, f"{name} 必须是非空一维数组")
    require(np.all((raw >= 0) & (raw <= 1)), f"{name} 每项必须在 [0,1]")
    result = raw.astype(float)
    require(not np.any((raw > 0) & (result == 0)), f"{name} 正概率转换下溢")
    total = math.fsum(result.tolist())
    require(math.isfinite(total) and abs(total - 1) <= ATOL, f"{name} 总和必须为1；不自动归一化")
    return result

def count_table(values):
    raw = array(values, "counts")
    require(raw.shape == (2, 2), "counts 必须是 (2,2)，行=缺陷/合格，列=报警/不报警")
    require(raw.dtype.kind in "iu", "counts 必须使用整数类型")
    vals = [integer(v) for v in raw.flat]
    total = sum(vals)  # Python整数累加，不允许int64溢出后变成另一张表。
    require(0 < total <= MAX_COUNT, "总计数必须为正且不超过10^12")
    return np.array(vals, dtype=np.int64).reshape(2, 2)

def conditional_count(intersection, condition):
    k = integer(intersection, "intersection")
    n = integer(condition, "condition")
    require(k <= n, "交集计数不能超过条件计数")
    if n == 0:
        raise ZeroDivisionError("条件计数为0，经验条件概率未定义，不能填成0")
    p = k / n
    return probability(p, "conditional result")

def optional_ratio(k, n):
    k = integer(k); n = integer(n)
    require(k <= n, "交集计数不能超过条件计数")
    return None if n == 0 else conditional_count(k, n)

def summarize_table(values):
    t = count_table(values)
    a, b, c, d = [int(x) for x in t.flat]
    n = a + b + c + d
    return {"counts": t.tolist(), "n": n, "alarm_count": a+c,
            "p_defect": conditional_count(a+b, n), "p_alarm": conditional_count(a+c, n),
            "p_alarm_given_defect": optional_ratio(a, a+b),
            "p_alarm_given_clean": optional_ratio(c, c+d),
            "p_defect_given_alarm": optional_ratio(a, a+c),
            "p_defect_given_no_alarm": optional_ratio(b, b+d)}

def _safe_products(x, y):
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        z = x * y
    require(np.all(np.isfinite(z)), "乘积产生非有限中间量")
    require(not np.any((x > 0) & (y > 0) & (z == 0)), "正概率乘积下溢为0；请改用精确分数或对数实现")
    require(not np.any((z > 0) & (z < np.finfo(float).tiny)), "正概率乘积进入次正规范围；本教学实现保守拒绝")
    return z

def bayes(priors, likelihoods):
    """有限互斥完备假设；likelihoods[i]=P(证据|第i个假设)。"""
    p = probability_vector(priors, "priors")
    raw = array(likelihoods, "likelihoods")
    require(raw.ndim == 1 and raw.shape == p.shape, "likelihoods 必须与 priors 同形的一维数组")
    require(np.all((raw >= 0) & (raw <= 1)), "likelihoods 每项必须在 [0,1]")
    likelihoods = raw.astype(float)
    require(not np.any((raw > 0) & (likelihoods == 0)), "likelihoods 正概率转换下溢")
    weights = _safe_products(p, likelihoods)
    raw_evidence = math.fsum(weights.tolist())
    require(math.isfinite(raw_evidence) and 0 <= raw_evidence <= 1 + ATOL, "证据概率中间量无效")
    # Match the documented input sum tolerance. Only an upper roundoff excess
    # within ATOL is clipped for the reported probability. The actual weight sum
    # remains the posterior denominator, so the normalized update stays coherent.
    evidence = min(raw_evidence, 1.0)
    if raw_evidence == 0:
        raise ZeroDivisionError("模型中的证据概率为0，当前离散条件概率未定义")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        posterior = weights / raw_evidence
    require(np.all(np.isfinite(posterior)), "后验产生非有限值")
    require(not np.any((weights > 0) & (posterior == 0)), "后验正权重下溢；请改用精确分数或对数实现")
    posterior = probability_vector(posterior, "posterior")
    return {"posterior": posterior.tolist(), "evidence": evidence, "raw_evidence": raw_evidence,
            "evidence_roundoff_adjustment": evidence - raw_evidence, "weights": weights.tolist()}

def detector_model(prior=.01, sensitivity=.8, false_alarm=.05):
    p = probability(prior, "prior")
    s = probability(sensitivity, "sensitivity")
    f = probability(false_alarm, "false_alarm")
    # Flatten order = [D∩A, D∩A^c, D^c∩A, D^c∩A^c].
    weights = _safe_products(np.array([p,p,1-p,1-p]), np.array([s,1-s,f,1-f]))
    return probability_vector(weights, "joint probabilities")

def simulate_devices(n=100000, seed=SEED, prior=.01, sensitivity=.8, false_alarm=.05):
    n = integer(n, "n", 1, MAX_DRAW)
    seed = integer(seed, "seed", 0, 2**64-1)
    # Check all parameters and joint arithmetic before drawing any random values.
    detector_model(prior, sensitivity, false_alarm)
    p = probability(prior); s = probability(sensitivity); f = probability(false_alarm)
    rng = np.random.default_rng(seed)
    defect = rng.random(n) < p
    alarm = rng.random(n) < np.where(defect, s, f)
    a = int(np.count_nonzero(defect & alarm))
    b = int(np.count_nonzero(defect & ~alarm))
    c = int(np.count_nonzero(~defect & alarm))
    d = int(np.count_nonzero(~defect & ~alarm))
    require(a+b+c+d == n, "模拟四格计数不守恒")
    return summarize_table([[a,b],[c,d]])

def repeated_simulations(n, repeats=400, seed=SEED+1, prior=.01, sensitivity=.8, false_alarm=.05):
    n = integer(n, "n", 1, MAX_DRAW)
    repeats = integer(repeats, "repeats", 1, 10000)
    seed = integer(seed, "seed", 0, 2**64-1)
    weights = detector_model(prior, sensitivity, false_alarm)
    counts = np.random.default_rng(seed).multinomial(n, weights, size=repeats)
    require(counts.shape == (repeats,4), "模拟输出形状不对")
    require(np.all(counts >= 0) and np.all(counts.sum(axis=1) == n), "模拟输出计数不守恒")
    denominators = counts[:,0] + counts[:,2]
    estimates = [optional_ratio(int(k),int(m)) for k,m in zip(counts[:,0],denominators)]
    valid = sorted(x for x in estimates if x is not None)
    def empirical_quantile(q):
        if not valid:
            return None
        # 第ceil(qR)个次序统计量；这里q严格处于(0,1]。
        value = valid[max(0, math.ceil(q*len(valid))-1)]
        return probability(value, "empirical quantile")
    summary = {"n": n, "repeats": repeats, "seed": seed,
               "valid_runs": len(valid), "zero_alarm_runs": repeats-len(valid),
               "alarm_count_min": int(denominators.min()),
               "alarm_count_median": float(np.median(denominators)),
               "alarm_count_max": int(denominators.max()),
               "mean_available": None if not valid else math.fsum(valid)/len(valid),
               "q025_available": empirical_quantile(.025), "q50_available": empirical_quantile(.5),
               "q975_available": empirical_quantile(.975),
               "interpretation": "empirical spread across independent runs with positive alarm count; not a confidence interval"}
    return {"summary": summary, "counts": counts.tolist(), "posterior_estimates": estimates}

def independence_report(values):
    t = count_table(values)
    a,b,c,d = [int(x) for x in t.flat]
    n = a+b+c+d
    joint = Fraction(a,n)
    product_marginals = Fraction(a+b,n)*Fraction(a+c,n)
    determinant = a*d-b*c
    return {"counts":t.tolist(), "joint":str(joint), "product_marginals":str(product_marginals),
            "joint_minus_product":str(joint-product_marginals), "integer_cross_product_difference":determinant,
            "exact_independence": determinant == 0}

def load_data():
    # CSV是文本容器。先验证十进制词法和字段，再显式解析为整数。
    def parse_int(value, signed=False):
        text = value[1:] if signed and value.startswith("-") else value
        require(bool(text) and text.isascii() and text.isdigit(), "CSV整数格式错误")
        return int(value)
    with (BASE/'data/device_counts.csv').open(newline='',encoding='utf-8') as f:
        reader = csv.DictReader(f)
        require(reader.fieldnames == ['state','alarm','no_alarm'], "固定表CSV字段错误")
        rows = list(reader)
    require(len(rows)==2 and [r['state'] for r in rows]==['defect','clean'], "固定表行标签错误")
    table = count_table([[parse_int(r['alarm']),parse_int(r['no_alarm'])] for r in rows])
    require(table.tolist() == [[80,20],[495,9405]], "教学固定表已改变，请另存变体并重算答案")
    with (BASE/'data/conditional_independence.csv').open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f); require(reader.fieldnames==['regime','x','y','count'], "条件独立CSV字段错误")
        rows=list(reader)
    expected=[(h,x,y,c) for h,cs in [('high',[64,16,16,4]),('low',[4,16,16,64])]
              for (x,y),c in zip([(1,1),(1,0),(0,1),(0,0)],cs)]
    actual=[(r['regime'],parse_int(r['x']),parse_int(r['y']),parse_int(r['count'])) for r in rows]
    require(actual==expected, "条件独立固定数据已改变")
    groups={h:count_table(np.array([c for hh,x,y,c in actual if hh==h],dtype=np.int64).reshape(2,2))
            for h in ['high','low']}
    with (BASE/'data/uncorrelated.csv').open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f); require(reader.fieldnames==['x','y','count'], "不相关CSV字段错误")
        rows=list(reader)
    triples=[(parse_int(r['x'],True),parse_int(r['y'],True),parse_int(r['count'])) for r in rows]
    require(triples==[(-1,1,1),(0,0,1),(1,1,1)], "不相关固定数据已改变")
    return table,groups,triples

def fraction_checks():
    f=Fraction
    prior=f(1,100); sensitivity=f(4,5); false_alarm=f(1,20)
    evidence=prior*sensitivity+(1-prior)*false_alarm
    posterior=prior*sensitivity/evidence
    require(evidence==f(23,400) and posterior==f(16,115), "精确主表检查失败")
    table,groups,triples=load_data()
    require(f(80,575)==posterior and f(20,9425)==f(4,1885), "分母核对失败")
    # 独立的整数枚举：81张0..2四格表，排除全0；不用生产公式作参考。
    tested=0
    for cells in product(range(3),repeat=4):
        n=sum(cells)
        if n==0: continue
        a,b,c,d=cells
        result=summarize_table(np.array(cells,dtype=np.int64).reshape(2,2))
        if a+c:
            require(result['p_defect_given_alarm']==float(f(a,a+c)), "枚举后验不一致")
        else:
            require(result['p_defect_given_alarm'] is None, "空分母必须None")
        if a+b and c+d and a+c:
            ref=bayes([float(f(a+b,n)),float(f(c+d,n))],[float(f(a,a+b)),float(f(c,c+d))])
            require(abs(ref['posterior'][0]-float(f(a,a+c)))<=ATOL, "Bayes与计数枚举不一致")
        ind=independence_report(np.array(cells,dtype=np.int64).reshape(2,2))
        expected=(f(a,n)==f(a+b,n)*f(a+c,n))
        require(ind['exact_independence']==expected, "交叉乘积检查不一致")
        tested+=1
    total=sum(c for x,y,c in triples)
    mx=sum(f(x*c,total) for x,y,c in triples); my=sum(f(y*c,total) for x,y,c in triples)
    cov=sum(f(c,total)*(x-mx)*(y-my) for x,y,c in triples)
    require(mx==0 and my==f(2,3) and cov==0, "不相关反例精确检查失败")
    # 事件X=0与Y=0的交集1/3，而边际乘积1/9，故仍不独立。
    require(f(1,3)!=f(1,3)*f(1,3), "依赖反例失败")
    return {"prior":str(prior),"evidence":str(evidence),"posterior_alarm":str(posterior),
            "posterior_no_alarm":"4/1885","odds_prior":"1/99","likelihood_ratio":"16",
            "odds_posterior":"16/99","enumerated_integer_tables":tested,
            "group_high":independence_report(groups['high']),"group_low":independence_report(groups['low']),
            "pooled":independence_report(groups['high']+groups['low']),
            "zero_covariance_example":{"mean_x":str(mx),"mean_y":str(my),"covariance":str(cov),
                                       "intersection":"1/3","marginal_product":"1/9"},
            "selection_example":{"unconditional_joint":"1/4","unconditional_product":"1/4",
                                 "conditional_joint":"1/2","conditional_product":"1/4"}}

def boundary_checks():
    cases=[]
    def reject(name, function):
        try: function()
        except (ValueError, ZeroDivisionError) as error:
            cases.append({"case":name,"status":"rejected_as_expected","message":str(error)})
        else: raise RuntimeError(f"应该拒绝却未拒绝：{name}")
    bad=[True,"0.5",complex(.5,0),float('nan'),float('inf'),-0.01,1.01,object()]
    for i,value in enumerate(bad): reject(f"probability_bad_{i}",lambda value=value:probability(value))
    for i,value in enumerate([[True,.5],[".5",".5"],np.array([.5,.5],dtype=object),[np.nan,0],[[.5,.5]],[],[.3,.3],[-.1,1.1]]):
        reject(f"vector_bad_{i}",lambda value=value:probability_vector(value))
    for i,value in enumerate([[[True,0],[0,1]],[[1.,0.],[0.,1.]],[["1",0],[0,1]],np.ones((2,2),dtype=object),[[1,-1],[0,1]],[[0,0],[0,0]],[1,2,3,4],[[1,2],[3]],[[1,np.inf],[0,1]],[[MAX_COUNT,1],[0,0]]]):
        reject(f"table_bad_{i}",lambda value=value:count_table(value))
    for i,args in enumerate([(0,0),(2,1),(True,2),("1",2),(1,2.),(-1,2)]):
        reject(f"ratio_bad_{i}",lambda args=args:conditional_count(*args))
    reject("bayes_zero_evidence",lambda:bayes([.5,.5],[0,0]))
    reject("bayes_shape",lambda:bayes([.5,.5],[.2]))
    reject("bayes_bool_likelihood",lambda:bayes([.5,.5],[True,.2]))
    reject("bayes_object_likelihood",lambda:bayes([.5,.5],np.array([.2,.4],dtype=object)))
    reject("bayes_nonfinite_likelihood",lambda:bayes([.5,.5],[float('nan'),.2]))
    reject("bayes_likelihood_gt_one",lambda:bayes([.5,.5],[1.1,.2]))
    reject("bayes_positive_product_underflow",lambda:bayes([1e-300,1.],[1e-300,0.]))
    reject("bayes_subnormal_product",lambda:bayes([1e-300,1.],[1e-20,.5]))
    for i,args in enumerate([(True,0), (0,0), (2.,0), (10,-1), (10,True),(MAX_DRAW+1,0)]):
        reject(f"simulate_bad_{i}",lambda args=args:simulate_devices(*args))
    reject("repeat_zero",lambda:repeated_simulations(20,repeats=0))
    reject("repeat_bool",lambda:repeated_simulations(20,repeats=True))
    reject("detector_underflow",lambda:detector_model(1e-300,1e-300,.1))
    # Legitimate boundaries are recorded alongside rejections.
    require(conditional_count(0,5)==0 and conditional_count(5,5)==1,"合法比例边界失败")
    cases.append({"case":"ratio_endpoints","status":"accepted_as_expected"})
    require(bayes([0,1],[1,.2])['posterior']==[0,1],"零先验边界失败")
    cases.append({"case":"zero_prior","status":"accepted_as_expected"})
    require(summarize_table([[0,5],[0,5]])['p_defect_given_alarm'] is None,"空报警必须None")
    cases.append({"case":"no_observed_alarm_returns_none","status":"accepted_as_expected"})
    zero=repeated_simulations(5,10,SEED,prior=.1,sensitivity=0,false_alarm=0)['summary']
    require(zero['zero_alarm_runs']==10 and zero['q50_available'] is None,"零报警重复结果错误")
    cases.append({"case":"all_zero_alarm_repeats","status":"accepted_as_expected"})
    end=simulate_devices(10,SEED,prior=1,sensitivity=1,false_alarm=0)
    require(end['counts']==[[10,0],[0,0]] and end['p_alarm_given_clean'] is None,"端点模拟错误")
    cases.append({"case":"deterministic_endpoint_model","status":"accepted_as_expected"})
    require(independence_report([[0,0],[5,5]])['exact_independence'],"退化事件独立性错误")
    cases.append({"case":"probability_zero_event_independent","status":"accepted_as_expected"})
    first = bayes([14/67,11/67,11/67,12/67,19/67], [.875,.875,.3125,.125,.5625])["posterior"]
    repeated = bayes(first, [1]*5)
    require(np.allclose(repeated["posterior"], first, rtol=0, atol=1e-15), "全1似然顺序更新不应改变后验")
    cases.append({"case":"sequential_unit_likelihood_roundoff","status":"accepted_as_expected"})
    require(repeated["raw_evidence"] > 1 and repeated["evidence"] == 1 and
            abs(repeated["evidence_roundoff_adjustment"]) <= ATOL,
            "证据上界舍入修正必须透明且有界")
    cases.append({"case":"bounded_evidence_roundoff_adjustment","status":"accepted_as_expected"})
    try:
        bayes([.5,.50000000001],[1,1])
    except ValueError:
        cases.append({"case":"prior_sum_beyond_tolerance","status":"rejected_as_expected"})
    else:
        raise ValueError("实质错误的总概率仍必须拒绝")
    return cases

def run_experiment(write_outputs=True):
    require(type(write_outputs) is bool,"write_outputs 必须是布尔控制开关")
    table,groups,triples=load_data()
    fixed=summarize_table(table)
    update=bayes([.01,.99],[.8,.05])
    exact=fraction_checks()
    simulation=simulate_devices()
    repeats=[repeated_simulations(n,400,SEED+1+i) for i,n in enumerate([200,2000,20000])]
    rare=repeated_simulations(5,400,SEED+10)
    curves=[]
    for p in [.001,.01,.05,.1,.5]:
        b=bayes([p,1-p],[.8,.05]); curves.append({"prior":p,"evidence":b['evidence'],"posterior":b['posterior'][0]})
    require(abs(fixed['p_defect_given_alarm']-16/115)<ATOL,"固定表主答案错误")
    require(abs(update['posterior'][0]-16/115)<ATOL,"Bayes主答案错误")
    report={"unit":"018","synthetic_only":True,"seed":SEED,"numpy":np.__version__,
            "fixed_table":fixed,"bayes":update,"exact_checks":exact,"single_simulation":simulation,
            "repeated_summaries":[x['summary'] for x in repeats],"rare_evidence_summary":rare['summary'],
            "base_rate_sweep":curves,"boundary_checks":boundary_checks(),
            "limitations":["Known synthetic probabilities, not estimated device performance",
                           "Finite repetition spread is not a confidence interval",
                           "Conditioning is not intervention; no causal effect is estimated"]}
    # Reject any nonfinite JSON numeric output; None deliberately denotes undefined ratios.
    serialized=json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n"
    if write_outputs:
        out=BASE/'outputs'; out.mkdir(exist_ok=True)
        (out/'experiment-report.json').write_text(serialized,encoding='utf-8')
        with (out/'base-rate-sweep.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=['prior','evidence','posterior']);w.writeheader();w.writerows(curves)
        with (out/'repeated-simulations.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f);w.writerow(['n','run','defect_alarm','defect_no_alarm','clean_alarm','clean_no_alarm','posterior_available'])
            for experiment in repeats+[rare]:
                for i,(counts,value) in enumerate(zip(experiment['counts'],experiment['posterior_estimates']),1):
                    w.writerow([experiment['summary']['n'],i,*counts,'' if value is None else value])
    return report

if __name__ == '__main__':
    result=run_experiment()
    print(json.dumps({"fixed_posterior":result['fixed_table']['p_defect_given_alarm'],
                      "single_simulation":result['single_simulation'],
                      "repeated_summaries":result['repeated_summaries'],
                      "rare_evidence_summary":result['rare_evidence_summary'],
                      "exact_integer_tables":result['exact_checks']['enumerated_integer_tables'],
                      "boundary_cases":len(result['boundary_checks']),"status":"passed"},ensure_ascii=False,indent=2,allow_nan=False))
