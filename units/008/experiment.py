"""第008讲：函数代数与数学表达的离线核验。

直接运行 python experiment.py。只用标准库，不访问网络，不训练模型。
Fraction 用于小整数与有理数精确计算；math 用于有明确容差的浮点对数核验。
有限检查不代替讲义中的一般证明。程序定义的辅助函数可先作为已准备工具使用。
"""
from pathlib import Path
from fractions import Fraction
import csv
import json
import math

BASE = Path(__file__).resolve().parent
REL_TOL = 1e-12
ABS_TOL = 1e-12


def f(x):
    """受限预测函数：只有闭区间 [0, 4] 内有限数值才是合法输入。"""
    if not 0 <= x <= 4 or not math.isfinite(x):
        raise ValueError('f 的定义域是有限实数区间 [0, 4]')
    return 2 * x + 1


def g(t):
    """平方数值。此函数不是声称真实分钟数平方后仍是分钟。"""
    return t * t


def cancelled_original(x):
    """原函数 (x²-1)/(x-1)：保留 x≠1，即使化简式能在 1 取值。"""
    if x == 1:
        raise ValueError('原式分母在 x = 1 时为零')
    return (x * x - 1) / (x - 1)


def log_real(u, base=math.e):
    """本实验实数对数接口：拒绝非法定义域和非有限机器输入。

    这里只核验有限、适中数值；不承诺所有合法实数都能由 float 稳定表示。
    """
    if not math.isfinite(u) or not math.isfinite(base):
        raise ValueError('实验只接收有限数值')
    if u <= 0 or base <= 0 or base == 1:
        raise ValueError('实数对数要求真数 > 0，底数 > 0 且底数 != 1')
    if base == 2:
        return math.log2(u)
    return math.log(u) / math.log(base)


def is_close(a, b):
    return math.isclose(a, b, rel_tol=REL_TOL, abs_tol=ABS_TOL)


def read_observations():
    """字符串直接变成分数，不先经过浮点近似。"""
    with (BASE / 'data' / 'observations.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    return [{'sample_id': r['sample_id'], 'x': Fraction(r['x']),
             'y': Fraction(r['observed_y']),
             'expected': Fraction(r['expected_prediction'])} for r in rows]


def error_summary(xs, ys):
    """Python 位置 j = 0,...,n-1，对应数学索引 i = j+1。"""
    if not xs or len(xs) != len(ys):
        raise ValueError('输入和观测必须同长度且至少有一个样本')
    predictions = []
    residuals = []
    squares = []
    for j in range(len(xs)):
        prediction = f(xs[j])
        residual = prediction - ys[j]
        predictions.append(prediction)
        residuals.append(residual)
        squares.append(residual * residual)
    squared_sum = sum(squares, Fraction(0))
    return {'predictions': predictions, 'residuals': residuals, 'squares': squares,
            'residual_sum': sum(residuals, Fraction(0)), 'squared_sum': squared_sum,
            'mse': squared_sum / len(xs)}


def alternative(x):
    """五个整数点与 f 一致，但点之间不同的另一条函数。"""
    if not 0 <= x <= 4:
        raise ValueError('只比较共同定义域 [0, 4]')
    product = Fraction(1)
    for j in range(5):
        product *= x - j
    return 2 * x + 1 + product


def quantifier_demo():
    """逐项穷尽这个有限集合；注意两个循环次序对应不同承诺。"""
    xs = [0, 1, 2]
    bs = [0, -2, -4]
    witnesses = []
    each_x_has_b = True
    for x in xs:
        matching = [b for b in bs if 2 * x + b == 0]
        witnesses.append({'x': x, 'valid_b': matching})
        if not matching:
            each_x_has_b = False
    shared_b = []
    for b in bs:
        works_for_all = True
        for x in xs:
            if 2 * x + b != 0:
                works_for_all = False
        if works_for_all:
            shared_b.append(b)
    return {'forall_x_exists_b': each_x_has_b,
            'exists_b_forall_x': bool(shared_b),
            'witnesses': witnesses, 'shared_b': shared_b}


def log_identity_checks():
    """合法输入上的有限数值验证。一般规则的理由仍是讲义中的推导。"""
    bases = [0.5, 2.0, 10.0, math.e]
    us = [0.125, 0.5, 1.0, 2.0, 3.0, 8.0]
    vs = [0.25, 1.0, 2.0, 5.0]
    powers = [-2, 0, 0.5, 1, 3]
    checks = []
    def compare(name, left, right):
        assert is_close(left, right), (name, left, right)
        checks.append({'identity': name, 'absolute_difference': abs(left - right)})
    for base in bases:
        for u in us:
            for v in vs:
                compare('product', log_real(u*v, base), log_real(u, base)+log_real(v, base))
                compare('quotient', log_real(u/v, base), log_real(u, base)-log_real(v, base))
            for r in powers:
                compare('power', log_real(u**r, base), r*log_real(u, base))
            compare('inverse', base**log_real(u, base), u)
            compare('change_base', log_real(u, base), math.log(u)/math.log(base))
    return {'count': len(checks), 'all_passed': True,
            'max_absolute_difference': max(c['absolute_difference'] for c in checks),
            'relative_tolerance': REL_TOL, 'absolute_tolerance': ABS_TOL,
            'bases': bases, 'finite_grid_only': True}


def expect_rejection(name, action):
    """预期的拒绝应明确出现 ValueError；不把所有异常静默吞掉。"""
    try:
        action()
    except ValueError:
        return name
    raise AssertionError('本应拒绝却没有拒绝：' + name)


def domain_checks():
    passed = []
    for value in [-1, Fraction(-1, 1000), Fraction(4001, 1000), 5]:
        passed.append(expect_rejection('f_outside_' + str(value), lambda v=value: f(v)))
    assert f(Fraction(0)) == 1 and f(Fraction(4)) == 9
    passed.append('f_closed_endpoints_accepted')
    assert g(f(Fraction(1))) == 9 and f(g(Fraction(1))) == 3
    passed.append('composition_order_distinct')
    assert g(f(Fraction(3))) == 49
    passed.append(expect_rejection('reverse_composition_intermediate_invalid', lambda: f(g(Fraction(3)))))
    assert f(g(Fraction(-2))) == 9 and f(g(Fraction(2))) == 9
    passed.append('reverse_composition_endpoints')
    passed.append(expect_rejection('cancelled_hole_preserved', lambda: cancelled_original(Fraction(1))))
    for x in [Fraction(-2), Fraction(0), Fraction(1,2), Fraction(2)]:
        assert cancelled_original(x) == x + 1
    passed.append('cancellation_exact_on_original_domain')
    for u, base in [(7,1), (1,0), (1,-2), (0,2), (-4,2)]:
        passed.append(expect_rejection(f'log_invalid_{u}_{base}', lambda a=u,b=base: log_real(a,b)))
    for u, base in [(math.nan,2), (math.inf,2), (2,math.inf)]:
        passed.append(expect_rejection('log_nonfinite', lambda a=u,b=base: log_real(a,b)))
    passed.append(expect_rejection('empty_mean_rejected', lambda: error_summary([], [])))
    passed.append(expect_rejection('length_mismatch_rejected', lambda: error_summary([Fraction(1)], [])))
    assert log_real(8, Fraction(1,2)) == -3
    passed.append('base_between_zero_and_one_accepted')
    assert math.prod([]) == 1 and sum([]) == 0
    passed.append('empty_sum_and_product_conventions')
    return passed


def run_experiment(write_outputs=True):
    rows = read_observations()
    xs = [r['x'] for r in rows]
    ys = [r['y'] for r in rows]
    summary = error_summary(xs, ys)
    assert summary['predictions'] == [r['expected'] for r in rows]
    assert summary['residuals'] == [0, 1, -1, 1, 0]
    assert summary['squared_sum'] == 3 and summary['mse'] == Fraction(3,5)
    # 文字的乘2加1、公式、CSV对应表、实际执行在共同输入上相互核对。
    half_input = Fraction(3,2)
    assert f(half_input) == 4
    assert all(alternative(x) == f(x) for x in xs)
    counter_x = Fraction(1,2)
    counter_difference = alternative(counter_x) - f(counter_x)
    assert counter_difference == Fraction(105,32)
    assert alternative(counter_x) == Fraction(169,32)
    quantifiers = quantifier_demo()
    assert quantifiers['forall_x_exists_b'] and not quantifiers['exists_b_forall_x']
    odd_sums = []
    for n in range(1, 101):
        total = sum(2*i-1 for i in range(1,n+1))
        assert total == n*n
        odd_sums.append({'n': n, 'sum': total, 'square': n*n})
    # 错误的“对数和拆开”在合法输入上被一个精确值反例推翻。
    wrong_log_left = log_real(1+1, 2)
    wrong_log_right = log_real(1,2) + log_real(1,2)
    assert wrong_log_left == 1 and wrong_log_right == 0
    # log(x²) 在 x=-2 合法，但 2log(x) 不合法；正确的是 2log(|x|)。
    assert log_real((-2)**2,2) == 2*log_real(abs(-2),2)
    exact_tenth = Fraction('0.1')
    float_tenth = Fraction(0.1)
    assert exact_tenth == Fraction(1,10) and float_tenth != exact_tenth
    report = {
        'unit': '008', 'task': 'function_algebra_and_proof_audit',
        'synthetic_rows': len(rows), 'domain': '[0,4]',
        'predictions': [str(x) for x in summary['predictions']],
        'residuals': [str(x) for x in summary['residuals']],
        'residual_sum': str(summary['residual_sum']),
        'sum_of_squares': str(summary['squared_sum']),
        'square_of_sum': str(summary['residual_sum']**2),
        'mse_exact': str(summary['mse']),
        'four_representations': {'extra_input': '3/2', 'prediction': str(f(half_input)), 'matched': True},
        'composition_at_one': {'g_after_f': 9, 'f_after_g': 3},
        'quantifiers': quantifiers,
        'finite_grid_counterexample': {'grid_matches': True, 'x': '1/2',
            'f': str(f(counter_x)), 'alternative': str(alternative(counter_x)),
            'difference': str(counter_difference)},
        'log_identities': log_identity_checks(),
        'false_log_sum': {'left': wrong_log_left, 'right': wrong_log_right},
        'fraction_construction': {'decimal_string': str(exact_tenth), 'float_input': str(float_tenth)},
        'domain_checks': domain_checks(),
        'odd_sum_check': {'tested_n': [1,100], 'count': 100, 'last_sum': odd_sums[-1]['sum'],
                          'general_proof': 'See lecture induction argument; finite checks are not that proof.'},
        'limitations': ['synthetic data only', 'no model training or deployment claim',
            'logarithm checks use a finite moderate-value grid and explicit tolerances',
            'real exponent construction not proved in this unit',
            'small exact computations do not validate arbitrary numerical implementations']
    }
    if write_outputs:
        out = BASE/'outputs';out.mkdir(exist_ok=True)
        (out/'experiment_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        with (out/'four_representations.csv').open('w',encoding='utf-8',newline='') as stream:
            w=csv.writer(stream);w.writerow(['sample_id','math_index_i','python_index_j','x','observed_y','prediction','residual','squared_residual'])
            for j,r in enumerate(rows):w.writerow([r['sample_id'],j+1,j,str(r['x']),str(r['y']),str(summary['predictions'][j]),str(summary['residuals'][j]),str(summary['squares'][j])])
    return report


def print_report(report):
    print('合法输入范围：[0,4]；四种表达新增输入 3/2，预测 = 4')
    print('预测：', ', '.join(report['predictions']))
    print('带方向差：', ', '.join(report['residuals']))
    print('平方差之和：', report['sum_of_squares'], '；和的平方：', report['square_of_sum'])
    print('精确平均平方差：', report['mse_exact'])
    print('复合顺序在 x=1：', report['composition_at_one'])
    print('每个 x 有一个 b：', report['quantifiers']['forall_x_exists_b'])
    print('存在同一个 b 适合全部 x：', report['quantifiers']['exists_b_forall_x'])
    print('五点都相同，在 x=1/2 的精确差：', report['finite_grid_counterexample']['difference'])
    print('合法对数有限数值核验：', report['log_identities']['count'], '项通过')
    print('定义域与边界核验：', len(report['domain_checks']), '项通过')
    print('奇数和精确核验 n=1 至 100：通过；一般证明仍见讲义归纳步骤')
    print('脚本核验通过；不把有限测试当成普遍证明。')


if __name__ == '__main__':
    print_report(run_experiment())
