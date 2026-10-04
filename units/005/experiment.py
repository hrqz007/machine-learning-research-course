"""第 5 讲：用数组保持样本对应关系。全部数据为合成教学数据。"""
from pathlib import Path
import argparse
import json
import platform
import timeit
import numpy as np

HERE = Path(__file__).resolve().parent


def as_real_array(values, name):
    """只接受实数数组；不把文字、布尔或复数悄悄转成实数。"""
    # 先检查尚未转换的列表/元组，避免 True 与数字混合后被静默变成 1。
    # 遇到内层列表就重复同一检查；已有 ndarray 只能检查当前 dtype，
    # 无法恢复它过去由哪些 Python 类型创建。
    def reject_boolean_leaves(value):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must not contain booleans")
        if isinstance(value, (list, tuple)):
            for child in value:
                reject_boolean_leaves(child)
    reject_boolean_leaves(values)
    arr = np.asarray(values)
    if arr.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numbers")
    arr = np.asarray(arr, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite numbers")
    return arr


def predict_loop(x, weight, bias):
    """单特征循环基准；输入和输出都约定为一维。"""
    x = as_real_array(x, "x")
    if x.ndim != 1 or x.size == 0:
        raise ValueError("x must have shape (n,) with n > 0")
    return np.array([weight * float(value) + bias for value in x])


def predict_array(x, weight, bias):
    x = as_real_array(x, "x")
    if x.ndim != 1 or x.size == 0:
        raise ValueError("x must have shape (n,) with n > 0")
    return weight * x + bias


def mae_checked(prediction, target):
    """每个样本恰好一个误差；先拒绝意外广播，再计算 MAE。"""
    prediction = as_real_array(prediction, "prediction")
    target = as_real_array(target, "target")
    if prediction.ndim != 1 or target.ndim != 1:
        raise ValueError("prediction and target must both have shape (n,)")
    if prediction.shape != target.shape or prediction.size == 0:
        raise ValueError("prediction and target must have equal non-empty shapes")
    return float(np.mean(np.abs(prediction - target)))


def weighted_predict(X, weights, bias):
    """X: (n,d)，weights: (d,)，输出: (n,)。不修改输入。"""
    X = as_real_array(X, "X")
    weights = as_real_array(weights, "weights")
    b = as_real_array(bias, "bias")
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must have shape (n,d) with n > 0 and d > 0")
    if weights.ndim != 1 or weights.shape != (X.shape[1],):
        raise ValueError("weights must have shape (d,)")
    if b.ndim != 0:
        raise ValueError("bias must be a scalar")
    contributions = X * weights
    prediction = contributions.sum(axis=1) + float(b)
    assert prediction.shape == (X.shape[0],)
    return prediction


def weighted_loop(X, weights, bias):
    """独立写出加权和，作为小数据核验基准。"""
    result = []
    for row in X:
        total = float(bias)
        for value, weight in zip(row, weights):
            total += float(value) * float(weight)
        result.append(total)
    return np.array(result, dtype=np.float64)


def require_value_error(call):
    try:
        call()
    except ValueError:
        return True
    raise AssertionError("Expected ValueError was not raised")


def timing_experiment():
    """同一预测与 MAE 工作量；输入准备和正确性检查放在计时外。"""
    records = []
    for n in (4, 100, 10000, 100000):
        x_array = np.linspace(20.0, 120.0, n)
        y_array = 2.0 * x_array + 10.0
        x_list, y_list = x_array.tolist(), y_array.tolist()

        def loop_work():
            predictions = [2.0 * x + 10.0 for x in x_list]
            return sum(abs(p - y) for p, y in zip(predictions, y_list)) / n

        def array_work():
            predictions = 2.0 * x_array + 10.0
            return float(np.abs(predictions - y_array).mean())

        assert np.isclose(loop_work(), array_work(), rtol=0.0, atol=1e-12)
        number = 100 if n <= 100 else 5
        loop_work(); array_work()  # 预热，不计入样本
        loop_samples = timeit.repeat(loop_work, number=number, repeat=5)
        array_samples = timeit.repeat(array_work, number=number, repeat=5)
        records.append({
            "n": n, "repeats": 5, "calls_per_repeat": number,
            "loop_ms_per_call": float(np.median(loop_samples) * 1000 / number),
            "array_ms_per_call": float(np.median(array_samples) * 1000 / number),
            "input_preparation_included": False,
        })
    return records


def run_checks(include_timing=True):
    checks = []
    def passed(name):
        checks.append({"name": name, "passed": True})

    x = np.array([50, 60, 80, 90], dtype=np.float64)
    y = np.array([110, 130, 170, 190], dtype=np.float64)
    rules = {"A": (2.0, 0.0), "B": (2.0, 10.0), "C": (1.5, 40.0)}
    scores = {}
    for name, (weight, bias) in rules.items():
        expected = predict_loop(x, weight, bias)
        actual = predict_array(x, weight, bias)
        assert actual.shape == expected.shape == (4,)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)
        scores[name] = mae_checked(actual, y)
    assert scores == {"A": 10.0, "B": 0.0, "C": 7.5}
    passed("house_predictions_loop_array_and_mae")

    # B 已在先前课程按训练数据选定；以下只复核既定测试结果。
    test_x = np.array([65.0, 85.0]); test_y = np.array([144.0, 174.0])
    test_score = mae_checked(predict_array(test_x, 2.0, 10.0), test_y)
    baseline_score = mae_checked(np.full(test_y.shape, y.mean()), test_y)
    assert test_score == 5.0 and baseline_score == 15.0
    passed("fixed_test_b_5_and_training_mean_baseline_15")

    X = np.array([[50.0, 1.0], [60.0, 2.0], [80.0, 1.0]])
    weights = np.array([2.0, 5.0]); bias = 10.0
    original = X.copy()
    result = weighted_predict(X, weights, bias)
    assert result.shape == (3,)
    np.testing.assert_allclose(result, [115, 140, 175], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(result, weighted_loop(X, weights, bias), rtol=0.0, atol=1e-12)
    np.testing.assert_array_equal(X, original)
    passed("non_square_3_by_2_prediction_and_no_mutation")
    single = weighted_predict(X[:1], weights, bias)
    assert single.shape == (1,) and single[0] == 115.0
    passed("single_sample_retains_batch_axis")
    one_feature = weighted_predict(X[:, :1], [2.0], 10.0)
    np.testing.assert_array_equal(one_feature, [110, 130, 170])
    passed("single_feature_retains_feature_axis")
    assert X.sum(axis=0).shape == (2,)
    assert X.sum(axis=1).shape == (3,)
    np.testing.assert_array_equal(X.sum(axis=0), [190, 4])
    np.testing.assert_array_equal(X.sum(axis=1), [51, 62, 81])
    np.testing.assert_array_equal(X.mean(axis=0, keepdims=True).shape, [1, 2])
    passed("axis_reductions_and_keepdims")

    # 这里故意造错：广播能运行，但样本配对已错。
    bad_difference = result[:, None] - result
    assert bad_difference.shape == (3, 3)
    assert float(np.abs(bad_difference).mean()) == 80.0 / 3.0
    require_value_error(lambda: mae_checked(result[:, None], result))
    require_value_error(lambda: mae_checked([1.0, 2.0], [1.0]))
    passed("broadcast_bug_reproduced_and_contract_rejects_it")

    require_value_error(lambda: weighted_predict(X[0], weights, bias))
    require_value_error(lambda: weighted_predict(X, [2.0], bias))
    require_value_error(lambda: weighted_predict(X, weights, [10.0]))
    require_value_error(lambda: weighted_predict(np.empty((0, 2)), weights, bias))
    require_value_error(lambda: weighted_predict(np.array([[np.nan, 2]]), weights, bias))
    require_value_error(lambda: weighted_predict(np.array([[np.inf, 2]]), weights, bias))
    require_value_error(lambda: weighted_predict([["50", "1"]], weights, bias))
    require_value_error(lambda: mae_checked([], []))
    require_value_error(lambda: weighted_predict([[True, False]], weights, bias))
    require_value_error(lambda: weighted_predict([[True, 1]], weights, bias))
    require_value_error(lambda: weighted_predict([[np.bool_(True), 1.0]], weights, bias))
    require_value_error(lambda: weighted_predict([[50.0, 1.0]], [True, 5.0], bias))
    require_value_error(lambda: mae_checked([True, 2], [1, 2]))
    require_value_error(lambda: mae_checked([1, 2], [1, np.bool_(False)]))
    passed("invalid_dimensions_empty_nonfinite_strings_booleans_rejected")

    a = np.array([50.0, 60.0, 80.0, 90.0])
    view = a[1:3]; copy = a[1:3].copy(); selected = a[[1, 2]]
    assert np.shares_memory(a, view)
    assert not np.shares_memory(a, copy)
    assert not np.shares_memory(a, selected)
    view[0] = 600
    assert a[1] == 600 and copy[0] == selected[0] == 60
    passed("slice_view_explicit_copy_and_advanced_index_copy")

    narrow = np.array([120], dtype=np.int8)
    wrapped = narrow + np.array([10], dtype=np.int8)
    safe = narrow.astype(np.int64) + 10
    assert int(wrapped[0]) == -126 and int(safe[0]) == 130
    truncation = np.array([1, 2], dtype=np.int64); truncation[0] = 1.9
    assert truncation[0] == 1
    v32 = np.array([100000000.0, 1.0, -100000000.0], dtype=np.float32)
    sum32 = float(v32.sum()); sum64 = float(v32.sum(dtype=np.float64))
    assert sum32 == 0.0 and sum64 == 1.0
    rounded = np.array([100000001.0], dtype=np.float32).astype(np.float64)
    assert rounded[0] == 100000000.0
    assert not np.array_equal(np.array([0.1 + 0.2]), np.array([0.3]))
    np.testing.assert_allclose([0.1 + 0.2], [0.3], rtol=0.0, atol=1e-12)
    passed("dtype_overflow_truncation_accumulation_and_tolerance")

    finite_scores = {"training_mae_wan": scores, "test_b_mae_wan": test_score,
                     "test_baseline_mae_wan": baseline_score,
                     "weighted_prediction_wan": result.tolist(),
                     "broadcast_wrong_mae_wan": float(np.abs(bad_difference).mean()),
                     "sum_float32": sum32, "sum_float64_accumulator": sum64}
    return {"unit": "005", "status": "passed", "python": platform.python_version(),
            "numpy": np.__version__, "checks": checks, "results": finite_scores,
            "timing": timing_experiment() if include_timing else [],
            "limitations": ["CPU and offline only", "Timing is machine-specific, not a promised speedup", "Anaconda installation and Jupyter browser UI are not tested by this script"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="test-result.json", help="输出 JSON 文件名")
    parser.add_argument("--skip-timing", action="store_true")
    args = parser.parse_args()
    result = run_checks(include_timing=not args.skip_timing)
    output = Path(args.output)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": len(result["checks"]),
                      "results": result["results"], "output": output.name}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
