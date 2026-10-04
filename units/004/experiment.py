"""第四讲：函数化的预测实验。标准库、CPU、离线、手工合成数据。

从任意工作目录运行 python <本文件路径>。默认读取脚本旁的 data/，
写 outputs/result.json；--data-dir 和 --output-dir 可指定其他目录。
计算函数不读写文件；run_experiment 负责读写；导入本模块不会启动实验。
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def require_finite_number(value, name):
    """接受普通 int/float（不含 bool），返回可用的有限 float。"""
    if type(value) not in (int, float):
        raise TypeError(name + " must be an ordinary int or float, not bool/text")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(name + " is outside the supported float range") from error
    if not math.isfinite(number):
        raise ValueError(name + " must be finite")
    return number


def predict_one(area, weight, bias):
    """输入面积和参数，返回预测万元；不读取外部权重、不修改输入。"""
    area = require_finite_number(area, "area")
    weight = require_finite_number(weight, "weight")
    bias = require_finite_number(bias, "bias")
    prediction = weight * area + bias
    return require_finite_number(prediction, "prediction result")


def mean_absolute_error(predictions, labels):
    """非空等长 list/tuple，元素为有限普通数值；位置必须已经配对。

    使用易追踪的逐项求和。差值或累计溢出时明确拒绝；不是任意精度算法。
    """
    if type(predictions) not in (list, tuple) or type(labels) not in (list, tuple):
        raise TypeError("predictions and labels must be lists or tuples")
    if len(predictions) == 0:
        raise ValueError("at least one prediction is required")
    if len(predictions) != len(labels):
        raise ValueError("predictions and labels must have equal length")
    total = 0.0
    for index in range(len(predictions)):
        prediction = require_finite_number(predictions[index], "prediction[" + str(index) + "]")
        label = require_finite_number(labels[index], "label[" + str(index) + "]")
        error = require_finite_number(abs(prediction - label), "absolute error")
        total = require_finite_number(total + error, "error total")
    return require_finite_number(total / len(predictions), "MAE result")


def schema_fields(kind):
    """每次返回新列表，避免调用者改坏共享的模式配置。"""
    if kind == "train":
        return ["id", "area_m2", "price_wan"]
    if kind == "test_inputs":
        return ["id", "area_m2"]
    if kind == "test_labels":
        return ["id", "price_wan"]
    if kind == "predictions":
        return ["id", "prediction_wan"]
    raise ValueError("unknown table kind: " + str(kind))


def validate_records(records, kind):
    """严格检查内存记录，返回新记录；CSV 之外的调用也不能绕过契约。"""
    fields = schema_fields(kind)
    if type(records) is not list:
        raise TypeError(kind + " records must be a list")
    if len(records) == 0:
        raise ValueError(kind + " requires at least one record")
    result = []
    seen_ids = []
    for index in range(len(records)):
        row = records[index]
        context = kind + " record " + str(index + 1)
        if type(row) is not dict:
            raise TypeError(context + " must be a dictionary")
        if set(row) != set(fields):
            raise ValueError(context + " has wrong fields; expected " + str(fields))
        identifier = row["id"]
        if type(identifier) is not str or not identifier.strip():
            raise ValueError(context + " id must be nonempty text")
        if identifier != identifier.strip():
            raise ValueError(context + " id must not have surrounding whitespace")
        if identifier in seen_ids:
            raise ValueError(context + " has duplicate id " + identifier)
        seen_ids.append(identifier)
        converted = {"id": identifier}
        for field in fields:
            if field != "id":
                number = require_finite_number(row[field], context + " " + field)
                if field == "area_m2" and number <= 0:
                    raise ValueError(context + " area_m2 must be positive")
                converted[field] = number
        result.append(converted)
    return result


def load_table(path, kind):
    """读 UTF-8 CSV：字段名及顺序严格匹配；数值转换仅在此边界发生。

    空白行按 csv 模块规则忽略；缺值、额外值、坏引号、重复编号均拒绝。
    错误上下文用文件名和 CSV 物理行号，不主动公开整个文件路径。
    """
    path = Path(path)
    fields = schema_fields(kind)
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        try:
            if reader.fieldnames != fields:
                raise ValueError(path.name + " header must be exactly " + str(fields))
            for raw in reader:
                context = path.name + " line " + str(reader.line_num)
                missing_cell = False
                for value in raw.values():
                    if value is None:
                        missing_cell = True
                if None in raw or missing_cell:
                    raise ValueError(context + " has missing or extra cells")
                row = {"id": raw["id"]}
                for field in fields:
                    if field != "id":
                        try:
                            number = float(raw[field])
                        except ValueError as error:
                            raise ValueError(context + " " + field + " is not numeric") from error
                        row[field] = require_finite_number(number, context + " " + field)
                # Check with row context now; whole-file duplicate checks follow.
                try:
                    validate_records([row], kind)
                except (TypeError, ValueError) as error:
                    raise ValueError(context + ": " + str(error)) from error
                rows.append(row)
        except csv.Error as error:
            raise ValueError(path.name + " line " + str(reader.line_num) + " invalid CSV") from error
    try:
        return validate_records(rows, kind)
    except (TypeError, ValueError) as error:
        raise ValueError(path.name + ": " + str(error)) from error


def default_candidates():
    """固定候选，不从测试集提炼；每次返回独立的新字典。"""
    return [
        {"name": "A", "weight": 2.0, "bias": 0.0},
        {"name": "B", "weight": 2.0, "bias": 10.0},
        {"name": "C", "weight": 1.5, "bias": 40.0},
    ]


def validate_model(model):
    if type(model) is not dict or set(model) != {"name", "weight", "bias"}:
        raise ValueError("model needs exactly name, weight, bias")
    if type(model["name"]) is not str or not model["name"].strip():
        raise ValueError("model name must be nonempty text")
    return {
        "name": model["name"],
        "weight": require_finite_number(model["weight"], "model weight"),
        "bias": require_finite_number(model["bias"], "model bias"),
    }


def fit_candidates(train_records, candidates):
    """只收训练数据。MAE 最小者胜；相同分数时保留候选顺序中的首个。"""
    train = validate_records(train_records, "train")
    if type(candidates) is not list or len(candidates) == 0:
        raise ValueError("candidates must be a nonempty list")
    scores = []
    names = []
    best_model = None
    best_score = None
    for candidate in candidates:
        model = validate_model(candidate)
        if model["name"] in names:
            raise ValueError("candidate names must be unique")
        names.append(model["name"])
        predictions = []
        labels = []
        for row in train:
            predictions.append(predict_one(row["area_m2"], model["weight"], model["bias"]))
            labels.append(row["price_wan"])
        score = mean_absolute_error(predictions, labels)
        scores.append({"name": model["name"], "train_mae_wan": score})
        if best_score is None or score < best_score:
            best_model = model
            best_score = score
    return {"best_model": best_model, "candidate_scores": scores}


def predict_records(model, input_records):
    """输入表只含 id 与面积；返回含 id 的新预测记录。"""
    model = validate_model(model)
    inputs = validate_records(input_records, "test_inputs")
    predictions = []
    for row in inputs:
        predictions.append({
            "id": row["id"],
            "prediction_wan": predict_one(row["area_m2"], model["weight"], model["bias"]),
        })
    return predictions


def evaluate(prediction_records, label_records):
    """按 id 对齐后算 MAE。顺序可不同；缺失、额外、重复 id 都报错。"""
    predictions = validate_records(prediction_records, "predictions")
    labels = validate_records(label_records, "test_labels")
    labels_by_id = {}
    for row in labels:
        labels_by_id[row["id"]] = row["price_wan"]
    prediction_ids = []
    for row in predictions:
        prediction_ids.append(row["id"])
    if set(prediction_ids) != set(labels_by_id):
        raise ValueError("prediction ids and label ids must match exactly")
    prediction_values = []
    label_values = []
    details = []
    for row in predictions:
        actual = labels_by_id[row["id"]]
        prediction_values.append(row["prediction_wan"])
        label_values.append(actual)
        details.append({"id": row["id"], "prediction_wan": row["prediction_wan"],
                        "label_wan": actual,
                        "absolute_error_wan": abs(row["prediction_wan"] - actual)})
    return {"mae_wan": mean_absolute_error(prediction_values, label_values), "rows": details}


def fit_mean_baseline(train_records):
    """沿用第三讲：用训练价格均值拟合常数基线，不读取测试标签。"""
    train = validate_records(train_records, "train")
    total = 0.0
    for row in train:
        total = require_finite_number(total + row["price_wan"], "training price total")
    return require_finite_number(total / len(train), "baseline mean")


def input_fingerprint(path):
    """内容摘要用于识别输入版本；不是加密或真实性证明。"""
    path = Path(path)
    return {"filename": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def run_experiment(data_dir, output_dir):
    """组织读入、训练选择、封存预测、评价与 JSON 写入，并返回结果。"""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    train = load_table(data_dir / "train.csv", "train")
    inputs = load_table(data_dir / "test_inputs.csv", "test_inputs")
    train_ids = []
    for row in train:
        train_ids.append(row["id"])
    for row in inputs:
        if row["id"] in train_ids:
            raise ValueError("training and test ids must be disjoint in this task")
    fitted = fit_candidates(train, default_candidates())
    predictions = predict_records(fitted["best_model"], inputs)
    baseline = fit_mean_baseline(train)
    # Test labels are opened only after model selection and prediction.
    labels = load_table(data_dir / "test_labels.csv", "test_labels")
    evaluation = evaluate(predictions, labels)
    baseline_predictions = []
    for row in inputs:
        baseline_predictions.append({"id": row["id"], "prediction_wan": baseline})
    baseline_evaluation = evaluate(baseline_predictions, labels)
    fingerprints = []
    for filename in ["train.csv", "test_inputs.csv", "test_labels.csv"]:
        fingerprints.append(input_fingerprint(data_dir / filename))
    result = {
        "unit": "004", "data_kind": "handmade synthetic teaching data",
        "train_count": len(train), "test_count": len(inputs),
        "selection_metric": "training MAE; ties choose first candidate",
        "candidate_scores": fitted["candidate_scores"],
        "best_model": fitted["best_model"],
        "test_mae_wan": evaluation["mae_wan"],
        "test_predictions": evaluation["rows"],
        "baseline_prediction_wan": baseline,
        "baseline_test_mae_wan": baseline_evaluation["mae_wan"],
        "input_files": fingerprints,
    }
    # Serialize before creating output: a non-finite result cannot be written as NaN.
    serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(serialized + "\n", encoding="utf-8")
    return result


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Unit 004 reproducible function experiment")
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs")
    args = parser.parse_args()
    result = run_experiment(args.data_dir, args.output_dir)
    print(json.dumps({"best_model": result["best_model"]["name"],
                      "test_mae_wan": result["test_mae_wan"],
                      "baseline_test_mae_wan": result["baseline_test_mae_wan"]},
                     ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
