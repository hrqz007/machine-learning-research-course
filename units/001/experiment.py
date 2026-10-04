"""第一讲：受限候选训练、冻结、预测和评价。合成数据，仅作教学。

python experiment.py --phase train
python experiment.py --phase predict
python experiment.py --phase evaluate
python experiment.py --phase all  # 复跑整条已公开答案的教学流程
python experiment.py --self-test
只使用 Python 标准库；所有默认输入均相对于本文件。
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math

HERE = Path(__file__).resolve().parent
CANDIDATES = {"A": (2.0, 0.0), "B": (2.0, 10.0), "C": (1.5, 40.0)}


def read_records(filename):
    """把一张 CSV 表读成记录列表；本讲只允许唯一且非空的 id。"""
    with open(filename, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("数据不能为空")
    ids = [row.get("id", "") for row in rows]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("id 必须非空且不能重复")
    for row in rows:
        for key in row:
            if key != "id":
                row[key] = float(row[key])
                if not math.isfinite(row[key]):
                    raise ValueError("数值必须有限")
    return rows


def predict(area, w, b):
    """面积×斜率+截距。单位：平方米×万元/平方米+万元=万元。"""
    return w * area + b


def mae(predictions, labels):
    """平均绝对误差；不把带方向误差直接相加。"""
    if len(predictions) != len(labels) or not predictions:
        raise ValueError("预测和标签必须等长且非空")
    return sum(abs(prediction - label)
               for prediction, label in zip(predictions, labels)) / len(labels)


def fit_candidates(training):
    """只接收训练记录。测试答案不在本函数的输入中。"""
    labels = [row["price_wan"] for row in training]
    if not labels:
        raise ValueError("训练数据不能为空")
    scores = {}
    for name, (w, b) in CANDIDATES.items():
        estimates = [predict(row["area_m2"], w, b) for row in training]
        scores[name] = mae(estimates, labels)
    # 若恰好并列，按名称顺序选取。这个约定也必须在看测试答案前确定。
    best = min(scores, key=lambda name: (scores[name], name))
    w, b = CANDIDATES[best]
    return {"candidate": best, "w": w, "b": b,
            "training_mae_wan": scores,
            "baseline_price_wan": sum(labels) / len(labels)}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def train(data_dir, output_dir):
    """选择模型后把参数写到磁盘。之后评价读取这份记录。"""
    model = fit_candidates(read_records(data_dir / "train.csv"))
    model["training_sha256"] = digest(data_dir / "train.csv")
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "frozen-model.json"
    if target.exists() and json.loads(target.read_text()) != model:
        raise RuntimeError("已有不同的冻结记录；请使用新的 --output 目录保留实验历史")
    target.write_text(json.dumps(model, ensure_ascii=False, indent=2) + "\n")
    return model


def read_frozen(data_dir, output_dir):
    path = output_dir / "frozen-model.json"
    if not path.exists():
        raise RuntimeError("请先运行 --phase train，记录并冻结模型")
    model = json.loads(path.read_text())
    if model["training_sha256"] != digest(data_dir / "train.csv"):
        raise RuntimeError("训练文件已经变化；请保留旧结果，并在新输出目录重新开始")
    return model


def make_predictions(data_dir, output_dir):
    model = read_frozen(data_dir, output_dir)
    inputs = read_records(data_dir / "test_inputs.csv")
    result = [{"id": row["id"],
               "prediction_wan": predict(row["area_m2"], model["w"], model["b"]),
               "baseline_wan": model["baseline_price_wan"]} for row in inputs]
    (output_dir / "predictions.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def evaluate(data_dir, output_dir):
    model = read_frozen(data_dir, output_dir)
    path = output_dir / "predictions.json"
    if not path.exists():
        raise RuntimeError("请先运行 --phase predict，留下测试预测再打开答案")
    predictions = json.loads(path.read_text())
    inputs = read_records(data_dir / "test_inputs.csv")
    expected = [{"id": r["id"], "prediction_wan": predict(r["area_m2"], model["w"], model["b"]),
                 "baseline_wan": model["baseline_price_wan"]} for r in inputs]
    if predictions != expected:
        raise RuntimeError("预测记录与冻结模型或输入不一致；不要在看答案后修改预测")
    labels_by_id = {row["id"]: row["price_wan"] for row in read_records(data_dir / "test_labels.csv")}
    if set(labels_by_id) != {row["id"] for row in predictions}:
        raise ValueError("测试输入与标签的 id 不一致")
    values = [row["prediction_wan"] for row in predictions]
    labels = [labels_by_id[row["id"]] for row in predictions]
    baseline = [row["baseline_wan"] for row in predictions]
    result = {"selected_candidate": model["candidate"],
              "test_predictions_wan": values, "test_labels_wan": labels,
              "absolute_errors_wan": [abs(a-b) for a,b in zip(values,labels)],
              "test_mae_wan": mae(values, labels),
              "baseline_test_mae_wan": mae(baseline, labels),
              "scope": "两条公开的合成教学记录，不是真实房价能力估计"}
    (output_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def self_test():
    training = [{"id": "H"+str(i+1), "area_m2": x, "price_wan": y}
                for i,(x,y) in enumerate([(50,110),(60,130),(80,170),(90,190)])]
    model = fit_candidates(training)
    assert model["candidate"] == "B"
    assert model["training_mae_wan"] == {"A":10.0,"B":0.0,"C":7.5}
    assert model["baseline_price_wan"] == 150.0
    assert predict(70, 2, 10) == 150.0
    assert mae([140,180],[144,174]) == 5.0
    assert mae([150,150],[144,174]) == 15.0
    assert mae([90,110],[100,100]) == 10.0  # 抵消反例
    assert fit_candidates(list(reversed(training))) == model  # 行顺序不改变答案
    for bad_predictions,bad_labels in [([],[]),([1],[1,2])]:
        try: mae(bad_predictions,bad_labels)
        except ValueError: pass
        else: raise AssertionError("应拒绝无效长度")
    print("SELF_TEST_PASS: 9 explicit checks")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["train","predict","evaluate","all"], default="all")
    parser.add_argument("--data", type=Path, default=HERE/"data")
    parser.add_argument("--output", type=Path, default=HERE/"results")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    actions = {"train":train,"predict":make_predictions,"evaluate":evaluate}
    phases = ["train","predict","evaluate"] if args.phase == "all" else [args.phase]
    for phase in phases:
        print(phase.upper())
        print(json.dumps(actions[phase](args.data,args.output),ensure_ascii=False,indent=2))

if __name__ == "__main__":
    main()
