"""可重跑的检查清单。只用标准库；失败立即停止，成功才写测试报告。

运行 python test_experiment.py。故障数据只写入 TemporaryDirectory；不会改 data/。
expect_error 只捕获指定错误类型，未预期异常照常传播，不能伪装成通过。
"""
import argparse
import copy
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import experiment as e


def run_tests(report_path):
    checks = []

    def check(name, actual, expected):
        if actual != expected:
            raise AssertionError(name + ": expected " + repr(expected) + ", got " + repr(actual))
        checks.append({"name": name, "status": "passed"})

    def expect_error(name, error_type, function, arguments):
        try:
            function(*arguments)
        except error_type:
            checks.append({"name": name, "status": "passed", "expected_error": error_type.__name__})
        else:
            raise AssertionError(name + ": expected " + error_type.__name__)

    check("prediction_known_answer", e.predict_one(70, 2, 10), 150.0)
    check("numeric_core_allows_zero_area", e.predict_one(0, 2, 10), 10.0)
    check("numeric_core_allows_negative_area", e.predict_one(-2, 2, 10), 6.0)
    check("prediction_float", e.predict_one(1.5, 2.0, 0.5), 3.5)
    for index in range(3):
        for bad in [True, "70", None, [70]]:
            args = [70, 2, 10]
            args[index] = bad
            expect_error("prediction_reject_type_" + str(index) + "_" + type(bad).__name__,
                         TypeError, e.predict_one, args)
        for label, bad in [("nan", float("nan")), ("inf", float("inf")), ("negative_inf", float("-inf"))]:
            args = [70, 2, 10]
            args[index] = bad
            expect_error("prediction_reject_" + label + "_" + str(index), ValueError, e.predict_one, args)
    expect_error("prediction_product_overflow", ValueError, e.predict_one, [1e308, 10, 0])
    expect_error("prediction_sum_overflow", ValueError, e.predict_one, [1e308, 1, 1e308])
    expect_error("prediction_oversized_int", ValueError, e.predict_one, [10 ** 1000, 1, 0])
    e.weight = 999
    e.bias = 999
    check("prediction_ignores_module_global_names", e.predict_one(70, 2, 10), 150.0)
    check("mae_known_answer", e.mean_absolute_error([140, 180], [144, 174]), 5.0)
    check("mae_single_sample", e.mean_absolute_error([140], [144]), 4.0)
    check("mae_tuple_inputs", e.mean_absolute_error((1, 4), (2, 2)), 1.5)
    check("mae_identity", e.mean_absolute_error([2, 3], [2, 3]), 0.0)
    check("mae_joint_permutation", e.mean_absolute_error([180, 140], [174, 144]), 5.0)
    check("mae_joint_translation", e.mean_absolute_error([240, 280], [244, 274]), 5.0)
    expect_error("mae_empty", ValueError, e.mean_absolute_error, [[], []])
    expect_error("mae_length_mismatch", ValueError, e.mean_absolute_error, [[1], [1, 2]])
    expect_error("mae_reject_text_container", TypeError, e.mean_absolute_error, ["12", [1, 2]])
    expect_error("mae_reject_dict_container", TypeError, e.mean_absolute_error, [{0: 1}, [1]])
    for label, bad in [("bool", True), ("text", "1"), ("none", None)]:
        expect_error("mae_prediction_" + label, TypeError, e.mean_absolute_error, [[bad], [1]])
        expect_error("mae_label_" + label, TypeError, e.mean_absolute_error, [[1], [bad]])
    for label, bad in [("nan", float("nan")), ("inf", float("inf")), ("negative_inf", float("-inf"))]:
        expect_error("mae_prediction_" + label, ValueError, e.mean_absolute_error, [[bad], [1]])
        expect_error("mae_label_" + label, ValueError, e.mean_absolute_error, [[1], [bad]])
    expect_error("mae_difference_overflow", ValueError, e.mean_absolute_error, [[1e308], [-1e308]])
    expect_error("mae_total_overflow", ValueError, e.mean_absolute_error, [[1e308, 1e308], [0, 0]])
    first = [140, 180]
    second = [144, 174]
    e.mean_absolute_error(first, second)
    check("mae_does_not_mutate", [first, second], [[140, 180], [144, 174]])
    root = Path(__file__).resolve().parent
    train = e.load_table(root / "data" / "train.csv", "train")
    inputs = e.load_table(root / "data" / "test_inputs.csv", "test_inputs")
    labels = e.load_table(root / "data" / "test_labels.csv", "test_labels")
    check("train_four_rows", len(train), 4)
    check("csv_numeric_conversion", type(train[0]["area_m2"]).__name__, "float")
    candidates = e.default_candidates()
    original = copy.deepcopy([train, candidates])
    fitted = e.fit_candidates(train, candidates)
    check("candidate_scores", fitted["candidate_scores"], [
        {"name": "A", "train_mae_wan": 10.0}, {"name": "B", "train_mae_wan": 0.0},
        {"name": "C", "train_mae_wan": 7.5}])
    check("selection_B", fitted["best_model"]["name"], "B")
    check("fit_does_not_mutate", [train, candidates], original)
    check("training_baseline", e.fit_mean_baseline(train), 150.0)
    check("baseline_single_sample", e.fit_mean_baseline([train[0]]), 110.0)
    fresh = e.default_candidates()
    fresh[0]["bias"] = 900
    check("candidate_config_not_shared", e.default_candidates()[0]["bias"], 0.0)
    tied = [{"name": "first", "weight": 2, "bias": 10}, {"name": "second", "weight": 2, "bias": 10}]
    check("tie_first_candidate", e.fit_candidates(train, tied)["best_model"]["name"], "first")
    expect_error("fit_empty_train", ValueError, e.fit_candidates, [[], candidates])
    expect_error("fit_empty_candidates", ValueError, e.fit_candidates, [train, []])
    expect_error("fit_duplicate_names", ValueError, e.fit_candidates, [train, [candidates[0], candidates[0]]])
    expect_error("fit_nonfinite_weight", ValueError, e.fit_candidates,
                 [train, [{"name": "bad", "weight": float("inf"), "bias": 0}]])
    expect_error("schema_unknown", ValueError, e.schema_fields, ["trainingg"])
    expect_error("records_require_list", TypeError, e.validate_records, ["bad", "train"])
    expect_error("records_require_dict", TypeError, e.validate_records, [[123], "train"])
    expect_error("records_boolean_area", TypeError, e.validate_records,
                 [[{"id": "X", "area_m2": True}], "test_inputs"])
    predictions = e.predict_records(fitted["best_model"], inputs)
    check("predictions", predictions, [{"id": "T1", "prediction_wan": 140.0},
                                        {"id": "T2", "prediction_wan": 180.0}])
    check("evaluation", e.evaluate(predictions, labels)["mae_wan"], 5.0)
    check("evaluation_reordered_labels", e.evaluate(predictions, list(reversed(labels)))["mae_wan"], 5.0)
    check("evaluation_single_sample", e.evaluate([predictions[0]], [labels[0]])["mae_wan"], 4.0)
    expect_error("evaluate_missing_id", ValueError, e.evaluate, [predictions, [labels[0]]])
    expect_error("evaluate_wrong_id", ValueError, e.evaluate,
                 [predictions, [{"id": "T1", "price_wan": 144}, {"id": "X", "price_wan": 174}]])
    expect_error("evaluate_duplicate_label", ValueError, e.evaluate, [predictions, [labels[0], labels[0]]])
    expect_error("evaluate_duplicate_prediction", ValueError, e.evaluate, [[predictions[0], predictions[0]], labels])
    expect_error("predict_refuses_label_field", ValueError, e.predict_records,
                 [fitted["best_model"], [{"id": "T1", "area_m2": 65, "price_wan": 144}]])

    def index_failure():
        values = [10, 20]
        return values[2]

    def unexpected_failure():
        return 1 / 0

    expect_error("deliberate_index_error", IndexError, index_failure, [])
    try:
        expect_error("must_not_claim_pass", ValueError, unexpected_failure, [])
    except ZeroDivisionError:
        check("harness_does_not_swallow_unknown_error", True, True)
    else:
        raise AssertionError("unknown error was swallowed")

    with TemporaryDirectory(prefix="ml_unit004_") as temporary:
        temp = Path(temporary)
        missing = temp / "absent.csv"
        expect_error("deliberate_missing_path", FileNotFoundError, e.load_table, [missing, "train"])
        failures = [
            ("empty_file", ""),
            ("header_only", "id,area_m2,price_wan\n"),
            ("wrong_header", "id,area,price_wan\nH1,50,110\n"),
            ("reordered_header", "area_m2,id,price_wan\n50,H1,110\n"),
            ("duplicate_header", "id,area_m2,area_m2\nH1,50,110\n"),
            ("extra_header", "id,area_m2,price_wan,note\nH1,50,110,x\n"),
            ("extra_cell", "id,area_m2,price_wan\nH1,50,110,x\n"),
            ("missing_cell", "id,area_m2,price_wan\nH1,50\n"),
            ("blank_numeric_cell", "id,area_m2,price_wan\nH1,,110\n"),
            ("nonnumeric", "id,area_m2,price_wan\nH1,unknown,110\n"),
            ("nan", "id,area_m2,price_wan\nH1,nan,110\n"),
            ("inf", "id,area_m2,price_wan\nH1,50,inf\n"),
            ("negative_inf", "id,area_m2,price_wan\nH1,-inf,110\n"),
            ("overflow_token", "id,area_m2,price_wan\nH1,1e309,110\n"),
            ("zero_area", "id,area_m2,price_wan\nH1,0,110\n"),
            ("negative_area", "id,area_m2,price_wan\nH1,-1,110\n"),
            ("empty_id", "id,area_m2,price_wan\n,50,110\n"),
            ("whitespace_id", "id,area_m2,price_wan\n H1 ,50,110\n"),
            ("duplicate_id", "id,area_m2,price_wan\nH1,50,110\nH1,60,130\n"),
            ("malformed_quote", 'id,area_m2,price_wan\n"H1,50,110\n'),
        ]
        for name, content in failures:
            fixture = temp / (name + ".csv")
            fixture.write_text(content, encoding="utf-8")
            expect_error("csv_" + name, ValueError, e.load_table, [fixture, "train"])
        good = temp / "quoted.csv"
        good.write_text('id,area_m2,price_wan\n"H,1",50,110\n', encoding="utf-8")
        check("csv_quoted_comma_supported", e.load_table(good, "train")[0]["id"], "H,1")
        good.write_text('id,area_m2,price_wan\r\n单样本,50,110\r\n', encoding="utf-8")
        check("csv_utf8_crlf_single", len(e.load_table(good, "train")), 1)
        result = e.run_experiment(root / "data", temp / "output")
        check("full_result_metric", [result["train_count"], result["test_count"], result["test_mae_wan"],
                                     result["baseline_test_mae_wan"]], [4, 2, 5.0, 15.0])
        saved = json.loads((temp / "output" / "result.json").read_text(encoding="utf-8"))
        check("json_roundtrip", saved, result)
        check("output_contains_no_absolute_internal_path", str(root) in json.dumps(saved), False)
        check("result_filenames_only", saved["input_files"][0]["filename"], "train.csv")
        check("rerun_same_result", e.run_experiment(root / "data", temp / "output"), result)
        copied = temp / "data"
        shutil.copytree(root / "data", copied)
        (copied / "test_labels.csv").write_text("id,price_wan\nT1,0\nT2,0\n", encoding="utf-8")
        changed_labels = e.run_experiment(copied, temp / "changed")
        check("test_labels_cannot_change_model", changed_labels["best_model"], result["best_model"])
        check("test_labels_cannot_change_train_scores", changed_labels["candidate_scores"], result["candidate_scores"])
        check("test_labels_cannot_change_baseline", changed_labels["baseline_prediction_wan"], 150.0)
        check("test_labels_change_only_evaluation", changed_labels["test_mae_wan"], 160.0)
        (copied / "test_inputs.csv").write_text("id,area_m2\nH1,65\nT2,85\n", encoding="utf-8")
        expect_error("reject_train_test_id_overlap", ValueError, e.run_experiment, [copied, temp / "invalid"])
        check("failed_run_does_not_write_result", (temp / "invalid" / "result.json").exists(), False)
        command = [sys.executable, str(root / "experiment.py"), "--output-dir", str(temp / "cli")]
        process = subprocess.run(command, cwd=temp, text=True, capture_output=True, check=True)
        check("cli_from_unrelated_cwd", json.loads(process.stdout),
              {"best_model": "B", "test_mae_wan": 5.0, "baseline_test_mae_wan": 15.0})
        check("cli_result_equal", json.loads((temp / "cli" / "result.json").read_text(encoding="utf-8")), result)
        custom = subprocess.run([sys.executable, str(root / "experiment.py"),
                                 "--data-dir", str(root / "data"), "--output-dir", str(temp / "custom")],
                                cwd=temp, text=True, capture_output=True, check=True)
        check("cli_explicit_data_dir", json.loads(custom.stdout)["test_mae_wan"], 5.0)
        optimized_code = "import sys; sys.path.insert(0, sys.argv[1]); import experiment; experiment.predict_one(True, 2, 10)"
        optimized = subprocess.run([sys.executable, "-O", "-c", optimized_code, str(root)],
                                   cwd=temp, text=True, capture_output=True)
        check("optimized_mode_still_rejects_invalid_input", optimized.returncode != 0 and "TypeError" in optimized.stderr, True)
        import_code = "import sys; sys.path.insert(0, sys.argv[1]); import experiment"
        imported = subprocess.run([sys.executable, "-c", import_code, str(root)], cwd=temp,
                                  text=True, capture_output=True, check=True)
        check("import_has_no_stdout", imported.stdout, "")
        help_run = subprocess.run([sys.executable, str(root / "experiment.py"), "--help"],
                                  cwd=temp, text=True, capture_output=True, check=True)
        check("cli_help_available", "--data-dir" in help_run.stdout, True)
    report = {"unit": "004", "status": "passed", "checked_on": "2026-10-04",
              "python_version": platform.python_version(), "dependency_scope": "Python standard library only",
              "check_count": len(checks), "checks": checks,
              "coverage": ["core contracts", "CSV schema and malformed inputs", "ID-aligned evaluation",
                           "training-only selection", "JSON roundtrip", "CLI from unrelated working directory",
                           "input validation retained with Python -O", "unknown errors propagate"],
              "limitations": ["finite ordinary int/float convertible to float; no arbitrary precision promise",
                              "small explicit tests do not prove all inputs correct",
                              "macOS and Windows runtime behavior not tested here",
                              "Jupyter browser UI and external socket transport not tested"]}
    Path(report_path).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("PASS:", len(checks), "explicit checks")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=Path(__file__).resolve().parent / "test-result.json")
    args = parser.parse_args()
    run_tests(args.report)
