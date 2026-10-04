"""Unit 006: offline, audited table exploration. All records are synthetic.
Run from any working directory: python path/to/experiment.py
Raw files are read only. Derived files go to outputs/ (or --output-dir).
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE / "data" / "raw"


def raw_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(RAW.glob("*.csv"))}


def read_raw():
    # Keep exact text first: unknown markers and whitespace remain inspectable.
    return pd.read_csv(RAW / "listings.csv", dtype="string",
                       keep_default_na=False, encoding="utf-8")


def clean_listings(raw):
    """Only remove exact original-field copies, then parse declared fields."""
    audit = []
    original_columns = raw.columns.tolist()
    work = raw.copy()
    work["source_csv_line"] = np.arange(2, len(raw) + 2)
    duplicate_mask = work.duplicated(subset=original_columns, keep="first")
    for _, row in work.loc[duplicate_mask].iterrows():
        audit.append({"step": "exact_duplicate", "event_id": row["event_id"],
                      "source_csv_line": int(row["source_csv_line"]),
                      "field": "all_original_fields", "before": "repeat",
                      "after": "removed_from_derived_table",
                      "reason": "same event and all six original values identical"})
    work = work.loc[~duplicate_mask].copy()
    # A conflicting repeated event is NOT silently removed.
    if work["event_id"].duplicated().any():
        raise ValueError("Conflicting event_id after exact-record deduplication")
    for col in ["district_code", "area_sqm"]:
        stripped = work[col].str.strip()
        for idx in work.index[work[col] != stripped]:
            audit.append({"step": "strip_whitespace", "event_id": work.at[idx, "event_id"],
                          "source_csv_line": int(work.at[idx, "source_csv_line"]),
                          "field": col, "before": work.at[idx, col], "after": stripped.at[idx],
                          "reason": "declared field format excludes surrounding spaces"})
        work[col] = stripped
    missing = work["area_sqm"].isin(["", "未知"])
    for _, row in work.loc[missing].iterrows():
        audit.append({"step": "missing_marker", "event_id": row["event_id"],
                      "source_csv_line": int(row["source_csv_line"]),
                      "field": "area_sqm", "before": row["area_sqm"], "after": "missing",
                      "reason": "declared missing marker, not zero"})
    # Unexpected text raises instead of being silently coerced into missingness.
    work["area_sqm"] = pd.to_numeric(work["area_sqm"].mask(missing), errors="raise").astype("Float64")
    work["list_price_wan"] = pd.to_numeric(work["list_price_wan"], errors="raise").astype("Float64")
    work["listed_at"] = pd.to_datetime(work["listed_at"], format="%Y-%m-%d", errors="raise")
    if not work.loc[work["area_sqm"].notna(), "area_sqm"].gt(0).all():
        raise ValueError("Observed area must be positive; investigate rather than delete")
    if not work["list_price_wan"].gt(0).all():
        raise ValueError("List price must be positive in this teaching dictionary")
    return work.reset_index(drop=True), pd.DataFrame(audit)


def load_auxiliary():
    candidates = pd.read_csv(RAW / "district_candidates.csv", dtype={"district_code": "string"})
    registry = pd.read_csv(RAW / "district_registry.csv", dtype={"district_code": "string"})
    splits = pd.read_csv(RAW / "split_manifest.csv", dtype="string")
    return candidates, registry, splits


def safe_merge(clean, registry):
    """Many events may refer to one verified district. Preserve all events."""
    if clean["district_code"].isna().any() or registry["district_code"].isna().any():
        raise ValueError("Missing join key requires a separate explicit policy")
    merged = clean.merge(registry, on="district_code", how="left",
                         validate="many_to_one", indicator=True)
    assert len(merged) == len(clean)
    assert merged["event_id"].is_unique
    return merged


def grouped_summary(merged):
    result = merged.groupby("district_code", dropna=False).agg(
        event_count=("event_id", "size"),
        observed_area_count=("area_sqm", "count"),
        observed_price_count=("list_price_wan", "count"),
        mean_area_sqm=("area_sqm", "mean"),
        mean_price_wan=("list_price_wan", "mean"))
    result["missing_area_count"] = result["event_count"] - result["observed_area_count"]
    result["missing_area_fraction"] = result["missing_area_count"] / result["event_count"]
    return result


def train_only_imputation(clean, splits):
    # Split is a pre-specified manifest, never inferred from outcomes.
    model_input = clean.merge(splits, on="event_id", how="left",
                              validate="one_to_one", indicator="split_match")
    if not model_input["split_match"].eq("both").all():
        raise ValueError("Every event must have a split assignment")
    if not model_input["split"].isin(["train", "test"]).all():
        raise ValueError("Unknown split assignment")
    train = model_input.loc[model_input["split"].eq("train")].copy()
    test = model_input.loc[model_input["split"].eq("test")].copy()
    if set(train["property_id"]) & set(test["property_id"]):
        raise ValueError("The new-property demonstration forbids cross-split properties")
    learned_mean = train["area_sqm"].mean()
    if pd.isna(learned_mean):
        raise ValueError("Training area has no observations; cannot estimate a mean")
    model_input["area_was_missing"] = model_input["area_sqm"].isna()
    model_input["area_imputed_sqm"] = model_input["area_sqm"].fillna(learned_mean)
    return model_input, float(learned_mean)


def run(output_dir=None):
    out = Path(output_dir) if output_dir else HERE / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    before_hashes = raw_hashes()
    raw = read_raw()
    clean, audit = clean_listings(raw)
    candidates, registry, splits = load_auxiliary()
    conflict_rows = candidates.loc[candidates.duplicated("district_code", keep=False)].copy()
    rejected = False
    try:
        clean.merge(candidates, on="district_code", how="left", validate="many_to_one")
    except pd.errors.MergeError:
        rejected = True
    # Intentionally unsafe counterexample; never used as the clean output.
    unsafe = clean.merge(candidates, on="district_code", how="left")
    merged = safe_merge(clean, registry)
    grouped = grouped_summary(merged)
    model_input, learned_mean = train_only_imputation(clean, splits)
    counts, edges = np.histogram(clean["area_sqm"].dropna().to_numpy(dtype=float),
                                bins=[40, 80, 120, 160, 200, 240, 280, 320])
    observed_prices = clean["list_price_wan"]
    summary = {
        "raw_rows": len(raw), "clean_events": len(clean),
        "unique_properties": int(clean["property_id"].nunique()),
        "exact_duplicate_rows_removed": len(raw)-len(clean),
        "area_observed_count": int(clean["area_sqm"].count()),
        "area_missing_count": int(clean["area_sqm"].isna().sum()),
        "price_observed_count": int(observed_prices.count()),
        "verified_merge_rows": len(merged),
        "merge_both": int(merged["_merge"].eq("both").sum()),
        "merge_left_only": int(merged["_merge"].eq("left_only").sum()),
        "unsafe_merge_rows": len(unsafe),
        "unsafe_mean_price_wan": float(unsafe["list_price_wan"].mean()),
        "safe_mean_price_wan": float(observed_prices.mean()),
        "safe_price_sum_wan": float(observed_prices.sum()),
        "inner_join_rows": int(merged["_merge"].eq("both").sum()),
        "inner_join_mean_price_wan": float(merged.loc[merged["_merge"].eq("both"), "list_price_wan"].mean()),
        "many_to_one_conflict_rejected": rejected,
        "train_events": int(model_input["split"].eq("train").sum()),
        "test_events": int(model_input["split"].eq("test").sum()),
        "train_observed_area_count": int(model_input.loc[model_input["split"].eq("train"), "area_sqm"].count()),
        "train_area_mean_sqm": learned_mean,
        "all_observed_area_mean_sqm_counterexample": float(clean["area_sqm"].mean()),
        "area_hist_counts": counts.tolist(), "area_hist_edges": edges.tolist(),
        "price_quantiles_linear": {str(q): float(observed_prices.quantile(q, interpolation="linear")) for q in [0, .25, .5, .75, 1]},
        "hand_quantiles_linear": {str(q): float(pd.Series([50,60,100,130]).quantile(q, interpolation="linear")) for q in [.25, .5, .75]},
        "raw_files_unchanged": before_hashes == raw_hashes(),
        "synthetic_data": True, "models_fitted": 0,
    }
    checks = []
    def check(name, condition):
        if not bool(condition):
            raise AssertionError(name)
        checks.append({"name": name, "passed": True})
    check("raw rows 16", len(raw) == 16)
    check("clean rows 14", len(clean) == 14)
    check("unique properties 12", clean["property_id"].nunique() == 12)
    check("unique event key", clean["event_id"].is_unique)
    check("duplicate original lines 16 and 17", audit.loc[audit.step.eq("exact_duplicate"), "source_csv_line"].tolist() == [16,17])
    check("two re-listing events preserved", {"L013","L014"}.issubset(set(clean.event_id)))
    check("H01 twice H02 twice", clean.property_id.value_counts().loc[["H01","H02"]].tolist() == [2,2])
    check("missing IDs L004 L010", clean.loc[clean.area_sqm.isna(), "event_id"].tolist() == ["L004","L010"])
    check("observed area count 12", clean.area_sqm.count() == 12)
    check("price count 14", clean.list_price_wan.count() == 14)
    check("area sum 1220", np.isclose(clean.area_sqm.sum(),1220))
    check("price sum 2888", np.isclose(clean.list_price_wan.sum(),2888))
    check("correct mean 2888/14", np.isclose(summary["safe_mean_price_wan"],2888/14))
    check("conflicting N has two candidate rows", len(conflict_rows)==2 and set(conflict_rows.district_code)=={"N"})
    check("cardinality rejected", rejected)
    check("safe left merge 14", len(merged)==14)
    check("matched 13 unmatched 1", summary["merge_both"]==13 and summary["merge_left_only"]==1)
    check("unmatched event L012", merged.loc[merged._merge.eq("left_only"), "event_id"].tolist()==["L012"])
    check("unsafe merge 19", len(unsafe)==19)
    check("unsafe sum 3576", np.isclose(unsafe.list_price_wan.sum(),3576))
    check("unsafe mean 3576/19", np.isclose(summary["unsafe_mean_price_wan"],3576/19))
    check("inner join mean 176", np.isclose(summary["inner_join_mean_price_wan"],176))
    check("group event counts", grouped.event_count.to_dict()=={"E":3,"N":5,"S":3,"W":2,"X":1})
    check("group observed area counts", grouped.observed_area_count.to_dict()=={"E":3,"N":5,"S":1,"W":2,"X":1})
    check("south missing fraction 2/3", np.isclose(grouped.loc["S","missing_area_fraction"],2/3))
    check("south area mean uses one observation", grouped.loc["S","mean_area_sqm"]==80)
    check("group counts reconcile", grouped.event_count.sum()==len(clean))
    check("histogram counts", counts.tolist()==[5,4,2,0,0,0,1])
    check("histogram includes all observed values", counts.sum()==12)
    check("quartile hand q25", np.isclose(summary["hand_quantiles_linear"]["0.25"],57.5))
    check("quartile hand q50", np.isclose(summary["hand_quantiles_linear"]["0.5"],80))
    check("quartile hand q75", np.isclose(summary["hand_quantiles_linear"]["0.75"],107.5))
    check("split 10 and 4", summary["train_events"]==10 and summary["test_events"]==4)
    check("train observed count 9", summary["train_observed_area_count"]==9)
    check("train mean 670/9", np.isclose(learned_mean,670/9))
    check("all data mean counterexample 1220/12", np.isclose(clean.area_sqm.mean(),1220/12))
    check("both missing values use train mean", np.allclose(model_input.loc[model_input.area_was_missing,"area_imputed_sqm"].astype(float),670/9))
    check("exploration retains missing values", clean.area_sqm.isna().sum()==2)
    check("legitimate large event retained", clean.loc[clean.event_id.eq("L012"),"area_sqm"].iloc[0]==300)
    check("no cross-split property overlap", not (set(model_input.loc[model_input.split.eq("train"),"property_id"]) & set(model_input.loc[model_input.split.eq("test"),"property_id"])))
    # Boundary tests for important failure modes.
    conflicting_raw = raw.copy()
    conflicting_raw.loc[14,"list_price_wan"] = "999"
    try:
        clean_listings(conflicting_raw)
    except ValueError:
        check("conflicting repeated event is rejected", True)
    else:
        raise AssertionError("Conflicting event must fail")
    bad_text = raw.iloc[:14].copy()
    bad_text.loc[0,"area_sqm"] = "fifty"
    try:
        clean_listings(bad_text)
    except ValueError:
        check("unexpected numeric token is rejected", True)
    else:
        raise AssertionError("Invalid numeric token must fail")
    check("raw hashes unchanged", before_hashes==raw_hashes())
    # Detailed audit + decision ledger. Raw files are never overwritten.
    clean.to_csv(out / "clean_events.csv", index=False, encoding="utf-8")
    merged.to_csv(out / "verified_join.csv", index=False, encoding="utf-8")
    grouped.to_csv(out / "group_summary.csv", encoding="utf-8")
    audit.to_csv(out / "cleaning_audit.csv", index=False, encoding="utf-8")
    conflict_rows.to_csv(out / "district_conflicts.csv", index=False, encoding="utf-8")
    model_input.to_csv(out / "imputation_demo.csv", index=False, encoding="utf-8")
    decisions = [
        {"decision":"preserve_relistings", "event_ids":["L013","L014"], "reason":"Different listing dates and stable event IDs; target is listing events."},
        {"decision":"use_verified_registry", "key":"N", "reason":"Provided teaching registry establishes source A; never arbitrary keep-first."},
        {"decision":"preserve_unmatched_event", "event_id":"L012", "reason":"X has no registry entry; left join retains event and marks left_only."},
        {"decision":"preserve_large_area", "event_id":"L012", "reason":"Synthetic data specification confirms 300 sqm; size alone is not evidence of error."},
        {"decision":"learn_imputation_on_train_only", "value":learned_mean, "reason":"Separate model-input demo; exploratory observations remain unchanged."}]
    for filename, data in [("summary.json", summary), ("raw_hashes.json", before_hashes),
                           ("decisions.json", decisions)]:
        (out / filename).write_text(json.dumps(data, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    result={"status":"passed","checked_on":"2026-10-04", "test_count":len(checks), "tests":checks,
            "versions":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__},
            "script":"experiment.py", "raw_hashes_unchanged":before_hashes==raw_hashes(),
            "limitations":["Synthetic teaching data, no market inference or model-performance claim.","Anaconda installation and browser Jupyter UI not exercised in this environment."]}
    (out / "test-result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return {"raw":raw,"clean":clean,"merged":merged,"groups":grouped,"model_input":model_input,
            "summary":summary,"tests":result,"audit":audit,"unsafe":unsafe}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE/"outputs")
    args=parser.parse_args()
    result=run(args.output_dir)
    print(json.dumps({"status":"passed", "test_count":result["tests"]["test_count"],
                      "summary":result["summary"]}, ensure_ascii=False,indent=2))
