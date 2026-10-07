#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT="${1:-outputs/rebuild}"
mkdir -p "$OUT"
"$PYTHON_BIN" experiment.py --out "$OUT/result.json"
"$PYTHON_BIN" test_experiment.py --report "$OUT/result.json" --out "$OUT/tests.json"
"$PYTHON_BIN" -O test_experiment.py --report "$OUT/result.json" --out "$OUT/tests-optimized.json"
"$PYTHON_BIN" plots.py --report "$OUT/result.json" --directory "$OUT/figures"
"$PYTHON_BIN" execute_notebook.py --out "$OUT/experiment.ipynb"
"$PYTHON_BIN" build_pdf.py --directory "$OUT/pdf"
