#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/outputs/mpl}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PWD/outputs/cache}"
export JUPYTER_RUNTIME_DIR="${JUPYTER_RUNTIME_DIR:-$PWD/outputs/jupyter}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$JUPYTER_RUNTIME_DIR"
"$PYTHON_BIN" experiment.py --out outputs/rebuild/result.json
"$PYTHON_BIN" test_experiment.py --report outputs/rebuild/result.json --out outputs/rebuild/test-result.json
"$PYTHON_BIN" -O test_experiment.py --report outputs/rebuild/result.json --out outputs/rebuild/test-result-optimized.json
"$PYTHON_BIN" plots.py --report outputs/rebuild/result.json --directory outputs/rebuild/figures
"$PYTHON_BIN" execute_notebook.py --out outputs/rebuild/experiment.ipynb
"$PYTHON_BIN" build_pdf.py --directory outputs/rebuild/pdf
