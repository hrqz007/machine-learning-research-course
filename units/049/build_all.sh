#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python3}"
mkdir -p "$HERE/outputs/tmp" "$HERE/outputs/cache" "$HERE/outputs/mpl" "$HERE/outputs/ipython"
export TMPDIR="$HERE/outputs/tmp" XDG_CACHE_HOME="$HERE/outputs/cache" MPLCONFIGDIR="$HERE/outputs/mpl" IPYTHONDIR="$HERE/outputs/ipython"
"$PYTHON" "$HERE/experiment.py" --out "$HERE/outputs/result.json"
"$PYTHON" "$HERE/audit.py" --out "$HERE/outputs/audit.json"
"$PYTHON" -O "$HERE/experiment.py" --out "$HERE/outputs/result-optimized.json"
"$PYTHON" -O "$HERE/audit.py" --out "$HERE/outputs/audit-optimized.json"
"$PYTHON" "$HERE/plots.py" --report "$HERE/outputs/result.json" --directory "$HERE/outputs/figures"
"$PYTHON" "$HERE/build_notebook.py" --out "$HERE/outputs/notebook/learning_guarantees.ipynb"
"$PYTHON" "$HERE/execute_notebook.py" "$HERE/outputs/notebook/learning_guarantees.ipynb"
"$PYTHON" "$HERE/build_pdf.py" --directory "$HERE/outputs/pdf"
