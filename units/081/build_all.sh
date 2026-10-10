#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-python}"
export MPLCONFIGDIR="$PWD/outputs/mpl"
export XDG_CACHE_HOME="$PWD/outputs/cache"
export IPYTHONDIR="$PWD/outputs/ipython"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export LOKY_MAX_CPU_COUNT=2
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$IPYTHONDIR"
"$PYTHON_BIN" generate_data.py --directory outputs/rebuild/data
"$PYTHON_BIN" experiment.py --out outputs/rebuild/experiment-result.json
"$PYTHON_BIN" test_experiment.py --out outputs/rebuild/test-result.json
"$PYTHON_BIN" -O test_experiment.py --out outputs/rebuild/test-result-optimized.json
"$PYTHON_BIN" plots.py --directory outputs/rebuild/figures
"$PYTHON_BIN" execute_notebook.py --out outputs/rebuild/experiment.ipynb
"$PYTHON_BIN" build_pdf.py --directory outputs/rebuild/pdf
