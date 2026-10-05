#!/usr/bin/env bash
# Rebuild within this standalone unit. Install dependencies separately.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
PYTHON="${PYTHON:-python}"
mkdir -p outputs/rebuild/tmp outputs/rebuild/mpl outputs/rebuild/cache outputs/rebuild/ipython
export TMPDIR="${TMPDIR:-$HERE/outputs/rebuild/tmp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$HERE/outputs/rebuild/mpl}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HERE/outputs/rebuild/cache}"
export IPYTHONDIR="${IPYTHONDIR:-$HERE/outputs/rebuild/ipython}"
"$PYTHON" experiment.py --out outputs/rebuild/result.json
"$PYTHON" -O experiment.py --out outputs/rebuild/result-O.json
"$PYTHON" audit.py --report outputs/rebuild/result.json --out outputs/rebuild/audit.json
"$PYTHON" plots.py --report outputs/rebuild/result.json --directory outputs/rebuild/figures
"$PYTHON" build_notebook.py
"$PYTHON" execute_notebook.py experiment.ipynb
"$PYTHON" build_pdf.py --directory outputs/rebuild/pdf
printf '\nRebuilt outputs in outputs/rebuild and refreshed experiment.ipynb.\n'
