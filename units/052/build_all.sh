#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
PYTHON="${PYTHON:-python}"
mkdir -p outputs/rebuild/{tmp,mpl,cache,ipython}
export TMPDIR="${TMPDIR:-$HERE/outputs/rebuild/tmp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$HERE/outputs/rebuild/mpl}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HERE/outputs/rebuild/cache}"
export IPYTHONDIR="${IPYTHONDIR:-$HERE/outputs/rebuild/ipython}"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$PYTHON" experiment.py --out outputs/rebuild/result.json
"$PYTHON" -O experiment.py --out outputs/rebuild/result-O.json
"$PYTHON" audit.py --report outputs/rebuild/result.json --out outputs/rebuild/audit.json
"$PYTHON" -O audit.py --report outputs/rebuild/result-O.json --out outputs/rebuild/audit-O.json
"$PYTHON" plots.py --report outputs/rebuild/result.json --directory outputs/rebuild/figures
"$PYTHON" build_notebook.py --out outputs/rebuild/unexecuted.ipynb
"$PYTHON" execute_notebook.py --notebook outputs/rebuild/unexecuted.ipynb --out outputs/rebuild/executed.ipynb
"$PYTHON" build_pdf.py --directory outputs/rebuild/pdf
"$PYTHON" verify_rebuild.py
printf '\nRebuilt outputs are in outputs/rebuild. Shipped files preserved.\n'
