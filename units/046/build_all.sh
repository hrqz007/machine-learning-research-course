#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
"$PYTHON" "$ROOT/experiment.py"
"$PYTHON" "$ROOT/audit.py" --out "$ROOT/outputs/full"
"$PYTHON" -O "$ROOT/audit.py" --out "$ROOT/outputs/full-optimized"
"$PYTHON" "$ROOT/plots.py"
"$PYTHON" "$ROOT/execute_notebook.py" --out "$ROOT/outputs/fresh-kernel.ipynb"
if [[ "${BUILD_PDF:-0}" == 1 ]]; then
  "$PYTHON" "$ROOT/build_pdf.py" --directory "$ROOT/outputs/pdf" --figure-directory "$ROOT/outputs/figures"
fi
