#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export MPLCONFIGDIR="$PWD/outputs/mpl"
export XDG_CACHE_HOME="$PWD/outputs/cache"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
python generate_data.py --directory outputs/generated-data
python experiment.py --out outputs/result.json
python test_experiment.py --out outputs/tests.json
python -O test_experiment.py --out outputs/tests-optimized.json
python plots.py --directory outputs/figures
python execute_notebook.py --out outputs/experiment.ipynb
python build_pdf.py --directory outputs/pdf
