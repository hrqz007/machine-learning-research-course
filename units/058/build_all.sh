#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python experiment.py --out outputs/rebuild/result.json
python test_experiment.py --report outputs/rebuild/result.json --out outputs/rebuild/test-result.json
python plots.py --report outputs/rebuild/result.json --directory outputs/rebuild/figures
python execute_notebook.py --out outputs/rebuild/experiment.ipynb
python build_pdf.py --directory outputs/rebuild/pdf
