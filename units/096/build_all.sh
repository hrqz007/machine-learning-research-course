#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python experiment.py
python draw_mechanism.py
python -m unittest -v test_experiment.py
python -O -m unittest -v test_experiment.py
python execute_notebook.py
python build_pdf.py
