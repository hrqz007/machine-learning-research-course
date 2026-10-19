#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python experiment.py
python -m unittest -v test_experiment.py
python -O -m unittest -v test_experiment.py
python execute_notebook.py
# PDF prerequisites: requirements in build-requirements.txt, Node.js, npm install.
python build_pdf.py --directory outputs/pdf
