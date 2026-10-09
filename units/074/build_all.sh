#!/usr/bin/env bash
set -euo pipefail
# 切换到脚本目录，避免依赖调用者所在目录。
cd "$(dirname "$0")"
# 输出写入新目录，保留冻结材料。
mkdir -p outputs/rebuild
python generate_data.py --directory outputs/rebuild/data
python experiment.py --out outputs/rebuild/result.json
python test_experiment.py --report outputs/rebuild/result.json --out outputs/rebuild/test.json
python -O test_experiment.py --report outputs/rebuild/result.json --out outputs/rebuild/test-optimized.json
python plots.py --report outputs/rebuild/result.json --directory outputs/rebuild/figures
python execute_notebook.py --out outputs/rebuild/experiment.ipynb
python build_pdf.py --directory outputs/rebuild/pdf
