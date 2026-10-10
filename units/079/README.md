# 第079课 变分推断与ELBO

按既定166单元大纲继续，详细先修和适用边界见讲义。全部数据和图原创，无需付费接口或联网下载数据。

## 自学顺序

1. [lecture.pdf](lecture.pdf) / [lecture.md](lecture.md)：动态机制、数学推导、五幅机制图、应用与误区。
2. [lab.pdf](lab.pdf) / [lab.md](lab.md)：独立操作目的、信息权限、预测与验收。
3. [experiment.ipynb](experiment.ipynb)：已在新Python进程内用IPython逐单元真实执行，含文本和内嵌图。
4. [answers.pdf](answers.pdf) / [answers.md](answers.md)：逐题推导、参考结果与反例。

## 离线复现

Python 3.12（3.11也可尝试）；安装requirements.txt后，在本目录运行：

```bash
python generate_data.py --directory outputs/generated-data
python experiment.py --out outputs/result.json
python test_experiment.py --out outputs/tests.json
python -O test_experiment.py --out outputs/tests-optimized.json
python plots.py --directory outputs/figures
python execute_notebook.py --out outputs/experiment.ipynb
```

[variational.py](variational.py)是核心实现；experiment.py组织固定协议。默认实验读取data中的冻结数据，生成outputs下副本不会自动替换输入。数据SHA256见data/generation.json。测试使用显式unittest断言，普通与-O两种模式各9项。源链接、核验范围见[sources.md](sources.md)和source-checks.json。

Notebook执行器在启动的新Python进程中创建IPython InteractiveShell，从上到下执行全部单元并捕获真实输出；受当前环境限制，没有启动Jupyter socket内核，也没有声称做过浏览器交互测试。Notebook文件可以在常规Jupyter中打开；如改为常规内核重跑，应重新检查所有输出。

PDF重建沿用本地MathJax/WeasyPrint：安装build-requirements.txt，在本目录npm install，并准备Pango和Noto CJK字体，再执行python build_pdf.py --directory outputs/pdf。默认重建仅写outputs，build_all.sh串联全部操作。安装完成后重建不联网。

## 冻结证据与边界

experiment-result.json、test-result.json、test-result-optimized.json为实跑结果；verification.json记录Notebook及PDF检查。五幅PNG/SVG说明在figures/README.md。数据字典见data/README.md。数值结果仅针对明确的人工模型与固定协议，不声称生产规模或真实总体表现。
