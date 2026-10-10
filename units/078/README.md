# 第078课 Monte Carlo与MCMC

严格承接原纲078；直接先修019、021–022、024、076。基础段包括Monte Carlo积分和重要性采样；深入段包括平稳/详细平衡与遍历条件、MH、Gibbs和误差/混合诊断。

## 自学顺序

1. [完整正文PDF](lecture.pdf) / [Markdown](lecture.md)：从抽样机制到推导与场景，五幅原创图和七道练习。
2. [独立实验PDF](lab.pdf) / [Markdown](lab.md)：解析对照、受控实验、逐步操作与验收。
3. [真实执行Notebook](experiment.ipynb)：新Python进程内IPython顺序计算、内嵌图、诊断和测试。
4. [详细答案PDF](answers.pdf) / [Markdown](answers.md)：完整推导、固定结果、错误解释与评分。

## 离线复现

Python3.11/3.12安装requirements.txt后在本目录执行：

```bash
python generate_data.py --directory outputs/generated-data
python experiment.py --out outputs/result.json
python test_experiment.py --out outputs/tests.json
python -O test_experiment.py --out outputs/tests-optimized.json
python plots.py --directory outputs/figures
python execute_notebook.py --out outputs/experiment.ipynb
```

samplers.py为带中文注释的核心实现。固定原始数据见[data说明](data/README.md)；不用网络、不调用收费模型。experiment-result.json和两份test-result文件保存已执行结果。PDF可安装build-requirements.txt及npm install后运行python build_pdf.py --directory outputs/pdf，依赖系统Pango与Noto CJK字体。build_all.sh将重建产物写outputs，不覆盖已验收文件。

## 实验边界

刻意保留tiny步长组不混合的失败结果。ESS为简化单链正序列截断估计，Rhat为经典split版本，均非Stan高级诊断的复刻。理论条件、有限运行诊断与现实模型适用性分别讨论。运行环境、页数、字体和全页视觉检查证据见[verification.json](verification.json)；来源和核对范围见[sources.md](sources.md)。
