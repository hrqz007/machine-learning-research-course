# 第077课 精确概率推断

本讲按既定166单元大纲承接前课，提供完整中文自学材料。直接先修：019、021、076。

## 阅读顺序

1. [lecture.pdf](lecture.pdf) / [lecture.md](lecture.md)：从零解释、手算、五幅机制图、练习。
2. [lab.pdf](lab.pdf) / [lab.md](lab.md)：独立实验设计、操作目的与预期结果。
3. [experiment.ipynb](experiment.ipynb)：新内核中真实顺序执行的Notebook，含结果与内嵌图。
4. [answers.pdf](answers.pdf) / [answers.md](answers.md)：逐题推导、参考结果与误区纠正。

## 离线复现

Python 3.12或3.11。安装requirements.txt后在本目录执行：

```bash
python experiment.py --out outputs/result.json
python test_experiment.py --out outputs/tests.json
python -O test_experiment.py --out outputs/tests-optimized.json
python plots.py --directory outputs/figures
python execute_notebook.py --out outputs/experiment.ipynb
```

数值算法仅使用标准库；绘图需matplotlib；Notebook用nbclient启动当前解释器的临时内核。网络不参与实验，数据由[generate_data.py](generate_data.py)完全重建。安装依赖是复现前的准备步骤。

PDF重建沿用已有单元的本地MathJax/WeasyPrint方式：安装build-requirements.txt并在本目录执行npm install；系统需Pango与Noto CJK字体。执行python build_pdf.py --directory outputs/pdf。build_all.sh串联所有步骤，默认重建仅写outputs目录。

## 文件与边界

- [exact_inference.py](exact_inference.py)：逐行教学注释的核心实现；[experiment.py](experiment.py)组织确定性实验。
- [data/README.md](data/README.md)：变量、状态及表字段解释；generation.json含生成数据字节数与SHA256。
- [experiment-result.json](experiment-result.json)、[test-result.json](test-result.json)、[test-result-optimized.json](test-result-optimized.json)：冻结实测结果。
- [sources.md](sources.md)、[source-checks.json](source-checks.json)：一手来源与实际核验范围。
- [verification.json](verification.json)：本次执行、Notebook和PDF验收记录。
- [figures/README.md](figures/README.md)：五幅PNG与SVG原创示意图。

这是小型离散模型的教学实现；不把浮点枚举结果当作现实因果证据，也不声称生产规模性能。图、表和代码均为本课程原创；外部参考仅用于定义核对。
