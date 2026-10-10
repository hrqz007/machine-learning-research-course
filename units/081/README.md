# 第081课 异常检测与密度估计

严格承接既定166单元大纲，完整中文自学材料与已执行离线实验。

## 阅读顺序

1. [lecture.pdf](lecture.pdf) / [lecture.md](lecture.md)：机制、推导、五幅原创图与结论边界。
2. [lab.pdf](lab.pdf) / [lab.md](lab.md)：先写预测，再手算、运行与审计。
3. [experiment.ipynb](experiment.ipynb)：新Python进程内IPython逐单元真实顺序执行，含输出与内嵌图片。
4. [answers.pdf](answers.pdf) / [answers.md](answers.md)：详细推导、冻结实测结果与错误解释。

## 离线复现

Python 3.12；首次安装requirements.txt后，实验本身无网络调用，无在线模型与付费API。在本目录执行：

```bash
python generate_data.py --directory outputs/generated-data
python experiment.py --out outputs/result.json
python test_experiment.py --out outputs/test-result.json
python -O test_experiment.py --out outputs/test-result-optimized.json
python plots.py --directory outputs/figures
python execute_notebook.py --out outputs/experiment.ipynb
```

低资源环境建议设置OPENBLAS_NUM_THREADS=1、LOKY_MAX_CPU_COUNT=2；如缓存目录不可写，设置MPLCONFIGDIR、XDG_CACHE_HOME与IPYTHONDIR到自己的outputs子目录。Notebook执行器真实运行每个单元并保存stdout、stderr和display输出；本环境不允许Jupyter内核socket绑定，所以不声称使用了网络内核或测试了浏览器交互。

PDF重建依赖build-requirements.txt、Node.js与本地MathJax（package.json），以及Pango和Noto CJK字体。安装构建依赖和npm依赖后，运行python build_pdf.py --directory outputs/pdf。build_all.sh完整重建到outputs/rebuild。已有根目录文件是冻结交付，不被默认命令覆盖。

## 证据与文件

- [generate_data.py](generate_data.py)、[data/README.md](data/README.md)：数据来源、许可、职责、种子与哈希。
- [experiment.py](experiment.py)、[experiment-result.json](experiment-result.json)：预声明流程与实测结果。
- [test_experiment.py](test_experiment.py)、[test-result.json](test-result.json)、[test-result-optimized.json](test-result-optimized.json)：正常和优化解释器下显式unittest。
- [plots.py](plots.py)、[figures/README.md](figures/README.md)：五幅原创PNG与SVG及重建方式。
- [sources.md](sources.md)、[source-checks.json](source-checks.json)：一手资料链接与实际核验范围。
- [verification.json](verification.json)：代码、Notebook、PDF页数与视觉验收。

## 本课边界

真实负例来自Digits数字0，数字6仅是预定义任务外类别，不代表有害。训练与校准只使用正常样本；目标5%是可交换条件下的边际误报预算，不是本次固定校准集、每个子群或部署系统的保证。四种方法全部报告，不据测试表现宣布最终部署选型。

核心实现：[anomaly.py](anomaly.py)。
