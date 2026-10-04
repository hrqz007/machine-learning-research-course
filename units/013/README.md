# 第十三讲 一元导数与局部变化

从一个参数的平方误差出发，理解导数定义、切线、乘积与链式法则，再用精确分数和浮点差分核对。

- 正文：`lecture.pdf`；可读源稿：`lecture.md`
- 独立实验指南：`lab.pdf`
- 九题逐步详解：`answers.pdf`
- 分步实验：`experiment.ipynb`
- 标准库核心与独立脚本：`experiment.py`
- 原创三行教学数据与说明：`data/`
- 11幅原创解释图：`figures/`

在本目录运行 `python experiment.py`，输出自动写入 `outputs/report.json` 和 `outputs/difference_scan.csv`。也可用 `--output my_results` 另存。核心仅需 Python 3.12 标准库；Notebook 绘图需要 Matplotlib，打开 Notebook 需要 JupyterLab/IPython。实际版本记录在 `test-result.json`。`requirements.txt` 仅列 Notebook 的绘图依赖。

先纸笔计算 L(3/2)=1/6、L'(3/2)=-4/3，再从清空内核顺序运行 Notebook。脚本通过205项精确核对与35项边界/基本结果检查；验证函数使用显式异常，在 `python -O` 下仍执行。实际运行记录及范围见 `verification.json`，一手来源见 `source-checks.json`。

数据和实验是固定的无量纲合成例子；不代表真实学习任务表现。差分扫描不是可导性证明，也不是通用自适应求导器。极小步长被拒绝时保留失败原因。零误差和失败项不能悄悄替换成对数曲线上的数值。

Notebook 使用新 Python 进程中的真实进程内内核执行并保留输出。浏览器UI、外进程socket传输、学习者本机Anaconda安装未测试。跨平台浮点小数和最佳步长可能不同。详情见实验指南。
