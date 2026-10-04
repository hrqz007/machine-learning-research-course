# 面向研究的机器学习课程

从零基础逐步进入可推导、可实现、可检验的机器学习研究实践。

> **制作中。** 课程地图规划了 136 讲主体和 6 条高级分支共 30 个专题单元。本仓库目前提供完整课程大纲，以及第 001–015 讲的完整教学包。其余单元尚未完整发布，制作与验收状态见下方进度清单。规划目录不代表正文或实验已经完成。

## 从这里开始

- [课程大纲（PDF，48 页）](catalog/curriculum.pdf)：课程范围、先修关系、学习路线、逐讲目标与验收任务
- [第 001 讲：机器学习到底在学什么](units/001/README.md)
  - [正文 PDF（24 页）](units/001/lecture.pdf)
  - [独立实验指南 PDF（9 页）](units/001/lab.pdf)
  - [可运行 Notebook](units/001/experiment.ipynb)
  - [Python 实验脚本](units/001/experiment.py)
  - [教学数据及说明](units/001/data/README.md)
  - [执行与检查记录](units/001/verification.json)

- [第 002 讲：任务定义与数据生成](units/002/README.md)
  - [正文 PDF（16 页）](units/002/lecture.pdf) · [实验指南（5 页）](units/002/lab.pdf) · [详细答案（4 页）](units/002/answers.pdf)
  - [Notebook](units/002/experiment.ipynb) · [Python 脚本](units/002/experiment.py) · [验证记录](units/002/verification.json)
- [第 003 讲：Python 执行与基本运算](units/003/README.md)
  - [正文 PDF（17 页）](units/003/lecture.pdf) · [实验指南（6 页）](units/003/lab.pdf)
  - [Notebook](units/003/experiment.ipynb) · [Python 脚本](units/003/experiment.py) · [验证记录](units/003/verification.json)

- [第 004 讲：函数与可调试程序](units/004/README.md)
  - [正文 PDF（17 页）](units/004/lecture.pdf) · [实验指南（8 页）](units/004/lab.pdf)
  - [Notebook](units/004/experiment.ipynb) · [Python 脚本](units/004/experiment.py) · [122 项测试](units/004/test_experiment.py) · [验证记录](units/004/verification.json)
- [第 005 讲：NumPy 数组与数值计算](units/005/README.md)
  - [正文（20 页）](units/005/lecture.pdf) · [实验指南（6 页）](units/005/lab.pdf) · [详解（4 页）](units/005/answers.pdf)
  - [Notebook](units/005/experiment.ipynb) · [Python 脚本](units/005/experiment.py) · [验证记录](units/005/verification.json)
- [第 006 讲：读表清洗与可视化](units/006/README.md)
  - [正文（22 页）](units/006/lecture.pdf) · [实验指南（5 页）](units/006/lab.pdf) · [详解（4 页）](units/006/answers.pdf) · [报告示例（2 页）](units/006/example_report.pdf)
  - [Notebook](units/006/experiment.ipynb) · [Python 脚本](units/006/experiment.py) · [验证记录](units/006/verification.json)
- [第 007 讲：第一个可重现实验](units/007/README.md)
  - [正文 PDF（10 页）](units/007/lecture.pdf) · [实验指南（5 页）](units/007/lab.pdf)
  - [Notebook](units/007/experiment.ipynb) · [Python 脚本](units/007/experiment.py) · [验证记录](units/007/verification.json)

- [第 008 讲：函数代数与数学表达](units/008/README.md)
  - [正文（19 页）](units/008/lecture.pdf) · [实验指南（4 页）](units/008/lab.pdf) · [详解（4 页）](units/008/answers.pdf)
  - [Notebook](units/008/experiment.ipynb) · [Python 脚本](units/008/experiment.py) · [验证记录](units/008/verification.json)

- [第 009 讲：向量内积与距离](units/009/README.md)：[正文](units/009/lecture.pdf) · [实验指南](units/009/lab.pdf) · [Notebook](units/009/experiment.ipynb)
- [第 010 讲：矩阵与线性变换](units/010/README.md)：[正文](units/010/lecture.pdf) · [实验指南](units/010/lab.pdf) · [Notebook](units/010/experiment.ipynb)
- [第 011 讲：线性方程与最小二乘几何](units/011/README.md)：[正文](units/011/lecture.pdf) · [实验指南](units/011/lab.pdf) · [Notebook](units/011/experiment.ipynb)
- [第 012 讲：特征值与奇异值分解](units/012/README.md)：[正文](units/012/lecture.pdf) · [实验指南](units/012/lab.pdf) · [Notebook](units/012/experiment.ipynb)

- [第 013 讲：一元导数与局部变化](units/013/README.md)：[正文](units/013/lecture.pdf) · [实验指南](units/013/lab.pdf) · [Notebook](units/013/experiment.ipynb)

- [第 014 讲：多元导数与链式法则](units/014/README.md)：[正文](units/014/lecture.pdf) · [实验指南](units/014/lab.pdf) · [Notebook](units/014/experiment.ipynb)

- [第 015 讲：曲率与局部近似](units/015/README.md)：[正文](units/015/lecture.pdf) · [实验指南](units/015/lab.pdf) · [Notebook](units/015/experiment.ipynb)

- [完整单元导航（JSON）](catalog/curriculum.json)
- [逐单元制作与发布状态](manifest.json)
- [运行环境说明](shared/README.md)

## 怎样学习

先阅读正文并完成手算，再运行实验，对照输出与预期，最后完成练习并检查失败条件。按具体问题补齐先修，不必把全部目录读完才开始一个有边界的研究练习。

第 1 讲实验只需 Python 3.10 或更新版本，核心脚本只使用标准库。进入 units/001 后运行：

```bash
python experiment.py --phase train
python experiment.py --phase predict
python experiment.py --phase evaluate
python experiment.py --self-test
```

Notebook 的计算单元已在新进程中的 IPython InProcessKernel 顺序执行；Jupyter 浏览器界面、外进程内核通信及真实 Anaconda 安装尚未测试。图源重建所需的可选依赖见该讲说明。

## 内容与验收

每讲按问题、图解、数学推导、手算、实现、诊断、练习与详解展开。完成的教学包包含正文 PDF、独立实验指南、Notebook、Python 脚本、教学数据或合法下载说明、来源及验证记录。

玩具数据只用于对应机制的教学，不构成现实性能或新的研究贡献证据。来源记录链接原始资料，不包含受版权保护教材的整章复制。

## 使用说明

仓库公开，方便在线学习与交流。目前未附加开源许可证；公开可读不代表授予未明确约定的再许可权。
