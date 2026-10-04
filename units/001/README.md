# 001 机器学习到底在学什么

先读 lecture.pdf，再按 lab.pdf 做纸笔与代码实验。正文是已完成的原第一讲，实验完整复现其中四行训练与两行教学测试数据。所有房价均为合成记录，不用于估价。

## 文件

- lecture.pdf：24 页原版正文，含图解、纸笔实验、练习与详解
- lab.pdf：独立实验指南，含运行步骤、故障注入、练习与详解
- experiment.ipynb：从上到下运行的 Notebook
- experiment.py：相同数据流程的命令行脚本，仅需 Python 标准库
- data/：训练、测试输入与单独标签；详见 data/README.md
- figures/ 与 make_figures.py：本实验指南原始教学图与可重建脚本
- source-checks.json：一手资料核验记录
- verification.json：实际执行和 PDF 检查记录

## 最小运行方式

Python 3.10 或更新版本。脚本本身没有第三方依赖。保留本目录结构，在终端运行：

```bash
python experiment.py --phase train
python experiment.py --phase predict
python experiment.py --phase evaluate
python experiment.py --self-test
```

第一次练习不要在预测前打开 data/test_labels.csv。答案也已在教程公开，故它只是隔离流程练习，不是保密盲测。Notebook 使用 Jupyter 的重启内核后全部运行。若用已有 Anaconda，选择带有 Jupyter 的环境即可。

预期候选 B，训练 MAE 为 A 10、B 0、C 7.5；测试预测 140、180，测试 MAE 5；固定训练均值基线 150 的测试 MAE 15。单位均遵循数据字典。

results/ 在运行时生成。改变训练数据后请用新的 --output 目录保留新旧实验；不覆盖旧冻结记录来掩盖改变。环境实测信息见 verification.json；共享 environment.yml 是课程建议环境，不把未逐平台测试的安装步骤称为已验证。

## 可选图源重建

实验脚本与 Notebook 的计算不依赖 matplotlib。只有重新生成 figures/ 才需要安装 matplotlib 与 Noto Sans CJK 字体。make_figures.py 当前是 Linux 制作脚本，字体路径为 /usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc；其他平台需把该路径改成本机已安装的 Noto Sans CJK 字体位置。图已随课程提供，无需重建即可完成实验。
