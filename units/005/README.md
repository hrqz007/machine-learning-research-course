# 第五讲 NumPy 数组与数值计算

核心问题：怎样用数组表示一批样本，并确保向量化没有算错？

先修为第三讲 Python 执行与基本运算、第四讲函数与可调试程序。本讲从零解释数组形状、轴、索引、切片、广播、逐元素计算与归约，加入视图、数值类型、精度和成本的初步认识。双特征预测只使用逐元素乘法与按行求和，不要求矩阵乘法基础。

## 学习顺序

1. 阅读 lecture.pdf（另附lecture.md），完成纸笔计算与 14 道练习。
2. 按 lab.pdf（另附lab.md） 打开 experiment.ipynb，逐格运行后重启内核并运行全部。
3. 执行 experiment.py，用新进程复核数值与边界检查。
4. 对照 answers.pdf（另附answers.md） 的逐步解答，记录自己第一次失败的原因。

## 文件

- lecture.pdf / lecture.md：完整主讲稿，含 10 幅彩图和参考资料。
- lab.pdf / lab.md：独立实验指南，含环境检查、预期输出、故障处理和提交清单。
- answers.pdf / answers.md：14 道练习的完整解答。
- experiment.ipynb：19 个循序渐进的代码单元，已保留一次真实内核顺序执行输出。
- experiment.py：预测、评分、输入检查、10 组自动验证与重复计时。
- data：三份小型合成数据 CSV 与数据字典。
- figures：10 幅 PNG 教学图，附在讲稿对应位置。
- source-checks.json：官方文档核对与资料完整性记录。
- test-result.json：脚本实测结果、版本、计时与 Notebook 核验范围。
- build_figures.py：可选插图生成脚本。

## 运行

在本讲目录启动终端，用已激活的 Anaconda Python 环境运行：

```bash
python experiment.py --output my-test-result.json
```

只检验数值与边界：

```bash
python experiment.py --skip-timing --output my-checks-only.json
```

正常结果为 status=passed、checks=10。A、B、C 训练 MAE 为 10、0、7.5 万元；已披露的固定测试中 B 为 5 万元，训练均值基线为 15 万元；双特征预测为 115、140、175 万元。Notebook 应重启内核后按顺序运行全部，而不是依赖已有变量。

数值实验只依赖 NumPy，已核验 Python 3.12.14、NumPy 2.3.5。运行 Notebook 还需 Jupyter 与 Python 内核。所有数据本地提供，CPU 离线运行，无 API、付费服务或大规模下载。

## 可选插图重建

插图生成使用 Matplotlib 3.10.8 与 Noto Sans CJK 字体。Linux 默认检查 /usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc；其他系统应先安装可用中文字体，并在 build_figures.py 中指定相应字体文件。正常学习无需重建插图，直接使用 figures 中已有 PNG 即可。

```bash
python build_figures.py
```

## 数据与验证边界

全部数据为合成教学数据，没有真实房源或个人信息。双特征规则事先指定，用于验证数组实现，不代表真实估价或因果结论。固定测试数据已在前序课程披露，本讲只复算，不能宣称它仍是未知封存测试。

脚本在新 Python 进程中实际执行。Notebook 使用新 Python 进程中的真实 ipykernel InProcessKernel 按序执行，19 个代码单元无未处理异常，输出已保存。此记录不包含浏览器 Jupyter 界面、socket 内核传输或学习者本机的 Anaconda 安装测试。

计时是当前 CPU 环境的一次重复测量，不承诺固定加速倍数。输入准备排除在计时外；小数组的固定开销、大数组的内存成本和不同机器负载都会影响结果。

输入类型检查对普通列表/元组会在数组转换前逐层拒绝布尔叶子，包括与数值混合的bool或np.bool_；已有ndarray只根据当前dtype检查，无法恢复早先转换丢失的原始类型。
