# 第077课 实验：三种方法回答同一后验问题

## 1 固定模型、目标与验收

本实验只做推断。数据是原创chain.json和star.json；没有现实设备记录，没有从样本学习参数，没有下载依赖数据。主查询为P(X0=1|E=1)，随后扩展到X0至X3全部边缘。对照方法有完整状态穷举、通用二元因子变量消元、链专用前向后向和积。

在运行之前，先写出预期：给定末端亮灯会提高四个状态为高负荷的后验；三种方法应在10的负12次方以内一致；换消元顺序只改代价，不改后验；零概率证据应明确拒绝。星图实验只比较两条事先指定顺序，不从运行时间噪声挑胜者。

| 检查对象 | 可重复验收 |
|---|---|
| 链的四个后验 | 穷举、消元、和积逐个比较 |
| 证据质量 | 三种方法均得0.40432 |
| 星图同一查询 | 好坏顺序均得P(L5=1)=0.45 |
| 临时表峰值 | 先叶后中心4项，先中心128项 |
| 不可能证据 | 明确ValueError，不返回伪概率 |

## 2 准备与第一个结果

进入本目录，使用Python 3.12或3.11。核心算法只依赖标准库；绘图和Notebook需要requirements.txt。默认输出写outputs，以保留随课参考报告。

```bash
# 首次准备环境时安装运行依赖。
python -m pip install -r requirements.txt
# 将原创参数重建到新目录，与冻结data文件对照。
python generate_data.py --directory outputs/generated-data
# 真实执行三方法比较与星图顺序实验。
python experiment.py --out outputs/result.json
```

预期输出包含chain_p1_given_e1的四个数，约0.502572、0.559458、0.664820、0.846755。max_method_error应接近浮点舍入级，不要求恰好为0。evidence_probability是0.40432。若你得到0.5附近的证据概率，应检查是否错把emission当成后验，或把transition的行列读反。

<div class="page-break"></div>

## 3 看见中间因子，才算理解消元

```python
from exact_inference import chain_factors, eliminate  # 组模型与做推断分工明确。
factors = chain_factors([.6,.4], [[.85,.15],[.25,.75]], 4, [.1,.9])
posterior, mass, trace = eliminate(factors, 'X0', {'E':1}, ['X3','X2','X1'])
print(posterior.table)  # 查询X0的两个归一化概率。
print(mass)  # 未丢弃尺度，故仍能读出证据概率。
for step in trace:
    print(step)  # 每步检查消去变量、临时范围、条目数和结果范围。
```

factors是多张局部表的列表，scope是每张表的变量顺序。evidence字典固定E=1；order只列隐藏变量，不能包含X0或E。手工顺序从右端向左消元，最大临时范围含两个变量。将顺序改为X1、X2、X3，检查临时表有时扩展到三个变量，但后验仍相同。

继续检查任意轴顺序：构造范围(A,B)的f与范围(B,A)的g，手算A=1,B=0时的乘积。正确代码会用f的键(1,0)乘g的键(0,1)。按数组外形直接逐项乘可能悄悄给出错误，却仍得到非负、和为1的分布。因此只检查归一化远远不够。

## 4 消息与星图实验

从chain_sum_product取得forward、backward及marginals。选择i=1，把forward[1][1]×backward[1][1]手工相乘，再除两状态总和；与marginals[1][1]对照。不要把一个方向的消息单独当成后验。

星图使用六个叶子，查询L5。分别执行先L0至L4后C、先C后L0至L4两种顺序。记录最大临时表项数，不把几毫秒的运行波动当作理论复杂度证据。两者Z均为4096；先中心第一次乘积128项、消去后64项，这两个数字的对象不同。

```bash
# 显式unittest覆盖任意顺序、轴对齐、非法输入及零概率证据。
python test_experiment.py --out outputs/tests.json
# 优化模式下测试仍然运行，不能依赖被删除的语言级assert。
python -O test_experiment.py --out outputs/tests-optimized.json
# 重画五幅机制图与后验图。
python plots.py --directory outputs/figures
```

<div class="page-break"></div>

## 5 Notebook与可读材料

```bash
# 用当前解释器的新内核顺序执行，不下载数据。
python execute_notebook.py --out outputs/experiment.ipynb
# 可选：准备PDF的Python与本地公式组件。
python -m pip install -r build-requirements.txt
npm install
# 重建讲义、实验、答案三份PDF。
python build_pdf.py --directory outputs/pdf
```

Notebook会打印因子范围、消元trace、四处前后向消息、星图两个顺序的成本，以及正确拒绝不可能证据的过程；最后运行六组测试。每个代码单元须有执行序号并无error输出。截图好看但未执行的Notebook不算完成实验。

PDF还依赖系统Pango和Noto CJK中文字体。数值代码不要求PDF组件。图片可同时查看figures的PNG与SVG，核对图3消息方向、图5填边和图4纵轴含义。重建PDF后应逐页观察文字是否溢出、符号是否缺失、公式与图注是否清楚。

## 6 失败注入与报告写法

把末端emission改为[0,0]，查询E=1。现在任何状态下都不可能亮灯，证据质量为0；eliminate、enumerate_query与chain_sum_product都应拒绝。这是定义域保护，不是算法收敛失败。

把某个因子所有值乘7，后验应保持不变，而未归一化质量乘7。对于原先归一化的贝叶斯网络，改变后的因子乘积不再直接是归一化联合概率；此时返回质量只能称相应未归一化模型质量，不能继续不加说明叫证据概率。此实验用于检查尺度处理。

提交实验记录包括：变量及状态字典、主查询定义、手算q、三种方法数值对照、两个消元trace、星图范围变化、普通与优化模式报告、已执行Notebook，以及一个明确拒绝路径。说明当前实现只支持二元离散因子，并未做通用有环sum-product或大规模稳定对数运算。

反思问题：如果三种方法都得到同一个后验，究竟确认了什么？答案是该人工模型下计算的一致性；模型是否可信、参数是否由适当数据学习、推断是否具有现实行动意义，都需要独立证据。详解见answers.md。
