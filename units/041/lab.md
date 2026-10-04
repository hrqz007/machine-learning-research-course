<style>table {break-inside:avoid} figure {break-inside:avoid}</style>

# 第041讲 独立实验指南

本指南可以不打开正文直接执行。目标是交付可重跑的回归比较证据：先让四行手算、梯度和独立求解路线吻合，再比较求解精度与真实成本，最后核验模型选择隔离和预测区间的限制。数据均为原创合成数据，不需要注册外部账号，不访问在线数据服务。

## 1 准备运行环境

推荐Python3.12，新建虚拟环境。在本单元文件夹打开终端，Windows PowerShell的激活命令与Linux/macOS不同。不要将教材目录作为覆盖输出的目标。

```text
python -m venv .venv
```

Linux/macOS激活：

```text
source .venv/bin/activate
```

Windows PowerShell激活：

```text
.venv\Scripts\Activate.ps1
```

激活后安装固定依赖。若本机命令是python3，请统一替换python。不要为安装依赖而改动数据或Notebook中的摘要。

```text
python -m pip install -r requirements.txt
python -c "import numpy,scipy,sklearn; print(numpy.__version__,scipy.__version__,sklearn.__version__)"
```

requirements.txt记录作者实际使用的数值和Notebook依赖。environment.yml是可选conda配置；可使用 `conda env create -f environment.yml` 再 `conda activate ml-unit041`，但作者未实测Anaconda或conda安装流程。PDF已提供，不要求学习者安装教材构建工具。

主脚本依自身位置寻找默认data目录。为防止覆盖任意已交付资产，建议把结果保存在单元目录以外的一个新目录；Notebook专用notebook_results目录允许替换它自己的运行结果。下面命令使用单元目录之外的results041目录；也可以指定任何已存在的外部结果目录。

## 2 最简命令与应当看到的结果

以下以从单元目录运行、输出到同级新目录为例。首次创建后无需重复mkdir。

```text
mkdir ../results041
python experiment.py --output ../results041/report.json
python audit.py --output ../results041/audit.json
python benchmark.py --output ../results041/timing.json
python -O audit.py --output ../results041/audit-optimized.json
```

experiment.py保存完整科学结果，终端只显示小摘要：单位041、选中λ=0.1、标准路径12条未达标，以及故意大步长路径的objective_increase_rejected。完整JSON含所有监测状态和覆盖重复记录，不只保留这四个数。

默认手算应得到 $w^1=(3/8,5/8)$、$w^2=(29/64,49/64)$；三个总目标依次 $7/4$、$29/64$、$1597/4096$。OLS参照 $(1/2,1)$，Ridge λ=1/2参照 $(5/11,9/11)$。这是当前data/hand.csv的答案，不能用于改过的数据。

完整audit会在新的普通/-O进程和不同工作目录执行脚本，再比较全部JSON摘要；还真实调用优化模式检查坏尾字段、单轮预算和驻点手算初值。耗时应允许几秒到几十秒，取决于机器。`--quick`跳过子进程重跑，只用于交互过程中快速复查，不能代替完整验收。

## 3 文件和角色先分清

|文件或字段|解决的问题|
|---|---|
|data/hand.csv|完整两轮纸笔、独立有理数核验|
|data/benchmark.csv|三病例五优化器，同目标数值比较|
|data/selection.csv|训练72、验证48、封存测试120|
|data/config.json|预声明阈值、候选、预算、种子|
|experiment-result.json|确定性科学参照，包含失败与完整轨迹|
|benchmark-result.json|作者实际计时原始样本，重跑不要求时间相同|
|test-result.json|Fraction、库、拒绝路径与新进程证据|
|verification.json|公开交付文件字节、摘要与验收范围|

不要把benchmark的病例当作selection的训练集。不要把生成真值元数据传给选择器。test字段只评价锁定后的模型；选择函数参数中不存在测试数组。

## 4 实验A 从纸笔走到程序

先不运行优化器。按四行顺序算预测、残差、半平方损失、平均损失和两个平均梯度贡献。惩罚只加一次。完成第一轮后，同时更新两个坐标，然后在同一四行上重新前向；这张新表才是第二轮的起点。

用audit.json中的exact_hand.states核对每个分数。它来自Fraction标量运算和二维消元，独立于NumPy主梯度。必须解释差异出现在哪一行、哪一个平均因子或哪一个局部导数；不能只把程序输出抄到纸上。

验收：三张逐行表的全部数值吻合；第二轮后总梯度是 $(-7/128,-17/128)$，并未到驻点。若有人把第三行起始标签0改为1，默认参照和Notebook契约都应失效，而非继续显示旧答案。

## 5 实验B 检查梯度与参照

查看hand.gradient_checks。按point分别列h与relative_error；再和wrong_sign_error比较。挑三个非驻点是为了避免某坐标恰为零的盲区。平方目标的中央差分不存在三阶截断项，但浮点消减仍会在小h处放大。

审计另外执行32组固定种子的二进制有理小问题。最大差分误差阈值是 $10^{-8}$。库对照使用NumPy增广最小二乘和scikit-learn的SVD Ridge，核对alpha=nλ；复制所有行后，平均目标与系数应不变。

验收：不能只交“check passed”。说明比较的是哪个标量目标、哪个点、哪个扰动、什么误差尺度、哪个独立参照。查看reference_rank与reference_singular_values，区分秩亏的不可识别和近共线的数值困难。

## 6 实验C 同一停止条件下读完整轨迹

在optimizers.runs中按case、lambda、method筛选。五方法为gd、diagonal_gd、minibatch、adam、newton。共同门槛是完整梯度相对残差不超过1e−6，最多400个记录轮次。小批量每轮8个16行子集，不能把400轮读成400次单步。

对每条路径至少记录status、final.epoch、relative_gradient、quadratic_gap、coefficient_distance、prediction_distance_rms、cost与完整trace。按gradient_rows/n画曲线；这是本实现梯度样本访问量，不是FLOPs也不是时间。

默认regular OLS中GD16步、Adam263步达标；collinear OLS中四种一阶路线都耗尽预算，Newton达到阈值。scaled OLS中对角GD11步达标，普通GD耗尽预算。不要只选其中一张有利图称“新方法更好”。

验收：保留全部30条路径，明确12条预算耗尽。不能把失败删除、改成收敛、替换成另一个种子，或用无限延长预算后得到的结果偷偷覆盖原配置。

## 7 实验D 失败与真实成本

查看optimizers.injected_failure。故意大步长使用η=3/L；首候选增加目标并被拒绝。已提交参数仍为零，但候选目标和成本都必须保留。

核算3次完整目标/梯度为何各360行：1次独立参照评价、1次初始评价、1次失败候选。另有Gram构造、谱求解、增广最小二乘和预测距离监测。不同字段不能简单相加当作FLOPs，因为矩阵工作量与每行维度不同。

benchmark.py独立测31个配置，包含30条标准路径及1条故障路径。每条1次暖机、3次正式测量，BLAS限为单线程。比较median_ns、min_ns、max_ns和每次sample.status。所有失败测量仍保留。

验收：写清计时范围是诊断求解调用，不包括读文件、中心化、模型选择、覆盖模拟、画图或写JSON。不能要求计时数值逐字节相同；应比较相同协议与科学结果。也不能拿未达标GD的更短时间与高精度结果直接作胜负结论。

## 8 实验E 选择锁与负面结果

阅读selection.lock.candidates。训练统计量只拟合一次，候选λ已预声明。通过数值门槛后按精确验证MSE选最小值，完全相等再选较大的λ。锁定后不重训。

默认λ=0.1获选。封存测试MSE约0.775054，OLS基线约0.767754。请写一句如实结论：验证选择在这次测试上没有击败OLS。不要据此回到验证前更改候选列表；新的探索须另存完整协议。

在副本里给全部测试标签加25，重新运行；选择锁与其摘要应完全相同，测试MSE应改变。另将测试特征第一列乘9，选择锁也应不变。audit.py已实际实现这两种检查。不能仅用代码文本中没有test字样就宣布不存在泄漏。

## 9 实验F 区间对象与失败边界

coverage.rows保留600次独立训练重复。该模拟没有选择Ridge，而是32行、含截距的正确OLS模型。每行有fitted、s、mean_half、prediction_half、new_y、shifted_y和四个覆盖标记。

验收应重算覆盖计数：真实均值580、同分布未来观测562、错误把均值区间当预测区间247、未来噪声SD变为1.5时321。分母均为600。用约0.0089的名义Monte Carlo标准误解释抽样波动，不把0.9367与0.95的差异当作自动否定。

随后写清：一个区间没有覆盖不代表实现错误；整体覆盖不错不保证所有子群；未来噪声偏移是统计假设破坏；t区间不自动适用于验证选择后的Ridge；回归系数不是干预效应。

## 10 Notebook的真实执行顺序

Notebook第一格用标准库核验契约文件的完整字节摘要，未通过前不导入数值模块、不生成结果。请在新的内核中从第一格运行全部单元，不跳格、不沿用旧变量。完整Notebook会重新跑科学实验、quick数值审计、实际计时和12幅图，并将运行报告保存到notebook_results。

代码明确以PNG字节显示图片，不能只看到“Figure对象”的文本就认为图形渲染成功。所有图片均根据本次运行的数据生成；Notebook中的计时也是本次实测，可能不同于讲义所引作者计时。

作者使用新Python进程内的真实ipykernel InProcessKernel完成顺序执行，并测试契约文件篡改第一格即拒绝。未实测浏览器Jupyter UI或Anaconda安装。常规桌面使用者可在自己的Jupyter环境中打开；若需安装Notebook/Lab前端，遵照该产品官方文档，前端不属于本包已核验依赖。

## 11 自定义输入要另存

复制data目录到新的外部目录，再改配置或CSV。不要直接改交付文件后跳过Notebook摘要。

```text
python experiment.py --data-dir ../my041-data --output ../results041/custom.json
```

--config可显式覆盖配置路径，但不会与默认配置合并。任何坏尾字段、重复JSON键、布尔/字符串数值、错维度、非有限值、过小非零原数都应拒绝。合法max_epochs=1仍要成功写出真实未收敛状态，不能写完报告后因摘要索引第二轮而崩溃。

输出父目录必须先存在。禁止输出到data、源码、PDF、Notebook、图或其它交付资产；禁止符号链接路径、符号链接父目录和多硬链接文件。原子写入失败应保留旧字节。普通与-O模式的拒绝行为须相同。

## 12 最终提交与评分标准

提交一份简短研究报告及可重跑文件：数据版本和划分、目标公式、完整两轮手算、梯度检查、独立参照、全部算法状态、真实成本与计时边界、选择锁及测试报告、覆盖实验、失败解释和可证伪下一问题。不要只交图片，也不要只交运行日志。

建议评分：数学及行级账本25%，独立代码核验20%，公平预算和失败记录20%，选择隔离15%，区间和结论边界15%，可重跑与可读性5%。出现测试标签选参、删失败、伪造计时或未检查的因果结论时，应先修复证据链，而非用更漂亮排版补偿。
