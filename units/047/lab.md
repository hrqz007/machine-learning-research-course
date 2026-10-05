# 第047讲 实验手册

校准与预测不确定性：先锁定规则，再看评价数据

本手册可独立使用。你将完成三个任务：手算四行温度更新；用独立校准标签拟合一个温度；用训练、校准、评价三份数据构造split conformal区间。全部是合成机制。目标不是追求最低ECE，而是说明每个数值的对象、数据来源和有效条件。

## 1. 文件与运行环境

先解压整个单元，保留相对路径。`lecture.pdf`是详细推导，`answers.pdf`给出本手册全部问题的解答。`experiment.py`负责科学结果；`calibration.py`与`conformal.py`提供基础函数；`audit.py`提供独立数值参照；`plots.py`产生十张真实图。四份CSV、协议JSON、环境、Notebook及PDF构建脚本均随包提供。

建议Python 3.12的隔离环境。数值依赖固定为NumPy 2.3.5、SciPy 1.17.0、scikit-learn 1.8.0、Matplotlib 3.10.8。安装命令在你的课程副本内执行，不修改系统Python：

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

日常学习直接打开PDF不需要Node或字体。需要从Markdown重建PDF时，再安装`build-requirements.txt`、Node.js、MathJax以及Noto CJK字体，具体见README。生成结果写入`outputs/`或新目录，不覆盖原始CSV和交付PDF。

## 2. 固定协议，避免先看结果再改实验

主配置为`data/protocol.json`。开始前记下：分类种子4705，校准400行、评价4000行，5个固定等宽箱；回归主种子4707，60/99/400三份数据；另120次独立重复用4708；漏覆盖率0.1；噪声压力测试放大2.5倍；区间展示预先选择前24行。

原始分数$z=2(1.2x-0.3)$固定。温度拟合只允许看到400行校准logit与标签。回归直线只能用60行训练数据，99行仅算残差，400行仅评价。所有真概率、真均值列都是解释用的“实验室真相”，不得作为实际拟合输入。

第一次运行不要更改协议。完成后可复制整份单元到新目录研究扩展，明确标注新实验。不能看过评价集后改温度范围、箱数或选展示样本，再把它称为同一个封存实验。

## 3. 完整执行与结果核对

在单元目录执行：

```bash
python experiment.py --out outputs/result.json
python audit.py --out outputs/audit.json
python -O experiment.py --out outputs/result-optimized.json
python -O audit.py --out outputs/audit-optimized.json
python plots.py --report outputs/result.json --directory outputs/figures
python generate_data.py --directory outputs/regenerated-data
```

`audit.py`针对交付默认协议，不是任意自定义实验的自动科学认证。`-O`会移除Python的`assert`；本讲重要检查使用显式异常，因此仍应工作。主报告较大，因为保留全部概率、分箱、99个残差和120次重复，而不是只保存最终几个平均值。

默认观察：温度约1.968054732；评价NLL约0.559406713、Brier约0.189806309；主回归$q$约0.936792199；主覆盖0.89、噪声改变覆盖0.5575。120次原机制平均覆盖约0.905958333，改变噪声后约0.578604167。小数末位在不同数值库或体系结构下可能差异；请用讲义定义的容差检查，不伪造相同字节。

## 4. 纸笔任务A：同一四行的三次前向

给定$z=(-m,-m,m,m)$，$m=2\log3$，$y=(0,1,1,1)$。初始$a=1$，学习率0.5。

1. 写出每行$u=az$、$p=\sigma(u)$、交叉熵损失。
2. 分别计算$\partial\ell/\partial p$、$\partial p/\partial u$、$\partial u/\partial a$；相乘，再除以4。
3. 汇总四行平均损失、梯度、Hessian。更新一次$a$。
4. 用新$a$重算全部四行，完成第二次更新；再执行一次新前向。
5. 与`report['hand']['states']`逐行比较，不能只比最后损失。

![图A 三次前向和逐样本梯度贡献，样本2为错误且自信的预测。](figures/02_hand_update_chain.png)

请回答：为什么第2行的梯度贡献与其余三行方向相反？为什么更新后第1行损失上升，但平均损失下降？详细答案见问题5至7。

## 5. 任务B：可靠性与适当评分

先不要调用任何ECE库。按照固定箱边界，将正类概率与标签配对，手算每个非空箱的平均概率、正类比例、权重和绝对差。再将分箱输入换成最高置信度与是否正确，重做一次。

```python
from calibration import evaluate
from math import log
z = [-2*log(3), -2*log(3), 2*log(3), 2*log(3)]
y = [0, 1, 1, 1]
for a in [1, .5]:
    out = evaluate(z, y, a, bins=5)
    print(a, out['binary_Brier'], out['accuracy'])
    print(out['positive_class_reliability'])
    print(out['top_label_reliability'])
```

预期：$a=1/2$时Brier为0.1875、准确率0.75、正类ECE0.25、最高置信度ECE接近0。一个“校准好了”的笼统结论不足以描述这些结果。若空箱均值是`null`，这是正确的缺值表达，不要替换成0再画线。

## 6. 任务C：证明评价标签没有参与选择

读取主报告中的`temperature.frozen_choice`和`choice_sha256`。它们应在评价指标计算前固定。把评价集标签翻转、解释用真概率改成$1-p$，只重新运行`temperature_report`。比较选择记录的完整规范JSON字节以及摘要；两者应完全不变。评价指标当然可以改变。

同时检查校准损失曲线：只用校准集寻找最低处，不能再把评价损失曲线最低位置作为正式温度。SciPy独立求解与二分导数根可以相互核对；它们验证数值求解，不验证现实概率是否正确。

![图B 真实独立评价集可靠性图与箱内样本数。](figures/05_evaluation_reliability.png)

读图后记录：每一箱有多少行？原先极端分数经过温度缩放后去了哪里？ECE降低是否伴随准确率提高？如果没有，解释为什么。

## 7. 任务D：从残差到预测区间

从`data/regression_train.csv`单独拟合直线。可用中心化公式独立核对：

$$w=\frac{\sum_i(x_i-\bar x)(y_i-\bar y)}{\sum_i(x_i-\bar x)^2},\quad b=\bar y-w\bar x.$$

随后只用99行校准数据算绝对残差，排序后取第90小值。选择一个评价点，逐步写出$x\to b+wx\to[\widehat f-q,\widehat f+q]\to1\{Y\in C(x)\}$。下一步才计算400行的平均覆盖。

```python
from conformal import finite_sample_quantile
print(finite_sample_quantile(list(range(1, 10)), .1))
print(finite_sample_quantile(list(range(1, 10)), .05))
print(finite_sample_quantile(list(range(1, 11)), .2))
```

预期分别为第9小值9、无穷区间、第9小值9。第二种返回`infinite=True`且`quantile=None`。如果程序返回9而没有无限标记，就丢失了95%目标所需的秩。

## 8. 任务E：负面结果必须留下

对照主报告与全部120次重复，填写：总体覆盖、两个预声明分组覆盖、噪声改变覆盖、宽度、错误复用的覆盖。计算标准差时分母用119，MCSE再除以$\sqrt{120}$。不要把400个共享校准过程的测试点当成400次独立训练实验。

解释错误复用1近邻：先在校准标签上训练，然后又用同样标签算残差。残差为0，区间宽度为0，这不能视为模型“不确定性消失”。请写出它违反秩证明的哪一个前提。

![图C 边际覆盖、分组覆盖和噪声改变并列报告，不删掉不理想结果。](figures/09_marginal_and_shift.png)

## 9. Notebook与全包重建

`calibration_uncertainty.ipynb`是带真实执行输出的Notebook。首格先验证源、数据与环境字节，之后才导入教学代码。标准Jupyter可打开学习；交付验证使用新的Python进程中的InProcessKernel，每次从空状态顺序执行，未声称测试浏览器或外部网络传输。

```bash
python build_notebook.py
python execute_notebook.py calibration_uncertainty.ipynb
```

这会在学习副本中重建并重执行Notebook。要保留原件，先复制整份单元。`build_all.sh`按顺序重建数据、结果、审计、十图、Notebook与三份PDF，输出放在`outputs/`。目录位置改变不应改变默认科学结果；PDF元数据可能使整文件字节不一致，不能用PDF时间戳当作科学差异。

## 10. 二十道练习

1. 用一句话分别定义正类校准、最高置信度校准和预测区间覆盖。
2. 四行例子中，$T=1$的两个非空正类箱是什么？逐项计算ECE。
3. $T=2$时两种ECE分别是多少？为何不同？
4. 展开Brier的条件期望，解释不可约项以及常数预测的局限。
5. 手算初始第2行全部局部导数与平均梯度贡献。
6. 写出第一次同步更新与下一次前向的四行概率、损失。
7. 完成第二次更新和第三次前向；说明为什么尚未最优。
8. 推导四行解析最优温度，计算最小NLL与Brier。
9. 有限正温度为什么不改准确率？无限温度为何要另说？
10. 原始logit极大时，为何不用先算概率再取对数？
11. 翻转评价标签后，哪部分记录必须不变，哪部分允许变化？
12. 一张只有可靠性曲线、没有样本数的图缺少什么？空箱如何表示？
13. 用三个数据集合写出split conformal的完整流程，并解释区间的对象。
14. 分别计算$n=99,\alpha=0.1$与$n=9,\alpha=0.05$的秩及区间类型。
15. 解释为什么证明里有$n+1$，而不是$n$；并列影响哪条结论？
16. 留出分数8时，普通插值90%分位数是多少？为什么与修正秩不同？
17. 对分数1至10、$\alpha=0.2$比较直接第$k$小值和NumPy的`higher`配方。
18. 从120个重复结果计算原机制覆盖均值、SD、MCSE，并说明独立单位。
19. 为什么总体约90%、困难组约83%不矛盾？噪声改变后保证为何失效？
20. 诊断1近邻校准复用；提出一个合法修复，并说明修复后仍不保证什么。

## 11. 交作业清单

提交完整命令、软件版本、协议与数据摘要、四行三次前向表、选择记录摘要、主区间计算链、120次完整结果与汇总、对负面结果的解释。每张图附一句“它支持什么”和一句“它不支持什么”。扩展结果放新文件，注明哪些规则在看到评价标签之前决定；不要覆盖本讲默认数据。
