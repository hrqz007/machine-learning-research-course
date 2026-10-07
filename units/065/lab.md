# 第065讲 对偶与核技巧实验指南

## 1 实验问题与证据链

本讲不搜索最佳核，而是检验一个精确命题：齐次二次核k(x,z)=(x^Tz)^2与显式特征phi=(x1²,sqrt2 x1x2,x2²)是否产生同一个SVM问题和相同的预测分数？检验之前，先用两个点推完对偶，并专门处理不存在内部支持向量时的偏置恢复。

交付物包括原始/对偶目标、KKT残差、显式/核分数差、正常与-O测试记录、六张图和已执行Notebook。只报告测试准确率1.0不够，因为许多错误实现也可能在这个简单任务上分对大多数点。

实验使用原创非个人数据和CPU，不下载数据。正式C固定为2、degree固定2、gamma固定1、coef0固定0，没有依据validation或test改变配置。二点手算另用C=1和0.1作为解析检查。

## 2 独立环境与文件

进入包含experiment.py的本讲目录后运行：

```bash
conda env create -f environment.yml
conda activate ml065
python -c "import numpy, scipy, sklearn; print(numpy.__version__, scipy.__version__, sklearn.__version__)"
```

验证环境为Python3.12.14、NumPy2.3.5、SciPy1.17.0、scikit-learn1.8.0。首次创建环境需联网，正式实验与已有依赖下的PDF构建均离线。已有隔离环境可以按requirements.txt安装，不要不记录地升级其中一个库后把数值差都归因于公式。

dual.py是本讲重点：显式映射、Gram检查、小规模对偶QP、偏置恢复与KKT审计。margin.py是独立原始求解器；experiment.py把各路线接起来；plots.py重建六图；test_experiment.py实施显式异常与数值检查。

阅读PDF不需要构建依赖。完整重建需Markdown、WeasyPrint、MathJax3.2.2、Pango和Noto CJK字体，见build-requirements.txt与package.json。build_all.sh只调用现有环境，在outputs中生成新结果，不安装软件也不覆盖冻结文件。

## 3 数据字典与冻结协议

train.csv为16行、validation.csv为24行、test.csv为80行。id是唯一标识，不进入模型。x1和x2的绝对值由0.55到1.45均匀抽样，符号组合覆盖四个象限。标签规则为y=-sign(x1)sign(x2)：同号负类、异号正类。

各文件独立生成，没有标签噪声。因此这是为展示非线性交互而设计的小任务，不能把它的满分结果外推到现实含噪高维任务。数据生成器、固定种子65021和摘要记录都公开提供。

```bash
python generate_data.py --directory outputs/regenerated-data
```

common.load_data同时验证摘要与确定性生成字节，并检查ID不重叠。不要仅修改generation.json让一份新CSV看起来“验证通过”。如果研究问题确实需要新数据，应明确变更生成机制并作为新实验保存。

## 4 纸笔核对两个点

取x=(-1,1)、y=(-1,1)，等式约束要求alpha1=alpha2=a。推导D(a)=2a-2a²，C=1时最优a=0.5，w=1、b=0，原始/对偶都为0.5。

```python
import numpy as np
from dual import solve_dual
X = np.array([[-1.], [1.]])
y = np.array([-1., 1.])
print(solve_dual(X @ X.T, y, 1.0))
```

把C改为0.1，检查alpha都等于C，bias_recovery.method变成bound_interval，区间为[-0.8,0.8]。取中点b=0后，两行hinge都是0.8，P=D=0.18。请自己代入另一个区间内b，确认目标仍然相同。

不要在没有内部支持向量时用空数组mean来恢复b，也不要直接挑alpha=C的点套m=1。本讲的边界实验就是用来暴露这种常见遗漏。

## 5 手算显式特征映射

取x=(1,2)、z=(3,4)，先计算点积11，平方得121。再计算phi(x)=(1,2sqrt2,4)、phi(z)=(9,12sqrt2,16)，内积为9+48+64=121。

```python
from dual import polynomial_features, gram
X = np.array([[1.,2.],[3.,4.]])
Phi = polynomial_features(X)
print(gram(X))
print(Phi @ Phi.T)
```

如果中间特征写成x1*x2而漏sqrt2，这个等式会失败。不要只看最终分类是否一样；本讲先检查核矩阵，再检查目标，再检查训练与新点分数，这样能更准确地定位错误层次。

## 6 主实验与数值验收

```bash
python experiment.py --out outputs/my-run/result.json
```

报告包含自写核对偶、显式原始QP、显式线性SVC、预计算核SVC和原生poly SVC对照。原生核显式指定degree=2、gamma=1、coef0=0，以免默认值改变数学对象。固定C=2没有超参数选择。

冻结运行的原始值和对偶值均约1.061602211689，对偶间隙约1.69e-14；独立原始求解与对偶值差约1.02e-13。显式SVC与预计算核SVC在测试上的最大分数差约1.78e-15，自写对偶与显式SVC差约8.96e-8。不同平台可在合理容差内波动，不应手工把尾数改成教材。

审查audit字段：equality_residual检查y^T alpha；box_violation检查0≤alpha≤C；margin_complementarity与slack_complementarity对应两组KKT乘积。间隙很小但约束违反很大，不能当作最优性证明。

## 7 检查PSD与反例

```python
from dual import validate_gram
# 这个矩阵对称且元素为正，但并不半正定
validate_gram([[1,2],[2,1]])
```

预期这里抛出ValueError，不是返回一个“修好”的核。它有负特征值-1，程序不能默默截断后继续把结果称为原问题。正式二次Gram因为显式特征只有三维，出现近零特征值正常；检查会按照矩阵尺度留浮点容差。

一次通过PSD检查不证明候选函数在所有输入上都是合法核。本讲的普遍合法性来自明确构造phi，而特征值检查用于核验当前数值矩阵和发现反例。

## 8 图与Notebook

```bash
python plots.py --report outputs/my-run/result.json --directory outputs/my-run/figures
python execute_notebook.py --out outputs/my-run/experiment.ipynb
```

六图依次支持两点对偶、KKT角色、Gram与秩、显式表征、核边界和分数等价/非法相似度。图4右侧只是三维phi的二维投影，不能把纵横坐标图当成完整特征空间。

Notebook在新Python进程的真实IPython InProcessKernel执行，实际重跑计算并嵌入六张PNG。浏览器UI和外部套接字内核传输没有测试。检查所有代码格有编号、无error输出，图本身保存而非只链接外部路径。

## 9 普通与优化模式测试

```bash
python test_experiment.py --report outputs/my-run/result.json --out outputs/my-run/tests.json
python -O test_experiment.py --report outputs/my-run/result.json --out outputs/my-run/tests-optimized.json
```

测试将从头训练，校验解析两点解、无内部SV偏置区间、矩形预测Gram、PSD反例、原始/对偶与KKT条件、显式/核分数和数据篡改拒绝。所有断言使用显式异常逻辑，不会被-O删掉。

若SLSQP失败，先看success消息和实际残差，不要删除审计后宣称完成。矩阵必须有限、对称、方阵且PSD，标签必须两类负1/正1，C必须正有限。本讲是小规模透明教学求解器，不保证大规模运行性能。

## 10 最终报告模板

先给原始与对偶的实际目标及间隙，再列可行性和互补条件残差。接着说明为什么没有内部支持向量时需要偏置区间，以及本次核的显式特征是什么。最后给测试分数等价误差与限定于合成任务的准确率结论。

必须能回答：预测80个新点时为什么需要80×16核矩阵而不是80×80？为什么一个对称相似度可能非法？为什么两个实现准确率相同仍不足以确认相同模型？若这些问题说得清楚，运行通过才真正转化成了理解。
