# 046 分类指标与排序评价实验

这是一份可单独操作的实验手册。你将先对固定的12行不平衡数据手算，再用真实 scikit-learn 核验，最后只改变阳性率，观察 ROC 与 PR 的不同反应。本实验不训练模型，不选择部署阈值，不通过换数据追求更好成绩。预计纸笔与代码各需约一小时，基础较弱时可以分两次完成。

## 1 交付物和运行环境

阅读材料为 `lecture.pdf`、本实验册 `lab.pdf`、`answers.pdf`，三者均有可编辑 Markdown。代码入口为 `experiment.py`；逐步交互版本为 `experiment.ipynb`；`audit.py` 给出独立精确参照；`plots.py` 重建9张图。固定数据位于 `data/review_scores.csv`。`data_integrity.json` 保存这份数据的 SHA256，避免实验中意外改变案例。

在解压后的 `046` 目录打开终端。推荐使用 Anaconda/Miniconda：

```bash
conda env create -f environment.yml
conda activate ml046-metrics
python -c "import sys, numpy, sklearn; print(sys.version); print(numpy.__version__, sklearn.__version__)"
```

也可以在 Python 3.12 的独立虚拟环境中安装 `requirements.txt`。Notebook 浏览界面需要 JupyterLab；`environment.yml` 已列出它。无浏览界面时，随包 `execute_notebook.py` 可在新的 Python 进程中启动真实的进程内 IPython 内核，依次运行全部单元并保存输出，它没有测试浏览器或网络内核传输。

已验证数值版本为 Python 3.12.14、numpy 2.3.5、scipy 1.17.0、scikit-learn 1.8.0、matplotlib 3.10.8、nbformat 5.11.1、ipykernel 7.4.0。版本输出不同不必立即断言错误，但应先对照接口端点与结果，再决定能否沿用报告。

## 2 先不运行代码 完成纸笔记录

| 编号 | 标签 | 分数 |
|---|---:|---:|
| A | 1 | 0.95 |
| B | 0 | 0.90 |
| C | 1 | 0.80 |
| D | 0 | 0.80 |
| E | 0 | 0.70 |
| F | 0 | 0.60 |
| G | 1 | 0.55 |
| H | 0 | 0.50 |
| I | 0 | 0.40 |
| J | 0 | 0.30 |
| K | 0 | 0.20 |
| L | 0 | 0.10 |

实验题 L1：使用“分数大于等于阈值”的规则，在 0.80 和 0.55 两个阈值分别把每行归入 TP/FP/FN/TN。先写编号，再数格子。对每个阈值计算 accuracy、precision、recall、F1、FPR。不要用小数近似替代全部分数运算。

检查点：0.80 的四格应为 TP = 2、FP = 2、FN = 1、TN = 7。如果不符，先查 C、D 的等号，再查矩阵的行列方向。0.55 时预测阳性有7件，不能把 $3/7$ 错写成 $3/12$。

实验题 L2：从正无穷开始，依次把阈值降到全部不同分数。每一步只写“新进入哪几行”，再更新四格。C、D 必须一起进入；不能在它们之间虚构一个阈值。表应有12行，其中一个起点和11个有限阈值。完整答案在答案册。

## 3 运行主实验并读懂输出

```bash
python experiment.py
```

默认产生 `outputs/experiment-result.json` 与 `outputs/threshold-sweep.csv`。不依赖调用者的当前工作目录；默认数据和输出位置由脚本自身位置确定。若希望把结果放在新目录，可传 `--out`：

```bash
python experiment.py --out ./outputs/run-a
python -O experiment.py --out ./outputs/run-optimized
```

`-O` 是 Python 优化模式，会去掉普通 `assert`；本讲输入检查和核验均未依赖 `assert`。从其他工作目录运行时，可对脚本给出你本机解压路径的绝对路径。程序不支持把输出目录指向课程根目录或其他源码子目录，避免覆盖课程正文和数据。

终端会给出 AUC、AP 与梯形 PR 面积。按本包固定数据，三者依次应为 0.7962962963、0.6428571429、0.6269841270。它们不是同一个量。JSON 包含完整曲线数组、每个阈值、四种平均方式、先验比例变化和立方变换。

实验题 L3：打开 `threshold-sweep.csv`，找到 0.70 和 0.60。解释 precision 为什么下降而 recall 不变。再看 0.60 到 0.55，解释 precision 为什么上升。这能反驳“降低阈值必然降低精确率”中的哪一个字？

## 4 直接调用库 不只看封装结果

下面代码是 Notebook 中的核心核验，使用真实 sklearn 接口。`load_case()` 返回固定数据，不进行拟合。

```python
from experiment import load_case
from sklearn.metrics import (
    confusion_matrix, precision_recall_fscore_support,
    roc_curve, precision_recall_curve,
    roc_auc_score, average_precision_score, auc)
ids, y, s = load_case()
pred = s >= 0.80
print(confusion_matrix(y, pred, labels=[0, 1]))
print(precision_recall_fscore_support(
    y, pred, labels=[0, 1], pos_label=1,
    average='binary', zero_division=0))
fpr, tpr, roc_t = roc_curve(
    y, s, pos_label=1, drop_intermediate=False)
p, r, pr_t = precision_recall_curve(
    y, s, pos_label=1, drop_intermediate=False)
print(roc_auc_score(y, s))
print(average_precision_score(y, s, pos_label=1))
print(auc(r, p))
```

这里 `pred` 是布尔预测，`s` 是连续评分。混淆矩阵需要 `pred`；排序曲线和 AUC/AP 需要 `s`。若把已经阈值化的 `pred` 传给 AUC，得到的是一个仅有两种分数的退化排序，原本的细分顺序已丢失。

实验题 L4：打印三个 ROC 数组的长度和全部阈值，再打印三个 PR 数组。解释：为什么 ROC 第一个阈值是 `inf`？为什么 PR 的两个坐标数组比阈值多一个元素？为什么不能把 `roc_t[i]` 和 `pr_t[i]` 当作同一工作点？

该版本完整 ROC 三数组长度均为12。PR 的坐标数组长度为12，阈值数组长度为11。PR 有限阈值从 0.10 上升到 0.95，末尾坐标为 precision = 1、recall = 0，且没有对应阈值。记住这个点是画图约定，不是空警报精确率的统计测量。

## 5 独立精确参照

先运行快速主例核验，再运行完整的有限枚举：

```bash
python audit.py --quick --out ./outputs/quick
python audit.py --out ./outputs/full
python -O audit.py --out ./outputs/full-optimized
```

`Fraction` 直接从 CSV 的整数百分数建立精确比较，不通过先算浮点再转分数“伪装精确”。参照独立枚举每个阈值入选的记录；另枚举所有阳性阴性对，按胜1、平1/2、负0计数。曲线积分结果与配对结果要先精确一致，再对照真实库的浮点结果。

完整测试覆盖1134个4行小例子，包含所有同时有两类的标签向量和三级评分向量。为什么选择这个规模？因为本讲最容易出错的是相等评分、阈值端点和排序模式，小规模可穷举给出完整检查，而不是随机碰运气。它不证明任意输入规模都正确，也不证明目标总体的模型表现。

实验题 L5：不用看程序，列出 A、C、G 分别对9个阴性的胜、平、负数，得到 AUC 的分子。再解释全同分评分为什么 AUC = 1/2，AP = 1/4。单类评价集为什么被本包装函数拒绝？

## 6 改变阳性率 保持类内评分不变

在这个实验中不要删行、重排标签或重新抽样。只给两个真实类别施加各自统一的权重：

```python
from experiment import prior_weights, library_metrics, counts_at
for pi in [0.10, 0.25, 0.50]:
    w = prior_weights(y, s, pi)
    curve = library_metrics(y, s, w)
    print(pi, counts_at(y, s, 0.80, w),
          curve['auc'], curve['ap'])
```

原阳性率为0.25；阳性权重为 $\pi/0.25$，阴性权重为 $(1-\pi)/0.75$。这是明确指定新混合比例的加权评价。它不改变类内分数分布，因此 ROC 重合。若真实分布变化同时影响每类内部特征和分数，这个结论不能保证。

实验题 L6：当阳性率为0.10，写出每类权重、0.80阈值的加权四格和精确率。使用 $P=\pi a/[\pi a+(1-\pi)b]$ 独立重算一次。找出三个增加召回率的步骤，算 AP = 29/60。解释为什么这不是模型“自动变差”的证据。

![实验图1 阳性率变化的实际输出。ROC重合依赖类条件评分分布不变；PR显示同一误报率在不同人群混合比例下的警报可靠性。](figures/07_prevalence.png)

## 7 比较平均方式和概率损失

```python
from experiment import average_metrics
for name, values in average_metrics(y, s).items():
    print(name, values)
```

实验题 L7：对同一个0.80混淆矩阵，分别算阳性 F1、宏 F1、微 F1、支持度加权 F1。不要把宏 F1 写成宏 precision 与宏 recall 的调和平均。解释四个值分别回答什么问题。

实验题 L8：计算 `s**3`。比较原评分与变换后的 AUC、AP、固定0.80阈值的混淆矩阵，以及映射阈值 $0.80^3$ 的混淆矩阵。再看 `experiment-result.json` 的 `monotone` 字段，它还包含把评分当作概率时的 log loss 和 Brier。解释为什么排序没变但概率损失会变。

本实验没有用这些损失训练模型，也没有声称原评分已经校准；“作为概率计算损失”是一个反例工具，说明 AUC 不检验概率数值。数值分数一旦被解释为概率，错误置信程度就会进入损失。

## 8 图像与真Notebook运行

```bash
python plots.py
jupyter lab experiment.ipynb
```

图像默认写入 `outputs/figures`，不会改动已交付的正文图。Notebook 中从上到下执行，不能只运行依赖旧内核变量的后半段。它会显示记录、完整扫描、库接口、精确参照、加权实验和9张实际图。打开时若已有输出，仍需用 Restart Kernel and Run All 检验自己环境，而不是把静态输出当成执行证明。

不使用浏览器也可运行：

```bash
python execute_notebook.py --out ./outputs/fresh-kernel.ipynb
python -O execute_notebook.py --out ./outputs/fresh-kernel-opt.ipynb
```

这个命令创建新的真实进程内 IPython 内核，执行全部代码单元，收集文本和图像输出，并在失败时报告具体单元。它不覆盖源 Notebook，不测试 JupyterLab 浏览器、网络socket或扩展插件。

实验题 L9：逐张读9幅图，回答讲义 G1 至 G9。再核对图中的阈值、计数与 CSV。特别注意 ROC 的横轴分母是9、PR 的横轴是 recall，以及图9的面积指标越大越好而概率损失越小越好。

## 9 可选重建PDF

数值实验不需要 PDF 构建依赖。若要重建排版，另安装 `build-requirements.txt` 中的 Python 包，以及 Node.js 和 `package.json` 中的 MathJax。系统需提供 Pango、Noto Sans/Serif CJK 中文字体与 DejaVu Sans Mono。此流程不依赖 XeLaTeX。

```bash
python -m pip install -r build-requirements.txt
npm install
python build_pdf.py --directory ./outputs/pdf
```

`build_pdf.py` 在本机用 MathJax 生成公式 SVG，再用 WeasyPrint 排版中文。只处理可信课程 Markdown，不是用于接收任意外来 HTML/TeX 的安全沙箱。正文 `.md`、图像、CSS、公式脚本和依赖声明全部随包提供。构建后仍需把PDF逐页渲染检查，不能以生成成功代替视觉验证。

## 10 验收清单与排错

实验题 L10：提交一段不超过250字的中文报告，至少包括样本单位、阳性定义与数量、阈值规则、四格、阳性 P/R/F1、AUC/AP 的定义、平均方式以及不能外推的限制。给出一个会让该报告失效的协议改变，例如测试集选阈值或更换阳性率后直接比较 AP。

完成标准如下：

- 纸笔四格与程序一致，所有12行阈值记录无遗漏，同分成组
- AUC = 43/54、AP = 9/14、梯形PR面积 = 79/126 三条精确锚点正确
- 0.80阈值的四种F1平均方式均解释清楚，而不只抄数
- 阳性率变化时明确类条件评分分布不变，重算出加权精确率和AP
- 正常进程与 `-O` 完整核验通过；Notebook在新内核按序执行无错误
- 报告不把12行合成例子说成真实泛化评估，不把AUC当单阈值成绩

遇到 `fixed data hash mismatch`，先核对 CSV 是否被意外修改，不要直接改哈希来消除错误。若故意做新实验，复制到独立文件并明确新协议。遇到单类输入错误，检查数据切分和类别定义；不能强行填造一个缺失类别。遇到数组长度不一致，优先查 PR 额外端点。遇到公式渲染依赖缺失，可以先完成数值实验；随包已有可读PDF与Markdown。

