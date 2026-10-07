# 第063讲 提升树的实际训练实验指南

## 1 实验任务与交付物

本实验从一组带缺失和类别特征的原创合成数据出发，实际训练四个直方图提升树候选与四个逻辑回归候选。你要交付一份能重现的记录，说明每份数据的用途、选择的配置、早停轮数、最终概率指标，以及校准是否在这次固定测试中改善。独立阅读本指南即可运行，不需要先翻正文才能知道文件路径和命令。

目标不是跑出“校准必胜”的结果。冻结运行恰好展示了校准略差这一有用反例。不要为了匹配自己的预期修改结果，也不要拿测试结果重新选配置。实验仅使用本地CPU和随包CSV，不下载数据，不请求任何付费服务。

## 2 建立独立环境

在Anaconda Prompt或终端进入本讲目录，也就是能看到experiment.py的目录。推荐按environment.yml创建环境；已经有Python 3.12独立环境时可用requirements.txt安装固定版本。首次创建需要联网，从官方Conda/PyPI来源获取包；安装完成后本讲数值实验无网络调用。

```bash
conda env create -f environment.yml
conda activate ml063
python --version
python -c "import sklearn; print(sklearn.__version__)"
```

本讲验证的Python是3.12.14，scikit-learn是1.8.0。X_val与y_val是在fit中传入的外部早停接口，太旧的版本可能报unexpected keyword argument。不要删除参数来“修好”程序，那会改变数据协议；应先核对环境。实验不需要GPU。

普通读者不必重建PDF。若要从Markdown重建完整三份PDF，还需build-requirements.txt、package.json列出的MathJax以及系统Pango、Noto Sans/Serif CJK字体。build_all.sh只调用现有环境，不自行联网安装，也不会覆盖冻结PDF或报告。首次安装这些可选构建依赖需要额外准备。

## 3 理解数据字典与生成过程

共有五个互不重叠的CSV：train为480行，stop为120行，tune和calibration各160行，test为240行。id由数据角色和行号组成，只用于识别与防止重叠，不是预测特征。x1是标准正态随机数，x2来自负2到2的均匀分布，category为三个无序类别编号，y为0或1。

x2大约18%的位置被置成NaN。生成器先计算真实的x2、类别与缺失标志，再按照预先规定的非线性logit生成Bernoulli标签。模型看不到被遮盖的x2。缺失标志在这个玩具总体中有信息，因此线性基线增加缺失指示列，而提升树使用原生缺失支持。

data/README.md给出完整列说明，generate_data.py是独立的可读生成程序。先练习把数据再生成到新目录，逐文件比较摘要，不要覆盖课程冻结数据：

```bash
python generate_data.py --directory outputs/regenerated-data
```

固定版本的NumPy与固定种子使本次字节可复现。common.load_data同时检查记录摘要、生成器字节和ID互斥性。若你确实想研究另一份数据，应复制一份实验目录，修改生成器、重新记录协议，并把它命名为新实验；不要只改CSV来绕过完整性检查。

## 4 先完成纸笔检查再训练

四个样本的标签为0、0、1、1，初始logit全是0。请在纸上写出每行p、g=p-y和h=p(1-p)，累加左右两叶统计量。分别代入L2为0和1，计算叶值与分裂增益。你应得到未正则化叶值负2与正2、增益2；L2为1时叶值负2/3与正2/3、增益2/3。

可以在Python中检查，但先保留自己的中间步骤：

```python
from boosting import derivatives, leaf_weight, split_gain
g, h = derivatives([0, 0, 1, 1], [0, 0, 0, 0])
print(g, h)
print(leaf_weight(1, .5, 1))
print(split_gain(1, .5, -1, .5, 1))
```

boosting.py故意只实现这些透明的小计算，没有伪装成另一个完整提升树库。正式训练由scikit-learn执行。你要分清教学公式、实现默认值和正式实验配置三层。

## 5 运行主实验并定位选择逻辑

```bash
python experiment.py --out outputs/my-run/result.json
```

脚本打印所选提升树候选索引和三组最终指标，并把详细报告写入新文件。它不会把测试指标反馈给搜索。四个树候选都使用最大150轮、最小叶12行、63个非缺失桶、外部stop数据和固定早停规则。

打开输出JSON，检查hist_candidates中的每个候选是否包含实际轮数、调参损失、训练/早停曲线和fit耗时。selected_hist_index应是tune_log_loss最小的索引。冻结结果选择索引2，即learning_rate=0.1、max_leaf_nodes=7、l2_regularization=1。不要把这一配置当成别的数据集的推荐常数。

线性基线通过ColumnTransformer建立两条预处理路径：数值列做训练均值插补、缺失指示与标准化，类别列做one-hot。它的四个C候选也只通过tune选择。校准器使用固定的sigmoid形式，只fit在calibration上。注意最终不再合并所有数据重训，因为那样会改变基模型分数分布，校准器也需重新设计。

## 6 检查早停与校准输出

先读取每个候选的n_iter对应记录。曲线长度应是iterations+1，因为第0轮常数模型也有一个分数。训练结束时刻不应自动等同于历史最小早停损失的时刻；本讲使用库返回的最后模型，没有人为回滚。

```bash
python plots.py --report outputs/my-run/result.json --directory outputs/my-run/figures
```

六张图分别解释数据角色、二阶正则化、早停曲线、候选预算、可靠性与最终指标。图的英文坐标与图注的中文解释互相对应。检查校准图时先看右侧每桶样本量，再解释左侧偏离对角线的幅度。

冻结测试log-loss为：未校准树0.588754、sigmoid树0.591923、逻辑回归0.631331。概率Brier分别为0.201192、0.202511、0.220339。尾数可能受平台微小浮点差异影响，fit耗时更不应逐位相等。如果数值明显不符，应先查版本、数据摘要、线程和完整运行日志。

## 7 普通测试与优化模式都要执行

```bash
python test_experiment.py --report outputs/my-run/result.json --out outputs/my-run/tests.json
python -O test_experiment.py --report outputs/my-run/result.json --out outputs/my-run/tests-optimized.json
```

两次运行均应status=passed。测试会再训练一遍，复核预测与候选曲线，不只是检查文件是否存在。它也检查导数有限差分、理论手算、数据篡改以及不安全输出链接。这里使用显式raise，不依赖-O模式会删除的assert。

报告中的checks数量包含带名字的数值和结构检查。阅读check_names比记住一个大数字更重要：一个“有300项检查”的程序仍可能遗漏核心泄漏问题。请定位CSV与摘要同时被改动的反例，并解释为什么只有摘要一致还不够。

## 8 运行Notebook并确认图真正嵌入

使用Jupyter可交互阅读experiment.ipynb。命令行重现使用随包执行器：

```bash
python execute_notebook.py --out outputs/my-run/experiment.ipynb
```

该命令在新Python进程内启动真实IPython InProcessKernel，重新运行代码、保留文本输出并嵌入六张PNG。它验证实际代码执行，不验证浏览器按钮和外部套接字传输。打开新Notebook，逐个检查代码格执行编号、实验表和图是否可见。只有Markdown中的图片路径而没有输出图，不算完成本实验的Notebook验收。

## 9 排障与改变实验的边界

若出现ModuleNotFoundError，确认当前python属于ml063，再按requirements.txt安装缺少依赖。若PDF出现方框，检查字体而不是删除中文；若MathJax找不到，检查NODE_PATH指向已安装依赖的node_modules。路径带空格时应在命令行加引号。

如果完整性检查失败，先把原始CSV与生成器恢复一致，不要关闭检查。输出目录若包含符号链接，报告程序会拒绝写入，这是为了避免误覆盖其他文件。使用一个普通的新建目录即可。

探索可以改变学习率、叶子数、校准方式或样本量，但每次只改变有明确问题的一部分，并记录预算。查看过test后作出的新选择需要新的封存评估。把旧测试结果用于启发假设可以，仍把它当成完全未见的最终检验不可以。

## 10 最终验收清单

提交你的运行报告、两份测试记录、已执行Notebook与六张图。附一页解释：每份数据允许参与的决定、候选次数与实际训练成本的区别、校准变差的可能原因、缺失与未知类别的处理策略。

最低验收是运行通过且能解释为什么test不能进入早停。进一步验收是能用手算证明叶值公式，逐行指出代码中的数据边界，并提出一个不复用现有测试结论的新实验设计。所有冻结文件均保留作为对照，个人实验应写到outputs下。
