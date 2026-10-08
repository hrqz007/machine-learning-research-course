# 第069讲 实验指导书

## 1 要完成的实验

本实验先手算四个二维点，再用从零SVD实现对六维数据降到二维，比较协方差谱、重构误差和库输出。最后故意改变单位与测试均值，并构造高方差无关输入，观察PCA的边界。预计需要90至150分钟，普通CPU即可。

交付时保留手算过程、重构曲线、库差值、两个反例及一段有条件的结论。不要仅截图一张漂亮的二维散点图。读lecture.pdf第2至6节后再运行，先核对矩阵形状，避免凭复制代码跳过推理。

## 2 环境与运行顺序

建议Python3.12，固定数值依赖见requirements.txt。首次安装需联网，所有数据均在包内，实验不需要外部数据服务。

```bash
python -m pip install -r requirements.txt
python experiment.py --out outputs/my-run/result.json
python test_experiment.py --report outputs/my-run/result.json --out outputs/my-run/tests.json
python -O test_experiment.py --report outputs/my-run/result.json --out outputs/my-run/tests-O.json
python plots.py --report outputs/my-run/result.json --directory outputs/my-run/figures
python execute_notebook.py --out outputs/my-run/experiment.ipynb
```

如需重建PDF，另安装build-requirements.txt与package.json中的构建依赖，确保系统具有Pango、Noto CJK和DejaVu字体。bash build_all.sh把新结果放outputs/rebuild。可设置PYTHON_BIN和NODE_PATH选择已有环境。脚本不自动安装，也不覆盖冻结材料。实际验证使用已有固定版本环境，未另行验证干净Conda安装。

## 3 数据字典和拟合边界

train.csv和test.csv分别包含320与160行，每行有id和x0至x5六个数值。数据由两维潜在变量线性组合、0.15标准差噪声及固定偏移生成。signal_train.csv和signal_test.csv分别有400与200行，每行包含id、x0、x1及y；x0是高方差无关变量，x1的符号决定标签。

生成器种子69021。main_与signal_前缀区分两个任务，训练与测试ID互不重叠。generation.json记录每个CSV的SHA256，读取还核对确定性生成器字节。generate_data.py --directory outputs/generated-data可在新目录重建。

PCA的mean、components、任何StandardScaler参数和逻辑回归参数均只能从各自训练数据学习。测试集用于计算既定方案的重构与预测表现。为了展示错误而用测试自身均值的那一项明确标为反例，不能混入正式结果。

## 4 纸笔工作

将(2,1)、(1,2)、(-1,-2)、(-2,-1)写成四行矩阵。证明均值为0，算出XᵀX=[[10,8],[8,10]]，再除以3。检验两个归一化方向(1,1)/√2与(1,-1)/√2的特征值。

把第一点投到第一轴，得到3/√2；重构为(1.5,1.5)。每点残差平方为0.5，总SSE为2。用(n-1)×丢弃特征值复核得到3×2/3=2。最后说明平均每样本平方距离为0.5，而每个标量元素MSE为0.25。本讲曲线采用前者。

## 5 阅读从零实现

打开pca.py，依次找出输入验证、训练均值、中心化、SVD、奇异值平方除以n-1、符号约定、transform和inverse_transform。请在纸上标注X为n×d、components为k×d、Z为n×k。

method='eigh'使用协方差分解，是独立计算路线。注意eigh输出升序，程序倒序排列特征值与对应特征向量。默认svd路线不显式构造协方差。对非常病态数据，形成XᵀX会恶化数值条件，因此库对照固定full SVD而不让自动求解器因版本改变选择路径。

调用fit后保存模型，再对测试数据调用transform；不要第二次fit测试数据。若只想重构，调用reconstruct，其内部使用已有训练均值和主轴。

## 6 数值验收

hand.sse应为2；协方差对角为10/3，非对角8/3。library_errors中的投影、特征值和测试重构差应小于1e-9。本次最大约3.6×10⁻¹⁵，是浮点舍入量级。

训练六维数据前两个特征值约28.969847和3.282490，累计方差比例0.997264。保留两维的训练平均平方残差约0.088192，测试约0.088299。检查reconstruction_curve每一行train_sse/(n-1)是否等于discarded_eigenvalue_sum。

训练SSE随k增加不应升高，因为允许的投影空间扩大；测试曲线也可以直接核验。对同一组按特征值嵌套的正交轴，添加轴同样不能增加任何固定测试点的平方投影残差，但测试误差下降不代表下游分类一定改善。

## 7 三个机制实验

单位实验：将第三列乘100。先看未经缩放的主轴载荷，再看训练集标准化后的载荷。写出两者对应的不同距离度量。不要简单写“标准化总是正确”，因为小方差噪声也可能被放大。

中心化实验：对测试第一列统一加3，正确做法继续减训练均值。两个主成分分数均值约为2.212和-0.974；错误地减测试自身均值会把均值人为归零。另比较中心化秩2 SSE约28.22与未中心化SVD约1020.80，说明穿过均值与穿过原点的差别。

预测反例：第一主成分解释99.8431%训练方差。两列输入的逻辑回归测试准确率0.99，PC1输入为0.48。标签由低方差方向决定，所以这不与PCA重构最优性矛盾。该反例使用独立生成机制，不能当成“PCA总是有害”的统计估计。

## 8 测试覆盖与故障排查

109项普通/-O显式检查覆盖完整报告重算、k=1至6的正交性、分数中心化、残差正交、谱SSE恒等式、库重构、平移与符号不变性、常数数据、非法输入和篡改。源代码使用显式异常，不依赖assert。

若两次结果主轴正负相反，先比较投影矩阵与重构；若特征值重复，单个主轴还有旋转不唯一性。若只有方差差一个n/(n-1)，检查协方差分母。若重构出现固定偏移，检查是否忘记加回训练均值。若测试分数均值总是严格零，检查是否错误地对测试数据重新fit或自中心化。

Notebook由新Python进程中的真实IPython InProcessKernel顺序执行，保留数值与7张图。Jupyter浏览器界面和外部socket内核传输未列为已验证。若缺模块，先检查当前解释器，不要修改数学公式来掩盖环境问题。

## 9 扩展与提交清单

可以另建实验改变噪声强度，观察谱的前两项与其余项如何接近；或者加入少量离群点，检查平方距离目标为什么对极端点敏感。保存新种子和新数据，不覆盖参考CSV。若用标签表现决定保留维数k，需要另设验证集，把PCA放进每次训练的Pipeline。

提交前确认：手算完整；矩阵维度明确；SSE单位与分母明确；主轴比较允许符号差异；测试只transform；图注没有把方差当作预测保证；扩展实验与参考实验分开。结论应说明PCA优化哪个空间、哪个距离和哪批数据。
