# 第075课实验 同样的分布，不同的因素

## 1 预先写下可检验主张

本实验有三条分开的主张：正交旋转后的载荷应给出相同观测协方差、密度和重构；PPCA可恢复与PCA相同的主子空间，但因子坐标需要对齐；不同观测噪声方差下，允许对角噪声的因子分析可能比同方差PPCA更合适。第三条是本次数据上的经验比较，前两条包含可证明的代数性质。

数据有train、audit、hetero_train、hetero_audit四份。前两份各列噪声标准差均0.25，后两份依次0.1、0.4、0.8、1.2。训练600行，审计300行。x0到x3是模型输入，z0和z1仅为生成真值，id是对象身份。没有真实业务含义，不能把图中的潜轴随意命名成心理或生理属性。

生成载荷固定为四行两列：[[1.4,0.2],[0.9,−0.6],[0.1,1.2],[−0.7,0.8]]。先读矩阵形状：每列描述一个因子影响所有四项读数的方向，每行描述某项读数同时受到两个因子的影响。

## 2 环境与完整执行

在本讲目录使用Python 3.12 CPU环境。安装依赖后，数值实验和Notebook无需网络。

```bash
# 依赖包含线性代数、因子分析库及Notebook执行组件。
python -m pip install -r requirements.txt
# 拟合模型、构造等价旋转、冻结对齐，再评价审计数据。
python experiment.py --out outputs/rebuild/result.json
# 测试代数恒等式、独立密度实现和边界输入。
python test_experiment.py \
  --report outputs/rebuild/result.json \
  --out outputs/rebuild/test.json
# 显式检查在优化模式下同样执行。
python -O test_experiment.py \
  --report outputs/rebuild/result.json \
  --out outputs/rebuild/test-optimized.json
```

报告应包含noise_variance、eigenvalues、五项等价变换差、原始与对齐因子误差、子空间距离、PCA与PPCA重构误差，以及异方差数据的审计密度。预期旋转不变项在10⁻¹⁰以下；它们由代数决定，容差只处理浮点误差。

## 3 实验A 看懂PPCA的五个输出

```python
# 数组库处理矩阵运算。
import numpy as np
# 数据加载会核验CSV摘要及生成器确定字节。
from common import load_data
# matrix只抽x列，不把真实z混入训练。
from experiment import matrix
# ppca返回均值、载荷、噪声方差、特征值和特征方向。
from latent import ppca, posterior
X = matrix(load_data()['train'])
mu, W, variance, values, directions = ppca(X, 2)
print(X.shape, W.shape, variance, values)  # 预期(600,4)、(4,2)。
```

检查噪声方差等于最后两个特征值的平均。预期约0.064848，对应生成真方差0.0625但不完全相同。有限样本不应要求估计值精确等于生成参数。把q改为4应报错，因为没有剩余方向估计正噪声；全常数输入也应明确拒绝。

## 4 实验B 在代码里旋转解释

构造60度旋转R，计算Wr=W@R。先看每个载荷数值确实改变，再比较W@W.T与Wr@Wr.T。不要只比较列名。用posterior分别计算两个模型的审计分数与重构，核对rotated_scores≈scores@R以及两份重构相同。

交付三个证据：一行代数证明；最大绝对数值差；两张载荷或分数图。三者互相补充。数值一致展示实现符合数学，代数说明结论不依赖这一份随机数据，图帮助理解为什么语义可能改变。

再把第一列载荷乘2、第二列乘0.5。若仍固定z协方差为I，观测协方差通常改变。将潜协方差同步改为diag(1/4,4)后才恢复原协方差。报告必须写清楚哪个实验保留原先验，哪个改变了先验。

## 5 实验C 恢复度量与对齐泄漏

原始因子MSE约4.124，对齐后约0.02381。代码仅在train的真实z上求正交对齐矩阵，冻结之后应用audit。若直接用audit真值拟合旋转再报audit恢复误差，评价对象同时参与了对齐参数学习；这会夸大泛化恢复表现。

本练习允许训练真值对齐，因为目的是合成模型审计，而非声称真实部署时可以知道z。真实数据没有z时，只能改用能观测的证据，例如留出密度、下游任务、额外传感器或干预，而不能伪造因素真值。

比较子空间投影距离和直接载荷逐项误差。解释为何旋转后的两个W逐项不相同，却有相同列空间投影。代码使用SVD按数值秩截断；请用全零载荷验证投影矩阵全零，避免把任意补充正交方向当真实因素。

## 6 实验D 重构与概率目标

普通PCA审计观测重构MSE约0.031041，PPCA后验信号重构约0.031932。后者略大是预期：后验向均值收缩，PCA直接做子空间最近投影。不要用这个微小差异说PPCA的概率模型被推翻。

异方差数据中，PPCA平均审计对数密度约−6.36945，因子分析约−6.06390，越高越好。同时查看因子分析noise_variance_与生成噪声平方，说明它们没有精确恢复。数据量、模型参数不唯一与估计误差都限制解释，不能只报告赢家。

## 7 产物与完成标准

```bash
# 从同一数据与报告生成载荷、坐标、谱和验证图。
python plots.py \
  --report outputs/rebuild/result.json \
  --directory outputs/rebuild/figures
# 新进程真实内核逐格执行，计算结果写入独立Notebook。
python execute_notebook.py --out outputs/rebuild/experiment.ipynb
# 重建四份人工数据与摘要，保留冻结材料。
python generate_data.py --directory outputs/rebuild/data
```

提交一份包含模型形状图、等价证明、五项数值差、对齐流程、两种噪声假设比较的报告。最后写一句你现在拒绝从良好重构中推出的过强结论，以及为支持它需要补充什么证据。答案应指向可识别性，而不是笼统说“模型可能有偏差”。

build_all.sh可重建本讲完整输出；PDF需额外构建依赖与中文字体。本实验只验证真实计算内核，不声称验证全部浏览器交互或任何线上服务。
