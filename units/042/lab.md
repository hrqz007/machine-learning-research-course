<style>p{text-align:left} table{break-inside:avoid} figure{break-inside:avoid}</style>

# 第042讲 实验指南：概率、优化与有限解

本实验的交付物不是一条损失曲线，而是一组能相互核对的证据：同四行的两轮局部链、独立有限MLE证书、分离反例、数值稳定性与真实库目标对齐。数据全部是原创合成数据；四行不构成现实泛化或校准实验。

## 1 环境与可重复运行

建议使用Python3.12的新环境。requirements.txt锁定NumPy2.3.5、SciPy1.17.0、scikit-learn1.8.0、Matplotlib3.10.8、JupyterLab4系列；environment.yml给出相同Conda入口。在本单元目录运行：

```bash
python -m pip install -r requirements.txt
python experiment.py --out outputs/result.json
python audit.py --out outputs/audit.json
python -O experiment.py --out outputs/result-O.json
python plots.py --report outputs/result.json --directory outputs/figures
```

默认CLI先核对data_integrity.json中的教学字节。修改原始数据做探索时，先复制CSV与model_spec.json到新目录，使用--data、--separable、--config显式指定副本；仍执行严格数值和形状检查。不要修改已验收的原始教材。自定义主数据不再自动得到本讲三次方程证书，输出finite_mle_certificate为null。

Notebook须在本单元目录、全新内核从首格顺序运行。首物理代码格仅用标准库验证固定契约，成功之后才导入数值库、计算或画图。图像应实际嵌入PNG；显示一行Figure对象文字不合格。本次制作真实测试的是新Python进程中的InProcessKernel与顺序执行，不声称已经验证浏览器Jupyter界面、socket传输、Windows或Anaconda安装过程。

## 2 先用纸笔，再读程序

读取observations.csv：B1至B4的x为−2、−1、1、2，y为0、1、0、1。参数顺序固定[b,w]。A的每行是[1,x]，不额外标准化。平均负对数似然F使用除以n，不是除以2n；单行损失本身也没有1/2。

先不要运行完整拟合。填写以下各项，再与Notebook逐行输出核对：

1. θ⁰=(0,0)的四个z、p、ℓ和ℓ/4。
2. 每行∂ℓ/∂p、∂p/∂z、稳定等价的∂ℓ/∂z，以及[1,x]。
3. 每行两个平均梯度贡献和2×2 Hessian贡献。
4. 汇总g=(0,−1/4)、H=diag(1/4,5/8)，一次同步更新得到θ¹=(0,1/8)。
5. 使用θ¹重算四行，而不是沿用θ⁰的概率。求新的总梯度后同步得到θ²。
6. 再把θ²代回全部四行。解释为什么中间两行损失增大而平均损失下降。

第一步可用分数给精确证明；第二步含指数，不能把打印出的六位小数当有理真值。audit.py另外计算110位Decimal参照，全部68次GD更新的69个状态和每行字段都交叉核对。

![图3 三个完整状态：总损失与每行损失回答不同问题。](figures/03_two_round_ledger.png)

## 3 同一目标的两种求解

运行默认GD与保护Newton。GD以η=.5、梯度L2阈值1e−10、最多400次更新；Newton每次解线性方程，尝试明确的正定移位序列，使用Armijo回溯。不要把Newton方向当完整概率目标的精确解。

本机实际结果：GD完成68次更新，Newton4次，都满足共同梯度阈值。未正则GD末斜率约.4196176248168132；Newton约.41961762499109795。两者与独立三次方程根的一致程度不同，但都按自己的容差据实停止。Newton路径小，并不等于实际耗时一定少；本单元未做墙钟基准。

cost中state_attempts在调用完整行状态函数前增加，state_successes仅在成功返回后增加。每次尝试对应n行前向、梯度、Hessian的计划工作；被拒绝候选也记账。中途异常的attempt不表示每行一定全部完成，不能把sample_*_attempts写成已完成FLOPs。求解和Cholesky尝试另外计数，初始状态只算一次，接受状态复用。

实际检查max_updates=1、独立最优点作初始参数、η=100首步被拒绝的情况。正确结果分别可能是预算耗尽、零次更新的梯度停止、零次提交但两次状态尝试。若没有第二轮，second_update须为null。无论普通还是-O，CLI都必须正常结束或明确失败，不能先写文件再因访问不存在的trace[2]崩溃。

## 4 统计存在性与优化停止分开

主例有独立有限解证书：对称性给b=0，再令t=eʷ得到t³−t−2=0。自己在t∈(1,2)二分，计算w=log t，并核对完整二维梯度。不要把库的输出再拿来证明库正确。

分离数据必须读取另一个separable.csv，不能偷偷换掉主例标签。沿θ=(0,s)，分别运行s=0、1、2、4、8、16、32、64，记录损失、梯度范数、参数范数、最小正确margin和Hessian特征值。损失和梯度越来越小，参数却继续增大。证明无有限MLE所需的是这条极限序列和所有有限参数的损失严格为正，而不是“某次优化没收敛”。

添加λ=.25斜率L2后，Newton斜率约.2905219321147506。截距不受惩罚。用只有0或只有1的标签调用fit应在任何内部状态计算前拒绝；解释为什么单类数据仍能用无穷截距逃避损失。常量特征配两类标签则可以有合法但不可识别的未正则模型，不能把它和单类问题混为一谈。

![图6 分离数据：小梯度可以与发散方向同时出现。](figures/06_complete_separation.png)

## 5 真实库、阈值与稳定性核验

对照scikit-learn1.8的lbfgs、无样本权重、fit_intercept=True。未正则C=np.inf；λ=.25、n=4时C=1/(nλ)=1。复制全部行以后C=.5，拟合斜率应保持；只复制数据不改C则改变了本讲λ，不能当一致性失败。

读取classes_再取正类概率列，核对decision_function、coef_、intercept_、n_iter_、所有warnings以及重新计算的本讲完整目标梯度。当前真实未正则路径发出忽略C与l1_ratio的UserWarning；记录它，不把它说成ConvergenceWarning，也不凭返回系数宣称无数值问题。

固定拟合概率，分别使用τ=.2、.5、.8。四行三个准确率恰好都为1/2。另设完全零得分的受控probe：自己的p≥.5规则判1，库默认predict判0。这个probe不是拟合模型近零截距的预测，不应混淆。

数值实验分两层：三个非驻点、五种h的中心差分验证梯度与Hv；再验证z=±40、±1000的loss、gradient、curvature。正确类z=40的损失、导数、曲率仍是可表示的小正/负尾项；z=1000处尾项不可表示时可以下溢为0，但必须明示，不写成有限参数的数学零损失。audit使用1100位Decimal保留e⁻¹⁰⁰⁰再与binary64表示比较。

## 6 交付与自查标准

提交原始输入副本、配置、完整result.json、audit.json、真实执行Notebook和简短实验解释。至少回答：

- 两次更新中每行前向、损失和导数怎样进入同一个同步更新？
- 主例为什么有有限MLE，分离例为什么没有？
- 为什么训练准确率一半仍能看到似然改善？
- λ、n、C怎样换算，截距是否惩罚？
- 相消、下溢和统计不存在是三类什么不同的故障？
- 正常停止、预算耗尽、拒绝候选和异常尝试怎样保留真实成本？

audit不仅看默认结果，还用54个坏原始值组合、每个配置字段、重复JSON键、单类/形状错误、输出符号链接/硬链接与原子失败进行验证。不要用assert承担用户输入保护，因为python -O会删除assert。输出仅允许JSON，不得覆盖输入或教材资产；序列化在创建输出目录之前完成，受控replace失败必须保留旧字节并清理临时文件。
