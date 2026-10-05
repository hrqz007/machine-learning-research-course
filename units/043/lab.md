# 第043讲 独立实验指南

## 1 实验目的与交付物

本实验用八行三类数据，验证softmax、Categorical负对数似然、逐样本反向链、两次同步更新以及熵与KL的关系。先写下预期再运行程序。不要把“没有异常”当科学验收，也不要把训练正确率当概率拟合是否正确的唯一依据。

完成后保留自己的手算纸、outputs/result.json、outputs/audit.json、从新内核运行的Notebook以及一段失败解释。正常输出之外还必须保留BFGS的precision-loss状态和只有1/2训练准确率的结果。本实验没有训练/验证/测试调参环节，八行全部用于机制学习；它不能评价现实泛化。

## 2 安装与进入目录

建议使用Python 3.12独立环境。Anaconda用户可在单元目录运行：

```bash
conda env create -f environment.yml
conda activate ml043-multiclass
jupyter lab
```

已有合适的Python环境，也可以运行：

```bash
python -m pip install -r requirements.txt
python experiment.py --out outputs/result.json
python audit.py --out outputs/audit.json
```

这份环境文件是重建入口，不是所有操作系统安装成功的保证。实际版本和执行方式见verification.json。PDF、图、Markdown可直接阅读，不需要先安装。PDF重建依赖另列build-requirements.txt和系统工具说明，不要求学习者为了运行数值实验安装排版软件。

先确认当前目录确有experiment.py及data/observations.csv。脚本默认输入相对于自身文件定位，所以可以从其他工作目录用绝对路径运行。Notebook则按本讲目录启动；首格检查数据、源码和环境契约后才导入数值库。

## 3 认识数据与列顺序

打开data/observations.csv，记录八行的id、x、y。x=−1的标签为0、0、1、2，x=1的标签为0、1、2、2。所有值均原创固定，无下载、随机抽样或隐含标签清洗。

|对象|形状|轴的含义|
|---|---|---|
|X|(8,1)|行是样本，列是唯一特征|
|y|(8,)|整数类别索引0、1、2|
|A|(8,2)|第0列全1，第1列是x|
|Theta|(2,3)|第0行b，第1行w；列顺序甲乙丙|
|Z、P、T|(8,3)|每行三个得分、概率或one-hot|
|gradient|(2,3)|与Theta一一对应|

请在纸上把M1的A行写成(1,−1)，M5写成(1,1)。问自己：如果把类别列改成丙、乙、甲，标签索引是否也必须一起改？必须，否则是另一个错误任务。

## 4 不运行优化器的归一化练习

在Notebook导入之后单独运行：

```python
p, logp = e.stable_softmax([[np.log(2), 0, np.log(3)]])
print(p)
print(-logp[0, 1])
```

预期p=(1/3,1/6,1/2)，真实类乙的损失log6。再给三个得分都加1000，比较概率最大绝对误差；使用容差判断，不要求任意平移逐bit不变。输入在进入浮点前可能已经丢失差异，任何后续算法都不能恢复那些未被表示的差异。

将两个样本组成(2,3)得分矩阵，对照SciPy的axis=1与axis=None。后者虽然全表之和为1，每行的和一般不等于1，不能当逐样本类别概率。记录错误发生在哪个轴，而不是只说“归一化有问题”。

## 5 手算第一轮再读代码

初始Theta全0，全部概率1/3。每行损失log3，真实类得分导数−2/3，另两类1/3。把每行得分梯度除8得到b贡献，乘x/8得到w贡献。八行求和应为

$$g_b=(-1/24,1/12,-1/24),\quad g_w=(1/8,0,-1/8).$$

学习率0.6后，b=(0.025,−0.05,0.025)，w=(−0.075,0,0.075)。阅读experiment.py的_state函数：A怎样补常数列，A@theta怎样形成Z，loss怎样取真实类别，gradient怎样用A.T@residual/n求和。每一步都要对应自己的纸笔对象。

M1与M2是两条有独立行身份但数值相同的教学记录。不要因为它们完全相同就删掉一行；删掉会改变类别计数和目标。真实数据中重复究竟是有效重复测量还是采集错误，要靠任务定义判断。

## 6 连续做两次同步更新

运行以下片段，暂时不用fit的返回轨迹：

```python
theta = np.array(cfg['initial_theta'])
hand = []
for k in range(3):
    state = e.row_ledger(X, y, theta)
    hand.append(state)
    print(k, state['theta'], state['objective'], state['gradient'])
    if k < 2:
        theta = theta - cfg['learning_rate'] * np.array(state['gradient'])
```

预期平均损失依次约1.098612288668110、1.076152315758658、1.062011537613334。第二次新b约(0.044881634194,−0.089763268388,0.044881634194)，w约(−0.134644902581,0,0.134644902581)。

然后打印每个state中的rows。每行需要核对logits、probabilities、log_probabilities、loss、local_dloss_dp、local_softmax_jacobian、stable_dloss_dz及gradient_contribution。全部行的gradient_contribution求和应等于该状态总gradient。把最后一个前向遗漏，会使“两轮计算”实际上只有一轮更新后的检验。

![实验图1 八行的三次前向。先解释两组损失为什么分开，再判断平均损失是否下降。](figures/04_two_round_rows.png)

## 7 有限最优点与库返回值

读取certificate，自己将 $b^*,w^*$ 代回八行，核验x=−1概率(0.5,0.25,0.25)、x=1概率(0.25,0.25,0.5)，平均损失1.5log2。这个参照来自分组频率，不依赖GD与BFGS互相同意。

默认GD预期85次更新达到梯度容差。BFGS预声明gtol为1e−11，实际返回success=False及precision loss。先保留message和迭代数，再检查reference_optimizer.state中的完整梯度与目标。数值接近证书可以支持“返回点接近最优”，却不能把库success字段改成True。

用最大概率类预测，训练准确率为1/2，乙类没有一次成为最大概率。请写一句话区分“概率模型正确实现”与“现实分类器足够好”。不要为了提高这八行训练正确率，改成每行记忆标签后仍称同一线性softmax模型。

## 8 信息论实验与零支持

读取information与reverse_information，按每一类展开三个求和。预期：H(q)≈1.029653014、H(q,p)≈1.121685864、D(q||p)≈0.092032850，反向≈0.085122826。

调用e.information([1,0,0],[0,.5,.5])，输出应标明交叉熵和KL为无穷，对应数值字段为null。调用同一个确定性分布与自身比较，熵和KL为0。这两个例子必须分开；不能用同一epsilon裁剪把它们混成有限近似。

![实验图2 逐渐缩小模型分给甲的概率。注意左图横轴为对数轴，图不包括精确的零边界。](figures/08_support_and_direction.png)

## 9 故意做错再定位

错误一，直接exp([1000,0,−1000])：预计溢出。错误二，先softmax再log：最小类可能下溢成0，log得到负无穷；直接log_softmax仍能返回有限log概率。错误三，输入p又做一次softmax：虽然行和仍为1，却是另一个分布。

错误四，只减max但直接log(sum(exp))：在(0,−40,−40)真实类为0时可能输出0损失，而本讲log1p路线保留约 $8.50\times10^{-18}$。本讲没有把所有极小值都变成非零；当真实值低于binary64表示范围，仍可能正常下溢。检查基准应该知道自己在验证数学还是浮点表示。

错误五，学习率设100：程序保留一个被拒绝的候选，报告step_rejected，不提交它。错误六，最大步数设0或1：合法返回当前或一步状态，second_update为null。为修改实验，复制配置到outputs/custom-config.json再用--config传入，不改原始契约文件。

## 10 完整重跑与科学结果比较

```bash
python experiment.py --out outputs/result.json
python -O experiment.py --out outputs/result-O.json
python audit.py --out outputs/audit.json
python -O audit.py --out outputs/audit-O.json
python plots.py --report outputs/result.json --directory outputs/figures
```

在新进程里比较两份完整JSON，而非只比打印摘要。作者实际在普通/-O及不同工作目录重跑。跨平台BLAS与数学库可能改变末位，所以公开验收先看容差与数学对象，不承诺所有机器JSON字节相同。

打开experiment.ipynb，重启内核并运行全部，确认每个代码格有执行计数、没有error输出、九张图是实际PNG。首格没有先导入experiment或展示旧图；它先核对绑定摘要。若源码或数据变了，先解释修改并重建Notebook，不能删除检查后仍宣称同一版已验收。

## 11 验收报告怎么写

报告按以下顺序写即可：问题与类别定义；固定数据及参数形状；两个完整更新；解析最优点；GD与BFGS状态；信息论分解和零支持；一个真实失败及原因；尚不能支持的结论。

必须写明：八行是机制例，没有独立泛化估计；模型概率合法不等于校准；默认BFGS不是成功状态；训练准确率只有1/2；Notebook实际验收方式不是浏览器UI和Anaconda跨平台安装。全部十八道题的计算与解释见answers.pdf。
