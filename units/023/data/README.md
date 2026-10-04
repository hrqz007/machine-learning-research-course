# 本讲的合成数据

所有内容均为人工设计的教学例子，不含真实设备、个人或业务数据。

- bernoulli.csv：8个不同的观测ID，value为(1,0,0,1,0,1,0,0)。模型假设它们是iid Bernoulli(p)；这里的ID不证明现实独立性。
- gaussian.csv：4个不同ID，value为(0,1,2,5)，单位是任意教学单位。用于拟合均值与正方差Gaussian，SSE为14。
- model_spec.json：独立的重复采样实验，不是从两份CSV训练出的现实数据生成模型。kind必须为synthetic_mle_repeated_sampling；seed=23023；repetitions=4000；sizes=[4,16,64]；p=0.375；mu=2；variance=3.5。variance字段是方差，生成Gaussian时取其平方根作为标准差。

每个n分别初始化Generator(PCG64(seed+n))。同一随机流先生成形状(R,n)的Bernoulli数据，再生成Gaussian数据。不同n没有共享前缀。R指重复批数，一行原始随机矩阵是一批完整实验；对每批求一次估计。

CSV输入：表头恰为observation_id,value。每条记录恰有两列，ID需唯一、无空格、为1至40字符的ASCII字母开头标识。值是无首尾空格的有限十进制文本，可有科学记数法；拒绝NaN、Infinity、空字段、额外列与无法表示的极小/极大数。数值API直接接收数组时拒绝文本，专用CSV解析层才负责合法十进制转换。

程序生成的replicate_estimates.csv字段：n是每批样本数；replicate是该n下从1开始的批编号；p_mle、mu_mle、variance_mle、variance_unbiased分别是该批估计。默认共12000行，不含原始随机矩阵。mean_variance_mle是这些批估计的平均，不是把全部观测合在一起拟合出的方差。

输入更改后用--bernoulli、--gaussian、--spec指定另存文件；程序输出以实际输入重算。Notebook为了防止图题与数据错配，会要求默认数据和配置。不要更改源数据后仍把保留输出当成新实验的结果。
