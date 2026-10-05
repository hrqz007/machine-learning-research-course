# ML044 原创合成数据

这些数据不对应真实个人、机构或业务。目的为逐行演算和有限机制实验，不代表现实预测效果。

- tiny.csv：固定纸笔例，3行。x=(-1,0,1)，曝光量均为1，计数y=(0,1,3)。参数顺序为截距b、斜率w。
- simulated.csv：560行，前320行train，后240行test。行号与split不进入特征。数据按protocol.json唯一一次生成。
- protocol.json：固定种子4402026，x均匀分布于[-2,2)，曝光量e均匀分布于[.5,2)，真率exp(.2+1.1x)。Poisson计数来自Poisson(e*真率)。另逐行独立抽Gamma(shape=2,scale=.5)乘子u，再抽Poisson(e*真率*u)，得到条件方差μ+μ²/2的过度离散计数。

CSV字段：row_id是唯一索引；split是固定训练/测试划分；x是无量纲可观察特征；exposure是正观察时长，单位为同一种标准时段；true_mean是生成机制的条件期望，只用于教学解释与画真曲线，绝不进入拟合；poisson_y和overdispersed_y是两种情景的非负整数观测值。每种情景只使用其本身的y列。

同一行的两种y共享x、e，但由不同抽样步骤产生，不能把它们当两份独立设计样本。所有行在各自情景内按条件独立生成。训练不看测试y，参数与拟合设置没有根据测试指标选择。测试图和评分在训练完成以后生成。

运行 `python generate_data.py --out outputs/regenerated.csv` 可重新生成。输出必须是新文件，不覆盖归档数据。数据随机数调用顺序为x、e、Poisson计数、Gamma乘子、混合Poisson计数。以归档CSV为跨版本权威输入；新版本随机数算法若变化，不假定字节相同。data_integrity.json保存三份输入文件SHA256，主程序先核验再读取。
