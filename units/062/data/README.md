# 原创数据说明

所有数据固定种子生成，不含个人或外部数据。训练、验证、测试分别160、100、220行，种子62001、62002、62003。输入x均匀分布于[-3,3]。回归均值为sin(1.6x)+0.25x，独立高斯噪声标准差0.16。分类概率为sigmoid(1.6sin(x)+0.35x)，y_class按此概率独立Bernoulli抽样。

id只作追踪；x为唯一模型输入。y_reg/y_class是两种任务的目标。true_mean与true_probability仅用于解释合成图，不参与拟合或选择。generation.json记录SHA-256；common.load_data同时核验生成器原始字节。
