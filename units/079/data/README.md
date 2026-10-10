# 数据字典与生成规则

correlated.json：一个总和读数x=[1]；A=[[1,1]]；prior_mean=[0,0]；prior_cov为2×2单位阵；noise_cov=[[0.25]]。所有量无量纲，0.25是噪声方差，标准差0.5。模型为z~N(m0,S0)、x|z~N(Az,R)，未知变量为两个贡献。

independent.json：两个分开读数x=[1,-1]，A为单位阵，噪声协方差0.25I，后验独立，用作族包含真值的对照。

没有现实个人数据、随机采样或下载。generate_data.py完整重建两文件；generation.json保存实际文件字节数与SHA256。实验读取本data目录，默认数据重建写outputs/generated-data以防覆盖。
