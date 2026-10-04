# 第025讲合成数据字典

全部为原创教学数值，不含真人、业务样本或私人数据。

- tiny_sample.csv：id为记录标识；value是任意同一单位的数值，默认[0,2]。只有这份CSV作为脚本--input输入。n<=6用于完全枚举n^n种有序索引样本。
- paired.csv：subject为同一个体的教学标识；before/after单位相同；四个增量均为1。这个极端例子用于精确区分配对与独立索引，不是实际干预效果证据。Notebook以严格默认内容核验后使用。
- grouped.csv：group表示独立采样单位；replicate为组内测量号；value为同一单位。3组、每组2行、组内完全相关，用于手算。Notebook核验默认内容后使用；主覆盖模拟另有24组且组内包含独立噪声。
- model_spec.json：kind为固定场景标签；seed为非负32位种子；normal_repetitions=3000为Gaussian外层重复次数；group_repetitions=1000为组模型外层次数；B=399为每次内层重采样次数；n=20为独立Gaussian样本量；G=24为独立组数；m=6为每组行数；mu=3为均值；sigma=0.5为独立测量噪声标准差；tau=2为组效应标准差。值的单位任意但一致。

Gaussian实验X_i~N(mu,sigma²)；组实验X_gj=mu+A_g+epsilon_gj，A_g~N(0,tau²)，epsilon_gj~N(0,sigma²)，不同组效应和所有噪声相互独立。组内共享A_g。两个实验复用sigma字段但含义都为其独立噪声标准差。

受限数值域和预算见README。对修改后的数据不沿用默认答案。无联网、无付费API、无外部数据下载。运行只在指定输出目录生成report.json，不修改任何输入数据。
