# 049 原始资料与核对范围

检索核对日期2026-10-05。理论定义与边界来自以下原始论文/高校课程资料；八状态总体、六行手算、追加批次、全部数据/代码/十图与数值结果均为原创教学构造。

1. [Hoeffding, W. (1963), Probability Inequalities for Sums of Bounded Random Variables](https://www.cs.rpi.edu/academics/courses/spring06/random/hoefding.pdf), Journal of the American Statistical Association 58(301),13–30。已阅读大学课程托管的原论文扫描页13–16，核对指数矩方法、Theorem1的[0,1]有界变量及Theorem2的一般范围。讲义自行推导学习损失的双侧形式，未复制论文图或段落。
2. [Balcan, Foundations of Machine Learning and Data Science, September16,2015](https://www.cs.cmu.edu/~ninamf/courses/806/lect-09-16.pdf)。核对有限可实现类的一致学习器样本复杂度、PAC量词、阈值和区间的增长/打散定义。
3. [Póczos, CMU10-715, Vapnik–Chervonenkis Theory](https://www.cs.cmu.edu/~epxing/Class/10715/lectures/VCTheory.pdf)。核对有限类union与无限类增长函数/对称化处理的不同。本文没有直接把样本模式数当成事前固定M。

作者独立数值参照包括：Fraction直接求总体风险；4096条有序序列枚举与组合计数交叉核对；每个规则的精确二项尾概率；80位Decimal界半径；100位Decimal解析最小值分布与实际SciPy二项CDF/SF；全部14000次主计数及6000次噪声/留出随机流重建。

初次核验中，朴素生存概率幂相减在一个极小PMF点损失精度。独立SciPy表达改为互补CDF配合log1p/expm1后解决，未放宽容差，原科学结果不变。统计抽样差异作为诊断保留，不依据它改变协议或随机种子。
