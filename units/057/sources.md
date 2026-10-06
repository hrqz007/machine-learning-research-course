# 第057讲 原始来源核对

访问日期2026-10-06。使用公开官方资料核对概念；图、推导、代码、数据和数值为原创。官网stable网页当前显示1.9.1，本包实际scikit-learn版本以requirements.txt为准，不依赖较新接口。

1. [scikit-learn Nearest Neighbors](https://scikit-learn.org/stable/modules/neighbors.html)。已读分类、回归、距离加权、暴力搜索与树搜索及维度影响章节。

2. [scikit-learn Nearest Neighbors Classification](https://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html)。已读示例的训练内StandardScaler管道、两种权重与决策区域解释。

3. [scikit-learn StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)。已核对训练均值尺度、常数列scale=1、异常值敏感性和ddof=0约定。

4. [scikit-learn Bias Variance Decomposition](https://scikit-learn.org/stable/auto_examples/ensemble/plot_bias_variance.html)。已读跨独立训练集的偏差方差和新观测噪声解释；本包原创40次近邻实验，未复制其树模型示例。

本包的稳定同距排序和二分类平局为教学实现明确约定；参考库对照在无距离平局的随机数据上进行，不能扩大为所有边界输入实现完全一致。
