# 第071课资料来源

核验日期2026-10-08。仅使用官方文档与作者或会议公开的原始论文。网页中的最新版本与实际执行的固定依赖版本可能不同；关键接口参数在代码中显式设置。手算推导与合成实验结果独立生成。

1. [KMeans official API](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
   核验内容：explicit n_init; init and stopping semantics。

2. [Arthur and Vassilvitskii 2007 original paper](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf)
   核验内容：D-squared initialization; expected approximation bound。

3. [scikit-learn clustering guide](https://scikit-learn.org/stable/modules/clustering.html#k-means)
   核验内容：geometry assumptions and local optima。
