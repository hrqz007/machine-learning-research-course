# 第072课资料来源

核验日期2026-10-08。仅使用官方文档与作者或会议公开的原始论文。网页中的最新版本与实际执行的固定依赖版本可能不同；关键接口参数在代码中显式设置。手算推导与合成实验结果独立生成。

1. [SciPy linkage official API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html)
   核验内容：linkage definitions; Euclidean Ward; implementation complexity。

2. [DBSCAN official API](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
   核验内容：min_samples includes self; eps and memory semantics。

3. [Ester et al 1996 original paper](https://cdn.aaai.org/KDD/1996/KDD96-037.pdf)
   核验内容：core, direct density reachability and global threshold limitations。

4. [scikit-learn clustering guide](https://scikit-learn.org/stable/modules/clustering.html)
   核验内容：density and hierarchy implementation limitations。
