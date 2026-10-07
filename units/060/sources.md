# 一手阅读来源

访问日期2026-10-07。软件接口固定scikit-learn1.8.0。正文推导、手算例子、数据、图表、实现和数值结果为原创，以下资料用于核对方法与接口。

1. BaggingClassifier：https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.BaggingClassifier.html 。核对bootstrap、概率聚合与OOB。
2. RandomForestClassifier：https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.RandomForestClassifier.html 。核对每节点max_features、树数、bootstrap、OOB及接口边界。
3. Breiman, Random Forests, 2001，作者发布论文：https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf 。阅读随机化、基学习器能力与相关性思想。本文等相关方差例子自行推导，不把简化公式冒充论文对所有分类森林的精确性能定律。
4. 官方OOB曲线例子：https://scikit-learn.org/1.8/auto_examples/ensemble/plot_ensemble_oob.html 。核对随树数观察OOB的用法，本文数据和代码另行编写。
5. 固定版本森林实现：https://github.com/scikit-learn/scikit-learn/blob/1.8.0/sklearn/ensemble/_forest.py 。核对bootstrap成员记录、计数样本权重与概率聚合实现。自建版本显式复制抽样行，不能承诺与库内部流程逐项相同。
6. 群组交叉验证：https://scikit-learn.org/1.8/modules/cross_validation.html#cross-validation-iterators-for-grouped-data 。核对独立单位与组隔离原则。
7. 官方集成学习概述：https://scikit-learn.org/1.8/modules/ensemble.html#bagging-meta-estimator 。用于定位Bagging、随机森林与其他集成机制。

未复制外部教材章节或实验图。不将本讲合成性能归因于上述来源，也不据此承诺真实科学任务性能。
