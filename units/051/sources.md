# 051 原始资料与实现核对

核对日期2026-10-05。以下提供理论和接口依据；四行手算、固定协议、合成数据、代码、图和数值结果为本讲重新制作，没有复制外部数据集或论文插图。

1. [Cawley与Talbot2010，On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation](https://jmlr.org/papers/v11/cawley10a.html)，[原PDF](https://jmlr.org/papers/volume11/cawley10a/cawley10a.pdf)。核对选择指标本身会过拟合，以及评价应包含完整选择过程。
2. [scikit-learn1.8 Nested versus non-nested cross-validation](https://scikit-learn.org/1.8/auto_examples/model_selection/plot_nested_cross_validation_iris.html)。核对内层搜索、外层评价的数据角色；本文不使用其Iris分数。
3. [Ridge](https://scikit-learn.org/1.8/modules/generated/sklearn.linear_model.Ridge.html)。实际独立实现按alpha=n_train×lambda对应平均惩罚目标，截距不惩罚。
4. [KFold](https://scikit-learn.org/1.8/modules/generated/sklearn.model_selection.KFold.html)、[GroupKFold](https://scikit-learn.org/1.8/modules/generated/sklearn.model_selection.GroupKFold.html)、[TimeSeriesSplit](https://scikit-learn.org/1.8/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)。核对折大小、组约束与gap语义，并实际保存各个splitter返回的索引。

独立科学参照：Fraction纸笔内层选择与三次前向；SciPy特殊函数构造基底、带主元QR重新核对全部81数据集内外层候选SSE；实际sklearn Ridge核对选中模型；实际GridSearchCV核对主数据五轮内层与最终选参；独立24点Gauss-Legendre求积核对同模型总体MSE。变换某轮外层标签及诊断oracle，分别检查相应选择的不变性。

以上核验用于发现实现错误，不把模拟结果变成无条件统计保证。
