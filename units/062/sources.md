# 一级资料与核验范围

核验日期：2026-10-07。公式、手算数据、合成数据、图与实现独立编写。

- [Friedman 2001 journal publication record](https://doi.org/10.1214/aos/1013203451)。状态：retrieved_metadata_only。核对范围：publication identification; full page iframe did not expose article body。

- [Friedman original manuscript mirrored by IIT Bombay](https://www.cse.iitb.ac.in/~soumen/readings/papers/Friedman1999GreedyFuncApprox.pdf)。状态：retrieved。核对范围：Algorithm 1, negative functional gradient, half-squared loss; preprint pagination differs from journal。

- [GradientBoostingRegressor 1.8.0 API](https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)。状态：retrieved。核对范围：frozen version parameters。

- [GradientBoostingClassifier 1.8.0 API](https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)。状态：retrieved。核对范围：frozen version classifier interface。

- [scikit-learn 1.8.0 implementation](https://github.com/scikit-learn/scikit-learn/blob/1.8.0/sklearn/ensemble/_gb.py)。状态：retrieved。核对范围：full-logit versus half-logit note and Newton leaf numerator/denominator。

库版本固定1.8.0，不依赖滚动stable文档的未来变化。资料检索和代码运行是两种独立证据；完整记录见source-checks.json。
