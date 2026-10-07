# 一级资料与核验范围

核验日期：2026-10-07。公式、手算数据、合成数据、图与实现独立编写。

- [Freund and Schapire AdaBoost original author-hosted paper](https://www.schapire.net/papers/FreundSc95.pdf)。状态：retrieved。核对范围：boosting weighted distribution and training-error bound; author-hosted manuscript states 1997 publication。

- [AdaBoostClassifier 1.8.0 API](https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)。状态：retrieved。核对范围：frozen version parameters and estimator behavior。

- [scikit-learn 1.8.0 implementation](https://github.com/scikit-learn/scikit-learn/blob/1.8.0/sklearn/ensemble/_weight_boosting.py)。状态：retrieved。核对范围：SAMME coefficient and sample-weight update。

库版本固定1.8.0，不依赖滚动stable文档的未来变化。资料检索和代码运行是两种独立证据；完整记录见source-checks.json。
