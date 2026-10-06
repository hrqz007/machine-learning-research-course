# 第056讲 原始来源与复现边界

访问日期2026-10-06。课程推导、代码、数值和图为原创；没有复制原站示例代码。

1. [JMLR Cawley and Talbot 2010](https://www.jmlr.org/papers/v11/cawley10a.html)。期刊摘要与书目信息已读；未将全文下载冒称完整阅读。

2. [scikit-learn nested CV](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)。正文与示例方法已读。

3. [scikit-learn threshold tuning](https://scikit-learn.org/stable/modules/classification_threshold.html)。正文中阈值与验证说明已读。

4. [scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)。预处理和泄漏章节已读。

5. [SciPy bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)。方法和paired参数说明已读。

官方stable网页显示版本可能高于实际安装版本；本包用scikit-learn1.8.0，未调用较新阈值估计器。Wilson区间公式在代码中直接实现，逐个区间不作同时推断。
