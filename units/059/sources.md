# 阅读来源与原创范围

访问核对日期为2026-10-07，接口固定scikit-learn1.8.0。正文风险推导、计数例子、扰动方案、代码和图是原创教学工作；下列一手资料用于核对方法名称、接口与使用边界。

1. DecisionTreeClassifier参数文档：https://scikit-learn.org/1.8/modules/generated/sklearn.tree.DecisionTreeClassifier.html 。核对max_depth、min_samples_leaf、min_samples_split、ccp_alpha、float32输入及概率含义。
2. 最小代价复杂度剪枝：https://scikit-learn.org/1.8/modules/tree.html#minimal-cost-complexity-pruning 。核对根归一化叶不纯度、有效α及最弱环节方法。它搜索固定已生长树的根子树，不是所有可能树。
3. 官方剪枝示例：https://scikit-learn.org/1.8/auto_examples/tree/plot_cost_complexity_pruning.html 。核对cost_complexity_pruning_path输出与训练风险随剪枝变化。本文使用独立验证选择，未复制该示例的具体数据和数值。
4. 群组交叉验证：https://scikit-learn.org/1.8/modules/cross_validation.html#cross-validation-iterators-for-grouped-data 。核对相关对象必须以群组方式隔离的评价边界。

版本固定有助于复现；资料网站之后可能更新。本文未复制教材章节或外部实验图，不将本次合成结果归因于外部作者。
