# 第058讲 来源与适用范围

访问核验日期：2026-10-06。实验库版本为scikit-learn 1.8.0；优先引用对应1.8文档。代码、数值推导、合成数据、手算示例和图片均为本讲自行编写。没有复制外部教材章节或声称复现论文实验。

1. [scikit-learn 1.8：Decision Trees](https://scikit-learn.org/1.8/modules/tree.html)。核验数值分裂、不纯度、叶频率、贪心过程和实现输入精度；本讲的严格正收益停止和平局规则另行明确，不声称逐位复制库内部行为。
2. [scikit-learn 1.8：DecisionTreeRegressor](https://scikit-learn.org/1.8/modules/generated/sklearn.tree.DecisionTreeRegressor.html)。核验squared_error准则、均值叶值及深度/最小叶样本参数。公式手推以本讲给定六行数据独立演算。
3. [scikit-learn 1.8：Understanding the decision tree structure](https://scikit-learn.org/1.8/auto_examples/tree/plot_unveil_tree_structure.html)。核验树结构、节点样本、子节点和路径的检查思路。本讲存储格式是独立的教学JSON，并非库内部序列化格式。
4. [scikit-learn 1.8：Common pitfalls and recommended practices](https://scikit-learn.org/1.8/common_pitfalls.html)。核验训练/测试隔离与数据泄漏的定义。所有测试文件都在拟合与协议冻结后读取，没有测试后调参。

补充：曾打开stable版本的分类器文档用于发现入口，但该页面现为1.9.1，不作为1.8实验行为的单独依据。1.8分类器API页面在本次网页工具中两次访问失败；相应内容改由可访问的1.8树指南以及本地1.8.0实际执行对照核验，未把失败记作成功。访问详情见source-checks.json。

本讲只证明当前教学实现的可检查性质，未声称提供新的算法、真实场景泛化结论或因果识别结果。
