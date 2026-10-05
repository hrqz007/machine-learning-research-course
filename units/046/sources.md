# 046 来源与协议

核对日期：2026年10月5日。代码实际使用 scikit-learn 1.8.0，文档固定到1.8版本。讲义的所有具体样本、分数、逐行推导、敏感性权重和图像均为课程原创计算，不来自真实个人或机构记录。

1. [ROC曲线官方接口](https://scikit-learn.org/1.8/modules/generated/sklearn.metrics.roc_curve.html)
   用于核对 `pos_label`、`sample_weight`、`drop_intermediate=False`、下降阈值、评分大于等于阈值和正无穷起点。讲义采用全点输出以保留教学扫描细节。
2. [PR曲线官方接口](https://scikit-learn.org/1.8/modules/generated/sklearn.metrics.precision_recall_curve.html)
   用于核对升序有限阈值、坐标比阈值多一项、末尾 precision=1/recall=0 且无阈值的画图端点。标签1为阳性。
3. [Average Precision官方接口](https://scikit-learn.org/1.8/modules/generated/sklearn.metrics.average_precision_score.html)
   用于核对召回增量加权、非线性插值的AP约定及其与梯形PR面积的区别。数值由独立Fraction参照与库双路径计算。
4. [Precision Recall Fscore Support官方接口](https://scikit-learn.org/1.8/modules/generated/sklearn.metrics.precision_recall_fscore_support.html)
   用于核对 `average`、`labels`、`pos_label`、`zero_division`、支持度与每类输出。报告使用 `labels=[0,1]` 和 `zero_division=0`。
5. [混淆矩阵官方接口](https://scikit-learn.org/1.8/modules/generated/sklearn.metrics.confusion_matrix.html)
   用于核对行是真实、列是预测的轴约定。课程保存原始计数，不默认归一化。
6. [Log loss官方接口](https://scikit-learn.org/1.8/modules/generated/sklearn.metrics.log_loss.html)
   用于核对自然对数概率损失的实际调用。条件期望的导数在讲义中逐步推导，不是借此引入训练任务。
7. [Brier score官方接口](https://scikit-learn.org/1.8/modules/generated/sklearn.metrics.brier_score_loss.html)
   二分类调用显式 `pos_label=1, scale_by_half=True`，对应单个阳性概率的均方误差。多分类尺度差异不在本讲包装函数支持范围。

## 数据来源与复现实质

`data/review_scores.csv` 是固定12行合成零件复核案例，整数评分百分位便于精确参照。无随机生成、无外部下载、无抽样重试、无模型拟合。C与D同分且标签不同，保留并列语义。数据SHA256位于 `data_integrity.json`。

阳性率变化通过对真实类别统一赋权实现；保持每类内部的经验评分分布，分别设目标阳性率0.10、0.25、0.50。该设定只支持受控先验敏感性，不保证真实部署类条件分布不变。概率立方变换也只是保序反例，不是训练或校准方法推荐。

## 核验边界

手算Fraction参照使用整数评分、完整阈值枚举、阳性阴性配对与明确半分并列。完整自检另外穷举4行、两类标签、三级评分的1134种配置，并检验合理域内的端点和输入错误。课程自检结果不是独立发布验收，也不证明真实总体的泛化能力。
