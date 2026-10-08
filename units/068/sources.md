# 一级资料与核对范围

检索日期2026-10-08。讲解与手算为原创，参考以下官方资料核对API和方法边界。

1. [scikit-learn 1.8 交叉验证](https://scikit-learn.org/1.8/modules/cross_validation.html)：训练、验证、测试的信息分工与Pipeline预处理隔离。
2. [scikit-learn 1.8 概率校准](https://scikit-learn.org/1.8/modules/calibration.html)：可靠性图、概率校准与区分能力、校准器独立数据要求。
3. [scikit-learn 1.8 log_loss](https://scikit-learn.org/1.8/modules/generated/sklearn.metrics.log_loss.html)：二分类自然对数损失与机器精度裁剪。
4. [scikit-learn 1.8 RandomForestClassifier](https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.RandomForestClassifier.html)：树数、深度、叶样本数及概率预测语义。

Wilson区间公式与配对百分位bootstrap在本讲完整列出并用代码独立复核；区间未声称覆盖重新选参和数据漂移。不引用第三方排行榜作为方法优劣证据。
