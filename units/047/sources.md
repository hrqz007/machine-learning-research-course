# 047 原始资料核对

核对日期：2026-10-05。以下用于核对定义、前提和库约定；全部数据、图和推导叙事为原创教学材料，没有复制论文图或使用外部数据集。

1. [Guo, Pleiss, Sun, Weinberger (2017), On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html)，[原论文PDF](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf)。核对最高类别置信度校准、ECE与正温度定义。本文使用独立合成模型，不移植论文CNN效果结论。
2. [Angelopoulos, Bates, A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification](https://arxiv.org/abs/2107.07511)，[PDF](https://arxiv.org/pdf/2107.07511)，核对第1节及附录D的独立校准、修正顺序统计量、边际覆盖与无穷情况。本文明确区分无并列精确秩覆盖和有并列下界。
3. [scikit-learn 1.8 calibration](https://scikit-learn.org/1.8/modules/calibration.html)。核对可靠性曲线与评分不只衡量校准这一边界；本文ECE为自定义显式实现。
4. [scikit-learn 1.8 brier_score_loss](https://scikit-learn.org/1.8/modules/generated/sklearn.metrics.brier_score_loss.html)。实际调用`pos_label=1, scale_by_half=True`确认二元0至1约定。
5. [NumPy 2.3 quantile](https://numpy.org/doc/2.3/reference/generated/numpy.quantile.html)。实际比较`linear`、`higher`位置约定；科学函数直接选择第k小值。

本讲额外独立参照：Decimal100位四行导数链；四行解析最优；SciPy有界标量最小化；中心化标量OLS重算120次实验；所有十种留出秩的有限枚举；实际sklearn 1近邻重现明确标记为错误的校准复用反例。库调用验证实现，不证明真实数据满足统计假设。
