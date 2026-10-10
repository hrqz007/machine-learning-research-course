# 一手来源与核验范围

核验日期：2026-10-10。技术定义优先采用项目官方文档；公式推导、教学例子、合成机制、测试与插图为课程原创。在线stable文档核验时指向1.9.1，实际冻结运行使用scikit-learn 1.8.0；只使用两者共有的公开接口，并以本地执行结果为准，不声称实跑1.9.1。

1. scikit-learn，Novelty and Outlier Detection：https://scikit-learn.org/stable/modules/outlier_detection.html 。核对新颖性与离群训练条件、LOF新样本评分注意事项。讲义不沿用可能被误读为未来概率保证的简略措辞；ν解释以具体API条目为准。
2. scikit-learn，LocalOutlierFactor：https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html 。核对novelty=True、score_samples及negative_outlier_factor_方向；LOF局部相对密度机制由官方用户指南交叉核对。原始研究为Breunig等（2000），LOF: identifying density-based local outliers，DOI 10.1145/342009.335388；本次出版商页面无法读取，未宣称逐页核验原论文。
3. scikit-learn，OneClassSVM：https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html 。核对ν训练错误上界/支持向量下界、gamma与score_samples偏移关系。独立干净校准阈值为本课程实现。
4. Angelopoulos, A. N. & Bates, S. A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification：https://arxiv.org/abs/2107.07511 。核验论文身份与主题；本课有限样本秩论证在正文独立给出，不声称该论文替真实Digits保证可交换性。
5. scikit-learn，Density Estimation：https://scikit-learn.org/stable/modules/density.html 。用于区分密度估计与决策分数；单高斯与近邻体积解释为课程推导。
6. scikit-learn，load_digits：https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html 。核对1797行、64维、像素0至16以及来自UCI原测试部分。
7. Alpaydin, E. & Kaynak, C. (1998). Optical Recognition of Handwritten Digits. UCI Machine Learning Repository：https://archive.ics.uci.edu/dataset/80/optical%2Brecognition%2Bof%2Bhandwritten%2Bdigits 。DOI https://doi.org/10.24432/C50P49 。核对来源、原数据规模、预处理背景与CC BY 4.0许可。本课选类、重新划分与像素归一化的变化记录在data/README.md。

本课不复制外部图表，也不将官方演示成绩作为本课程成绩。所有实验数值来自experiment-result.json。来源链接用于追溯定义与数据，不表示对未来软件版本或现实部署作出保证。
